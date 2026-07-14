"""Fail-closed inference for the hash-bound wind-v6 model bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CAPACITY_KWH, DATA_DIR, GROUPS, MODEL_DIR, SUBMISSION_DIR, TEST_DIR
from features import build_features
from postprocess import apply_post


RECIPE_KIND = "wind-v6-production-recipe"
RECIPE_SCHEMA_VERSION = 1
EXPECTED_RECIPE_HASH = (
    "9b89a7cb6e3bcef101da3f8e32c057278006e944176d416acac6fb37ca29870f"
)
EXPECTED_POSTPROCESS_HASH = (
    "698cb8dfd72cd3c1748c56822021c33a7fdc3847b2463b3693ed91cf642e369d"
)
EXPECTED_COLUMNS_HASH = (
    "393f05485537111054b534ee30521592325bcb9729bb182f4d4c6a5589a968c9"
)
EXPECTED_FEATURE_COUNT = 276
SEEDS = (42, 202, 777)
FAMILIES = ("kpx_group_1", "kpx_group_2", "pooled")
EXPECTED_MODEL_NAMES = {
    f"lgbm_v6_weighted_{family}_s{seed}.txt" for family in FAMILIES for seed in SEEDS
}
LEGACY_MODEL_FILES = (
    "lgbm_kpx_group_1.txt",
    "lgbm_kpx_group_1_s42.txt",
    "lgbm_kpx_group_1_s202.txt",
    "lgbm_kpx_group_1_s777.txt",
    "lgbm_kpx_group_2.txt",
    "lgbm_kpx_group_2_s42.txt",
    "lgbm_kpx_group_2_s202.txt",
    "lgbm_kpx_group_2_s777.txt",
    "lgbm_kpx_group_3.txt",
    "lgbm_pooled_s42.txt",
    "lgbm_pooled_s202.txt",
    "lgbm_pooled_s777.txt",
)
ROOT_RECIPE_KEYS = {
    "kind",
    "schema_version",
    "recipe",
    "promotion",
    "sources",
    "features",
    "targets",
    "training",
    "postprocess",
    "capacities",
    "versions",
}


class InferenceContractError(ValueError):
    """Raised before publishing output when any trusted input drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise InferenceContractError("recipe is not canonical JSON") from error


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_model_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise InferenceContractError("model path must be a basename")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path.name != value
        or path.suffix != ".txt"
        or ".." in path.parts
    ):
        raise InferenceContractError("model path must be a .txt basename")
    return value


def _require_keys(
    value: object, expected: set[str], field: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise InferenceContractError(f"{field} fields do not match schema")
    return value


def _validate_recipe(recipe: object) -> dict[str, object]:
    root = _require_keys(recipe, ROOT_RECIPE_KEYS, "recipe")
    if root["kind"] != RECIPE_KIND or root["schema_version"] != RECIPE_SCHEMA_VERSION:
        raise InferenceContractError("recipe identity does not match")
    if root["capacities"] != CAPACITY_KWH:
        raise InferenceContractError("capacity contract drifted")

    recipe_config = _require_keys(
        root["recipe"], {"name", "config", "config_sha256"}, "recipe config"
    )
    if (
        recipe_config["name"] != "v6-weighted-potential-q60-filter05-floor10"
        or recipe_config["config_sha256"] != EXPECTED_RECIPE_HASH
        or _canonical_hash(recipe_config["config"]) != EXPECTED_RECIPE_HASH
    ):
        raise InferenceContractError("weighted recipe hash drifted")

    promotion = _require_keys(
        root["promotion"],
        {
            "status",
            "artifact_sha256",
            "seeds",
            "mean_delta",
            "folds",
            "candidate_manifests",
            "fold24_best_iterations_sha256",
        },
        "promotion",
    )
    if promotion["status"] != "PASS" or promotion["seeds"] != list(SEEDS):
        raise InferenceContractError("promotion evidence does not pass")
    if not _is_sha256(promotion["artifact_sha256"]) or not _is_sha256(
        promotion["fold24_best_iterations_sha256"]
    ):
        raise InferenceContractError("promotion hashes are invalid")
    manifests = promotion["candidate_manifests"]
    if not isinstance(manifests, Mapping) or set(manifests) != {"fold23", "fold24"}:
        raise InferenceContractError("promotion manifests do not match schema")
    if not all(_is_sha256(value) for value in manifests.values()):
        raise InferenceContractError("promotion manifest hash is invalid")

    sources = _require_keys(root["sources"], {"training", "inference"}, "sources")
    inference_sources = sources["inference"]
    if not isinstance(inference_sources, Mapping) or set(inference_sources) != {
        "ldaps_test",
        "gfs_test",
        "sample_submission",
    }:
        raise InferenceContractError("inference sources do not match schema")
    expected_source_names = {
        "ldaps_test": "ldaps_test.csv",
        "gfs_test": "gfs_test.csv",
        "sample_submission": "sample_submission.csv",
    }
    for name, expected_path in expected_source_names.items():
        source = _require_keys(inference_sources[name], {"path", "sha256"}, name)
        if source["path"] != expected_path or not _is_sha256(source["sha256"]):
            raise InferenceContractError(f"{name} source declaration drifted")

    features = _require_keys(
        root["features"],
        {
            "path",
            "file_sha256",
            "count",
            "ordered_columns_sha256",
            "training_frame_sha256",
        },
        "features",
    )
    if (
        features["path"] != "feature_cols.txt"
        or features["count"] != EXPECTED_FEATURE_COUNT
        or features["ordered_columns_sha256"] != EXPECTED_COLUMNS_HASH
        or not _is_sha256(features["file_sha256"])
        or not _is_sha256(features["training_frame_sha256"])
    ):
        raise InferenceContractError("feature declaration drifted")

    postprocess = _require_keys(
        root["postprocess"],
        {"path", "file_sha256", "config", "config_sha256", "groups"},
        "postprocess",
    )
    if (
        postprocess["path"] != "post_params.json"
        or postprocess["config_sha256"] != EXPECTED_POSTPROCESS_HASH
        or _canonical_hash(postprocess["config"]) != EXPECTED_POSTPROCESS_HASH
        or not _is_sha256(postprocess["file_sha256"])
    ):
        raise InferenceContractError("postprocess declaration drifted")
    group_post = postprocess["groups"]
    if not isinstance(group_post, Mapping) or set(group_post) != set(GROUPS):
        raise InferenceContractError("postprocess groups drifted")
    for group in GROUPS:
        values = _require_keys(
            group_post[group], {"scale", "floor_ratio", "floor_kwh"}, group
        )
        if values != {
            "scale": 1.0,
            "floor_ratio": 0.10,
            "floor_kwh": 0.10 * CAPACITY_KWH[group],
        }:
            raise InferenceContractError(f"{group} postprocess drifted")

    training = _require_keys(
        root["training"],
        {
            "params",
            "seeds",
            "filter_ratio",
            "mismatch_frame_sha256",
            "fold24_manifest_key",
            "families",
        },
        "training",
    )
    if training["seeds"] != list(SEEDS) or not _is_sha256(
        training["fold24_manifest_key"]
    ):
        raise InferenceContractError("training identity drifted")
    families = training["families"]
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise InferenceContractError("model families drifted")
    declared_paths: list[str] = []
    for family in FAMILIES:
        family_record = _require_keys(
            families[family],
            {
                "row_count",
                "index_sha256",
                "target_sha256",
                "validation_iterations",
                "base_rounds",
                "full_rounds",
                "categorical_features",
                "models",
            },
            family,
        )
        expected_categorical = ["group_id"] if family == "pooled" else []
        if family_record["categorical_features"] != expected_categorical:
            raise InferenceContractError(f"{family} categorical feature drifted")
        iterations = family_record["validation_iterations"]
        if (
            not isinstance(iterations, list)
            or len(iterations) != len(SEEDS)
            or any(type(value) is not int or value <= 0 for value in iterations)
        ):
            raise InferenceContractError(f"{family} iterations are invalid")
        base_rounds = int(sum(iterations) / len(iterations))
        full_rounds = max(int(base_rounds * 1.2), 100)
        if (
            family_record["base_rounds"] != base_rounds
            or family_record["full_rounds"] != full_rounds
        ):
            raise InferenceContractError(f"{family} round policy drifted")
        models = family_record["models"]
        if not isinstance(models, list) or len(models) != len(SEEDS):
            raise InferenceContractError(f"{family} models do not match seeds")
        for model, seed in zip(models, SEEDS, strict=True):
            entry = _require_keys(
                model, {"path", "seed", "sha256", "feature_count", "rounds"}, "model"
            )
            path = validate_model_relative_path(entry["path"])
            if (
                entry["seed"] != seed
                or entry["sha256"] is None
                or not _is_sha256(entry["sha256"])
                or entry["rounds"] != full_rounds
                or entry["feature_count"]
                != (EXPECTED_FEATURE_COUNT + (1 if family == "pooled" else 0))
            ):
                raise InferenceContractError(f"{family} model declaration drifted")
            declared_paths.append(path)
    if (
        len(declared_paths) != len(set(declared_paths))
        or set(declared_paths) != EXPECTED_MODEL_NAMES
    ):
        raise InferenceContractError("active model path set drifted")
    return dict(root)


def _load_verified_recipe() -> dict[str, object]:
    recipe_path = MODEL_DIR / "recipe.json"
    sidecar_path = MODEL_DIR / "recipe.json.sha256"
    try:
        raw = recipe_path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="ascii")
    except OSError as error:
        raise InferenceContractError(
            f"production recipe is incomplete: {error}"
        ) from error
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar != f"{digest}  recipe.json\n":
        raise InferenceContractError("production recipe sidecar does not match")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceContractError("production recipe is malformed") from error
    if raw != _canonical_json_bytes(payload):
        raise InferenceContractError("production recipe bytes are not canonical")
    return _validate_recipe(payload)


def _verify_file(path: Path, expected_hash: object, field: str) -> None:
    if not _is_sha256(expected_hash) or _sha256_file(path) != expected_hash:
        raise InferenceContractError(f"{field} hash does not match")


def validate_submission_frame(
    frame: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    require_competition_shape: bool = False,
) -> pd.DataFrame:
    """Validate identity, order, finite predictions, floor and capacity bounds."""
    if list(frame.columns) != list(sample.columns):
        raise InferenceContractError("submission schema does not match sample")
    if len(frame) != len(sample):
        raise InferenceContractError("submission row count does not match sample")
    identity = ["forecast_id", "forecast_kst_dtm"]
    for column in identity:
        if not frame[column].astype(str).equals(sample[column].astype(str)):
            raise InferenceContractError("submission identity does not match sample")
        if frame[column].duplicated().any():
            raise InferenceContractError("submission identity is not unique")
    timestamps = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise InferenceContractError("submission timestamps are not strictly ordered")
    if (
        len(timestamps) > 1
        and not (timestamps.diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    ):
        raise InferenceContractError("submission timestamps are not hourly")
    if require_competition_shape:
        if len(frame) != 8_760:
            raise InferenceContractError("submission must contain 8,760 rows")
        if timestamps.iloc[0] != pd.Timestamp("2025-01-01 01:00:00") or timestamps.iloc[
            -1
        ] != pd.Timestamp("2026-01-01 00:00:00"):
            raise InferenceContractError("submission timestamp range drifted")
    validated = frame.copy()
    for group in GROUPS:
        numeric = pd.to_numeric(validated[group], errors="raise")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise InferenceContractError(f"{group} predictions are not finite")
        if (values < 0.0).any() or (values > CAPACITY_KWH[group]).any():
            raise InferenceContractError(f"{group} predictions violate bounds")
        if (values < 0.10 * CAPACITY_KWH[group]).any():
            raise InferenceContractError(f"{group} predictions violate floor")
        validated[group] = numeric
    return validated


def _load_models(
    recipe: Mapping[str, object], feature_columns: list[str]
) -> dict[str, list[lgb.Booster]]:
    families = recipe["training"]["families"]
    loaded: dict[str, list[lgb.Booster]] = {}
    for family in FAMILIES:
        expected_features = feature_columns + (
            ["group_id"] if family == "pooled" else []
        )
        models = []
        for entry in families[family]["models"]:
            path = MODEL_DIR / validate_model_relative_path(entry["path"])
            _verify_file(path, entry["sha256"], path.name)
            model = lgb.Booster(model_file=str(path))
            if model.num_feature() != len(expected_features):
                raise InferenceContractError(f"{path.name} feature count drifted")
            if tuple(model.feature_name()) != tuple(expected_features):
                raise InferenceContractError(f"{path.name} feature names drifted")
            if model.current_iteration() != entry["rounds"]:
                raise InferenceContractError(f"{path.name} rounds drifted")
            models.append(model)
        loaded[family] = models
    return loaded


def _ensemble_predict(models: list[lgb.Booster], features: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray([model.predict(features) for model in models], dtype=float)
    if (
        predictions.shape != (len(SEEDS), len(features))
        or not np.isfinite(predictions).all()
    ):
        raise InferenceContractError("model predictions are not finite or aligned")
    return predictions.mean(axis=0)


def main() -> None:
    recipe = _load_verified_recipe()
    sources = recipe["sources"]["inference"]
    source_paths = {
        "ldaps_test": TEST_DIR / "ldaps_test.csv",
        "gfs_test": TEST_DIR / "gfs_test.csv",
        "sample_submission": DATA_DIR / "sample_submission.csv",
    }
    for name, path in source_paths.items():
        _verify_file(path, sources[name]["sha256"], name)

    feature_record = recipe["features"]
    feature_path = MODEL_DIR / feature_record["path"]
    _verify_file(feature_path, feature_record["file_sha256"], "feature columns")
    feature_columns = feature_path.read_text(encoding="utf-8").splitlines()
    if (
        len(feature_columns) != EXPECTED_FEATURE_COUNT
        or len(set(feature_columns)) != len(feature_columns)
        or _canonical_hash(feature_columns) != feature_record["ordered_columns_sha256"]
    ):
        raise InferenceContractError("feature column schema drifted")

    post_record = recipe["postprocess"]
    _verify_file(
        MODEL_DIR / post_record["path"], post_record["file_sha256"], "postprocess"
    )
    models = _load_models(recipe, feature_columns)

    sample = pd.read_csv(
        DATA_DIR / "sample_submission.csv",
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        keep_default_na=False,
    )
    if list(sample.columns) != ["forecast_id", "forecast_kst_dtm", *GROUPS]:
        raise InferenceContractError("sample schema drifted")
    timestamps = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
    features = build_features(
        TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv"
    ).sort_index()
    if tuple(features.columns) != tuple(feature_columns):
        raise InferenceContractError("test feature schema drifted")
    aligned = features.reindex(pd.DatetimeIndex(timestamps))[feature_columns]
    if aligned.shape != (len(sample), EXPECTED_FEATURE_COUNT):
        raise InferenceContractError("test feature matrix shape drifted")
    if aligned.isna().all(axis=1).any():
        raise InferenceContractError("test feature matrix contains an all-NaN row")

    submission = sample.copy()
    for group in ("kpx_group_1", "kpx_group_2"):
        raw = _ensemble_predict(models[group], aligned)
        capacity = CAPACITY_KWH[group]
        clipped = np.clip(raw, 0.0, capacity)
        values = post_record["groups"][group]
        submission[group] = apply_post(
            clipped, capacity, values["scale"], values["floor_kwh"]
        )
    pooled_features = aligned.copy()
    pooled_features["group_id"] = 3
    group = "kpx_group_3"
    capacity = CAPACITY_KWH[group]
    raw = _ensemble_predict(models["pooled"], pooled_features) * capacity
    clipped = np.clip(raw, 0.0, capacity)
    values = post_record["groups"][group]
    submission[group] = apply_post(
        clipped, capacity, values["scale"], values["floor_kwh"]
    )
    validate_submission_frame(submission, sample, require_competition_shape=True)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "submission.csv"
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        submission.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        if not temporary_path.read_bytes().startswith(b"\xef\xbb\xbf"):
            raise InferenceContractError("temporary submission is missing UTF-8 BOM")
        reread = pd.read_csv(temporary_path, encoding="utf-8-sig")
        validate_submission_frame(reread, sample, require_competition_shape=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    final_bytes = output_path.read_bytes()
    if not final_bytes.startswith(b"\xef\xbb\xbf"):
        raise InferenceContractError("final submission is missing UTF-8 BOM")
    final = pd.read_csv(output_path, encoding="utf-8-sig")
    validate_submission_frame(final, sample, require_competition_shape=True)
    for filename in LEGACY_MODEL_FILES:
        (MODEL_DIR / filename).unlink(missing_ok=True)

    print(f"submission: {output_path} ({_sha256_file(output_path)})")
    for group in GROUPS:
        values = final[group]
        print(
            f"{group}: min={values.min():.6f}, mean={values.mean():.6f}, "
            f"max={values.max():.6f}"
        )


if __name__ == "__main__":
    main()
