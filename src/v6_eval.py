"""Frozen wind-v5 baseline evaluator with fail-closed provenance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CAPACITY_KWH, GROUPS, ROOT, TRAIN_DIR
from features import build_features
from metrics import group_ficr, group_nmae
from scada import build_mismatch_mask, build_potential


BASELINE_RECIPE = "v5-c1-potential-q60-filter05-floor10"
BASELINE_PARAMS = {
    "objective": "quantile",
    "alpha": 0.60,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}
TRAIN_FILTER_RATIO = 0.05
FLOOR_RATIO = 0.10
G12 = ("kpx_group_1", "kpx_group_2")
G3 = "kpx_group_3"
EXPERIMENT_DIR = ROOT / ".omx" / "experiments" / "wind-v6"
CACHE_SCHEMA_VERSION = 2
ANCHOR_TOLERANCE = 0.00015
BASELINE_ANCHORS = {"fold23": 0.6316, "fold24": 0.6380}
EXPECTED_ROWS = {
    "fold23": {
        "g1_train": 6215,
        "g2_train": 6174,
        "g1_valid": 8757,
        "g2_valid": 8758,
    },
    "fold24": {
        "g1_train": 12516,
        "g2_train": 12455,
        "pooled_train": 30153,
        "g1_valid": 8778,
        "g2_valid": 8778,
        "g3_valid": 8778,
    },
}
BASELINE_POSTPROCESS_CONFIG = {
    "version": 1,
    "kind": "capacity_clip_then_floor",
    "clip_min_ratio": 0.0,
    "clip_max_ratio": 1.0,
    "floor_ratio": FLOOR_RATIO,
}
BASELINE_RECIPE_CONFIG = {
    "version": 1,
    "name": BASELINE_RECIPE,
    "training": {
        "target": "scada_potential",
        "exclude_scada_mismatch": True,
        "filter_ratio": TRAIN_FILTER_RATIO,
    },
    "validation": {
        "target": "raw_generation_label",
        "fold_strategy": "calendar_year_holdout",
    },
    "model": {
        "library": "lightgbm",
        "params": dict(BASELINE_PARAMS),
        "num_boost_round": 5000,
        "early_stopping_rounds": 200,
        "seed_aggregation": "arithmetic_mean",
    },
    "capacities": dict(CAPACITY_KWH),
    "group_strategy": {
        "version": 1,
        "kpx_group_1": "solo_kwh",
        "kpx_group_2": "solo_kwh",
        "kpx_group_3": "pooled_normalized_with_group_id",
    },
    "postprocess": dict(BASELINE_POSTPROCESS_CONFIG),
}


class ProvenanceError(ValueError):
    """Raised when fold data cannot prove its identity and alignment."""


@dataclass(frozen=True)
class TrainingBundle:
    features: pd.DataFrame
    labels: pd.DataFrame
    potential: pd.DataFrame
    mismatch: pd.DataFrame
    data: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_hash: str
    data_hashes: dict[str, str]
    derived_data_hashes: dict[str, str]


@dataclass(frozen=True)
class FoldProvenance:
    recipe: str
    recipe_config: dict[str, object]
    recipe_hash: str
    postprocess_config: dict[str, object]
    postprocess_hash: str
    cache_schema_version: int
    train_years: tuple[int, ...]
    valid_year: int
    groups: tuple[str, ...]
    seeds: tuple[int, ...]
    feature_hash: str
    data_hashes: dict[str, str]
    derived_data_hashes: dict[str, str]
    row_counts: dict[str, int]
    validation_index_hashes: dict[str, str]
    target_hashes: dict[str, str]
    prediction_hashes: dict[str, str]
    manifest_key: str


@dataclass(frozen=True)
class FoldPredictions:
    model_predictions: dict[str, pd.Series]
    validation_targets: dict[str, pd.Series]
    provenance: FoldProvenance


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_object(value: Mapping[str, object], field_name: str) -> dict[str, object]:
    """Return a detached, JSON-canonical mapping suitable for provenance."""
    try:
        normalized = json.loads(
            json.dumps(
                dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
    except (TypeError, ValueError) as error:
        raise ProvenanceError(f"{field_name} must be JSON serializable") from error
    if not isinstance(normalized, dict):
        raise ProvenanceError(f"{field_name} must be a JSON object")
    return normalized


def recipe_fingerprint(config: Mapping[str, object]) -> str:
    """Hash every structured behavior field in a recipe or postprocess config."""
    return _canonical_hash(_json_object(config, "recipe config"))


def _pandas_hash(obj: pd.DataFrame | pd.Series | pd.Index) -> str:
    digest = hashlib.sha256()
    if isinstance(obj, pd.Index):
        digest.update(str(obj.dtype).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(obj, index=False).values.tobytes())
        return digest.hexdigest()
    if isinstance(obj, pd.DataFrame):
        schema = [(str(column), str(dtype)) for column, dtype in obj.dtypes.items()]
        digest.update(
            json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        )
    else:
        digest.update(str(obj.name).encode("utf-8"))
        digest.update(str(obj.dtype).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(obj, index=True).values.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_hash(features: pd.DataFrame) -> str:
    """Hash feature names, dtypes, index, and values."""
    return _pandas_hash(features)


def manifest_key(
    *,
    recipe: str,
    recipe_hash: str,
    postprocess_hash: str,
    cache_schema_version: int,
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
    feature_hash: str,
    data_hashes: Mapping[str, str],
    derived_data_hashes: Mapping[str, str] | None = None,
) -> str:
    """Return the identity of a requested fold before predictions exist."""
    return _canonical_hash(
        {
            "recipe": recipe,
            "recipe_hash": recipe_hash,
            "postprocess_hash": postprocess_hash,
            "cache_schema_version": int(cache_schema_version),
            "train_years": list(train_years),
            "valid_year": valid_year,
            "groups": list(groups),
            "seeds": list(seeds),
            "feature_hash": feature_hash,
            "data_hashes": dict(sorted(data_hashes.items())),
            "derived_data_hashes": dict(sorted((derived_data_hashes or {}).items())),
        }
    )


def build_provenance(
    *,
    recipe: str,
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
    recipe_config: Mapping[str, object],
    postprocess_config: Mapping[str, object],
    feature_hash: str,
    data_hashes: Mapping[str, str],
    derived_data_hashes: Mapping[str, str] | None = None,
    row_counts: Mapping[str, int],
    model_predictions: Mapping[str, pd.Series],
    validation_targets: Mapping[str, pd.Series],
) -> FoldProvenance:
    """Build hashes that make a fold independently auditable."""
    group_tuple = tuple(groups)
    if tuple(model_predictions) != group_tuple:
        raise ProvenanceError("model prediction group keys do not match groups")
    if tuple(validation_targets) != group_tuple:
        raise ProvenanceError("validation target group keys do not match groups")
    normalized_recipe = _json_object(recipe_config, "recipe config")
    normalized_postprocess = _json_object(postprocess_config, "postprocess config")
    recipe_hash_value = recipe_fingerprint(normalized_recipe)
    postprocess_hash_value = recipe_fingerprint(normalized_postprocess)
    data_hash_dict = dict(sorted(data_hashes.items()))
    derived_hash_dict = dict(sorted((derived_data_hashes or {}).items()))
    seed_tuple = tuple(int(seed) for seed in seeds)
    return FoldProvenance(
        recipe=recipe,
        recipe_config=normalized_recipe,
        recipe_hash=recipe_hash_value,
        postprocess_config=normalized_postprocess,
        postprocess_hash=postprocess_hash_value,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        train_years=tuple(train_years),
        valid_year=int(valid_year),
        groups=group_tuple,
        seeds=seed_tuple,
        feature_hash=feature_hash,
        data_hashes=data_hash_dict,
        derived_data_hashes=derived_hash_dict,
        row_counts={key: int(value) for key, value in sorted(row_counts.items())},
        validation_index_hashes={
            group: _pandas_hash(model_predictions[group].index) for group in group_tuple
        },
        target_hashes={
            group: _pandas_hash(validation_targets[group]) for group in group_tuple
        },
        prediction_hashes={
            group: _pandas_hash(model_predictions[group]) for group in group_tuple
        },
        manifest_key=manifest_key(
            recipe=recipe,
            recipe_hash=recipe_hash_value,
            postprocess_hash=postprocess_hash_value,
            cache_schema_version=CACHE_SCHEMA_VERSION,
            train_years=tuple(train_years),
            valid_year=int(valid_year),
            groups=group_tuple,
            seeds=seed_tuple,
            feature_hash=feature_hash,
            data_hashes=data_hash_dict,
            derived_data_hashes=derived_hash_dict,
        ),
    )


def validate_fold_predictions(
    fold: FoldPredictions,
    *,
    expected_recipe: str | None = None,
    expected_seeds: tuple[int, ...] | None = None,
    expected_feature_hash: str | None = None,
    expected_data_hashes: Mapping[str, str] | None = None,
    expected_derived_data_hashes: Mapping[str, str] | None = None,
    expected_recipe_hash: str | None = None,
    expected_postprocess_hash: str | None = None,
) -> None:
    """Fail unless fold values and all declared provenance agree exactly."""
    provenance = fold.provenance
    groups = provenance.groups
    if not groups:
        raise ProvenanceError("fold has no group keys")
    if tuple(fold.model_predictions) != groups:
        raise ProvenanceError("model prediction group keys do not match provenance")
    if tuple(fold.validation_targets) != groups:
        raise ProvenanceError("validation target group keys do not match provenance")
    if not provenance.seeds:
        raise ProvenanceError("fold has no seeds")
    if provenance.cache_schema_version != CACHE_SCHEMA_VERSION:
        raise ProvenanceError("cache schema version does not match evaluator")
    if provenance.recipe_hash != recipe_fingerprint(provenance.recipe_config):
        raise ProvenanceError("recipe config fingerprint does not match")
    if provenance.postprocess_hash != recipe_fingerprint(provenance.postprocess_config):
        raise ProvenanceError("postprocess config fingerprint does not match")

    expected_key = manifest_key(
        recipe=provenance.recipe,
        recipe_hash=provenance.recipe_hash,
        postprocess_hash=provenance.postprocess_hash,
        cache_schema_version=provenance.cache_schema_version,
        train_years=provenance.train_years,
        valid_year=provenance.valid_year,
        groups=groups,
        seeds=provenance.seeds,
        feature_hash=provenance.feature_hash,
        data_hashes=provenance.data_hashes,
        derived_data_hashes=provenance.derived_data_hashes,
    )
    if provenance.manifest_key != expected_key:
        raise ProvenanceError("manifest key does not match fold metadata")

    for group in groups:
        prediction = fold.model_predictions[group]
        target = fold.validation_targets[group]
        row_count_key = f"g{group[-1]}_valid"
        if provenance.row_counts.get(row_count_key) != len(prediction):
            raise ProvenanceError(f"{group} validation row count does not match")
        if not isinstance(prediction, pd.Series) or not isinstance(target, pd.Series):
            raise ProvenanceError(
                f"{group} predictions and targets must be indexed Series"
            )
        if prediction.shape != target.shape:
            raise ProvenanceError(f"{group} prediction and target shapes differ")
        if not prediction.index.equals(target.index):
            raise ProvenanceError(f"{group} validation index differs from target index")
        if (
            not prediction.index.is_unique
            or not prediction.index.is_monotonic_increasing
        ):
            raise ProvenanceError(f"{group} validation index must be unique and sorted")
        if provenance.validation_index_hashes.get(group) != _pandas_hash(
            prediction.index
        ):
            raise ProvenanceError(f"{group} validation index hash does not match")
        if provenance.target_hashes.get(group) != _pandas_hash(target):
            raise ProvenanceError(f"{group} target hash does not match")
        if provenance.prediction_hashes.get(group) != _pandas_hash(prediction):
            raise ProvenanceError(f"{group} prediction hash does not match")

    if expected_recipe is not None and provenance.recipe != expected_recipe:
        raise ProvenanceError("recipe does not match requested recipe")
    if expected_seeds is not None and provenance.seeds != tuple(expected_seeds):
        raise ProvenanceError("seeds do not match requested seeds")
    if (
        expected_feature_hash is not None
        and provenance.feature_hash != expected_feature_hash
    ):
        raise ProvenanceError("feature hash does not match requested features")
    if expected_data_hashes is not None and provenance.data_hashes != dict(
        sorted(expected_data_hashes.items())
    ):
        raise ProvenanceError("data hashes do not match requested data")
    if (
        expected_derived_data_hashes is not None
        and provenance.derived_data_hashes
        != dict(sorted(expected_derived_data_hashes.items()))
    ):
        raise ProvenanceError("derived data hashes do not match requested data")
    if (
        expected_recipe_hash is not None
        and provenance.recipe_hash != expected_recipe_hash
    ):
        raise ProvenanceError("recipe fingerprint does not match requested recipe")
    if (
        expected_postprocess_hash is not None
        and provenance.postprocess_hash != expected_postprocess_hash
    ):
        raise ProvenanceError("postprocess fingerprint does not match requested config")


def assert_fold_alignment(
    baseline: FoldPredictions,
    candidate: FoldPredictions,
    *,
    require_feature_hash: bool = True,
) -> None:
    """Require candidates to use the baseline fold, seeds, data, and row ordering."""
    validate_fold_predictions(baseline)
    validate_fold_predictions(candidate)
    base = baseline.provenance
    other = candidate.provenance
    if base.train_years != other.train_years or base.valid_year != other.valid_year:
        raise ProvenanceError("fold years differ")
    if base.groups != other.groups:
        raise ProvenanceError("group keys differ")
    if base.seeds != other.seeds:
        raise ProvenanceError("seeds differ")
    if base.data_hashes != other.data_hashes:
        raise ProvenanceError("data hashes differ")
    if require_feature_hash and base.feature_hash != other.feature_hash:
        raise ProvenanceError("feature hashes differ")
    if (
        base.postprocess_config != other.postprocess_config
        or base.postprocess_hash != other.postprocess_hash
    ):
        raise ProvenanceError("postprocess provenance differs")
    base_validation_rows = {
        key: value for key, value in base.row_counts.items() if key.endswith("_valid")
    }
    other_validation_rows = {
        key: value for key, value in other.row_counts.items() if key.endswith("_valid")
    }
    if base_validation_rows != other_validation_rows:
        raise ProvenanceError("validation row metadata differs")
    for group in base.groups:
        base_prediction = baseline.model_predictions[group]
        other_prediction = candidate.model_predictions[group]
        if base_prediction.shape != other_prediction.shape:
            raise ProvenanceError(f"{group} validation shapes differ")
        if not base_prediction.index.equals(other_prediction.index):
            raise ProvenanceError(f"{group} validation index differs")
        if base.target_hashes[group] != other.target_hashes[
            group
        ] or not baseline.validation_targets[group].equals(
            candidate.validation_targets[group]
        ):
            raise ProvenanceError(f"{group} validation targets differ")


def apply_floor10(values: np.ndarray | pd.Series, capacity: float):
    """Apply the evaluation-threshold floor without changing safe predictions."""
    floor = FLOOR_RATIO * capacity
    if isinstance(values, pd.Series):
        return values.clip(lower=floor)
    return np.maximum(np.asarray(values), floor)


def gate_scores(
    baseline: dict[str, float],
    candidate: dict[str, float],
    min_mean_delta: float = 0.001,
) -> dict:
    deltas = {fold: candidate[fold] - baseline[fold] for fold in ("fold23", "fold24")}
    mean_delta = sum(deltas.values()) / 2.0
    passed = (
        all(delta > 0.0 for delta in deltas.values()) and mean_delta >= min_mean_delta
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "mean_delta": mean_delta,
    }


def blend_predictions(
    potential_predictions: dict[str, np.ndarray],
    actual_label_predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> dict[str, np.ndarray]:
    """Blend model families; each weight belongs to the actual-label family."""
    return {
        group: (1.0 - weights[group]) * potential_predictions[group]
        + weights[group] * actual_label_predictions[group]
        for group in potential_predictions
    }


def load_scada_targets(
    labels: pd.DataFrame, *, cache_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild SCADA-derived targets from raw CSVs for this process."""
    _ = cache_dir  # Retained only for call-site compatibility; disk caches are unsafe.
    potential = build_potential(labels)
    mismatch = build_mismatch_mask(labels)
    return potential.reindex(labels.index), mismatch.reindex(labels.index)


_BUNDLE: TrainingBundle | None = None


def load_bundle() -> TrainingBundle:
    """Load the exact production feature/target bundle and its source hashes."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    labels_path = TRAIN_DIR / "train_labels.csv"
    labels = pd.read_csv(
        labels_path,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
        index_col="kst_dtm",
    ).sort_index()

    features = build_features(
        TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv"
    ).sort_index()

    potential, mismatch = load_scada_targets(labels)
    data = features.join(labels, how="inner")
    source_paths = {
        "ldaps_train": TRAIN_DIR / "ldaps_train.csv",
        "gfs_train": TRAIN_DIR / "gfs_train.csv",
        "train_labels": labels_path,
        "scada_vestas": TRAIN_DIR / "scada_vestas_train.csv",
        "scada_unison": TRAIN_DIR / "scada_unison_train.csv",
    }
    data_hashes = {name: _file_hash(path) for name, path in source_paths.items()}
    derived_data_hashes = {
        "potential_frame": _pandas_hash(potential),
        "mismatch_frame": _pandas_hash(mismatch),
    }
    _BUNDLE = TrainingBundle(
        features=features,
        labels=labels,
        potential=potential,
        mismatch=mismatch,
        data=data,
        feature_columns=tuple(features.columns),
        feature_hash=feature_hash(features),
        data_hashes=dict(sorted(data_hashes.items())),
        derived_data_hashes=dict(sorted(derived_data_hashes.items())),
    )
    return _BUNDLE


def _solo_train_frame(
    bundle: TrainingBundle, group: str, train_years: tuple[int, ...]
) -> pd.DataFrame:
    capacity = CAPACITY_KWH[group]
    frame = bundle.data.dropna(subset=[group])
    train = frame[frame.index.year.isin(train_years)].copy()
    train["_target"] = bundle.potential[f"{group}_potential"].reindex(train.index)
    mismatch = bundle.mismatch[f"{group}_mismatch"].reindex(train.index).fillna(False)
    train = train[~mismatch].dropna(subset=["_target"])
    return train[train["_target"] >= TRAIN_FILTER_RATIO * capacity]


def _pooled_train_frame(
    bundle: TrainingBundle, train_years: tuple[int, ...]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frames = []
    columns = bundle.feature_columns
    for group in GROUPS:
        frame = bundle.data.dropna(subset=[group]).copy()
        frame["_target"] = bundle.potential[f"{group}_potential"].reindex(frame.index)
        mismatch = (
            bundle.mismatch[f"{group}_mismatch"].reindex(frame.index).fillna(False)
        )
        frame = frame[~mismatch].dropna(subset=["_target"])
        normalized = frame[list(columns)].copy()
        normalized["normalized_target"] = frame["_target"] / CAPACITY_KWH[group]
        normalized["group_id"] = int(group[-1])
        frames.append(normalized)
    pooled = pd.concat(frames)
    pooled = pooled[pooled.index.year.isin(train_years)]
    pooled = pooled[pooled["normalized_target"] >= TRAIN_FILTER_RATIO]
    return pooled, columns + ("group_id",)


def _fit_ensemble(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    seeds: tuple[int, ...],
    *,
    categorical: tuple[str, ...] = (),
) -> np.ndarray:
    predictions = []
    for seed in seeds:
        params = dict(BASELINE_PARAMS, seed=seed)
        train_set = lgb.Dataset(
            train_features,
            train_target,
            categorical_feature=list(categorical) if categorical else "auto",
        )
        valid_set = lgb.Dataset(
            validation_features, validation_target, reference=train_set
        )
        model = lgb.train(
            params,
            train_set,
            5000,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        predictions.append(
            model.predict(validation_features, num_iteration=model.best_iteration)
        )
    return np.mean(predictions, axis=0)


def _fold_name(valid_year: int) -> str:
    return f"fold{str(valid_year)[-2:]}"


def assert_baseline_fingerprint(fold_name: str, row_counts: Mapping[str, int]) -> None:
    expected = EXPECTED_ROWS.get(fold_name)
    actual = {key: int(value) for key, value in row_counts.items()}
    if expected is None or actual != expected:
        raise ProvenanceError(
            f"{fold_name} row fingerprint drift: expected {expected}, got {actual}"
        )


def assert_score_anchor(
    fold_name: str, score: float, *, seeds: tuple[int, ...]
) -> None:
    if tuple(int(seed) for seed in seeds) != (42,):
        return
    expected = BASELINE_ANCHORS.get(fold_name)
    if (
        expected is None
        or not np.isfinite(score)
        or abs(score - expected) > ANCHOR_TOLERANCE
    ):
        raise ProvenanceError(
            f"{fold_name} score anchor drift: expected {expected} +/- "
            f"{ANCHOR_TOLERANCE}, got {score}"
        )


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"fold-predictions-{key}.npz",
        cache_dir / f"fold-manifest-{key}.json",
    )


def write_prediction_cache(
    fold: FoldPredictions, cache_dir: Path = EXPERIMENT_DIR
) -> Path:
    """Persist predictions plus checksums; validation targets remain live-only."""
    validate_fold_predictions(fold)
    cache_dir.mkdir(parents=True, exist_ok=True)
    provenance = fold.provenance
    predictions_path, manifest_path = _cache_paths(cache_dir, provenance.manifest_key)
    arrays: dict[str, np.ndarray] = {}
    for group in provenance.groups:
        suffix = group.rsplit("_", 1)[-1]
        arrays[f"g{suffix}_index_ns"] = fold.model_predictions[group].index.asi8
        arrays[f"g{suffix}_prediction"] = fold.model_predictions[group].to_numpy()
    temporary_predictions = predictions_path.with_suffix(".tmp.npz")
    with temporary_predictions.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_predictions.replace(predictions_path)

    manifest = asdict(provenance)
    manifest.update(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "predictions_file": predictions_path.name,
        }
    )
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def _read_prediction_cache(
    *,
    cache_dir: Path,
    recipe: str,
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
    recipe_fingerprint: str,
    postprocess_config: Mapping[str, object],
    feature_fingerprint: str,
    data_hashes: Mapping[str, str],
    derived_data_hashes: Mapping[str, str] | None = None,
    row_counts: Mapping[str, int],
    live_validation_targets: Mapping[str, pd.Series],
) -> FoldPredictions | None:
    normalized_postprocess = _json_object(postprocess_config, "postprocess config")
    postprocess_hash = _canonical_hash(normalized_postprocess)
    normalized_derived_hashes = dict(sorted((derived_data_hashes or {}).items()))
    key = manifest_key(
        recipe=recipe,
        recipe_hash=recipe_fingerprint,
        postprocess_hash=postprocess_hash,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        feature_hash=feature_fingerprint,
        data_hashes=data_hashes,
        derived_data_hashes=normalized_derived_hashes,
    )
    predictions_path, manifest_path = _cache_paths(cache_dir, key)
    if not predictions_path.exists() and not manifest_path.exists():
        return None
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise ProvenanceError("prediction cache is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "recipe",
            "recipe_config",
            "recipe_hash",
            "postprocess_config",
            "postprocess_hash",
            "cache_schema_version",
            "schema_version",
            "train_years",
            "valid_year",
            "groups",
            "seeds",
            "feature_hash",
            "data_hashes",
            "derived_data_hashes",
            "row_counts",
            "validation_index_hashes",
            "target_hashes",
            "prediction_hashes",
            "manifest_key",
            "predictions_file",
        }
        if not isinstance(manifest, dict) or not required <= manifest.keys():
            raise ProvenanceError("prediction cache manifest is missing provenance")
        if (
            manifest["schema_version"] != CACHE_SCHEMA_VERSION
            or manifest["cache_schema_version"] != CACHE_SCHEMA_VERSION
        ):
            raise ProvenanceError("prediction cache schema version does not match")

        manifest_recipe_config = _json_object(
            manifest["recipe_config"], "recipe config"
        )
        if _canonical_hash(manifest_recipe_config) != recipe_fingerprint:
            raise ProvenanceError("prediction cache recipe fingerprint does not match")
        expected_metadata = {
            "recipe": recipe,
            "recipe_hash": recipe_fingerprint,
            "postprocess_config": normalized_postprocess,
            "postprocess_hash": postprocess_hash,
            "train_years": list(train_years),
            "valid_year": valid_year,
            "groups": list(groups),
            "seeds": list(seeds),
            "feature_hash": feature_fingerprint,
            "data_hashes": dict(sorted(data_hashes.items())),
            "derived_data_hashes": normalized_derived_hashes,
            "row_counts": {
                name: int(count) for name, count in sorted(row_counts.items())
            },
            "manifest_key": key,
            "predictions_file": predictions_path.name,
        }
        if any(manifest[field] != value for field, value in expected_metadata.items()):
            raise ProvenanceError("prediction cache metadata does not match request")
        if tuple(live_validation_targets) != tuple(groups):
            raise ProvenanceError(
                "prediction cache live validation target groups do not match"
            )

        validation_targets: dict[str, pd.Series] = {}
        for group in groups:
            target = live_validation_targets[group]
            if not isinstance(target, pd.Series):
                raise ProvenanceError(
                    f"prediction cache live validation target for {group} is not a Series"
                )
            if manifest["validation_index_hashes"].get(group) != _pandas_hash(
                target.index
            ):
                raise ProvenanceError(
                    f"prediction cache live validation index for {group} differs"
                )
            if manifest["target_hashes"].get(group) != _pandas_hash(target):
                raise ProvenanceError(
                    f"prediction cache live validation target for {group} differs"
                )
            validation_targets[group] = target

        model_predictions: dict[str, pd.Series] = {}
        with np.load(predictions_path, allow_pickle=False) as arrays:
            for group in groups:
                suffix = group.rsplit("_", 1)[-1]
                index = pd.DatetimeIndex(arrays[f"g{suffix}_index_ns"])
                if not index.equals(validation_targets[group].index):
                    raise ProvenanceError(
                        f"prediction cache live validation index for {group} differs"
                    )
                model_predictions[group] = pd.Series(
                    arrays[f"g{suffix}_prediction"],
                    index=index,
                    name=group,
                )

        provenance = build_provenance(
            recipe=recipe,
            train_years=train_years,
            valid_year=valid_year,
            groups=groups,
            seeds=seeds,
            recipe_config=manifest_recipe_config,
            postprocess_config=normalized_postprocess,
            feature_hash=feature_fingerprint,
            data_hashes=data_hashes,
            derived_data_hashes=normalized_derived_hashes,
            row_counts=row_counts,
            model_predictions=model_predictions,
            validation_targets=validation_targets,
        )
        serialized_provenance = json.loads(json.dumps(asdict(provenance)))
        if serialized_provenance != {
            field: manifest[field] for field in serialized_provenance
        }:
            raise ProvenanceError("prediction cache hashes do not match manifest")
        fold = FoldPredictions(model_predictions, validation_targets, provenance)
        validate_fold_predictions(
            fold,
            expected_recipe=recipe,
            expected_seeds=seeds,
            expected_feature_hash=feature_fingerprint,
            expected_data_hashes=data_hashes,
            expected_derived_data_hashes=normalized_derived_hashes,
            expected_recipe_hash=recipe_fingerprint,
            expected_postprocess_hash=postprocess_hash,
        )
        return fold
    except ProvenanceError:
        raise
    except Exception as error:
        raise ProvenanceError(f"prediction cache is malformed: {error}") from error


def fit_fold(
    recipe: str,
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
) -> FoldPredictions:
    """Fit or load one exact frozen-v5 validation fold."""
    train_years = tuple(train_years)
    groups = tuple(groups)
    seeds = tuple(int(seed) for seed in seeds)
    if recipe != BASELINE_RECIPE:
        raise ValueError(f"unsupported recipe: {recipe}")
    if not seeds:
        raise ValueError("at least one seed is required")
    expected_fold = {
        ((2022,), 2023, G12),
        ((2022, 2023), 2024, tuple(GROUPS)),
    }
    if (train_years, valid_year, groups) not in expected_fold:
        raise ValueError(
            "baseline evaluator supports only the frozen fold23/fold24 splits"
        )

    bundle = load_bundle()
    row_counts: dict[str, int] = {}
    solo_frames: dict[str, pd.DataFrame] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    for group in groups:
        source = bundle.data.dropna(subset=[group])
        validation = source[source.index.year == valid_year]
        validation_frames[group] = validation
        row_counts[f"g{group[-1]}_valid"] = len(validation)
        if group in G12:
            solo_frames[group] = _solo_train_frame(bundle, group, train_years)
            row_counts[f"g{group[-1]}_train"] = len(solo_frames[group])

    pooled_frame: pd.DataFrame | None = None
    pooled_columns: tuple[str, ...] = ()
    if G3 in groups:
        pooled_frame, pooled_columns = _pooled_train_frame(bundle, train_years)
        row_counts["pooled_train"] = len(pooled_frame)
    row_counts = dict(sorted(row_counts.items()))
    fold_name = _fold_name(valid_year)
    assert_baseline_fingerprint(fold_name, row_counts)
    validation_targets = {
        group: validation_frames[group][group].copy().rename(group) for group in groups
    }
    baseline_recipe_hash = recipe_fingerprint(BASELINE_RECIPE_CONFIG)

    cached = _read_prediction_cache(
        cache_dir=EXPERIMENT_DIR,
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        recipe_fingerprint=baseline_recipe_hash,
        postprocess_config=BASELINE_POSTPROCESS_CONFIG,
        feature_fingerprint=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
        derived_data_hashes=bundle.derived_data_hashes,
        row_counts=row_counts,
        live_validation_targets=validation_targets,
    )
    if cached is not None:
        return cached

    model_predictions: dict[str, pd.Series] = {}
    columns = list(bundle.feature_columns)
    for group in groups:
        capacity = CAPACITY_KWH[group]
        validation = validation_frames[group]
        validation_target = validation_targets[group]
        if group in G12:
            train = solo_frames[group]
            raw_prediction = _fit_ensemble(
                train[columns],
                train["_target"],
                validation[columns],
                validation_target,
                seeds,
            )
        else:
            if pooled_frame is None:
                raise ProvenanceError("pooled training frame is missing")
            validation_features = validation[columns].copy()
            validation_features["group_id"] = 3
            raw_prediction = (
                _fit_ensemble(
                    pooled_frame[list(pooled_columns)],
                    pooled_frame["normalized_target"],
                    validation_features[list(pooled_columns)],
                    validation_target / capacity,
                    seeds,
                    categorical=("group_id",),
                )
                * capacity
            )
        clipped = np.clip(raw_prediction, 0.0, capacity)
        model_predictions[group] = pd.Series(
            apply_floor10(clipped, capacity), index=validation.index, name=group
        )

    provenance = build_provenance(
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        recipe_config=BASELINE_RECIPE_CONFIG,
        postprocess_config=BASELINE_POSTPROCESS_CONFIG,
        feature_hash=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
        derived_data_hashes=bundle.derived_data_hashes,
        row_counts=row_counts,
        model_predictions=model_predictions,
        validation_targets=validation_targets,
    )
    fold = FoldPredictions(model_predictions, validation_targets, provenance)
    validate_fold_predictions(
        fold,
        expected_recipe=recipe,
        expected_seeds=seeds,
        expected_feature_hash=bundle.feature_hash,
        expected_data_hashes=bundle.data_hashes,
        expected_derived_data_hashes=bundle.derived_data_hashes,
        expected_recipe_hash=baseline_recipe_hash,
        expected_postprocess_hash=recipe_fingerprint(BASELINE_POSTPROCESS_CONFIG),
    )
    write_prediction_cache(fold)
    return fold


def score_fold(predictions: FoldPredictions) -> float:
    """Score only a fully indexed, hash-verified fold."""
    validate_fold_predictions(predictions)
    provenance = predictions.provenance
    if provenance.recipe == BASELINE_RECIPE:
        if provenance.recipe_config != _json_object(
            BASELINE_RECIPE_CONFIG, "baseline recipe config"
        ) or provenance.recipe_hash != recipe_fingerprint(BASELINE_RECIPE_CONFIG):
            raise ProvenanceError("baseline recipe config drift")
        if provenance.postprocess_config != _json_object(
            BASELINE_POSTPROCESS_CONFIG, "baseline postprocess config"
        ) or provenance.postprocess_hash != recipe_fingerprint(
            BASELINE_POSTPROCESS_CONFIG
        ):
            raise ProvenanceError("baseline postprocess config drift")
        assert_baseline_fingerprint(
            _fold_name(provenance.valid_year), provenance.row_counts
        )
    nmaes = []
    ficrs = []
    for group in predictions.provenance.groups:
        model_prediction = predictions.model_predictions[group].to_numpy()
        validation_target = predictions.validation_targets[group].to_numpy()
        capacity = CAPACITY_KWH[group]
        nmaes.append(group_nmae(model_prediction, validation_target, capacity))
        ficrs.append(group_ficr(model_prediction, validation_target, capacity))
    score = 0.5 * (1.0 - float(np.mean(nmaes))) + 0.5 * float(np.mean(ficrs))
    if not np.isfinite(score):
        raise ProvenanceError("fold score is not finite")
    return score


def run_stage7(seeds: tuple[int, ...], baseline_only: bool = False) -> int:
    """Reproduce the frozen baseline; remain fail-closed until a candidate exists."""
    seeds = tuple(int(seed) for seed in seeds)
    if not baseline_only:
        print(
            "stage7 candidate evaluation is not implemented yet; use --baseline-only",
            file=sys.stderr,
        )
        return 2
    try:
        fold23 = fit_fold(BASELINE_RECIPE, (2022,), 2023, G12, seeds)
        fold24 = fit_fold(BASELINE_RECIPE, (2022, 2023), 2024, tuple(GROUPS), seeds)
        assert_baseline_fingerprint("fold23", fold23.provenance.row_counts)
        assert_baseline_fingerprint("fold24", fold24.provenance.row_counts)
        scores = {"fold23": score_fold(fold23), "fold24": score_fold(fold24)}
        for fold_name, score in scores.items():
            assert_score_anchor(fold_name, score, seeds=seeds)
            print(f"[{fold_name}] frozen v5 C1 score={score:.6f}")
        result = {
            "status": "BASELINE_OK",
            "recipe": BASELINE_RECIPE,
            "seeds": list(seeds),
            "scores": scores,
            "manifest_keys": {
                "fold23": fold23.provenance.manifest_key,
                "fold24": fold24.provenance.manifest_key,
            },
        }
        print(json.dumps(result, sort_keys=True))
    except (
        ProvenanceError,
        ValueError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"stage7 baseline rejected: {error}", file=sys.stderr)
        return 2

    return 0
