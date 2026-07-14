"""Train the hash-bound wind-v6 production ensemble.

This script promotes only the canonical three-seed candidate that passed the
dual-fold evaluator.  It rebuilds the exact weighted targets on label-years
2022--2024, trains nine models, and publishes ``recipe.json`` last so inference
can never consume a partially written model bundle.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Iterable, Mapping, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, TEST_DIR
from scada import WeightCalibration, build_weighted_targets
from v6_eval import (
    BASELINE_PARAMS,
    BASELINE_POSTPROCESS_CONFIG,
    CANONICAL_SEEDS,
    FINAL_GATE_PATH,
    FoldPredictions,
    G12,
    TRAIN_FILTER_RATIO,
    TrainingBundle,
    WEIGHTED_RECIPE,
    WEIGHTED_RECIPE_CONFIG,
    _assert_scada_source_hashes,
    _canonical_hash,
    _pandas_hash,
    _pooled_train_frame,
    _solo_train_frame,
    fit_fold,
    load_bundle,
    read_final_gate_artifact,
    recipe_fingerprint,
)


PRODUCTION_RECIPE_KIND = "wind-v6-production-recipe"
PRODUCTION_RECIPE_SCHEMA_VERSION = 1
FULL_LABEL_YEARS = (2022, 2023, 2024)
ACTIVE_MODEL_NAMES = {
    family: tuple(f"lgbm_v6_weighted_{family}_s{seed}.txt" for seed in CANONICAL_SEEDS)
    for family in (*G12, "pooled")
}


class TrainingContractError(ValueError):
    """Raised when promotion evidence or a production artifact drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
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
        raise TrainingContractError("metadata must be canonical JSON") from error


def calculate_full_rounds(iterations: Iterable[int]) -> tuple[int, int]:
    """Apply the frozen two-stage truncation policy to raw seed iterations."""
    values = tuple(iterations)
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("validation iterations must be positive integers")
    base_rounds = int(sum(values) / len(values))
    full_rounds = max(int(base_rounds * 1.2), 100)
    return base_rounds, full_rounds


def _atomic_write(path: Path, raw: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _model_entry(
    path: Path, seed: int, feature_count: int, rounds: int
) -> dict[str, object]:
    model = lgb.Booster(model_file=str(path))
    if model.num_feature() != feature_count:
        raise TrainingContractError(f"saved model feature count drifted: {path.name}")
    if model.current_iteration() != rounds:
        raise TrainingContractError(f"saved model rounds drifted: {path.name}")
    return {
        "path": path.name,
        "seed": seed,
        "sha256": sha256_file(path),
        "feature_count": feature_count,
        "rounds": rounds,
    }


def _train_family_models(
    *,
    family: str,
    features: pd.DataFrame,
    target: pd.Series,
    rounds: int,
    categorical: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Train to temporary files, validate them, then activate all three."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[tuple[Path, Path, int]] = []
    try:
        for seed, filename in zip(
            CANONICAL_SEEDS, ACTIVE_MODEL_NAMES[family], strict=True
        ):
            final_path = MODEL_DIR / filename
            temporary_path = final_path.with_name(f".{filename}.{os.getpid()}.tmp")
            params = dict(BASELINE_PARAMS, seed=seed)
            train_set = lgb.Dataset(
                features,
                target,
                categorical_feature=list(categorical) if categorical else "auto",
            )
            model = lgb.train(params, train_set, num_boost_round=rounds)
            model.save_model(str(temporary_path))
            loaded = lgb.Booster(model_file=str(temporary_path))
            if loaded.num_feature() != features.shape[1]:
                raise TrainingContractError(
                    f"temporary model feature count drifted: {filename}"
                )
            if loaded.current_iteration() != rounds:
                raise TrainingContractError(
                    f"temporary model rounds drifted: {filename}"
                )
            if tuple(loaded.feature_name()) != tuple(features.columns):
                raise TrainingContractError(
                    f"temporary model feature names drifted: {filename}"
                )
            temporary_paths.append((temporary_path, final_path, seed))
        for temporary_path, final_path, _ in temporary_paths:
            os.replace(temporary_path, final_path)
        return [
            _model_entry(final_path, seed, features.shape[1], rounds)
            for _, final_path, seed in temporary_paths
        ]
    finally:
        for temporary_path, _, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _source_hashes() -> dict[str, dict[str, str]]:
    sources = {
        "ldaps_test": TEST_DIR / "ldaps_test.csv",
        "gfs_test": TEST_DIR / "gfs_test.csv",
        "sample_submission": TEST_DIR.parent / "sample_submission.csv",
    }
    return {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in sorted(sources.items())
    }


def _calibration_record(calibration: WeightCalibration) -> dict[str, object]:
    record = asdict(calibration)
    record["train_label_years"] = list(record["train_label_years"])
    record["turbine_columns"] = list(record["turbine_columns"])
    record["weights"] = list(record["weights"])
    return record


def _validate_gate_and_fold24(
    bundle: TrainingBundle,
) -> tuple[dict[str, object], FoldPredictions]:
    gate = read_final_gate_artifact(
        expected_feature_hash=bundle.feature_hash,
        expected_source_hashes=bundle.data_hashes,
    )
    fold24 = fit_fold(
        WEIGHTED_RECIPE,
        (2022, 2023),
        2024,
        tuple(GROUPS),
        CANONICAL_SEEDS,
    )
    gate_folds = cast(Mapping[str, Mapping[str, Mapping[str, object]]], gate["folds"])
    evidence = gate_folds["fold24"]["candidate"]
    if fold24.provenance.manifest_key != evidence["manifest_key"]:
        raise TrainingContractError("candidate fold24 manifest parity failed")
    declared_iterations = gate["fold24_candidate_best_iterations"]
    actual_iterations = {
        family: list(values)
        for family, values in fold24.provenance.best_iterations.items()
    }
    if declared_iterations != actual_iterations:
        raise TrainingContractError("candidate fold24 iteration parity failed")
    return gate, fold24


def main() -> None:
    bundle = load_bundle()
    gate, fold24 = _validate_gate_and_fold24(bundle)
    gate_folds = cast(Mapping[str, Mapping[str, Mapping[str, object]]], gate["folds"])

    weighted_targets, calibrations = build_weighted_targets(
        bundle.labels.index,
        train_label_years=FULL_LABEL_YEARS,
        target_label_years=FULL_LABEL_YEARS,
        groups=tuple(GROUPS),
    )
    _assert_scada_source_hashes(bundle, tuple(GROUPS))
    boundary = pd.Timestamp("2025-01-01 00:00:00")
    if (
        boundary not in weighted_targets.index
        or weighted_targets.loc[boundary].notna().any()
    ):
        raise TrainingContractError("2025 label boundary entered production targets")

    solo_frames = {
        group: _solo_train_frame(bundle, group, FULL_LABEL_YEARS, weighted_targets)
        for group in G12
    }
    pooled_frame, pooled_columns = _pooled_train_frame(
        bundle, FULL_LABEL_YEARS, weighted_targets
    )
    expected_rows = {
        "kpx_group_1": 18_213,
        "kpx_group_2": 18_131,
        "pooled": 46_331,
    }
    actual_rows = {
        **{group: len(frame) for group, frame in solo_frames.items()},
        "pooled": len(pooled_frame),
    }
    if actual_rows != expected_rows:
        raise TrainingContractError(
            f"full training row fingerprint drift: {actual_rows}"
        )

    iterations = cast(Mapping[str, list[int]], gate["fold24_candidate_best_iterations"])
    family_frames = {
        "kpx_group_1": (
            solo_frames["kpx_group_1"][list(bundle.feature_columns)],
            solo_frames["kpx_group_1"]["_target"],
            (),
        ),
        "kpx_group_2": (
            solo_frames["kpx_group_2"][list(bundle.feature_columns)],
            solo_frames["kpx_group_2"]["_target"],
            (),
        ),
        "pooled": (
            pooled_frame[list(pooled_columns)],
            pooled_frame["normalized_target"],
            ("group_id",),
        ),
    }

    training_families: dict[str, dict[str, object]] = {}
    for family, (features, target, categorical) in family_frames.items():
        raw_iterations = tuple(iterations[family])
        base_rounds, full_rounds = calculate_full_rounds(raw_iterations)
        models = _train_family_models(
            family=family,
            features=features,
            target=target,
            rounds=full_rounds,
            categorical=categorical,
        )
        target_evidence: pd.Series | pd.DataFrame
        if family == "pooled":
            target_evidence = pooled_frame[["normalized_target", "group_id"]]
        else:
            target_evidence = target
        training_families[family] = {
            "row_count": len(features),
            "index_sha256": _pandas_hash(features.index),
            "target_sha256": _pandas_hash(target_evidence),
            "validation_iterations": list(raw_iterations),
            "base_rounds": base_rounds,
            "full_rounds": full_rounds,
            "categorical_features": list(categorical),
            "models": models,
        }
        print(
            f"[{family}] rows={len(features):,}, base={base_rounds}, "
            f"full={full_rounds}, models={len(models)}"
        )

    feature_raw = "\n".join(bundle.feature_columns).encode("utf-8")
    floor_ratio = cast(float, BASELINE_POSTPROCESS_CONFIG["floor_ratio"])
    post_payload = {
        "config": BASELINE_POSTPROCESS_CONFIG,
        "config_sha256": recipe_fingerprint(BASELINE_POSTPROCESS_CONFIG),
        "groups": {
            group: {
                "scale": 1.0,
                "floor_ratio": floor_ratio,
                "floor_kwh": floor_ratio * CAPACITY_KWH[group],
            }
            for group in GROUPS
        },
    }
    post_raw = canonical_json_bytes(post_payload)
    _atomic_write(MODEL_DIR / "feature_cols.txt", feature_raw)
    _atomic_write(MODEL_DIR / "post_params.json", post_raw)

    calibration_targets = {}
    for group in GROUPS:
        column = f"{group}_weighted_potential"
        calibration_targets[group] = {
            "calibration": _calibration_record(calibrations[group]),
            "weighted_target_sha256": _pandas_hash(weighted_targets[column]),
            "weighted_non_null_count": int(weighted_targets[column].notna().sum()),
        }

    recipe = {
        "kind": PRODUCTION_RECIPE_KIND,
        "schema_version": PRODUCTION_RECIPE_SCHEMA_VERSION,
        "recipe": {
            "name": WEIGHTED_RECIPE,
            "config": WEIGHTED_RECIPE_CONFIG,
            "config_sha256": recipe_fingerprint(WEIGHTED_RECIPE_CONFIG),
        },
        "promotion": {
            "status": gate["status"],
            "artifact_sha256": sha256_file(FINAL_GATE_PATH),
            "seeds": gate["seeds"],
            "mean_delta": gate["mean_delta"],
            "folds": gate_folds,
            "candidate_manifests": {
                name: gate_folds[name]["candidate"]["manifest_key"]
                for name in ("fold23", "fold24")
            },
            "fold24_best_iterations_sha256": gate[
                "fold24_candidate_best_iterations_hash"
            ],
        },
        "sources": {
            "training": dict(sorted(bundle.data_hashes.items())),
            "inference": _source_hashes(),
        },
        "features": {
            "path": "feature_cols.txt",
            "file_sha256": hashlib.sha256(feature_raw).hexdigest(),
            "count": len(bundle.feature_columns),
            "ordered_columns_sha256": _canonical_hash(list(bundle.feature_columns)),
            "training_frame_sha256": bundle.feature_hash,
        },
        "targets": {
            "label_years": list(FULL_LABEL_YEARS),
            "label_year_boundary": "raw_scada_timestamp.ceil('h')",
            "weighted_frame_sha256": _pandas_hash(weighted_targets),
            "groups": calibration_targets,
            "excluded_boundary": "2025-01-01 00:00:00",
        },
        "training": {
            "params": BASELINE_PARAMS,
            "seeds": list(CANONICAL_SEEDS),
            "filter_ratio": TRAIN_FILTER_RATIO,
            "mismatch_frame_sha256": bundle.derived_data_hashes["mismatch_frame"],
            "fold24_manifest_key": fold24.provenance.manifest_key,
            "families": training_families,
        },
        "postprocess": {
            "path": "post_params.json",
            "file_sha256": hashlib.sha256(post_raw).hexdigest(),
            **post_payload,
        },
        "capacities": dict(CAPACITY_KWH),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    recipe_raw = canonical_json_bytes(recipe)
    recipe_path = MODEL_DIR / "recipe.json"
    recipe_digest = hashlib.sha256(recipe_raw).hexdigest()
    _atomic_write(
        MODEL_DIR / "recipe.json.sha256",
        f"{recipe_digest}  recipe.json\n".encode("ascii"),
    )
    _atomic_write(recipe_path, recipe_raw)
    print(f"production recipe: {recipe_path} ({recipe_digest})")


if __name__ == "__main__":
    main()
