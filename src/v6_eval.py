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
from scada import (
    WeightCalibration,
    build_mismatch_mask,
    build_potential,
    build_weighted_targets,
)


BASELINE_RECIPE = "v5-c1-potential-q60-filter05-floor10"
WEIGHTED_RECIPE = "v6-weighted-potential-q60-filter05-floor10"
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
CACHE_SCHEMA_VERSION = 3
FINAL_GATE_SCHEMA_VERSION = 1
FINAL_GATE_KIND = "wind-v6-final-gate"
CANONICAL_SEEDS = (42, 202, 777)
FINAL_GATE_PATH = EXPERIMENT_DIR / "final-gate.json"
ANCHOR_TOLERANCE = 0.00015
BASELINE_ANCHORS = {"fold23": 0.6316, "fold24": 0.6380}
WEIGHTED_ANCHORS = {
    "fold23": 0.633691786326,
    "fold24": 0.639226950590,
}
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
WEIGHTED_EXPECTED_ROWS = {
    "fold23": {
        "g1_train": 6214,
        "g2_train": 6176,
        "g1_valid": 8757,
        "g2_valid": 8758,
    },
    "fold24": {
        "g1_train": 12516,
        "g2_train": 12458,
        "pooled_train": 30158,
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
WEIGHTED_RECIPE_CONFIG = {
    "version": 1,
    "name": WEIGHTED_RECIPE,
    "training": {
        "target": "fold_safe_turbine_weighted_scada_potential",
        "exclude_scada_mismatch": True,
        "filter_ratio": TRAIN_FILTER_RATIO,
        "label_year_boundary": "raw_scada_timestamp.ceil('h')",
        "calibration": {
            "complete_power_and_wind": True,
            "min_power": 1.0,
            "min_wind_speed": 5.0,
            "min_group_output_capacity_ratio_per_interval": 0.10 / 6.0,
            "estimator": "n_times_median_turbine_share_then_mean_one",
        },
        "healthy": "power_gt_1_or_wind_lt_5",
        "min_healthy": {
            "kpx_group_1": 3,
            "kpx_group_2": 3,
            "kpx_group_3": 2,
        },
        "clip_each_10m": True,
        "hourly_aggregation": "mean_times_6",
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
    best_iterations: dict[str, tuple[int, ...]]
    best_iterations_hash: str
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


def _expected_best_iteration_families(groups: tuple[str, ...]) -> tuple[str, ...]:
    families = tuple(group for group in groups if group in G12)
    if G3 in groups:
        families += ("pooled",)
    if len(families) != len(groups):
        raise ProvenanceError("best iteration family keys do not match groups")
    return families


def _normalize_best_iterations(
    best_iterations: Mapping[str, object],
    *,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """Validate and preserve one positive iteration per seed and model family."""
    if not isinstance(best_iterations, Mapping):
        raise ProvenanceError("best iterations must be a mapping")
    expected_families = _expected_best_iteration_families(groups)
    if set(best_iterations) != set(expected_families):
        raise ProvenanceError("best iteration family keys do not match groups")
    normalized: dict[str, tuple[int, ...]] = {}
    for family in expected_families:
        values = best_iterations[family]
        if not isinstance(values, (list, tuple)):
            raise ProvenanceError(
                f"{family} best iteration evidence must be an ordered sequence"
            )
        if len(values) != len(seeds):
            raise ProvenanceError(
                f"{family} best iteration cardinality does not match seeds"
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in values
        ):
            raise ProvenanceError(f"{family} best iterations must be positive integers")
        normalized[family] = tuple(int(value) for value in values)
    return normalized


def best_iterations_fingerprint(
    seeds: tuple[int, ...], best_iterations: Mapping[str, tuple[int, ...]]
) -> str:
    """Bind ordered seed evidence without making it part of the pre-fit cache key."""
    return _canonical_hash(
        {
            "families": {
                family: list(values)
                for family, values in sorted(best_iterations.items())
            },
            "seeds": list(seeds),
        }
    )


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


def _raw_source_paths() -> dict[str, Path]:
    return {
        "ldaps_train": TRAIN_DIR / "ldaps_train.csv",
        "gfs_train": TRAIN_DIR / "gfs_train.csv",
        "train_labels": TRAIN_DIR / "train_labels.csv",
        "scada_vestas": TRAIN_DIR / "scada_vestas_train.csv",
        "scada_unison": TRAIN_DIR / "scada_unison_train.csv",
    }


def _current_source_hashes() -> dict[str, str]:
    return dict(
        sorted((name, _file_hash(path)) for name, path in _raw_source_paths().items())
    )


def _assert_scada_source_hashes(
    bundle: TrainingBundle, groups: tuple[str, ...]
) -> None:
    """Fail if SCADA bytes changed after weighted-target reconstruction."""
    sources = {
        "vestas": ("scada_vestas", TRAIN_DIR / "scada_vestas_train.csv"),
        "unison": ("scada_unison", TRAIN_DIR / "scada_unison_train.csv"),
    }
    makers = {"unison" if group == G3 else "vestas" for group in groups}
    for maker in sorted(makers):
        source_name, path = sources[maker]
        expected = bundle.data_hashes.get(source_name)
        if expected is None or _file_hash(path) != expected:
            raise ProvenanceError(
                f"SCADA source hash changed during weighted target reconstruction: "
                f"{source_name}"
            )


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
    best_iterations: Mapping[str, object],
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
    normalized_best_iterations = _normalize_best_iterations(
        best_iterations, groups=group_tuple, seeds=seed_tuple
    )
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
        best_iterations=normalized_best_iterations,
        best_iterations_hash=best_iterations_fingerprint(
            seed_tuple, normalized_best_iterations
        ),
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
    normalized_best_iterations = _normalize_best_iterations(
        provenance.best_iterations,
        groups=groups,
        seeds=provenance.seeds,
    )
    if normalized_best_iterations != provenance.best_iterations:
        raise ProvenanceError("best iteration evidence is not normalized")
    if provenance.best_iterations_hash != best_iterations_fingerprint(
        provenance.seeds, normalized_best_iterations
    ):
        raise ProvenanceError("best iteration fingerprint does not match")
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
    data_hashes = _current_source_hashes()
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
    bundle: TrainingBundle,
    group: str,
    train_years: tuple[int, ...],
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    capacity = CAPACITY_KWH[group]
    frame = bundle.data.dropna(subset=[group])
    train = frame[frame.index.year.isin(train_years)].copy()
    target_frame = bundle.potential if targets is None else targets
    suffix = "potential" if targets is None else "weighted_potential"
    train["_target"] = target_frame[f"{group}_{suffix}"].reindex(train.index)
    mismatch = bundle.mismatch[f"{group}_mismatch"].reindex(train.index).fillna(False)
    train = train[~mismatch].dropna(subset=["_target"])
    return train[train["_target"] >= TRAIN_FILTER_RATIO * capacity]


def _pooled_train_frame(
    bundle: TrainingBundle,
    train_years: tuple[int, ...],
    targets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frames = []
    columns = bundle.feature_columns
    target_frame = bundle.potential if targets is None else targets
    suffix = "potential" if targets is None else "weighted_potential"
    for group in GROUPS:
        frame = bundle.data.dropna(subset=[group]).copy()
        frame["_target"] = target_frame[f"{group}_{suffix}"].reindex(frame.index)
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
) -> tuple[np.ndarray, tuple[int, ...]]:
    predictions = []
    best_iterations = []
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
        best_iterations.append(int(model.best_iteration))
    return np.mean(predictions, axis=0), tuple(best_iterations)


def _fold_name(valid_year: int) -> str:
    return f"fold{str(valid_year)[-2:]}"


def assert_baseline_fingerprint(fold_name: str, row_counts: Mapping[str, int]) -> None:
    expected = EXPECTED_ROWS.get(fold_name)
    actual = {key: int(value) for key, value in row_counts.items()}
    if expected is None or actual != expected:
        raise ProvenanceError(
            f"{fold_name} row fingerprint drift: expected {expected}, got {actual}"
        )


def assert_weighted_fingerprint(fold_name: str, row_counts: Mapping[str, int]) -> None:
    expected = WEIGHTED_EXPECTED_ROWS.get(fold_name)
    actual = {key: int(value) for key, value in row_counts.items()}
    if expected is None or actual != expected:
        raise ProvenanceError(
            f"{fold_name} weighted row fingerprint drift: expected {expected}, "
            f"got {actual}"
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


def assert_weighted_score_anchor(
    fold_name: str, score: float, *, seeds: tuple[int, ...]
) -> None:
    if tuple(int(seed) for seed in seeds) != (42,):
        return
    expected = WEIGHTED_ANCHORS.get(fold_name)
    if expected is None or not np.isfinite(score) or abs(score - expected) > 1e-6:
        raise ProvenanceError(
            f"{fold_name} weighted score anchor drift: expected {expected} +/- "
            f"1e-06, got {score}"
        )


def _weighted_derived_hashes(
    bundle: TrainingBundle,
    weighted_targets: pd.DataFrame,
    calibrations: Mapping[str, WeightCalibration],
    solo_frames: Mapping[str, pd.DataFrame],
    pooled_frame: pd.DataFrame | None,
) -> dict[str, str]:
    """Hash every candidate-specific input after fold-safe target construction."""
    hashes = {
        "mismatch_frame": bundle.derived_data_hashes["mismatch_frame"],
        "weighted_targets_frame": _pandas_hash(weighted_targets),
    }
    for group, calibration in calibrations.items():
        prefix = f"g{group[-1]}"
        hashes[f"{prefix}_calibration_metadata"] = _canonical_hash(asdict(calibration))
        hashes[f"{prefix}_calibration_index"] = calibration.calibration_index_hash
        hashes[f"{prefix}_weights"] = calibration.weights_hash
        hashes[f"{prefix}_weighted_target"] = _pandas_hash(
            weighted_targets[f"{group}_weighted_potential"]
        )
    for group, train in solo_frames.items():
        prefix = f"g{group[-1]}"
        hashes[f"{prefix}_postfilter_train_index"] = _pandas_hash(train.index)
        hashes[f"{prefix}_postfilter_train_target"] = _pandas_hash(train["_target"])
    if pooled_frame is not None:
        hashes["pooled_postfilter_train_index"] = _pandas_hash(pooled_frame.index)
        hashes["pooled_postfilter_train_target"] = _pandas_hash(
            pooled_frame[["normalized_target", "group_id"]]
        )
    return dict(sorted(hashes.items()))


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
            "best_iterations",
            "best_iterations_hash",
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
            best_iterations=manifest["best_iterations"],
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
    """Fit or load one exact baseline or weighted-candidate validation fold."""
    train_years = tuple(train_years)
    groups = tuple(groups)
    seeds = tuple(int(seed) for seed in seeds)
    if recipe not in {BASELINE_RECIPE, WEIGHTED_RECIPE}:
        raise ValueError(f"unsupported recipe: {recipe}")
    if not seeds:
        raise ValueError("at least one seed is required")
    expected_fold = {
        ((2022,), 2023, G12),
        ((2022, 2023), 2024, tuple(GROUPS)),
    }
    if (train_years, valid_year, groups) not in expected_fold:
        raise ValueError(
            "stage7 evaluator supports only the frozen fold23/fold24 splits"
        )

    bundle = load_bundle()
    weighted_targets: pd.DataFrame | None = None
    calibrations: dict[str, WeightCalibration] = {}
    if recipe == WEIGHTED_RECIPE:
        weighted_targets, calibrations = build_weighted_targets(
            bundle.labels.index,
            train_label_years=train_years,
            target_label_years=train_years,
            groups=groups,
        )
        _assert_scada_source_hashes(bundle, groups)
    row_counts: dict[str, int] = {}
    solo_frames: dict[str, pd.DataFrame] = {}
    validation_frames: dict[str, pd.DataFrame] = {}
    for group in groups:
        source = bundle.data.dropna(subset=[group])
        validation = source[source.index.year == valid_year]
        validation_frames[group] = validation
        row_counts[f"g{group[-1]}_valid"] = len(validation)
        if group in G12:
            solo_frames[group] = _solo_train_frame(
                bundle, group, train_years, weighted_targets
            )
            row_counts[f"g{group[-1]}_train"] = len(solo_frames[group])

    pooled_frame: pd.DataFrame | None = None
    pooled_columns: tuple[str, ...] = ()
    if G3 in groups:
        pooled_frame, pooled_columns = _pooled_train_frame(
            bundle, train_years, weighted_targets
        )
        row_counts["pooled_train"] = len(pooled_frame)
    row_counts = dict(sorted(row_counts.items()))
    fold_name = _fold_name(valid_year)
    if recipe == BASELINE_RECIPE:
        assert_baseline_fingerprint(fold_name, row_counts)
        recipe_config = BASELINE_RECIPE_CONFIG
        derived_data_hashes = bundle.derived_data_hashes
    else:
        assert_weighted_fingerprint(fold_name, row_counts)
        if weighted_targets is None:
            raise ProvenanceError("weighted targets are missing")
        recipe_config = WEIGHTED_RECIPE_CONFIG
        derived_data_hashes = _weighted_derived_hashes(
            bundle,
            weighted_targets,
            calibrations,
            solo_frames,
            pooled_frame,
        )
    validation_targets = {
        group: validation_frames[group][group].copy().rename(group) for group in groups
    }
    current_recipe_hash = recipe_fingerprint(recipe_config)

    cached = _read_prediction_cache(
        cache_dir=EXPERIMENT_DIR,
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        recipe_fingerprint=current_recipe_hash,
        postprocess_config=BASELINE_POSTPROCESS_CONFIG,
        feature_fingerprint=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
        derived_data_hashes=derived_data_hashes,
        row_counts=row_counts,
        live_validation_targets=validation_targets,
    )
    if cached is not None:
        return cached

    model_predictions: dict[str, pd.Series] = {}
    best_iterations: dict[str, tuple[int, ...]] = {}
    columns = list(bundle.feature_columns)
    for group in groups:
        capacity = CAPACITY_KWH[group]
        validation = validation_frames[group]
        validation_target = validation_targets[group]
        if group in G12:
            train = solo_frames[group]
            raw_prediction, family_best_iterations = _fit_ensemble(
                train[columns],
                train["_target"],
                validation[columns],
                validation_target,
                seeds,
            )
            best_iterations[group] = family_best_iterations
        else:
            if pooled_frame is None:
                raise ProvenanceError("pooled training frame is missing")
            validation_features = validation[columns].copy()
            validation_features["group_id"] = 3
            normalized_prediction, family_best_iterations = _fit_ensemble(
                pooled_frame[list(pooled_columns)],
                pooled_frame["normalized_target"],
                validation_features[list(pooled_columns)],
                validation_target / capacity,
                seeds,
                categorical=("group_id",),
            )
            raw_prediction = normalized_prediction * capacity
            best_iterations["pooled"] = family_best_iterations
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
        recipe_config=recipe_config,
        postprocess_config=BASELINE_POSTPROCESS_CONFIG,
        feature_hash=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
        derived_data_hashes=derived_data_hashes,
        best_iterations=best_iterations,
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
        expected_derived_data_hashes=derived_data_hashes,
        expected_recipe_hash=current_recipe_hash,
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
    elif provenance.recipe == WEIGHTED_RECIPE:
        if provenance.recipe_config != _json_object(
            WEIGHTED_RECIPE_CONFIG, "weighted recipe config"
        ) or provenance.recipe_hash != recipe_fingerprint(WEIGHTED_RECIPE_CONFIG):
            raise ProvenanceError("weighted recipe config drift")
        if provenance.postprocess_config != _json_object(
            BASELINE_POSTPROCESS_CONFIG, "weighted postprocess config"
        ) or provenance.postprocess_hash != recipe_fingerprint(
            BASELINE_POSTPROCESS_CONFIG
        ):
            raise ProvenanceError("weighted postprocess config drift")
        assert_weighted_fingerprint(
            _fold_name(provenance.valid_year), provenance.row_counts
        )
    metrics = fold_metrics(predictions)
    nmaes = [values["nmae"] for values in metrics.values()]
    ficrs = [values["ficr"] for values in metrics.values()]
    score = 0.5 * (1.0 - float(np.mean(nmaes))) + 0.5 * float(np.mean(ficrs))
    if not np.isfinite(score):
        raise ProvenanceError("fold score is not finite")
    return score


def fold_metrics(predictions: FoldPredictions) -> dict[str, dict[str, float]]:
    """Return per-group components from hash-verified live validation targets."""
    validate_fold_predictions(predictions)
    metrics: dict[str, dict[str, float]] = {}
    for group in predictions.provenance.groups:
        model_prediction = predictions.model_predictions[group].to_numpy()
        validation_target = predictions.validation_targets[group].to_numpy()
        capacity = CAPACITY_KWH[group]
        nmae = float(group_nmae(model_prediction, validation_target, capacity))
        ficr = float(group_ficr(model_prediction, validation_target, capacity))
        score = 0.5 * (1.0 - nmae) + 0.5 * ficr
        if not np.isfinite([nmae, ficr, score]).all():
            raise ProvenanceError(f"{group} metrics are not finite")
        metrics[group] = {"nmae": nmae, "ficr": ficr, "score": score}
    return metrics


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
        raise ProvenanceError("final gate must be canonical JSON") from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ProvenanceError(f"{field_name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ProvenanceError(f"{field_name} must be a finite number")
    return number


def _validate_final_gate_payload(
    payload: Mapping[str, object],
    *,
    expected_feature_hash: str,
    expected_source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Strictly validate promotion evidence and recompute its score contract."""
    expected_fields = {
        "kind",
        "schema_version",
        "cache_schema_version",
        "status",
        "seeds",
        "candidate_recipe",
        "recipes",
        "hashes",
        "folds",
        "mean_delta",
        "fold24_candidate_best_iterations",
        "fold24_candidate_best_iterations_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ProvenanceError("final gate fields do not match schema")
    if payload["kind"] != FINAL_GATE_KIND:
        raise ProvenanceError("final gate kind does not match")
    if payload["schema_version"] != FINAL_GATE_SCHEMA_VERSION:
        raise ProvenanceError("final gate schema version does not match")
    if payload["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ProvenanceError("final gate cache schema version does not match")
    if payload["status"] != "PASS":
        raise ProvenanceError("final gate status is not PASS")
    if payload["seeds"] != list(CANONICAL_SEEDS):
        raise ProvenanceError("final gate seeds are not canonical")
    recipes = payload["recipes"]
    expected_recipes = {
        "baseline": BASELINE_RECIPE,
        "candidate": WEIGHTED_RECIPE,
    }
    if recipes != expected_recipes or payload["candidate_recipe"] != WEIGHTED_RECIPE:
        raise ProvenanceError("final gate recipes do not match")
    hashes = payload["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "recipes",
        "postprocess",
        "features",
        "sources",
    }:
        raise ProvenanceError("final gate hashes do not match schema")
    expected_recipe_hashes = {
        "baseline": recipe_fingerprint(BASELINE_RECIPE_CONFIG),
        "candidate": recipe_fingerprint(WEIGHTED_RECIPE_CONFIG),
    }
    if hashes["recipes"] != expected_recipe_hashes:
        raise ProvenanceError("final gate recipe hashes do not match")
    if hashes["postprocess"] != recipe_fingerprint(BASELINE_POSTPROCESS_CONFIG):
        raise ProvenanceError("final gate postprocess hash does not match")
    if hashes["features"] != expected_feature_hash or not _is_sha256(
        hashes["features"]
    ):
        raise ProvenanceError("final gate feature hash does not match")
    normalized_sources = dict(sorted(expected_source_hashes.items()))
    if hashes["sources"] != normalized_sources or not all(
        _is_sha256(value) for value in normalized_sources.values()
    ):
        raise ProvenanceError("final gate source hashes do not match")

    folds = payload["folds"]
    if not isinstance(folds, Mapping) or set(folds) != {"fold23", "fold24"}:
        raise ProvenanceError("final gate folds do not match")
    baseline_scores: dict[str, float] = {}
    candidate_scores: dict[str, float] = {}
    for fold_name, expected_groups in (
        ("fold23", G12),
        ("fold24", tuple(GROUPS)),
    ):
        fold = folds[fold_name]
        if not isinstance(fold, Mapping) or set(fold) != {
            "baseline",
            "candidate",
            "delta",
        }:
            raise ProvenanceError(f"{fold_name} fields do not match schema")
        for lane, destination in (
            ("baseline", baseline_scores),
            ("candidate", candidate_scores),
        ):
            evidence = fold[lane]
            if not isinstance(evidence, Mapping) or set(evidence) != {
                "score",
                "metrics",
                "manifest_key",
            }:
                raise ProvenanceError(f"{fold_name} {lane} fields do not match schema")
            if not _is_sha256(evidence["manifest_key"]):
                raise ProvenanceError(f"{fold_name} {lane} manifest key is invalid")
            metrics = evidence["metrics"]
            if not isinstance(metrics, Mapping) or set(metrics) != set(expected_groups):
                raise ProvenanceError(f"{fold_name} {lane} metric groups do not match")
            group_scores = []
            for group in expected_groups:
                values = metrics[group]
                if not isinstance(values, Mapping) or set(values) != {
                    "nmae",
                    "ficr",
                    "score",
                }:
                    raise ProvenanceError(
                        f"{fold_name} {lane} {group} metrics do not match schema"
                    )
                nmae = _finite_number(values["nmae"], f"{group} nmae")
                ficr = _finite_number(values["ficr"], f"{group} ficr")
                declared_group_score = _finite_number(values["score"], f"{group} score")
                computed_group_score = 0.5 * (1.0 - nmae) + 0.5 * ficr
                if not np.isclose(
                    declared_group_score, computed_group_score, rtol=0.0, atol=1e-12
                ):
                    raise ProvenanceError(f"{group} score does not match metrics")
                group_scores.append(computed_group_score)
            computed_score = float(np.mean(group_scores))
            declared_score = _finite_number(
                evidence["score"], f"{fold_name} {lane} score"
            )
            if not np.isclose(declared_score, computed_score, rtol=0.0, atol=1e-12):
                raise ProvenanceError(f"{fold_name} {lane} score does not match")
            destination[fold_name] = declared_score
        computed_delta = candidate_scores[fold_name] - baseline_scores[fold_name]
        declared_delta = _finite_number(fold["delta"], f"{fold_name} delta")
        if not np.isclose(declared_delta, computed_delta, rtol=0.0, atol=1e-12):
            raise ProvenanceError(f"{fold_name} delta does not match")

    gate = gate_scores(baseline_scores, candidate_scores)
    if gate["status"] != "PASS" or not np.isclose(
        _finite_number(payload["mean_delta"], "mean delta"),
        gate["mean_delta"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ProvenanceError("final gate result does not pass score contract")
    best_iteration_payload = payload["fold24_candidate_best_iterations"]
    if not isinstance(best_iteration_payload, Mapping):
        raise ProvenanceError("final gate best iterations must be a mapping")
    normalized_best_iterations = _normalize_best_iterations(
        best_iteration_payload,
        groups=tuple(GROUPS),
        seeds=CANONICAL_SEEDS,
    )
    serialized_best_iterations = {
        family: list(values) for family, values in normalized_best_iterations.items()
    }
    if payload["fold24_candidate_best_iterations"] != serialized_best_iterations:
        raise ProvenanceError("final gate best iterations are not normalized")
    expected_best_iterations_hash = best_iterations_fingerprint(
        CANONICAL_SEEDS, normalized_best_iterations
    )
    if (
        payload["fold24_candidate_best_iterations_hash"]
        != expected_best_iterations_hash
    ):
        raise ProvenanceError("final gate best iteration fingerprint does not match")
    return json.loads(_canonical_json_bytes(payload))


def read_final_gate_artifact(
    path: Path = FINAL_GATE_PATH,
    *,
    expected_feature_hash: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Read hash-bound canonical PASS evidence and compare it with current inputs."""
    path = Path(path)
    sidecar_path = path.with_name(f"{path.name}.sha256")
    try:
        raw = path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="ascii")
    except OSError as error:
        raise ProvenanceError(f"final gate artifact is incomplete: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar != f"{digest}  {path.name}\n":
        raise ProvenanceError("final gate SHA-256 sidecar does not match")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError("final gate JSON is malformed") from error
    if raw != _canonical_json_bytes(payload):
        raise ProvenanceError("final gate JSON bytes are not canonical")
    if expected_feature_hash is None or expected_source_hashes is None:
        bundle = load_bundle()
        if expected_feature_hash is None:
            expected_feature_hash = bundle.feature_hash
        if expected_source_hashes is None:
            expected_source_hashes = _current_source_hashes()
    return _validate_final_gate_payload(
        payload,
        expected_feature_hash=expected_feature_hash,
        expected_source_hashes=expected_source_hashes,
    )


def write_final_gate_artifact(
    payload: Mapping[str, object],
    path: Path = FINAL_GATE_PATH,
    *,
    expected_feature_hash: str,
    expected_source_hashes: Mapping[str, str],
) -> tuple[Path, Path]:
    """Atomically publish exact-seed PASS evidence and its byte digest."""
    normalized = _validate_final_gate_payload(
        payload,
        expected_feature_hash=expected_feature_hash,
        expected_source_hashes=expected_source_hashes,
    )
    raw = _canonical_json_bytes(normalized)
    path = Path(path)
    sidecar_path = path.with_name(f"{path.name}.sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = path.with_name(f".{path.name}.tmp")
    temporary_sidecar = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    digest = hashlib.sha256(raw).hexdigest()
    temporary_json.write_bytes(raw)
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    temporary_json.replace(path)
    temporary_sidecar.replace(sidecar_path)
    read_final_gate_artifact(
        path,
        expected_feature_hash=expected_feature_hash,
        expected_source_hashes=expected_source_hashes,
    )
    return path, sidecar_path


def _build_final_gate_payload(
    *,
    baseline_folds: Mapping[str, FoldPredictions],
    candidate_folds: Mapping[str, FoldPredictions],
    baseline_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
) -> dict[str, object]:
    gate = gate_scores(dict(baseline_scores), dict(candidate_scores))
    if gate["status"] != "PASS":
        raise ProvenanceError("failed gate cannot become promotion evidence")
    fold24_provenance = candidate_folds["fold24"].provenance
    payload: dict[str, object] = {
        "kind": FINAL_GATE_KIND,
        "schema_version": FINAL_GATE_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "status": "PASS",
        "seeds": list(CANONICAL_SEEDS),
        "candidate_recipe": WEIGHTED_RECIPE,
        "recipes": {
            "baseline": BASELINE_RECIPE,
            "candidate": WEIGHTED_RECIPE,
        },
        "hashes": {
            "recipes": {
                "baseline": recipe_fingerprint(BASELINE_RECIPE_CONFIG),
                "candidate": recipe_fingerprint(WEIGHTED_RECIPE_CONFIG),
            },
            "postprocess": recipe_fingerprint(BASELINE_POSTPROCESS_CONFIG),
            "features": fold24_provenance.feature_hash,
            "sources": dict(sorted(fold24_provenance.data_hashes.items())),
        },
        "folds": {
            fold_name: {
                "baseline": {
                    "score": baseline_scores[fold_name],
                    "metrics": fold_metrics(baseline_folds[fold_name]),
                    "manifest_key": baseline_folds[fold_name].provenance.manifest_key,
                },
                "candidate": {
                    "score": candidate_scores[fold_name],
                    "metrics": fold_metrics(candidate_folds[fold_name]),
                    "manifest_key": candidate_folds[fold_name].provenance.manifest_key,
                },
                "delta": gate["deltas"][fold_name],
            }
            for fold_name in ("fold23", "fold24")
        },
        "mean_delta": gate["mean_delta"],
        "fold24_candidate_best_iterations": {
            family: list(values)
            for family, values in fold24_provenance.best_iterations.items()
        },
        "fold24_candidate_best_iterations_hash": (
            fold24_provenance.best_iterations_hash
        ),
    }
    return _validate_final_gate_payload(
        payload,
        expected_feature_hash=fold24_provenance.feature_hash,
        expected_source_hashes=fold24_provenance.data_hashes,
    )


def run_stage7(
    seeds: tuple[int, ...], baseline_only: bool = False, screen: str | None = None
) -> int:
    """Reproduce v5 or compare the fold-safe weighted candidate under one gate."""
    seeds = tuple(int(seed) for seed in seeds)
    if baseline_only and screen is not None:
        print("--baseline-only cannot be combined with --screen", file=sys.stderr)
        return 2
    if not baseline_only and screen is None:
        screen = "weighted"
    if screen not in {None, "weighted"}:
        print(f"unsupported stage7 screen: {screen}", file=sys.stderr)
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
        if baseline_only:
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
            return 0

        candidate23 = fit_fold(WEIGHTED_RECIPE, (2022,), 2023, G12, seeds)
        candidate24 = fit_fold(
            WEIGHTED_RECIPE, (2022, 2023), 2024, tuple(GROUPS), seeds
        )
        assert_weighted_fingerprint("fold23", candidate23.provenance.row_counts)
        assert_weighted_fingerprint("fold24", candidate24.provenance.row_counts)
        assert_fold_alignment(fold23, candidate23)
        assert_fold_alignment(fold24, candidate24)
        candidates = {
            "fold23": score_fold(candidate23),
            "fold24": score_fold(candidate24),
        }
        for fold_name, score in candidates.items():
            assert_weighted_score_anchor(fold_name, score, seeds=seeds)
            print(f"[{fold_name}] weighted v6 score={score:.6f}")
        gate = gate_scores(scores, candidates)
        result: dict[str, object] = {
            **gate,
            "candidate_recipe": WEIGHTED_RECIPE,
            "seeds": list(seeds),
            "metrics": {
                "baseline": {
                    "fold23": fold_metrics(fold23),
                    "fold24": fold_metrics(fold24),
                },
                "candidate": {
                    "fold23": fold_metrics(candidate23),
                    "fold24": fold_metrics(candidate24),
                },
            },
            "manifest_keys": {
                "baseline": {
                    "fold23": fold23.provenance.manifest_key,
                    "fold24": fold24.provenance.manifest_key,
                },
                "candidate": {
                    "fold23": candidate23.provenance.manifest_key,
                    "fold24": candidate24.provenance.manifest_key,
                },
            },
            "best_iterations": {
                "baseline": {
                    "fold23": {
                        family: list(values)
                        for family, values in fold23.provenance.best_iterations.items()
                    },
                    "fold24": {
                        family: list(values)
                        for family, values in fold24.provenance.best_iterations.items()
                    },
                },
                "candidate": {
                    "fold23": {
                        family: list(values)
                        for family, values in candidate23.provenance.best_iterations.items()
                    },
                    "fold24": {
                        family: list(values)
                        for family, values in candidate24.provenance.best_iterations.items()
                    },
                },
            },
        }
        if seeds == CANONICAL_SEEDS and gate["status"] == "PASS":
            result = _build_final_gate_payload(
                baseline_folds={"fold23": fold23, "fold24": fold24},
                candidate_folds={
                    "fold23": candidate23,
                    "fold24": candidate24,
                },
                baseline_scores=scores,
                candidate_scores=candidates,
            )
            write_final_gate_artifact(
                result,
                FINAL_GATE_PATH,
                expected_feature_hash=candidate24.provenance.feature_hash,
                expected_source_hashes=candidate24.provenance.data_hashes,
            )
            sys.stdout.write(_canonical_json_bytes(result).decode("ascii"))
        else:
            print(json.dumps(result, sort_keys=True))
        return 0 if gate["status"] == "PASS" else 1
    except (
        ProvenanceError,
        ValueError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        scope = "baseline" if baseline_only else "weighted candidate"
        print(f"stage7 {scope} rejected: {error}", file=sys.stderr)
        return 2
