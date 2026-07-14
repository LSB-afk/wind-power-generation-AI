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
from features import add_context, build_features
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
CACHE_DIR = ROOT / "cache"
EXPERIMENT_DIR = ROOT / ".omx" / "experiments" / "wind-v6"
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


@dataclass(frozen=True)
class FoldProvenance:
    recipe: str
    train_years: tuple[int, ...]
    valid_year: int
    groups: tuple[str, ...]
    seeds: tuple[int, ...]
    feature_hash: str
    data_hashes: dict[str, str]
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
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
    feature_hash: str,
    data_hashes: Mapping[str, str],
) -> str:
    """Return the identity of a requested fold before predictions exist."""
    return _canonical_hash(
        {
            "recipe": recipe,
            "train_years": list(train_years),
            "valid_year": valid_year,
            "groups": list(groups),
            "seeds": list(seeds),
            "feature_hash": feature_hash,
            "data_hashes": dict(sorted(data_hashes.items())),
        }
    )


def build_provenance(
    *,
    recipe: str,
    train_years: tuple[int, ...],
    valid_year: int,
    groups: tuple[str, ...],
    seeds: tuple[int, ...],
    feature_hash: str,
    data_hashes: Mapping[str, str],
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
    data_hash_dict = dict(sorted(data_hashes.items()))
    return FoldProvenance(
        recipe=recipe,
        train_years=tuple(train_years),
        valid_year=int(valid_year),
        groups=group_tuple,
        seeds=tuple(int(seed) for seed in seeds),
        feature_hash=feature_hash,
        data_hashes=data_hash_dict,
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
            train_years=tuple(train_years),
            valid_year=int(valid_year),
            groups=group_tuple,
            seeds=tuple(int(seed) for seed in seeds),
            feature_hash=feature_hash,
            data_hashes=data_hash_dict,
        ),
    )


def validate_fold_predictions(
    fold: FoldPredictions,
    *,
    expected_recipe: str | None = None,
    expected_seeds: tuple[int, ...] | None = None,
    expected_feature_hash: str | None = None,
    expected_data_hashes: Mapping[str, str] | None = None,
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

    expected_key = manifest_key(
        recipe=provenance.recipe,
        train_years=provenance.train_years,
        valid_year=provenance.valid_year,
        groups=groups,
        seeds=provenance.seeds,
        feature_hash=provenance.feature_hash,
        data_hashes=provenance.data_hashes,
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
            raise ProvenanceError(f"{group} predictions and targets must be indexed Series")
        if prediction.shape != target.shape:
            raise ProvenanceError(f"{group} prediction and target shapes differ")
        if not prediction.index.equals(target.index):
            raise ProvenanceError(f"{group} validation index differs from target index")
        if not prediction.index.is_unique or not prediction.index.is_monotonic_increasing:
            raise ProvenanceError(f"{group} validation index must be unique and sorted")
        if provenance.validation_index_hashes.get(group) != _pandas_hash(prediction.index):
            raise ProvenanceError(f"{group} validation index hash does not match")
        if provenance.target_hashes.get(group) != _pandas_hash(target):
            raise ProvenanceError(f"{group} target hash does not match")
        if provenance.prediction_hashes.get(group) != _pandas_hash(prediction):
            raise ProvenanceError(f"{group} prediction hash does not match")

    if expected_recipe is not None and provenance.recipe != expected_recipe:
        raise ProvenanceError("recipe does not match requested recipe")
    if expected_seeds is not None and provenance.seeds != tuple(expected_seeds):
        raise ProvenanceError("seeds do not match requested seeds")
    if expected_feature_hash is not None and provenance.feature_hash != expected_feature_hash:
        raise ProvenanceError("feature hash does not match requested features")
    if expected_data_hashes is not None and provenance.data_hashes != dict(
        sorted(expected_data_hashes.items())
    ):
        raise ProvenanceError("data hashes do not match requested data")


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
    for group in base.groups:
        base_prediction = baseline.model_predictions[group]
        other_prediction = candidate.model_predictions[group]
        if base_prediction.shape != other_prediction.shape:
            raise ProvenanceError(f"{group} validation shapes differ")
        if not base_prediction.index.equals(other_prediction.index):
            raise ProvenanceError(f"{group} validation index differs")


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
    deltas = {
        fold: candidate[fold] - baseline[fold] for fold in ("fold23", "fold24")
    }
    mean_delta = sum(deltas.values()) / 2.0
    passed = all(delta > 0.0 for delta in deltas.values()) and mean_delta >= min_mean_delta
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
    labels: pd.DataFrame, *, cache_dir: Path = CACHE_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ignored SCADA frames, rebuilding each missing frame from raw files."""
    potential_path = cache_dir / "scada_potential.parquet"
    mismatch_path = cache_dir / "scada_mismatch.parquet"
    potential = (
        pd.read_parquet(potential_path)
        if potential_path.is_file()
        else build_potential(labels)
    )
    mismatch = (
        pd.read_parquet(mismatch_path)
        if mismatch_path.is_file()
        else build_mismatch_mask(labels)
    )
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

    base_path = CACHE_DIR / "train_base.parquet"
    ldaps_cache_path = CACHE_DIR / "train_ldaps_raw.parquet"
    if base_path.is_file() and ldaps_cache_path.is_file():
        base_features = pd.read_parquet(base_path)
        ldaps_raw = pd.read_parquet(ldaps_cache_path)
        available_at = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
            "forecast_kst_dtm"
        )["data_available_kst_dtm"]
        features = add_context(base_features, available_at).sort_index()
    else:
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
    data_hashes["potential_frame"] = _pandas_hash(potential)
    data_hashes["mismatch_frame"] = _pandas_hash(mismatch)
    _BUNDLE = TrainingBundle(
        features=features,
        labels=labels,
        potential=potential,
        mismatch=mismatch,
        data=data,
        feature_columns=tuple(features.columns),
        feature_hash=feature_hash(features),
        data_hashes=dict(sorted(data_hashes.items())),
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
        mismatch = bundle.mismatch[f"{group}_mismatch"].reindex(frame.index).fillna(False)
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
            model.predict(
                validation_features, num_iteration=model.best_iteration
            )
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


def assert_score_anchor(fold_name: str, score: float) -> None:
    expected = BASELINE_ANCHORS.get(fold_name)
    if expected is None or not np.isfinite(score) or abs(score - expected) > ANCHOR_TOLERANCE:
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
    """Persist predictions and a complete, hash-verifiable JSON manifest."""
    validate_fold_predictions(fold)
    cache_dir.mkdir(parents=True, exist_ok=True)
    provenance = fold.provenance
    predictions_path, manifest_path = _cache_paths(
        cache_dir, provenance.manifest_key
    )
    arrays: dict[str, np.ndarray] = {}
    for group in provenance.groups:
        suffix = group.rsplit("_", 1)[-1]
        arrays[f"g{suffix}_index_ns"] = fold.model_predictions[group].index.asi8
        arrays[f"g{suffix}_prediction"] = fold.model_predictions[group].to_numpy()
        arrays[f"g{suffix}_target"] = fold.validation_targets[group].to_numpy()
    temporary_predictions = predictions_path.with_suffix(".tmp.npz")
    with temporary_predictions.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_predictions.replace(predictions_path)

    manifest = asdict(provenance)
    manifest.update(
        {
            "schema_version": 1,
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
    feature_fingerprint: str,
    data_hashes: Mapping[str, str],
    row_counts: Mapping[str, int],
) -> FoldPredictions | None:
    key = manifest_key(
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        feature_hash=feature_fingerprint,
        data_hashes=data_hashes,
    )
    predictions_path, manifest_path = _cache_paths(cache_dir, key)
    if not predictions_path.exists() and not manifest_path.exists():
        return None
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise ProvenanceError("prediction cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "recipe",
        "train_years",
        "valid_year",
        "groups",
        "seeds",
        "feature_hash",
        "data_hashes",
        "row_counts",
        "validation_index_hashes",
        "target_hashes",
        "prediction_hashes",
        "manifest_key",
        "predictions_file",
    }
    if not required <= manifest.keys():
        raise ProvenanceError("prediction cache manifest is missing provenance")
    if manifest["manifest_key"] != key or manifest["predictions_file"] != predictions_path.name:
        raise ProvenanceError("prediction cache manifest key or file does not match")
    if manifest["row_counts"] != dict(row_counts):
        raise ProvenanceError("prediction cache row fingerprint does not match")

    model_predictions: dict[str, pd.Series] = {}
    validation_targets: dict[str, pd.Series] = {}
    with np.load(predictions_path, allow_pickle=False) as arrays:
        for group in groups:
            suffix = group.rsplit("_", 1)[-1]
            index = pd.DatetimeIndex(arrays[f"g{suffix}_index_ns"])
            model_predictions[group] = pd.Series(
                arrays[f"g{suffix}_prediction"],
                index=index,
                name=group,
            )
            validation_targets[group] = pd.Series(
                arrays[f"g{suffix}_target"],
                index=index,
                name=group,
            )
    provenance = build_provenance(
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        feature_hash=feature_fingerprint,
        data_hashes=data_hashes,
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
    )
    return fold


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
        raise ValueError("baseline evaluator supports only the frozen fold23/fold24 splits")

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

    cached = _read_prediction_cache(
        cache_dir=EXPERIMENT_DIR,
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        feature_fingerprint=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
        row_counts=row_counts,
    )
    if cached is not None:
        return cached

    model_predictions: dict[str, pd.Series] = {}
    validation_targets: dict[str, pd.Series] = {}
    columns = list(bundle.feature_columns)
    for group in groups:
        capacity = CAPACITY_KWH[group]
        validation = validation_frames[group]
        validation_target = validation[group].copy()
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
            raw_prediction = _fit_ensemble(
                pooled_frame[list(pooled_columns)],
                pooled_frame["normalized_target"],
                validation_features[list(pooled_columns)],
                validation_target / capacity,
                seeds,
                categorical=("group_id",),
            ) * capacity
        clipped = np.clip(raw_prediction, 0.0, capacity)
        model_predictions[group] = pd.Series(
            apply_floor10(clipped, capacity), index=validation.index, name=group
        )
        validation_targets[group] = validation_target.rename(group)

    provenance = build_provenance(
        recipe=recipe,
        train_years=train_years,
        valid_year=valid_year,
        groups=groups,
        seeds=seeds,
        feature_hash=bundle.feature_hash,
        data_hashes=bundle.data_hashes,
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
    )
    write_prediction_cache(fold)
    return fold


def score_fold(predictions: FoldPredictions) -> float:
    """Score only a fully indexed, hash-verified fold."""
    validate_fold_predictions(predictions)
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
    try:
        fold23 = fit_fold(BASELINE_RECIPE, (2022,), 2023, G12, seeds)
        fold24 = fit_fold(
            BASELINE_RECIPE, (2022, 2023), 2024, tuple(GROUPS), seeds
        )
        assert_baseline_fingerprint("fold23", fold23.provenance.row_counts)
        assert_baseline_fingerprint("fold24", fold24.provenance.row_counts)
        scores = {"fold23": score_fold(fold23), "fold24": score_fold(fold24)}
        for fold_name, score in scores.items():
            assert_score_anchor(fold_name, score)
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
    except (ProvenanceError, ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"stage7 baseline rejected: {error}", file=sys.stderr)
        return 2

    if not baseline_only:
        print(
            "stage7 candidate evaluation is not implemented yet; use --baseline-only",
            file=sys.stderr,
        )
        return 2
    return 0
