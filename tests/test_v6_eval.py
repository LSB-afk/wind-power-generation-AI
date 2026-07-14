from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from v6_eval import (
    WEIGHTED_EXPECTED_ROWS,
    WEIGHTED_RECIPE,
    WEIGHTED_RECIPE_CONFIG,
    FoldPredictions,
    ProvenanceError,
    TrainingBundle,
    apply_floor10,
    assert_baseline_fingerprint,
    assert_weighted_fingerprint,
    assert_fold_alignment,
    assert_score_anchor,
    blend_predictions,
    build_provenance,
    feature_hash,
    gate_scores,
    load_scada_targets,
    manifest_key,
    score_fold,
    validate_fold_predictions,
    write_prediction_cache,
)


def _synthetic_fold(
    index: pd.DatetimeIndex,
    *,
    recipe: str = "synthetic-baseline",
    seeds: tuple[int, ...] = (42,),
    feature_fingerprint: str = "feature-sha",
    data_hashes: dict[str, str] | None = None,
    derived_data_hashes: dict[str, str] | None = None,
    recipe_config: dict | None = None,
    postprocess_config: dict | None = None,
    row_counts: dict[str, int] | None = None,
    target_shift: float = 0.0,
    best_iterations: dict[str, tuple[int, ...]] | None = None,
) -> FoldPredictions:
    group = "kpx_group_1"
    model_predictions = {
        group: pd.Series(
            np.linspace(3000.0, 4000.0, len(index)), index=index, name=group
        )
    }
    validation_targets = {
        group: pd.Series(
            np.linspace(3200.0, 4200.0, len(index)) + target_shift,
            index=index,
            name=group,
        )
    }
    provenance = build_provenance(
        recipe=recipe,
        train_years=(2022,),
        valid_year=2023,
        groups=(group,),
        seeds=seeds,
        recipe_config=recipe_config or {"name": recipe, "version": 1},
        postprocess_config=postprocess_config
        or {"kind": "capacity_floor", "floor_ratio": 0.10, "version": 1},
        feature_hash=feature_fingerprint,
        data_hashes=data_hashes or {"labels": "labels-sha"},
        derived_data_hashes=derived_data_hashes or {"potential": "potential-sha"},
        best_iterations=best_iterations
        or {group: tuple(100 + offset for offset, _ in enumerate(seeds))},
        row_counts=row_counts or {"g1_train": 2, "g1_valid": len(index)},
        model_predictions=model_predictions,
        validation_targets=validation_targets,
    )
    return FoldPredictions(
        model_predictions=model_predictions,
        validation_targets=validation_targets,
        provenance=provenance,
    )


def test_gate_requires_both_folds_and_mean_gain():
    passed = gate_scores(
        {"fold23": 0.63, "fold24": 0.64},
        {"fold23": 0.631, "fold24": 0.6412},
    )
    assert passed["status"] == "PASS"

    failed = gate_scores(
        {"fold23": 0.63, "fold24": 0.64},
        {"fold23": 0.6299, "fold24": 0.65},
    )
    assert failed["status"] == "FAIL"


def test_gate_rejects_two_positive_deltas_below_mean_threshold():
    result = gate_scores(
        {"fold23": 0.0, "fold24": 0.0},
        {"fold23": 0.0009, "fold24": 0.0009},
    )

    assert result["status"] == "FAIL"
    assert result["mean_delta"] == pytest.approx(0.0009)


def test_gate_accepts_exact_mean_threshold():
    result = gate_scores(
        {"fold23": 0.0, "fold24": 0.0},
        {"fold23": 0.001, "fold24": 0.001},
    )

    assert result["status"] == "PASS"
    assert result["mean_delta"] == pytest.approx(0.001)


def test_blend_predictions_uses_actual_family_weight():
    potential = {"kpx_group_1": np.array([10.0, 20.0])}
    actual = {"kpx_group_1": np.array([20.0, 40.0])}

    got = blend_predictions(potential, actual, {"kpx_group_1": 0.25})

    np.testing.assert_allclose(got["kpx_group_1"], [12.5, 25.0])


def test_fold_alignment_requires_identical_hourly_validation_indices():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    baseline = _synthetic_fold(index)
    candidate = _synthetic_fold(index, recipe="candidate")

    assert_fold_alignment(baseline, candidate)

    shifted = _synthetic_fold(index.shift(1, freq="h"), recipe="candidate")
    with pytest.raises(ProvenanceError, match="validation index"):
        assert_fold_alignment(baseline, shifted)


def test_fold_alignment_requires_identical_seeds_and_data_hashes():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    baseline = _synthetic_fold(index)

    with pytest.raises(ProvenanceError, match="seeds"):
        assert_fold_alignment(
            baseline,
            _synthetic_fold(index, recipe="candidate", seeds=(202,)),
        )
    with pytest.raises(ProvenanceError, match="data hashes"):
        assert_fold_alignment(
            baseline,
            _synthetic_fold(
                index,
                recipe="candidate",
                data_hashes={"labels": "different-labels-sha"},
            ),
        )


def test_fold_alignment_requires_exact_live_targets_and_validation_rows():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    baseline = _synthetic_fold(index)

    with pytest.raises(ProvenanceError, match="validation targets"):
        assert_fold_alignment(
            baseline,
            _synthetic_fold(index, recipe="candidate", target_shift=1.0),
        )
    with pytest.raises(ProvenanceError, match="validation row metadata"):
        assert_fold_alignment(
            baseline,
            _synthetic_fold(
                index,
                recipe="candidate",
                row_counts={"g1_train": 2, "g1_valid": 3, "g2_valid": 0},
            ),
        )


def test_fold_alignment_requires_exact_structured_postprocess():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    baseline = _synthetic_fold(index)
    different_floor = _synthetic_fold(
        index,
        recipe="candidate",
        postprocess_config={
            "kind": "capacity_floor",
            "floor_ratio": 0.20,
            "version": 1,
        },
    )

    with pytest.raises(ProvenanceError, match="postprocess"):
        assert_fold_alignment(baseline, different_floor)


def test_fold_alignment_allows_recipe_specific_derived_training_targets():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    baseline = _synthetic_fold(index)
    candidate = _synthetic_fold(
        index,
        recipe="candidate",
        derived_data_hashes={"weighted_potential": "candidate-target-sha"},
    )

    assert_fold_alignment(baseline, candidate)


def test_floor10_only_raises_predictions_below_the_floor():
    capacity = 21600.0
    already_safe = np.array([2160.0, 2160.1, 5000.0, 21600.0])

    np.testing.assert_array_equal(apply_floor10(already_safe, capacity), already_safe)
    np.testing.assert_array_equal(
        apply_floor10(np.array([0.0, 2159.9]), capacity),
        np.array([2160.0, 2160.0]),
    )


def test_manifest_key_changes_with_seed_or_feature_name():
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    features = pd.DataFrame({"wind_speed": [4.0, 5.0]}, index=index)
    renamed_features = features.rename(columns={"wind_speed": "hub_wind_speed"})
    common = {
        "recipe": "v5-c1",
        "recipe_hash": "recipe-sha",
        "postprocess_hash": "postprocess-sha",
        "cache_schema_version": 3,
        "train_years": (2022,),
        "valid_year": 2023,
        "groups": ("kpx_group_1", "kpx_group_2"),
        "data_hashes": {"labels": "labels-sha"},
    }

    original = manifest_key(**common, seeds=(42,), feature_hash=feature_hash(features))
    changed_seed = manifest_key(
        **common, seeds=(202,), feature_hash=feature_hash(features)
    )
    changed_feature = manifest_key(
        **common, seeds=(42,), feature_hash=feature_hash(renamed_features)
    )
    changed_derived_target = manifest_key(
        **common,
        seeds=(42,),
        feature_hash=feature_hash(features),
        derived_data_hashes={"weighted_potential": "candidate-target-sha"},
    )

    assert len({original, changed_seed, changed_feature, changed_derived_target}) == 4


@pytest.mark.parametrize(
    "changed_hash",
    [
        "g1_calibration_index",
        "g1_weights",
        "weighted_targets_frame",
        "g1_postfilter_train_index",
        "g1_postfilter_train_target",
    ],
)
def test_candidate_cache_identity_covers_every_weighted_training_input(changed_hash):
    common = {
        "recipe": WEIGHTED_RECIPE,
        "recipe_hash": "recipe-sha",
        "postprocess_hash": "post-sha",
        "cache_schema_version": 3,
        "train_years": (2022,),
        "valid_year": 2023,
        "groups": ("kpx_group_1", "kpx_group_2"),
        "seeds": (42,),
        "feature_hash": "feature-sha",
        "data_hashes": {"labels": "label-sha", "scada": "scada-sha"},
    }
    evidence = {
        "g1_calibration_index": "a",
        "g1_weights": "b",
        "weighted_targets_frame": "c",
        "g1_postfilter_train_index": "d",
        "g1_postfilter_train_target": "e",
    }
    changed = dict(evidence)
    changed[changed_hash] += "-changed"

    original_key = manifest_key(**common, derived_data_hashes=evidence)
    changed_key = manifest_key(**common, derived_data_hashes=changed)

    assert changed_key != original_key


def test_recipe_fingerprint_covers_every_frozen_c1_behavior():
    import v6_eval

    original = deepcopy(v6_eval.BASELINE_RECIPE_CONFIG)
    variants = []

    changed_alpha = deepcopy(original)
    changed_alpha["model"]["params"]["alpha"] = 0.61
    variants.append(changed_alpha)

    changed_filter = deepcopy(original)
    changed_filter["training"]["filter_ratio"] = 0.06
    variants.append(changed_filter)

    changed_floor = deepcopy(original)
    changed_floor["postprocess"]["floor_ratio"] = 0.11
    variants.append(changed_floor)

    changed_capacity = deepcopy(original)
    changed_capacity["capacities"]["kpx_group_1"] += 1.0
    variants.append(changed_capacity)

    changed_strategy = deepcopy(original)
    changed_strategy["group_strategy"]["kpx_group_3"] = "solo"
    variants.append(changed_strategy)

    hashes = {v6_eval.recipe_fingerprint(original)}
    hashes.update(v6_eval.recipe_fingerprint(variant) for variant in variants)

    assert len(hashes) == 1 + len(variants)


def test_weighted_recipe_fingerprint_covers_boundary_calibration_and_minimums():
    import v6_eval

    original = deepcopy(WEIGHTED_RECIPE_CONFIG)
    variants = []
    for path, value in (
        (("training", "label_year_boundary"), "raw_timestamp_year"),
        (("training", "calibration", "min_power"), 2.0),
        (("training", "min_healthy", "kpx_group_1"), 2),
    ):
        changed = deepcopy(original)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        variants.append(changed)

    hashes = {v6_eval.recipe_fingerprint(original)}
    hashes.update(v6_eval.recipe_fingerprint(variant) for variant in variants)

    assert WEIGHTED_RECIPE == "v6-weighted-potential-q60-filter05-floor10"
    assert len(hashes) == 1 + len(variants)


def test_provenance_rejects_tampered_predictions_and_metadata():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    fold = _synthetic_fold(index)
    validate_fold_predictions(
        fold,
        expected_recipe="synthetic-baseline",
        expected_seeds=(42,),
        expected_feature_hash="feature-sha",
        expected_data_hashes={"labels": "labels-sha"},
    )
    assert np.isfinite(score_fold(fold))

    changed_predictions = {
        **fold.model_predictions,
        "kpx_group_1": fold.model_predictions["kpx_group_1"] + 1.0,
    }
    tampered_predictions = FoldPredictions(
        model_predictions=changed_predictions,
        validation_targets=fold.validation_targets,
        provenance=fold.provenance,
    )
    with pytest.raises(ProvenanceError, match="prediction hash"):
        validate_fold_predictions(tampered_predictions)

    tampered_recipe = FoldPredictions(
        model_predictions=fold.model_predictions,
        validation_targets=fold.validation_targets,
        provenance=replace(fold.provenance, recipe="other-recipe"),
    )
    with pytest.raises(ProvenanceError, match="manifest key"):
        validate_fold_predictions(tampered_recipe)

    tampered_row_count = FoldPredictions(
        model_predictions=fold.model_predictions,
        validation_targets=fold.validation_targets,
        provenance=replace(
            fold.provenance,
            row_counts={"g1_train": 2, "g1_valid": len(index) + 1},
        ),
    )
    with pytest.raises(ProvenanceError, match="validation row count"):
        validate_fold_predictions(tampered_row_count)


def test_prediction_cache_manifest_contains_complete_provenance(tmp_path):
    fold = _synthetic_fold(pd.date_range("2023-01-01", periods=2, freq="h"))

    manifest_path = write_prediction_cache(fold, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert {
        "recipe",
        "recipe_config",
        "recipe_hash",
        "postprocess_config",
        "postprocess_hash",
        "seeds",
        "feature_hash",
        "data_hashes",
        "row_counts",
        "prediction_hashes",
    } <= manifest.keys()
    assert manifest["manifest_key"] == fold.provenance.manifest_key
    assert manifest["schema_version"] == 3
    assert (tmp_path / manifest["predictions_file"]).is_file()


def test_prediction_cache_round_trip_revalidates_every_hash(tmp_path):
    import v6_eval

    fold = _synthetic_fold(pd.date_range("2023-01-01", periods=2, freq="h"))
    write_prediction_cache(fold, tmp_path)

    loaded = v6_eval._read_prediction_cache(
        cache_dir=tmp_path,
        recipe=fold.provenance.recipe,
        train_years=fold.provenance.train_years,
        valid_year=fold.provenance.valid_year,
        groups=fold.provenance.groups,
        seeds=fold.provenance.seeds,
        recipe_fingerprint=fold.provenance.recipe_hash,
        postprocess_config=fold.provenance.postprocess_config,
        feature_fingerprint=fold.provenance.feature_hash,
        data_hashes=fold.provenance.data_hashes,
        derived_data_hashes=fold.provenance.derived_data_hashes,
        row_counts=fold.provenance.row_counts,
        live_validation_targets={
            group: fold.validation_targets[group].copy()
            for group in fold.provenance.groups
        },
    )

    assert loaded is not None
    for group in fold.provenance.groups:
        pd.testing.assert_series_equal(
            loaded.model_predictions[group],
            fold.model_predictions[group],
            check_freq=False,
        )
        pd.testing.assert_series_equal(
            loaded.validation_targets[group],
            fold.validation_targets[group],
            check_freq=False,
        )


def test_prediction_cache_returns_live_targets_and_rejects_live_drift(tmp_path):
    import v6_eval

    fold = _synthetic_fold(pd.date_range("2023-01-01", periods=2, freq="h"))
    write_prediction_cache(fold, tmp_path)
    live_targets = {
        group: fold.validation_targets[group].copy() for group in fold.provenance.groups
    }
    kwargs = {
        "cache_dir": tmp_path,
        "recipe": fold.provenance.recipe,
        "train_years": fold.provenance.train_years,
        "valid_year": fold.provenance.valid_year,
        "groups": fold.provenance.groups,
        "seeds": fold.provenance.seeds,
        "recipe_fingerprint": fold.provenance.recipe_hash,
        "postprocess_config": fold.provenance.postprocess_config,
        "feature_fingerprint": fold.provenance.feature_hash,
        "data_hashes": fold.provenance.data_hashes,
        "derived_data_hashes": fold.provenance.derived_data_hashes,
        "row_counts": fold.provenance.row_counts,
    }

    loaded = v6_eval._read_prediction_cache(
        **kwargs, live_validation_targets=live_targets
    )

    assert loaded is not None
    for group in fold.provenance.groups:
        assert loaded.validation_targets[group] is live_targets[group]

    changed_values = {group: target + 1.0 for group, target in live_targets.items()}
    with pytest.raises(ProvenanceError, match="live validation target"):
        v6_eval._read_prediction_cache(**kwargs, live_validation_targets=changed_values)

    changed_indexes = {
        group: target.set_axis(target.index.shift(1, freq="h"))
        for group, target in live_targets.items()
    }
    with pytest.raises(ProvenanceError, match="live validation index"):
        v6_eval._read_prediction_cache(
            **kwargs, live_validation_targets=changed_indexes
        )


def test_prediction_cache_rejects_tampered_best_iteration_evidence(tmp_path):
    import v6_eval

    fold = _synthetic_fold(
        pd.date_range("2023-01-01", periods=2, freq="h"),
        seeds=(42, 202, 777),
        best_iterations={"kpx_group_1": (101, 102, 103)},
    )
    manifest_path = write_prediction_cache(fold, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["best_iterations"]["kpx_group_1"][1] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="prediction cache hashes"):
        v6_eval._read_prediction_cache(
            cache_dir=tmp_path,
            recipe=fold.provenance.recipe,
            train_years=fold.provenance.train_years,
            valid_year=fold.provenance.valid_year,
            groups=fold.provenance.groups,
            seeds=fold.provenance.seeds,
            recipe_fingerprint=fold.provenance.recipe_hash,
            postprocess_config=fold.provenance.postprocess_config,
            feature_fingerprint=fold.provenance.feature_hash,
            data_hashes=fold.provenance.data_hashes,
            derived_data_hashes=fold.provenance.derived_data_hashes,
            row_counts=fold.provenance.row_counts,
            live_validation_targets=fold.validation_targets,
        )


@pytest.mark.parametrize("corruption", ["json", "schema", "npz"])
def test_prediction_cache_wraps_malformed_artifacts_as_provenance_error(
    tmp_path, corruption
):
    import v6_eval

    fold = _synthetic_fold(pd.date_range("2023-01-01", periods=2, freq="h"))
    manifest_path = write_prediction_cache(fold, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions_path = tmp_path / manifest["predictions_file"]
    if corruption == "json":
        manifest_path.write_text("{not-json", encoding="utf-8")
    elif corruption == "schema":
        manifest["schema_version"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        predictions_path.write_bytes(b"not-an-npz")

    with pytest.raises(ProvenanceError, match="prediction cache"):
        v6_eval._read_prediction_cache(
            cache_dir=tmp_path,
            recipe=fold.provenance.recipe,
            train_years=fold.provenance.train_years,
            valid_year=fold.provenance.valid_year,
            groups=fold.provenance.groups,
            seeds=fold.provenance.seeds,
            recipe_fingerprint=fold.provenance.recipe_hash,
            postprocess_config=fold.provenance.postprocess_config,
            feature_fingerprint=fold.provenance.feature_hash,
            data_hashes=fold.provenance.data_hashes,
            derived_data_hashes=fold.provenance.derived_data_hashes,
            row_counts=fold.provenance.row_counts,
            live_validation_targets=fold.validation_targets,
        )


def test_scada_targets_always_build_from_raw(monkeypatch):
    import v6_eval

    labels = pd.DataFrame(
        {"kpx_group_1": [1.0]},
        index=pd.date_range("2023-01-01", periods=1, freq="h"),
    )
    expected_potential = pd.DataFrame(
        {"kpx_group_1_potential": [2.0]}, index=labels.index
    )
    expected_mismatch = pd.DataFrame(
        {"kpx_group_1_mismatch": [False]}, index=labels.index
    )
    calls = []
    monkeypatch.setattr(
        v6_eval,
        "build_potential",
        lambda frame: calls.append(("potential", frame)) or expected_potential,
    )
    monkeypatch.setattr(
        v6_eval,
        "build_mismatch_mask",
        lambda frame: calls.append(("mismatch", frame)) or expected_mismatch,
    )

    monkeypatch.setattr(
        v6_eval.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("stale parquet cache was consumed"),
    )

    potential, mismatch = load_scada_targets(labels)

    pd.testing.assert_frame_equal(potential, expected_potential)
    pd.testing.assert_frame_equal(mismatch, expected_mismatch)
    assert [name for name, _ in calls] == ["potential", "mismatch"]


def test_load_bundle_rebuilds_every_derived_frame_from_raw(monkeypatch):
    import v6_eval

    index = pd.date_range("2023-01-01", periods=2, freq="h")
    features = pd.DataFrame({"wind": [1.0, 2.0]}, index=index)
    labels = pd.DataFrame(
        {group: [3000.0, 4000.0] for group in v6_eval.GROUPS}, index=index
    )
    potential = pd.DataFrame(
        {f"{group}_potential": [3100.0, 4100.0] for group in v6_eval.GROUPS},
        index=index,
    )
    mismatch = pd.DataFrame(
        {f"{group}_mismatch": [False, False] for group in v6_eval.GROUPS},
        index=index,
    )
    calls = []
    monkeypatch.setattr(v6_eval, "_BUNDLE", None)
    monkeypatch.setattr(
        v6_eval.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("stale parquet cache was consumed"),
    )
    monkeypatch.setattr(v6_eval.pd, "read_csv", lambda *_args, **_kwargs: labels)
    monkeypatch.setattr(
        v6_eval,
        "build_features",
        lambda *paths: calls.append(("features", paths)) or features,
    )
    monkeypatch.setattr(
        v6_eval,
        "build_potential",
        lambda frame: calls.append(("potential", frame)) or potential,
    )
    monkeypatch.setattr(
        v6_eval,
        "build_mismatch_mask",
        lambda frame: calls.append(("mismatch", frame)) or mismatch,
    )
    monkeypatch.setattr(v6_eval, "_file_hash", lambda path: f"sha-{path.name}")

    bundle = v6_eval.load_bundle()

    pd.testing.assert_frame_equal(bundle.features, features)
    pd.testing.assert_frame_equal(bundle.potential, potential)
    pd.testing.assert_frame_equal(bundle.mismatch, mismatch)
    assert [call[0] for call in calls] == ["features", "potential", "mismatch"]


def test_weighted_candidate_rechecks_scada_hashes_after_target_reconstruction(
    monkeypatch,
):
    import v6_eval

    index = pd.DatetimeIndex(["2022-01-01 01:00", "2023-01-01 01:00"])
    labels = pd.DataFrame(
        {group: [3000.0, 4000.0] for group in v6_eval.G12}, index=index
    )
    data = labels.copy()
    potential = pd.DataFrame(
        {f"{group}_potential": [3000.0, 4000.0] for group in v6_eval.G12},
        index=index,
    )
    mismatch = pd.DataFrame(
        {f"{group}_mismatch": [False, False] for group in v6_eval.G12},
        index=index,
    )
    weighted = pd.DataFrame(
        {f"{group}_weighted_potential": [3000.0, np.nan] for group in v6_eval.G12},
        index=index,
    )
    bundle = TrainingBundle(
        features=pd.DataFrame(index=index),
        labels=labels,
        potential=potential,
        mismatch=mismatch,
        data=data,
        feature_columns=(),
        feature_hash="feature-sha",
        data_hashes={
            "scada_vestas": "before-vestas-sha",
            "scada_unison": "before-unison-sha",
        },
        derived_data_hashes={"mismatch_frame": "mismatch-sha"},
    )
    monkeypatch.setattr(v6_eval, "load_bundle", lambda: bundle)
    monkeypatch.setattr(
        v6_eval,
        "build_weighted_targets",
        lambda *_args, **_kwargs: (weighted, {}),
    )
    monkeypatch.setattr(v6_eval, "_file_hash", lambda _path: "changed-scada-sha")

    with pytest.raises(ProvenanceError, match="SCADA source hash changed"):
        v6_eval.fit_fold(
            v6_eval.WEIGHTED_RECIPE,
            (2022,),
            2023,
            v6_eval.G12,
            (42,),
        )


def test_baseline_guards_reject_row_or_anchor_drift():
    assert_baseline_fingerprint(
        "fold23",
        {
            "g1_train": 6215,
            "g2_train": 6174,
            "g1_valid": 8757,
            "g2_valid": 8758,
        },
    )
    assert_score_anchor("fold23", 0.6316, seeds=(42,))

    with pytest.raises(ProvenanceError, match="row fingerprint"):
        assert_baseline_fingerprint(
            "fold23",
            {
                "g1_train": 6214,
                "g2_train": 6174,
                "g1_valid": 8757,
                "g2_valid": 8758,
            },
        )
    with pytest.raises(ProvenanceError, match="score anchor"):
        assert_score_anchor("fold23", 0.6316 + 0.000151, seeds=(42,))

    assert_score_anchor("fold23", 0.0, seeds=(42, 202, 777))


def test_weighted_fingerprint_pins_candidate_training_rows():
    assert_weighted_fingerprint("fold23", WEIGHTED_EXPECTED_ROWS["fold23"])

    changed = dict(WEIGHTED_EXPECTED_ROWS["fold23"])
    changed["g1_train"] -= 1
    with pytest.raises(ProvenanceError, match="weighted row fingerprint"):
        assert_weighted_fingerprint("fold23", changed)


def test_score_fold_enforces_complete_fingerprint_for_baseline_recipe():
    import v6_eval

    fold = _synthetic_fold(
        pd.date_range("2023-01-01", periods=2, freq="h"),
        recipe=v6_eval.BASELINE_RECIPE,
        recipe_config=deepcopy(v6_eval.BASELINE_RECIPE_CONFIG),
        postprocess_config=deepcopy(v6_eval.BASELINE_POSTPROCESS_CONFIG),
    )

    with pytest.raises(ProvenanceError, match="row fingerprint"):
        score_fold(fold)


def test_stage7_default_path_enters_weighted_evaluation_without_baseline_ok(
    monkeypatch, capsys
):
    import v6_eval

    calls = []

    def reject(*args, **_kwargs):
        calls.append(args)
        raise ProvenanceError("sentinel baseline rejection")

    monkeypatch.setattr(v6_eval, "fit_fold", reject)

    assert v6_eval.run_stage7((42,), baseline_only=False) == 2
    captured = capsys.readouterr()
    assert calls and calls[0][0] == v6_eval.BASELINE_RECIPE
    assert "BASELINE_OK" not in captured.out


def test_stage7_converts_provenance_failure_to_code_2(monkeypatch, capsys):
    import v6_eval

    def reject(*_args, **_kwargs):
        raise ProvenanceError("prediction cache schema is malformed")

    monkeypatch.setattr(v6_eval, "fit_fold", reject)

    assert v6_eval.run_stage7((42,), baseline_only=True) == 2
    captured = capsys.readouterr()
    assert "BASELINE_OK" not in captured.out
    assert "prediction cache schema" in captured.err


def test_stage7_weighted_screen_compares_aligned_baseline_and_candidate(
    monkeypatch, capsys, tmp_path
):
    import v6_eval

    calls = []

    class StubFold:
        def __init__(self, recipe, valid_year, groups, seeds):
            self.recipe = recipe
            self.valid_year = valid_year
            expected = (
                v6_eval.EXPECTED_ROWS
                if recipe == v6_eval.BASELINE_RECIPE
                else v6_eval.WEIGHTED_EXPECTED_ROWS
            )
            self.provenance = type(
                "StubProvenance",
                (),
                {
                    "row_counts": expected[v6_eval._fold_name(valid_year)],
                    "manifest_key": f"{recipe}-{valid_year}",
                    "best_iterations": {
                        group: tuple(100 for _ in seeds)
                        for group in groups
                        if group in v6_eval.G12
                    }
                    | (
                        {"pooled": tuple(200 for _ in seeds)}
                        if v6_eval.G3 in groups
                        else {}
                    ),
                },
            )()

    def fit(recipe, train_years, valid_year, groups, seeds):
        calls.append((recipe, train_years, valid_year, groups, seeds))
        return StubFold(recipe, valid_year, groups, seeds)

    scores = {
        (v6_eval.BASELINE_RECIPE, 2023): 0.631623985226,
        (v6_eval.BASELINE_RECIPE, 2024): 0.638029787652,
        (v6_eval.WEIGHTED_RECIPE, 2023): 0.633691786326,
        (v6_eval.WEIGHTED_RECIPE, 2024): 0.639226950590,
    }
    alignments = []
    monkeypatch.setattr(v6_eval, "fit_fold", fit)
    monkeypatch.setattr(
        v6_eval, "score_fold", lambda fold: scores[(fold.recipe, fold.valid_year)]
    )
    monkeypatch.setattr(
        v6_eval,
        "assert_fold_alignment",
        lambda baseline, candidate: alignments.append((baseline, candidate)),
    )
    monkeypatch.setattr(v6_eval, "fold_metrics", lambda _fold: {"g": {}})
    artifact_path = tmp_path / "final-gate.json"
    artifact_path.write_bytes(b"existing-promotion-evidence\n")
    monkeypatch.setattr(v6_eval, "FINAL_GATE_PATH", artifact_path)

    assert v6_eval.run_stage7((42,), screen="weighted") == 0
    output = capsys.readouterr().out.strip().splitlines()
    result = json.loads(output[-1])

    assert [call[0] for call in calls] == [
        v6_eval.BASELINE_RECIPE,
        v6_eval.BASELINE_RECIPE,
        v6_eval.WEIGHTED_RECIPE,
        v6_eval.WEIGHTED_RECIPE,
    ]
    assert len(alignments) == 2
    assert result["status"] == "PASS"
    assert result["candidate_recipe"] == v6_eval.WEIGHTED_RECIPE
    assert result["mean_delta"] == pytest.approx(0.001632482019)
    assert artifact_path.read_bytes() == b"existing-promotion-evidence\n"


def _stub_main_inputs(monkeypatch, exp_runner):
    ldaps_raw = pd.DataFrame(columns=["forecast_kst_dtm", "data_available_kst_dtm"])
    monkeypatch.setattr(exp_runner, "build_cache", lambda: None)
    monkeypatch.setattr(
        exp_runner,
        "load_cache",
        lambda: (pd.DataFrame(), ldaps_raw, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(exp_runner, "add_context", lambda feat, _davail: feat)


def _forbid_legacy_bootstrap(monkeypatch, exp_runner):
    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"legacy bootstrap called: {name}")

        return fail

    for name in ("build_cache", "load_cache", "add_context"):
        monkeypatch.setattr(exp_runner, name, forbidden(name))


def test_main_routes_literal_stage7_and_propagates_result(monkeypatch):
    import exp_runner

    _forbid_legacy_bootstrap(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(
        exp_runner,
        "run_stage7",
        lambda seeds, baseline_only=False: calls.append((seeds, baseline_only)) or 17,
    )
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage7"])

    assert exp_runner.main() == 17
    assert calls == [((42, 202, 777), False)]


def test_main_parses_stage7_seed_and_baseline_only_flags(monkeypatch):
    import exp_runner

    _forbid_legacy_bootstrap(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(
        exp_runner,
        "run_stage7",
        lambda seeds, baseline_only=False: calls.append((seeds, baseline_only)) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp_runner.py", "stage7", "--seeds", "42", "--baseline-only"],
    )

    assert exp_runner.main() == 0
    assert calls == [((42,), True)]


def test_main_parses_weighted_screen_without_legacy_bootstrap(monkeypatch):
    import exp_runner

    _forbid_legacy_bootstrap(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(
        exp_runner,
        "run_stage7",
        lambda seeds, baseline_only=False, screen=None: calls.append(
            (seeds, baseline_only, screen)
        )
        or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp_runner.py", "stage7", "--seeds", "42", "--screen", "weighted"],
    )

    assert exp_runner.main() == 0
    assert calls == [((42,), False, "weighted")]


def test_main_rejects_invalid_stage7_flags_without_training(monkeypatch, capsys):
    import exp_runner

    _forbid_legacy_bootstrap(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(
        exp_runner,
        "run_stage7",
        lambda seeds, baseline_only=False: calls.append((seeds, baseline_only)) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage7", "--unknown-flag"])

    assert exp_runner.main() == 2
    assert calls == []
    assert "--unknown-flag" in capsys.readouterr().err


def test_main_routes_literal_stage6(monkeypatch):
    import exp_runner

    _stub_main_inputs(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(exp_runner, "stage6", calls.append)
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage6"])

    assert exp_runner.main() is None
    assert len(calls) == 1
    assert set(calls[0]) == {"ctx"}


def test_main_rejects_unknown_mode_without_stage_fallback(monkeypatch, capsys):
    import exp_runner

    _forbid_legacy_bootstrap(monkeypatch, exp_runner)
    stage6_calls = []
    stage7_calls = []
    monkeypatch.setattr(exp_runner, "stage6", stage6_calls.append)
    monkeypatch.setattr(exp_runner, "run_stage7", stage7_calls.append)
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage7-extra"])

    assert exp_runner.main() == 2
    assert stage6_calls == []
    assert stage7_calls == []
    assert "stage7-extra" in capsys.readouterr().err


def _best_iterations_hash(seeds, families):
    payload = json.dumps(
        {"families": families, "seeds": list(seeds)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _passing_final_gate_payload():
    import v6_eval

    seeds = [42, 202, 777]
    sources = {
        "gfs_train": "1" * 64,
        "ldaps_train": "2" * 64,
        "scada_unison": "3" * 64,
        "scada_vestas": "4" * 64,
        "train_labels": "5" * 64,
    }
    best_iterations = {
        "kpx_group_1": [111, 112, 113],
        "kpx_group_2": [211, 212, 213],
        "pooled": [311, 312, 313],
    }

    def group_metrics(nmae, ficr):
        return {
            "nmae": nmae,
            "ficr": ficr,
            "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
        }

    fold23_baseline = {group: group_metrics(0.20, 0.40) for group in v6_eval.G12}
    fold23_candidate = {group: group_metrics(0.198, 0.402) for group in v6_eval.G12}
    fold24_baseline = {group: group_metrics(0.20, 0.40) for group in v6_eval.GROUPS}
    fold24_candidate = {group: group_metrics(0.199, 0.401) for group in v6_eval.GROUPS}
    return {
        "kind": "wind-v6-final-gate",
        "schema_version": 1,
        "cache_schema_version": 3,
        "status": "PASS",
        "seeds": seeds,
        "candidate_recipe": v6_eval.WEIGHTED_RECIPE,
        "recipes": {
            "baseline": v6_eval.BASELINE_RECIPE,
            "candidate": v6_eval.WEIGHTED_RECIPE,
        },
        "hashes": {
            "recipes": {
                "baseline": v6_eval.recipe_fingerprint(v6_eval.BASELINE_RECIPE_CONFIG),
                "candidate": v6_eval.recipe_fingerprint(v6_eval.WEIGHTED_RECIPE_CONFIG),
            },
            "postprocess": v6_eval.recipe_fingerprint(
                v6_eval.BASELINE_POSTPROCESS_CONFIG
            ),
            "features": "6" * 64,
            "sources": sources,
        },
        "folds": {
            "fold23": {
                "baseline": {
                    "score": 0.60,
                    "metrics": fold23_baseline,
                    "manifest_key": "7" * 64,
                },
                "candidate": {
                    "score": 0.602,
                    "metrics": fold23_candidate,
                    "manifest_key": "8" * 64,
                },
                "delta": 0.002,
            },
            "fold24": {
                "baseline": {
                    "score": 0.60,
                    "metrics": fold24_baseline,
                    "manifest_key": "9" * 64,
                },
                "candidate": {
                    "score": 0.601,
                    "metrics": fold24_candidate,
                    "manifest_key": "a" * 64,
                },
                "delta": 0.001,
            },
        },
        "mean_delta": 0.0015,
        "fold24_candidate_best_iterations": best_iterations,
        "fold24_candidate_best_iterations_hash": _best_iterations_hash(
            seeds, best_iterations
        ),
    }


def test_provenance_preserves_seed_ordered_best_iterations_outside_cache_key():
    import v6_eval

    index = pd.date_range("2023-01-01", periods=2, freq="h")
    group = "kpx_group_1"
    predictions = {group: pd.Series([3000.0, 4000.0], index=index, name=group)}
    targets = {group: pd.Series([3100.0, 4100.0], index=index, name=group)}
    common = {
        "recipe": "synthetic",
        "train_years": (2022,),
        "valid_year": 2023,
        "groups": (group,),
        "seeds": (42, 202, 777),
        "recipe_config": {"name": "synthetic", "version": 1},
        "postprocess_config": {"floor_ratio": 0.1, "version": 1},
        "feature_hash": "feature-sha",
        "data_hashes": {"labels": "labels-sha"},
        "row_counts": {"g1_train": 2, "g1_valid": 2},
        "model_predictions": predictions,
        "validation_targets": targets,
    }

    first = v6_eval.build_provenance(**common, best_iterations={group: (101, 102, 103)})
    second = v6_eval.build_provenance(
        **common, best_iterations={group: (201, 202, 203)}
    )

    assert first.best_iterations == {group: (101, 102, 103)}
    assert first.best_iterations_hash != second.best_iterations_hash
    assert first.manifest_key == second.manifest_key


@pytest.mark.parametrize(
    "best_iterations, message",
    [
        ({"kpx_group_1": (101, 102)}, "cardinality"),
        ({"kpx_group_1": (101, True, 103)}, "positive integers"),
        ({"kpx_group_1": (101, 102.0, 103)}, "positive integers"),
        ({"pooled": (101, 102, 103)}, "family keys"),
    ],
)
def test_provenance_rejects_invalid_best_iteration_evidence(best_iterations, message):
    import v6_eval

    index = pd.date_range("2023-01-01", periods=1, freq="h")
    group = "kpx_group_1"
    prediction = {group: pd.Series([3000.0], index=index, name=group)}
    with pytest.raises(ProvenanceError, match=message):
        v6_eval.build_provenance(
            recipe="synthetic",
            train_years=(2022,),
            valid_year=2023,
            groups=(group,),
            seeds=(42, 202, 777),
            recipe_config={"name": "synthetic", "version": 1},
            postprocess_config={"floor_ratio": 0.1, "version": 1},
            feature_hash="feature-sha",
            data_hashes={"labels": "labels-sha"},
            row_counts={"g1_train": 1, "g1_valid": 1},
            model_predictions=prediction,
            validation_targets=prediction,
            best_iterations=best_iterations,
        )


def test_final_gate_artifact_round_trip_is_canonical_and_hash_bound(tmp_path):
    import v6_eval

    payload = _passing_final_gate_payload()
    artifact_path = tmp_path / "final-gate.json"
    json_path, sidecar_path = v6_eval.write_final_gate_artifact(
        payload,
        artifact_path,
        expected_feature_hash=payload["hashes"]["features"],
        expected_source_hashes=payload["hashes"]["sources"],
    )

    raw = json_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert raw == (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert sidecar_path.read_text(encoding="ascii") == (f"{digest}  final-gate.json\n")
    assert (
        v6_eval.read_final_gate_artifact(
            json_path,
            expected_feature_hash=payload["hashes"]["features"],
            expected_source_hashes=payload["hashes"]["sources"],
        )
        == payload
    )


def test_final_gate_reader_rejects_json_or_sidecar_tampering(tmp_path):
    import v6_eval

    payload = _passing_final_gate_payload()
    artifact_path = tmp_path / "final-gate.json"
    _, sidecar_path = v6_eval.write_final_gate_artifact(
        payload,
        artifact_path,
        expected_feature_hash=payload["hashes"]["features"],
        expected_source_hashes=payload["hashes"]["sources"],
    )
    original_json = artifact_path.read_bytes()
    artifact_path.write_bytes(original_json.replace(b'"PASS"', b'"FAIL"'))
    with pytest.raises(ProvenanceError, match="SHA-256"):
        v6_eval.read_final_gate_artifact(
            artifact_path,
            expected_feature_hash=payload["hashes"]["features"],
            expected_source_hashes=payload["hashes"]["sources"],
        )

    artifact_path.write_bytes(original_json)
    sidecar_path.write_text(f"{'0' * 64}  final-gate.json\n", encoding="ascii")
    with pytest.raises(ProvenanceError, match="SHA-256"):
        v6_eval.read_final_gate_artifact(
            artifact_path,
            expected_feature_hash=payload["hashes"]["features"],
            expected_source_hashes=payload["hashes"]["sources"],
        )


def test_final_gate_reader_recomputes_metrics_after_valid_digest_tampering(tmp_path):
    import v6_eval

    payload = _passing_final_gate_payload()
    artifact_path = tmp_path / "final-gate.json"
    _, sidecar_path = v6_eval.write_final_gate_artifact(
        payload,
        artifact_path,
        expected_feature_hash=payload["hashes"]["features"],
        expected_source_hashes=payload["hashes"]["sources"],
    )
    payload["folds"]["fold24"]["candidate"]["metrics"]["kpx_group_1"]["score"] += 0.01
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    artifact_path.write_bytes(raw)
    sidecar_path.write_text(
        f"{hashlib.sha256(raw).hexdigest()}  final-gate.json\n", encoding="ascii"
    )

    with pytest.raises(ProvenanceError, match="score does not match metrics"):
        v6_eval.read_final_gate_artifact(
            artifact_path,
            expected_feature_hash=payload["hashes"]["features"],
            expected_source_hashes=payload["hashes"]["sources"],
        )


def test_final_gate_default_reader_rehashes_current_raw_sources(monkeypatch, tmp_path):
    import v6_eval

    payload = _passing_final_gate_payload()
    artifact_path = tmp_path / "final-gate.json"
    v6_eval.write_final_gate_artifact(
        payload,
        artifact_path,
        expected_feature_hash=payload["hashes"]["features"],
        expected_source_hashes=payload["hashes"]["sources"],
    )
    stale_bundle = type(
        "StaleBundle",
        (),
        {
            "feature_hash": payload["hashes"]["features"],
            "data_hashes": payload["hashes"]["sources"],
        },
    )()
    monkeypatch.setattr(v6_eval, "load_bundle", lambda: stale_bundle)
    monkeypatch.setattr(v6_eval, "_file_hash", lambda _path: "0" * 64)

    with pytest.raises(ProvenanceError, match="source hashes"):
        v6_eval.read_final_gate_artifact(artifact_path)


def test_final_gate_writer_rejects_noncanonical_or_failed_payload_without_touching(
    tmp_path,
):
    import v6_eval

    artifact_path = tmp_path / "final-gate.json"
    artifact_path.write_bytes(b"existing-promotion-evidence\n")
    existing = artifact_path.read_bytes()
    for mutate in (
        lambda payload: payload.update(status="FAIL"),
        lambda payload: payload.update(seeds=[42]),
    ):
        payload = _passing_final_gate_payload()
        mutate(payload)
        with pytest.raises(ProvenanceError):
            v6_eval.write_final_gate_artifact(
                payload,
                artifact_path,
                expected_feature_hash=payload["hashes"]["features"],
                expected_source_hashes=payload["hashes"]["sources"],
            )
        assert artifact_path.read_bytes() == existing


def test_default_stage7_runs_only_weighted_candidate_and_writes_canonical_gate(
    monkeypatch, capsys, tmp_path
):
    import v6_eval

    calls = []

    class StubFold:
        def __init__(self, recipe, valid_year, groups, seeds):
            self.recipe = recipe
            self.valid_year = valid_year
            expected = (
                v6_eval.EXPECTED_ROWS
                if recipe == v6_eval.BASELINE_RECIPE
                else v6_eval.WEIGHTED_EXPECTED_ROWS
            )
            families = {
                group: tuple(100 + offset for offset, _ in enumerate(seeds))
                for group in groups
                if group in v6_eval.G12
            }
            if v6_eval.G3 in groups:
                families["pooled"] = tuple(
                    200 + offset for offset, _ in enumerate(seeds)
                )
            recipe_config = (
                v6_eval.BASELINE_RECIPE_CONFIG
                if recipe == v6_eval.BASELINE_RECIPE
                else v6_eval.WEIGHTED_RECIPE_CONFIG
            )
            self.provenance = type(
                "StubProvenance",
                (),
                {
                    "row_counts": expected[v6_eval._fold_name(valid_year)],
                    "manifest_key": ("b" if recipe == v6_eval.BASELINE_RECIPE else "c")
                    * 64,
                    "recipe_hash": v6_eval.recipe_fingerprint(recipe_config),
                    "postprocess_hash": v6_eval.recipe_fingerprint(
                        v6_eval.BASELINE_POSTPROCESS_CONFIG
                    ),
                    "feature_hash": "d" * 64,
                    "data_hashes": {
                        "gfs_train": "1" * 64,
                        "ldaps_train": "2" * 64,
                        "scada_unison": "3" * 64,
                        "scada_vestas": "4" * 64,
                        "train_labels": "5" * 64,
                    },
                    "best_iterations": families,
                    "best_iterations_hash": _best_iterations_hash(seeds, families),
                },
            )()

    def fit(recipe, train_years, valid_year, groups, seeds):
        calls.append(recipe)
        return StubFold(recipe, valid_year, groups, seeds)

    scores = {
        (v6_eval.BASELINE_RECIPE, 2023): 0.60,
        (v6_eval.BASELINE_RECIPE, 2024): 0.60,
        (v6_eval.WEIGHTED_RECIPE, 2023): 0.602,
        (v6_eval.WEIGHTED_RECIPE, 2024): 0.601,
    }

    def metrics(fold):
        groups = v6_eval.G12 if fold.valid_year == 2023 else tuple(v6_eval.GROUPS)
        if fold.recipe == v6_eval.BASELINE_RECIPE:
            nmae, ficr = 0.20, 0.40
        elif fold.valid_year == 2023:
            nmae, ficr = 0.198, 0.402
        else:
            nmae, ficr = 0.199, 0.401
        return {
            group: {
                "nmae": nmae,
                "ficr": ficr,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
            }
            for group in groups
        }

    artifact_path = tmp_path / "final-gate.json"
    monkeypatch.setattr(v6_eval, "FINAL_GATE_PATH", artifact_path)
    monkeypatch.setattr(v6_eval, "fit_fold", fit)
    monkeypatch.setattr(
        v6_eval, "score_fold", lambda fold: scores[(fold.recipe, fold.valid_year)]
    )
    monkeypatch.setattr(v6_eval, "assert_fold_alignment", lambda *_args: None)
    monkeypatch.setattr(v6_eval, "fold_metrics", metrics)

    assert v6_eval.run_stage7((42, 202, 777)) == 0
    output = capsys.readouterr().out
    final_line = output.splitlines(keepends=True)[-1]
    result = json.loads(final_line)

    assert calls == [
        v6_eval.BASELINE_RECIPE,
        v6_eval.BASELINE_RECIPE,
        v6_eval.WEIGHTED_RECIPE,
        v6_eval.WEIGHTED_RECIPE,
    ]
    assert result["candidate_recipe"] == v6_eval.WEIGHTED_RECIPE
    assert result["status"] == "PASS"
    assert artifact_path.is_file()
    assert final_line.encode("utf-8") == artifact_path.read_bytes()
    assert result == json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(result["fold24_candidate_best_iterations"]) == {
        "kpx_group_1",
        "kpx_group_2",
        "pooled",
    }


def test_custom_stage7_diagnostic_never_overwrites_promotion_evidence(
    monkeypatch, tmp_path
):
    import v6_eval

    artifact_path = tmp_path / "final-gate.json"
    artifact_path.write_bytes(b"existing-promotion-evidence\n")
    monkeypatch.setattr(v6_eval, "FINAL_GATE_PATH", artifact_path)
    monkeypatch.setattr(
        v6_eval,
        "fit_fold",
        lambda *_args, **_kwargs: pytest.fail("diagnostic fixture stops before fit"),
    )

    assert v6_eval.run_stage7((42,), screen="unsupported") == 2
    assert artifact_path.read_bytes() == b"existing-promotion-evidence\n"
