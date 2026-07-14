from copy import deepcopy
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from v6_eval import (
    FoldPredictions,
    ProvenanceError,
    apply_floor10,
    assert_baseline_fingerprint,
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
        "cache_schema_version": 2,
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
    assert manifest["schema_version"] == 2
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


def test_stage7_nonbaseline_path_fails_before_baseline_ok(monkeypatch, capsys):
    import v6_eval

    monkeypatch.setattr(
        v6_eval,
        "fit_fold",
        lambda *_args, **_kwargs: pytest.fail("baseline trained on nonbaseline path"),
    )

    assert v6_eval.run_stage7((42,), baseline_only=False) == 2
    captured = capsys.readouterr()
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
