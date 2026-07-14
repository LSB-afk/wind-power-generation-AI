from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import inference
import train
from config import CAPACITY_KWH, GROUPS


EXPECTED_RECIPE_KEYS = {
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
EXPECTED_ACTIVE_MODELS = {
    f"lgbm_v6_weighted_{family}_s{seed}.txt"
    for family in ("kpx_group_1", "kpx_group_2", "pooled")
    for seed in (42, 202, 777)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_frame(rows: int = 2) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 01:00:00", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "forecast_id": [
                f"forecast_{position:04d}" for position in range(1, rows + 1)
            ],
            "forecast_kst_dtm": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "kpx_group_1": [CAPACITY_KWH["kpx_group_1"] * 0.1] * rows,
            "kpx_group_2": [CAPACITY_KWH["kpx_group_2"] * 0.2] * rows,
            "kpx_group_3": [CAPACITY_KWH["kpx_group_3"] * 0.3] * rows,
        }
    )


def test_round_policy_uses_exact_two_stage_integer_truncation():
    assert train.calculate_full_rounds((100, 101, 102)) == (101, 121)
    assert train.calculate_full_rounds((1, 2, 2)) == (1, 100)


@pytest.mark.parametrize("values", [(), (0,), (True,), (1.5,), (np.int64(4), False)])
def test_round_policy_rejects_non_positive_or_non_builtin_integer_evidence(values):
    with pytest.raises(ValueError, match="positive integers"):
        train.calculate_full_rounds(values)


@pytest.mark.parametrize(
    "path",
    ["../model.txt", "nested/model.txt", "/tmp/model.txt", "model.bin", ""],
)
def test_inference_rejects_non_basename_model_paths(path):
    with pytest.raises(inference.InferenceContractError):
        inference.validate_model_relative_path(path)


def test_submission_validator_preserves_sample_identity_and_accepts_finite_bounds():
    sample = _sample_frame()
    result = inference.validate_submission_frame(sample.copy(), sample.copy())
    pd.testing.assert_frame_equal(result, sample)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("forecast_id", "changed", "identity"),
        ("forecast_kst_dtm", "2025-02-01 00:00:00", "identity"),
        ("kpx_group_1", np.nan, "finite"),
        ("kpx_group_2", np.inf, "finite"),
        ("kpx_group_3", CAPACITY_KWH["kpx_group_3"] + 1.0, "bounds"),
        ("kpx_group_1", CAPACITY_KWH["kpx_group_1"] * 0.1 - 1.0, "floor"),
    ],
)
def test_submission_validator_fails_closed(column, value, message):
    sample = _sample_frame()
    candidate = sample.copy()
    candidate.loc[0, column] = value
    with pytest.raises(inference.InferenceContractError, match=message):
        inference.validate_submission_frame(candidate, sample)


def test_inference_has_no_training_or_evaluator_dependency():
    source = (SRC / "inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(node.module or "")]
        )
    }
    assert imports.isdisjoint({"train", "scada", "v6_eval"})
    assert "TRAIN_DIR" not in source
    assert ".omx" not in source


def test_final_recipe_models_and_submission_are_self_consistent():
    model_dir = ROOT / "models"
    recipe_path = model_dir / "recipe.json"
    sidecar_path = model_dir / "recipe.json.sha256"
    raw = recipe_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert sidecar_path.read_text(encoding="ascii") == f"{digest}  recipe.json\n"
    recipe = json.loads(raw)
    assert raw == train.canonical_json_bytes(recipe)
    assert set(recipe) == EXPECTED_RECIPE_KEYS
    assert recipe["kind"] == "wind-v6-production-recipe"
    assert recipe["schema_version"] == 1
    assert recipe["promotion"]["status"] == "PASS"
    assert recipe["training"]["seeds"] == [42, 202, 777]
    assert recipe["features"]["count"] == 276
    assert recipe["features"]["file_sha256"] == _sha256(
        model_dir / recipe["features"]["path"]
    )

    declared_models = {
        model["path"]
        for family in recipe["training"]["families"].values()
        for model in family["models"]
    }
    assert declared_models == EXPECTED_ACTIVE_MODELS
    assert {path.name for path in model_dir.glob("lgbm*.txt")} == EXPECTED_ACTIVE_MODELS
    for family in recipe["training"]["families"].values():
        assert family["base_rounds"] == int(
            sum(family["validation_iterations"]) / len(family["validation_iterations"])
        )
        assert family["full_rounds"] == max(int(family["base_rounds"] * 1.2), 100)
        for model in family["models"]:
            path = model_dir / model["path"]
            assert path.is_file()
            assert model["sha256"] == _sha256(path)
            assert model["feature_count"] in {276, 277}

    submission_path = ROOT / "submissions" / "submission.csv"
    assert submission_path.read_bytes().startswith(b"\xef\xbb\xbf")
    sample = pd.read_csv(ROOT / "Data" / "sample_submission.csv", encoding="utf-8-sig")
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    validated = inference.validate_submission_frame(submission, sample)
    assert len(validated) == 8_760
    assert list(validated.columns) == list(sample.columns)
    for group in GROUPS:
        assert (
            validated[group]
            .between(0.10 * CAPACITY_KWH[group], CAPACITY_KWH[group])
            .all()
        )
