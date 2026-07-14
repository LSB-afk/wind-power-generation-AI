import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from v6_eval import blend_predictions, gate_scores, run_stage7


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


def test_blend_predictions_uses_actual_family_weight():
    potential = {"kpx_group_1": np.array([10.0, 20.0])}
    actual = {"kpx_group_1": np.array([20.0, 40.0])}

    got = blend_predictions(potential, actual, {"kpx_group_1": 0.25})

    np.testing.assert_allclose(got["kpx_group_1"], [12.5, 25.0])


def test_unimplemented_stage7_fails_closed():
    assert run_stage7((42, 202, 777)) == 2


def _stub_main_inputs(monkeypatch, exp_runner):
    ldaps_raw = pd.DataFrame(
        columns=["forecast_kst_dtm", "data_available_kst_dtm"]
    )
    monkeypatch.setattr(exp_runner, "build_cache", lambda: None)
    monkeypatch.setattr(
        exp_runner,
        "load_cache",
        lambda: (pd.DataFrame(), ldaps_raw, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(exp_runner, "add_context", lambda feat, _davail: feat)


def test_main_routes_literal_stage7_and_propagates_result(monkeypatch):
    import exp_runner

    _stub_main_inputs(monkeypatch, exp_runner)
    calls = []
    monkeypatch.setattr(
        exp_runner, "run_stage7", lambda seeds: calls.append(seeds) or 17
    )
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage7"])

    assert exp_runner.main() == 17
    assert calls == [exp_runner.SEEDS3]


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

    _stub_main_inputs(monkeypatch, exp_runner)
    stage6_calls = []
    stage7_calls = []
    monkeypatch.setattr(exp_runner, "stage6", stage6_calls.append)
    monkeypatch.setattr(exp_runner, "run_stage7", stage7_calls.append)
    monkeypatch.setattr(sys, "argv", ["exp_runner.py", "stage7-extra"])

    assert exp_runner.main() == 2
    assert stage6_calls == []
    assert stage7_calls == []
    assert "stage7-extra" in capsys.readouterr().err
