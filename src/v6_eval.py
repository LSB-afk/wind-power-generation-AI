"""Shared contracts for the wind-v6 evaluator."""

import sys

import numpy as np


def gate_scores(
    baseline: dict[str, float],
    candidate: dict[str, float],
    min_mean_delta: float = 0.001,
) -> dict:
    deltas = {
        fold: candidate[fold] - baseline[fold] for fold in ("fold23", "fold24")
    }
    mean_delta = sum(deltas.values()) / 2.0
    passed = (
        all(delta > 0.0 for delta in deltas.values())
        and mean_delta >= min_mean_delta
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "mean_delta": mean_delta,
    }


def blend_predictions(
    potential: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    weights: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        group: (1.0 - weights[group]) * potential[group]
        + weights[group] * actual[group]
        for group in potential
    }


def run_stage7(seeds: tuple[int, ...]) -> int:
    """Fail closed until the stage7 evaluator is implemented."""
    del seeds
    print("stage7 evaluator is not implemented yet", file=sys.stderr)
    return 2
