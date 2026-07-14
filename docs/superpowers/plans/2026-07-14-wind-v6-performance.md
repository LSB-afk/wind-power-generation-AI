# Wind v6 Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible dual-fold evaluator, find a leakage-safe recipe that beats v5 on both folds, and regenerate a validated DACON submission.

**Architecture:** Add a focused `v6_eval.py` module that owns immutable v5/v6 recipe evaluation, prediction blending, cache manifests, and the PASS/FAIL contract. Keep `exp_runner.py` as the CLI router, share the winning feature/config constants with production training, and make inference blend the exact saved model families selected by the evaluator.

**Tech Stack:** Python 3.13, pandas 2.2.3, NumPy 2.1.3, LightGBM 4.6.0, pytest 8.3.4.

## Global Constraints

- Do not add dependencies.
- Validation targets are always actual KPX labels.
- Baseline and candidate use identical folds, rows, seeds, post-processing, and actual arrays.
- fold23 trains on 2022 and validates on 2023 for groups 1/2; fold24 trains on 2022-2023 and validates on 2024 for all groups.
- PASS requires `delta24 > 0`, `delta23 > 0`, and `(delta24 + delta23) / 2 >= 0.0010`.
- Final confirmation uses seeds `42`, `202`, and `777`.
- Only the distribution-independent `0.10 * capacity` floor is allowed without cross-year evidence.
- Raw competition data and experiment caches remain untracked.

---

### Task 1: Evaluator contract and explicit CLI routing

**Files:**
- Create: `src/v6_eval.py`
- Create: `tests/test_v6_eval.py`
- Modify: `src/exp_runner.py:681-700`

**Interfaces:**
- Produces: `gate_scores(baseline: dict[str, float], candidate: dict[str, float], min_mean_delta: float = 0.001) -> dict`
- Produces: `blend_predictions(potential: dict[str, np.ndarray], actual: dict[str, np.ndarray], weights: dict[str, float]) -> dict[str, np.ndarray]`
- Produces: `run_stage7(seeds: tuple[int, ...]) -> int`
- `exp_runner.main()` calls `run_stage7(SEEDS3)` only for the literal mode `stage7`.

- [ ] **Step 1: Write failing pure-function and router tests**

```python
def test_gate_requires_both_folds_and_mean_gain():
    passed = gate_scores({"fold23": 0.63, "fold24": 0.64},
                         {"fold23": 0.631, "fold24": 0.6412})
    assert passed["status"] == "PASS"
    failed = gate_scores({"fold23": 0.63, "fold24": 0.64},
                         {"fold23": 0.6299, "fold24": 0.65})
    assert failed["status"] == "FAIL"

def test_blend_predictions_uses_actual_family_weight():
    potential = {"kpx_group_1": np.array([10.0, 20.0])}
    actual = {"kpx_group_1": np.array([20.0, 40.0])}
    got = blend_predictions(potential, actual, {"kpx_group_1": 0.25})
    np.testing.assert_allclose(got["kpx_group_1"], [12.5, 25.0])
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py`

Expected: collection fails because `v6_eval` does not exist.

- [ ] **Step 3: Implement the pure contract and explicit router**

```python
def gate_scores(baseline, candidate, min_mean_delta=0.001):
    deltas = {fold: candidate[fold] - baseline[fold] for fold in ("fold23", "fold24")}
    mean_delta = sum(deltas.values()) / 2.0
    passed = all(delta > 0.0 for delta in deltas.values()) and mean_delta >= min_mean_delta
    return {"status": "PASS" if passed else "FAIL", "baseline": baseline,
            "candidate": candidate, "deltas": deltas, "mean_delta": mean_delta}

def blend_predictions(potential, actual, weights):
    return {g: (1.0 - weights[g]) * potential[g] + weights[g] * actual[g]
            for g in potential}
```

Replace the catch-all stage6 fallback with explicit `elif mode == "stage6"`, `elif mode == "stage7"`, and an unknown-mode error that exits `2`.

- [ ] **Step 4: Run targeted tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

Commit only `src/v6_eval.py`, `src/exp_runner.py`, and `tests/test_v6_eval.py` with a Lore-format message whose `Tested:` trailer names the pytest command.

### Task 2: Frozen v5 OOF reproduction and provenance guard

**Files:**
- Modify: `src/v6_eval.py`
- Modify: `tests/test_v6_eval.py`

**Interfaces:**
- Produces: `load_bundle() -> TrainingBundle`
- Produces: `fit_fold(recipe: str, train_years: tuple[int, ...], valid_year: int, groups: tuple[str, ...], seeds: tuple[int, ...]) -> FoldPredictions`
- Produces: `score_fold(predictions: FoldPredictions) -> float`
- Produces: ignored cache files under `.omx/experiments/wind-v6/` with a JSON manifest containing recipe, seeds, feature hash, data hashes, row counts, and prediction hashes.

- [ ] **Step 1: Add failing tests for fold alignment, floor, cache keys, and provenance**

Use synthetic hourly indices to assert that baseline and candidate validation indices are identical, that floor10 never changes predictions already above the floor, and that changing a seed or feature name changes the manifest key.

- [ ] **Step 2: Run the targeted tests and confirm failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py`

Expected: failures for the new undefined functions.

- [ ] **Step 3: Implement the exact v5 recipe**

The immutable baseline is potential target, mismatch-only exclusion, `objective="quantile"`, `alpha=0.60`, filter ratio `0.05`, groups 1/2 solo, group 3 potential-pooled, and floor10. Build potential/mismatch frames from raw data when caches are missing. Record these row-count guards:

```python
EXPECTED_ROWS = {
    "fold23": {"g1_train": 6215, "g2_train": 6174, "g1_valid": 8757, "g2_valid": 8758},
    "fold24": {"g1_train": 12516, "g2_train": 12455, "pooled_train": 30153,
               "g1_valid": 8778, "g2_valid": 8778, "g3_valid": 8778},
}
```

Refuse to score and return evaluator code `2` if these fingerprints or the seed-42 score anchors drift beyond `0.00015` from fold23 `0.6316` and fold24 `0.6380`.

- [ ] **Step 4: Run synthetic tests, then reproduce the one-seed baseline**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7 --seeds 42 --baseline-only`

Expected: tests pass; stage7 reports anchors within tolerance and writes a cache manifest.

- [ ] **Step 5: Checkpoint and commit**

Record a performance-goal checkpoint with the baseline scores and commit the frozen evaluator with the exact commands in `Tested:`.

### Task 3: Fold-safe turbine-weighted potential candidate

**Files:**
- Modify: `src/scada.py`
- Modify: `src/v6_eval.py`
- Modify: `src/exp_runner.py`
- Create: `tests/test_scada.py`
- Modify: `tests/test_v6_eval.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- Produces: immutable `WeightCalibration` metadata with train label-years, ordered turbine
  columns, weights, calibration row count, calibration-index hash, and weights hash.
- Produces: `estimate_turbine_weights(...)`, `reconstruct_weighted_potential(...)`, and
  `build_weighted_targets(...)` from `src/scada.py`; evaluation and production must share them.
- Produces: candidate recipe `v6-weighted-potential-q60-filter05-floor10` in `src/v6_eval.py`.
- Produces: `stage7 --seeds 42 --screen weighted` with baseline/candidate scores, deltas, and
  machine-readable gate output.

- [ ] **Step 1: Write failing reconstruction and calendar-boundary tests**

Cover equal-weight parity with v5, all-healthy identity, a proportional-output example, missing
high/low-weight turbines, deterministic normalized weights, sentinel/NaN handling, unsorted input,
column ordering, exact-hour timestamps, and finite division behavior. Pin calendar membership to
`hour_end = timestamp.ceil("h")`, including `12/31 23:50`, so validation-year SCADA cannot affect
calibration or training targets. Pin minimum healthy turbines to `3/3/2`.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_scada.py tests/test_v6_eval.py`

Expected: missing weighted-target interfaces and candidate recipe fail before production edits.

- [ ] **Step 3: Implement one shared weighted-target builder**

Use calibration rows from training label-years only. Require every turbine's sanitized power and
wind speed to be present, every power `> 1`, every wind speed `>= 5`, and group output at least
`0.10 * capacity / 6`. Estimate robust shares with
`n * median(turbine_power / group_power)`, then renormalize to mean one. Reconstruct with weighted
healthy-turbine coverage, clip each 10-minute value before hourly aggregation, and return NaN
outside requested target label-years. Accept label indexes, never actual label values.

- [ ] **Step 4: Integrate candidate provenance and cache safety**

Hash calibration indexes/weights, weighted targets, post-filter training indexes/targets, and
candidate row counts. Keep raw-source hashes and actual validation targets identical to baseline;
allow candidate derived-target hashes and training counts to differ. Fold23 must never calibrate
group 3. Pin these expected rows:

```python
WEIGHTED_EXPECTED_ROWS = {
    "fold23": {"g1_train": 6214, "g2_train": 6176,
               "g1_valid": 8757, "g2_valid": 8758},
    "fold24": {"g1_train": 12516, "g2_train": 12458, "pooled_train": 30158,
               "g1_valid": 8778, "g2_valid": 8778, "g3_valid": 8778},
}
```

- [ ] **Step 5: Reproduce the corrected one-seed gate**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7 --seeds 42 --screen weighted`

Expected (within deterministic tolerance): fold23 `0.633691786326`, fold24 `0.639226950590`, both
deltas positive, mean delta `0.001632482019`, final status `PASS`.

- [ ] **Step 6: Record, review, and commit**

Append the candidate, rejected g13 variants, formula, boundary correction, and exact evidence to
`docs/experiments.md`. Run project tests/static checks, obtain an independent code review, then
commit with Lore trailers.

### Task 4: Three-seed final evaluator gate

**Files:**
- Modify: `src/v6_eval.py`
- Modify: `src/exp_runner.py`
- Modify: `tests/test_v6_eval.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- `run_stage7((42, 202, 777))` evaluates frozen v5 and exactly one weighted-potential candidate
  with identical raw sources, features, validation targets, seeds, and post-processing.
- Default `python src/exp_runner.py stage7` prints one final JSON result and exits `0` only when
  `gate_scores()` passes.

- [ ] **Step 1: Write the final-route and result-contract tests**

Assert default stage7 selects only the weighted recipe, reports per-fold/per-group NMAE and FICR,
compares exact actual targets, and returns non-zero for either-fold regression or mean gain below
`0.0010`.

- [ ] **Step 2: Run the canonical three-seed evaluator**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7`

Expected: exit `0`, both deltas positive, mean delta at least `0.0010`, and final JSON status
`PASS`. If this fails, stop promotion and use only the bounded contingency candidates documented
in the design; do not weaken the gate.

- [ ] **Step 3: Run regression/static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests`

Run: `python -m compileall -q src tests`

Run: `black --check src tests && mypy src`

- [ ] **Step 4: Record passing checkpoint and commit**

Run `omx performance-goal checkpoint --slug wind-v6-dual-fold --status pass` with the complete
three-seed fold scores/deltas and manifest keys. Commit evaluator/recipe/docs with Lore trailers.

### Task 5: Production retraining and inference parity

**Files:**
- Modify: `src/train.py`
- Modify: `src/inference.py`
- Modify: `models/feature_cols.txt`
- Modify/Create: selected `models/*.txt`
- Modify: `models/post_params.json`
- Create: `models/recipe.json`
- Create: `tests/test_submission.py`

**Interfaces:**
- Training imports the exact weighted-target implementation used by stage7 and saves the selected
  models plus `recipe.json` containing calibration metadata, target/training hashes, seeds, feature
  schema hash, rounds, model hashes, floor ratios, and final evaluator scores.
- Inference loads only files named by `recipe.json`, verifies model/feature hashes, applies floor10,
  and writes `submissions/submission.csv`.

- [ ] **Step 1: Write failing metadata and submission-validator tests**

Assert required recipe keys, model-file existence, exact sample column order, 8,760 rows, matching IDs/timestamps, unique rows, finite values, and bounds `[0, capacity]`.

- [ ] **Step 2: Run tests and confirm missing metadata failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_submission.py`

- [ ] **Step 3: Make training and inference consume the frozen weighted recipe**

Reuse the exact feature function, params, seeds, weighted target construction, grouping, and
post-processing confirmed by stage7. Full training uses label-years 2022-2024 only; the
`2025-01-01 00:00` boundary label remains excluded. Assert evaluator and production target hashes
match for identical year inputs.

- [ ] **Step 4: Retrain and generate the artifact**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/train.py`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/inference.py`

Expected: all selected models, metadata, and `submissions/submission.csv` are regenerated without warnings that invalidate rows.

- [ ] **Step 5: Validate production artifacts**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_submission.py`

Run: `git diff --check`

Expected: all checks pass.

### Task 6: Completion audit and handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/experiments.md`

**Interfaces:**
- Documents the exact command, baseline/candidate fold scores, model recipe, artifact hash, and local-vs-public limitation.

- [ ] **Step 1: Update reproducibility documentation**

Document `stage7`, full retraining/inference, the selected v6 recipe, both-fold evidence, and the SHA-256 of the generated CSV.

- [ ] **Step 2: Run the complete audit**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests`

Run: `python -m compileall -q src tests`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/inference.py`

Run: `git diff --check && git status --short`

Expected: evaluator PASS, tests/compile/diff checks pass, and the regenerated submission remains valid.

- [ ] **Step 3: Commit the final artifacts**

Use a Lore-format commit with `Constraint:`, rejected alternatives, confidence, scope risk, exact tests, and any known Public-LB gap.

- [ ] **Step 4: Complete the durable goal**

After the completion audit, mark the Codex goal complete, fetch the fresh goal snapshot, and pass it to `omx performance-goal complete --slug wind-v6-dual-fold` with the evaluator and artifact evidence.
