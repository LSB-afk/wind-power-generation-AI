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

### Task 3: Cross-target ensemble screening

**Files:**
- Modify: `src/v6_eval.py`
- Modify: `tests/test_v6_eval.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- Produces: actual-label v4 family predictions using bad-mask exclusion, q60/filter05, solo groups 1/2, pooled group 3, and floor10.
- Produces: `screen_blend_weights(...) -> list[dict]` over weights `{0.25, 0.50, 0.75}`.

- [ ] **Step 1: Write failing tests for group-wise blend-grid ranking**

Use fixed synthetic baseline/alternative predictions and actuals to prove that ranking is deterministic and never reads validation targets while constructing predictions.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py -k blend`

Expected: `screen_blend_weights` is missing.

- [ ] **Step 3: Implement actual-label family and bounded grid**

Train/cache actual-label OOF predictions on the exact baseline folds. Evaluate global actual-family weights `0.25`, `0.50`, and `0.75`; select the highest mean score among candidates with positive deltas on both folds. Do not search scale or floor.

- [ ] **Step 4: Run one-seed screening**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7 --seeds 42 --screen blend`

Expected: a table with baseline, each weight, fold deltas, and a selected or rejected result.

- [ ] **Step 5: Record evidence**

Append every screened score to `docs/experiments.md`; checkpoint failures as `fail`, not `pass`. Commit only if the harness behavior changed or a recipe is selected.

### Task 4: Physical feature-pack candidate if blending is insufficient

**Files:**
- Modify: `src/features.py`
- Modify: `src/v6_eval.py`
- Create: `tests/test_features.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- Produces: `add_physical_features(feat: pd.DataFrame) -> pd.DataFrame`
- Adds stable columns `phys_ws100_density_power`, `phys_ws80_density_power`, `phys_shear_100_80`, `phys_shear_100_10`, `phys_gust_factor`, `phys_ws100_u`, and `phys_ws100_v` when source columns are present.

- [ ] **Step 1: Write failing deterministic feature tests**

Construct a two-row frame with known wind speed, density, direction sine/cosine, gust, and shear inputs. Assert exact formulas, finite outputs, unchanged index, and identical train/test column order.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_features.py`

Expected: `add_physical_features` is missing.

- [ ] **Step 3: Implement the compact feature pack**

Use clipped denominators (`1e-3`) and no learned statistics. Call the function from `build_features` for both train and test.

- [ ] **Step 4: Screen with one seed**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7 --seeds 42 --screen physical`

Expected: candidate fold23/fold24 scores and deltas are printed. Retain the pack only if both deltas are positive.

- [ ] **Step 5: Test and record**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_features.py tests/test_v6_eval.py`

Update `docs/experiments.md` with accepted/rejected evidence and commit the feature pack only when accepted.

### Task 5: Metric-aligned bounded search if prior candidates fail

**Files:**
- Modify: `src/v6_eval.py`
- Modify: `tests/test_v6_eval.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- Produces: a bounded search over quantile alphas `{0.57, 0.60, 0.63}` and evaluation-aligned training weights `{uniform, 1 + y/cap}` with all other parameters frozen.

- [ ] **Step 1: Add failing tests for the exact candidate matrix**

Assert the matrix has exactly six unique candidates, preserves frozen base parameters, and contains no scale/floor tuning fields.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_v6_eval.py -k candidate_matrix`

- [ ] **Step 3: Implement and run one-seed bounded search**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7 --seeds 42 --screen metric`

Expected: six candidate rows with both fold deltas; select only a both-fold winner.

- [ ] **Step 4: Record and checkpoint evidence**

Update `docs/experiments.md` and the performance-goal ledger. Do not promote a setting that wins only one fold.

### Task 6: Three-seed final evaluator gate

**Files:**
- Modify: `src/v6_eval.py`
- Modify: `docs/experiments.md`

**Interfaces:**
- `run_stage7((42, 202, 777))` evaluates the frozen v5 baseline and exactly one selected v6 candidate and prints a final JSON object.

- [ ] **Step 1: Freeze the selected recipe constants**

Store the winning model family, feature-pack flag, alpha/weight setting, and group blend weights as immutable literals in `v6_eval.py`; remove exploratory selection from the default stage7 path.

- [ ] **Step 2: Run the canonical evaluator**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7`

Expected: exit `0`, both deltas positive, mean delta at least `0.0010`, and final JSON `status` equal to `PASS`.

- [ ] **Step 3: Run regression checks**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q`

Run: `python -m compileall -q src tests`

Expected: both commands succeed.

- [ ] **Step 4: Record passing checkpoint and commit**

Run `omx performance-goal checkpoint --slug wind-v6-dual-fold --status pass` with full fold scores/deltas and commit evaluator/recipe/docs using Lore trailers.

### Task 7: Production retraining and inference parity

**Files:**
- Modify: `src/train.py`
- Modify: `src/inference.py`
- Modify: `models/feature_cols.txt`
- Modify/Create: selected `models/*.txt`
- Modify: `models/post_params.json`
- Create: `models/recipe.json`
- Create: `tests/test_submission.py`

**Interfaces:**
- Training saves every selected model family with unambiguous prefixes and a `recipe.json` containing seeds, weights, feature hash, model files, floor ratios, and evaluator scores.
- Inference loads only `recipe.json`, verifies model/feature hashes, blends selected families, applies floor10, and writes `submissions/submission.csv`.

- [ ] **Step 1: Write failing metadata and submission-validator tests**

Assert required recipe keys, model-file existence, exact sample column order, 8,760 rows, matching IDs/timestamps, unique rows, finite values, and bounds `[0, capacity]`.

- [ ] **Step 2: Run tests and confirm missing metadata failure**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_submission.py`

- [ ] **Step 3: Make training and inference consume the frozen winning recipe**

Reuse the exact feature function, params, seeds, target construction, family names, and blend weights confirmed by stage7. Include the final `2025-01-01 00:00` label row only if the separately evaluated boundary candidate passed both folds; otherwise preserve the frozen year split.

- [ ] **Step 4: Retrain and generate the artifact**

Run: `PYTHONDONTWRITEBYTECODE=1 python src/train.py`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/inference.py`

Expected: all selected models, metadata, and `submissions/submission.csv` are regenerated without warnings that invalidate rows.

- [ ] **Step 5: Validate production artifacts**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_submission.py`

Run: `git diff --check`

Expected: all checks pass.

### Task 8: Completion audit and handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/experiments.md`

**Interfaces:**
- Documents the exact command, baseline/candidate fold scores, model recipe, artifact hash, and local-vs-public limitation.

- [ ] **Step 1: Update reproducibility documentation**

Document `stage7`, full retraining/inference, the selected v6 recipe, both-fold evidence, and the SHA-256 of the generated CSV.

- [ ] **Step 2: Run the complete audit**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q`

Run: `python -m compileall -q src tests`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/exp_runner.py stage7`

Run: `PYTHONDONTWRITEBYTECODE=1 python src/inference.py`

Run: `git diff --check && git status --short`

Expected: evaluator PASS, tests/compile/diff checks pass, and the regenerated submission remains valid.

- [ ] **Step 3: Commit the final artifacts**

Use a Lore-format commit with `Constraint:`, rejected alternatives, confidence, scope risk, exact tests, and any known Public-LB gap.

- [ ] **Step 4: Complete the durable goal**

After the completion audit, mark the Codex goal complete, fetch the fresh goal snapshot, and pass it to `omx performance-goal complete --slug wind-v6-dual-fold` with the evaluator and artifact evidence.
