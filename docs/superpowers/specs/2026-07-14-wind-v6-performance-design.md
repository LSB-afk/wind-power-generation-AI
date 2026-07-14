# Wind v6 Performance Design

## Goal and evidence contract

Produce a new DACON submission that is a lower-risk improvement over the v4 public score of
`0.63595` and the current v5 recipe. A candidate passes only when, under identical seeds and
data splits, it beats the frozen v5 recipe on both validation folds and improves the mean of the
two fold scores by at least `0.0010`. The final retraining, inference, schema, row count, time
ordering, finite-value, and capacity-bound checks must also pass.

The two folds are:

- fold23: train on 2022 and validate on 2023 for groups 1 and 2;
- fold24: train on 2022-2023 and validate on 2024 for all three groups.

All validation targets remain the actual KPX labels. SCADA-derived values may be used only as
training targets or training-row diagnostics.

## Baseline correction

The repository currently has two distinct reference points:

- v4 commit `b3c5dce` produced the submission associated with public score `0.63595`;
- current `main` commit `41168ab` contains v5 models and a v5 submission that has not been
  recorded as submitted.

The documented v5 final recipe is not directly reproduced by `stage6`, because its final C1
combination (potential-target solo models for groups 1/2 and a potential-target pooled model for
group 3) is absent from the stage output. `stage7` will therefore freeze and evaluate this exact
recipe before testing any v6 candidate.

## Candidate sequence

### 1. Cross-target ensemble (recommended first)

Train two leakage-safe model families on each fold:

- actual-label family: the public-proven v4 cleaning recipe;
- potential-label family: the current v5 recipe.

Search a small, fixed blend grid and accept only weights that improve both folds. This combines
the public robustness of actual-label learning with the fold23 gain of potential reconstruction,
without adding a dependency or relying on year-specific scale/floor tuning.

### 2. Physical feature pack

If blending alone does not pass, add a compact pack shared by training and inference:

- density-adjusted hub-height wind power proxies;
- 80-100 m and 10-100 m shear ratios/exponents;
- gust factors and directional wind components weighted by speed;
- within-forecast-block ramps for the most relevant hub-height variables.

Each feature must be derivable from the supplied forecast at prediction time. The full pack is
accepted only when its combined recipe improves both folds; individual feature fishing is
avoided.

### 3. Metric-aligned alternative

Only if the first two approaches fail, compare a bounded set of LightGBM objectives or
sample-weight rules that approximate the FICR tolerance bands. No post-hoc parameter may be
chosen from fold24 alone. A setting must win on both folds and then be confirmed with the
three-seed ensemble.

## Evaluator and data flow

`python src/exp_runner.py stage7` will:

1. load the existing cached forecast features and SCADA-derived training targets;
2. construct the exact v5 baseline and the selected v6 candidate with identical folds/seeds;
3. apply only the mathematically safe `0.10 * capacity` floor;
4. print per-fold, per-group, NMAE, FICR, score, deltas, and a final PASS/FAIL result;
5. exit non-zero when the performance contract is not met.

The winning recipe will then be represented once in production training code and reused by
inference through saved feature names, model names, ensemble weights, and post-processing
metadata. This prevents the experiment/production drift present in v5.

## Verification and stop condition

Screen candidates with one seed, confirm the winner with seeds `42`, `202`, and `777`, then run:

- the stage7 evaluator;
- targeted metric/feature/submission tests;
- Python compilation and diff checks;
- full `src/train.py` retraining;
- full `src/inference.py` generation;
- submission schema, timestamp, finite-value, and capacity-bound validation.

Stop when a candidate satisfies the dual-fold contract and the regenerated submission passes all
artifact checks. If no candidate satisfies it, retain the best verified existing recipe rather than
claiming an unproven improvement.
