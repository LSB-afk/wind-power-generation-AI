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

## Candidate decision

### Selected candidate: fold-safe turbine-weighted potential

A bounded seed-42 probe showed that the equal-turbine assumption in the v5 SCADA potential is
the most promising remaining bias. When a healthy turbine's output is approximately
`p_i = c * w_i`, the all-turbine potential is reconstructed as:

```text
sum(p_i for healthy i) * sum(w_i for all i) / sum(w_i for healthy i)
```

The weights are robust per-turbine output shares learned only from the training label-years.
Equal weights reduce exactly to v5, and an all-healthy row reduces to the observed group total.
The corrected leakage-safe probe produced:

- fold23 `0.633691786326`, delta `+0.002067801100`;
- fold24 `0.639226950590`, delta `+0.001197162938`;
- mean delta `+0.001632482019`, satisfying the one-seed gate.

Calendar membership is defined by `hour_end = raw_scada_timestamp.ceil("h")`, never by the raw
10-minute timestamp's year. This excludes the five `12/31 23:10`-`23:50` rows whose hourly label
belongs to the following validation year. The legacy healthy-count thresholds remain exactly
`3/3/2` for groups 1/2/3.

The weights, ordered turbine columns, calibration row/index hashes, weighted target hashes, and
post-filter training row/target hashes become candidate provenance. Evaluation and production
must import the same target builder.

### Rejected or deferred probes

- Adding temporal context for `ldaps_g13_ws10` improved both folds but only by a mean
  `+0.000494403`, below the contract.
- Combining that context with the weighted target weakened fold24 to only `+0.000138713`; the
  extra feature complexity is rejected.
- Cross-target blending, the broader physical pack, and metric-weight search are now contingency
  paths only if the selected recipe fails the three-seed gate. Continuing to search after a
  physically grounded both-fold winner would add selection risk without evidence of need.

## Evaluator and data flow

`python src/exp_runner.py stage7` will:

1. rebuild forecast features and SCADA-derived targets from the current raw competition files,
   using only hash-verified prediction caches;
2. construct the exact v5 baseline and the selected v6 candidate with identical folds/seeds;
3. apply only the mathematically safe `0.10 * capacity` floor;
4. print per-fold, per-group, NMAE, FICR, score, deltas, and a final PASS/FAIL result;
5. exit non-zero when the performance contract is not met.

The exact three-seed passing result is also persisted atomically as a hash-bound final-gate
artifact. It includes ordered per-seed validation `best_iteration` values for each candidate model
family. Full-data training preserves the existing refit policy separately from evaluator parity:
truncate the mean fold24 best iteration, multiply by `1.2`, truncate again, and use at least 100
rounds for every seed in that family. Production must recompute and verify this scalar from the
gate artifact rather than copying it manually.

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
