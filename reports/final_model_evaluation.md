# LANL Model Evaluation Status

## Status: no validated final model

The prior model's random sampled-holdout results are reproducible but are not
a temporal or production-distribution evaluation. They are retained below as
the **before** diagnostic; no "after" performance improvement is claimed
because the available data cannot support a positive-labeled future test.

## Dataset

- Auth events scanned: 1,051,430,459
- Matched red-team events: 702
- Observed benchmark prevalence: 0.00006677%
- Prepared sample: 702 red-team, 14,040 sampled
  normal (4.7619% positive)
- Sampling: all matched positives and a global reservoir of up to 20 normal
  events per positive
- Event-time range in prepared data: 30–5,011,041
- Latest known red-team timestamp: 2,557,047

## Legacy diagnostic ("before")

- Model: class-weighted logistic regression with one-hot categoricals and
  standardized numeric features.
- Split: stratified random 75/25, seed 42.
- Threshold: default 0.5, not selected on validation.
- ROC-AUC: 0.986189
- Average Precision: 0.640886
- Sampled precision / recall / F1: 0.558730 /
  1.000000 / 0.716904
- Confusion matrix (TN, FP, FN, TP): 3371,
  139, 0,
  176
- Sampled false-positive rate: 3.960114%
- Illustrative precision at observed prevalence using that FPR:
  0.00168594%; extrapolated, not measured.

## Before versus after

| Evaluation | ROC-AUC | Average Precision | Precision / Recall | Status |
|---|---:|---:|---:|---|
| Before: random, downsampled holdout | 0.986189 | 0.640886 | 0.558730 / 1.000000 | Reproduced; not trustworthy for future activity |
| After: temporal, natural-prevalence test | Not computed | Not computed | Not computed | No positive labels in future periods |

## Temporal, calibration, and operating-point assessment

A chronological 60/20/20 timestamp split has the following label coverage:

| Period | Rows | Red-team | Normal |
|---|---:|---:|---:|
| train earliest 60 percent | 8,564 | 702 | 7,862 |
| validation next 20 percent | 3,071 | 0 | 3,071 |
| test latest 20 percent | 3,107 | 0 | 3,107 |

The latest available red-team event is at timestamp 2,557,047.
Later periods contain no positive ground truth, so they cannot measure future
red-team recall, ROC-AUC, Average Precision, or precision. Calibration,
precision-at-recall, validation threshold selection, and confidence intervals
are therefore **not available for this particular 60/20/20 split**. A
separate historical operating-point experiment uses positive-timestamp
quantiles to obtain labeled validation and test windows; see
`temporal_threshold_evaluation.md`. Its threshold is not a prospective or
production recommendation.

## Features, leakage, and generalization

The model uses four raw authentication fields, hour-of-day, machine-account,
and same-computer indicators. No future-looking history features are present.
However, the identifiers needed for user/host/group analysis were removed from
the prepared CSV. The random holdout shares the full time range and cannot
establish generalization to future time or unseen entities. Dataset-specific
red-team simulation artifacts have not been ruled out.

`feature_importance.csv` contains exploratory logistic coefficients from the
legacy diagnostic only. It must not be interpreted as validated feature
importance. No SHAP summary, alternative-model leaderboard, calibrated
probabilities, or production-ready classifier is claimed.

## Conclusion and reproducibility

The legacy random holdout is not a trustworthy future-period result. A
historical chronological holdout within the available label period is
available in `temporal_threshold_evaluation.md`, but it is not an independent
post-label-period test. Later-period positive ground truth is still needed to
measure deployment-time performance. Do not deploy this model or use its
scores as probabilities of compromise.

Reproduce the audit and legacy diagnostic with:

```powershell
python -m soc_ml.cli audit-lanl --data data\lanl_auth_sample.csv --auth-events-scanned 1051430459 --redteam-events-matched 702 --report-dir reports
```

The audit is deterministic (seed 42). For inference with the baseline model,
score a raw auth file to obtain an uncalibrated ranking score:

```powershell
python -m soc_ml.cli score-lanl-auth --auth data\new_auth.txt.gz --model models\lanl_redteam.joblib --output data\new_auth_scores.csv.gz
```

The score is uncalibrated and no decision threshold is implicit. To use the
separately validation-selected historical threshold, first create the
`lanl_redteam_temporal.joblib` and `lanl_temporal_threshold.json` artifacts
with `tune-lanl-threshold`, then pass that JSON with `--threshold-file`. This
threshold is not production-validated.
