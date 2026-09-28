# LANL Temporal Operating-Point Experiment

## Purpose and caveats

This experiment selects a score threshold on an earlier chronological
validation window, then evaluates it once on a later chronological holdout.
The threshold is chosen as the **highest threshold that attains at least
90% red-team recall on validation**, reducing alerts while
meeting that recall floor there.

This is not a production evaluation. Normal events were globally downsampled
before this split, red-team labels stop before the authentication log ends,
and the test period is the latest period with available positive labels. The
reported precision values reflect these sampled periods. The score is not
calibrated, and the validation recall target is not guaranteed to transfer.

## Chronological windows

Contiguous split boundaries use the 50th and 75th percentiles of positive
event timestamps, so each labeled window has both classes. This retrospective
split is label-informed; it is used to obtain a measurable operating point
from the provided benchmark, not as a prospective deployment protocol.

| Split | Time start | Time end | Normal | Red-team | Rows |
|---|---:|---:|---:|---:|---:|
| Train | 30 | 1,072,461 | 2,750 | 351 | 3,101 |
| Validation | 1,072,594 | 1,153,081 | 233 | 175 | 408 |
| Test | 1,153,118 | 2,557,047 | 3,757 | 176 | 3,933 |

## Validation-selected threshold

| Metric | Validation |
|---|---:|
| Threshold | 0.909930 |
| ROC-AUC | 0.9752 |
| Average Precision | 0.9265 |
| Red-team precision | 0.9600 |
| Red-team recall | 0.9600 |
| Red-team F1 | 0.9600 |
| Normal precision | 0.9700 |
| Normal recall | 0.9700 |
| False-positive rate | 3.0043% |
| TP / FP / FN / TN | 168 / 7 / 7 / 226 |

## Untouched later-period holdout

| Metric | Default threshold 0.5 | Validation-selected threshold |
|---|---:|---:|
| Red-team precision | 0.5317 | 0.5847 |
| Red-team recall | 1.0000 | 0.8239 |
| Normal precision | 1.0000 | 0.9916 |
| Normal recall | 0.9587 | 0.9726 |
| Red-team F1 | 0.6943 | 0.6840 |
| False-positive rate | 4.1256% | 2.7415% |
| TP / FP / FN / TN | 176 / 155 / 0 / 3602 | 145 / 103 / 31 / 3654 |

Detailed metrics at the selected threshold:

| Metric | Test |
|---|---:|
| Threshold | 0.909930 |
| ROC-AUC | 0.9811 |
| Average Precision | 0.5322 |
| Red-team precision | 0.5847 |
| Red-team recall | 0.8239 |
| Red-team F1 | 0.6840 |
| Normal precision | 0.9916 |
| Normal recall | 0.9726 |
| False-positive rate | 2.7415% |
| TP / FP / FN / TN | 145 / 103 / 31 / 3654 |

The test metrics were not used to select this threshold. They are reported
once to show temporal transfer within the labeled benchmark interval. For
comparison, the threshold selection is based only on validation.

## Interpretation

Raising the threshold can reduce red-team recall and reduce false positives,
usually improving red-team precision. It can also increase false negatives,
which may lower normal-class precision. The two class precisions are therefore
not guaranteed to improve together. Compare both class precision values and
the confusion matrix against the operational workload and missed-detection
cost before using the setting. In this run, the selected cutoff improved
red-team precision and reduced recall, but normal-class precision decreased
slightly versus the 0.5 cutoff.

## Artifacts and reproducibility

- Model trained on the training window: `models\lanl_redteam_temporal.joblib`
- Validation-selected threshold and split metadata: `models\lanl_temporal_threshold.json`
- This report: `reports\temporal_threshold_evaluation.md`

Recreate this experiment with:

```powershell
python -m soc_ml.cli tune-lanl-threshold --data data\lanl_auth_sample.csv --model models\lanl_redteam_temporal.joblib --threshold-output models\lanl_temporal_threshold.json --report reports\temporal_threshold_evaluation.md --target-recall 0.90
```
