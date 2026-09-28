# LANL ML Pipeline Audit

## Scope and data

The prepared file contains 14,742 rows: 702 red-team and
14,040 normal. Its sampled red-team prevalence is
4.7619%. The preprocessing run scanned 1,051,430,459
auth events and matched 702 red-team events, for an
observed benchmark prevalence of 0.00006677%. Normal events in
the prepared file were selected by a global reservoir, giving a prepared
negative-to-positive ratio of 20.00:1. That
is a strong case-control sample, not a representative
alert stream.

The saved CSV spans timestamps 30 through 5,011,041; the latest
matched positive is at 2,557,047. There are 10
byte-for-byte duplicate prepared rows. Because the preparer discarded source
user and computer IDs, those rows cannot be safely deduplicated or attributed
to distinct entities from this CSV alone.

## How the old metrics were produced

`train_lanl_model` used a stratified, random 75/25 event-row split
(`random_state=42`). A one-hot encoder and scaler were fitted inside a sklearn
pipeline on the training fold only (good); logistic regression used
`class_weight='balanced'`. The holdout report used the classifier's default
0.5 threshold. ROC-AUC and Average Precision were computed on the random
holdout, where prevalence remained 4.7619%, not at the observed benchmark
prevalence. There was no threshold selection, calibration, hyperparameter
search, temporal split, group split, or independent test set.

Reproduction on this prepared sample:

| Metric | Random sampled holdout |
|---|---:|
| ROC-AUC | 0.986189 |
| Average Precision | 0.640886 |
| Precision at legacy 0.5 | 0.558730 |
| Recall at legacy 0.5 | 1.000000 |
| F1 at legacy 0.5 | 0.716904 |
| TP / FP / FN / TN | 176 / 139 / 0 / 3371 |
| Sampled-holdout FPR | 3.960114% |

Applying that sampled FPR and recall to the observed prevalence gives an
illustrative prior-adjusted precision of
0.00168594%. This is an extrapolation from only
a sampled negative holdout, not a measured production precision or a calibrated
probability. Its magnitude shows why sampled precision is misleading; it must
not be used as a performance guarantee.

## Methodological findings

1. **Random split leaks chronology across folds.** Authentication events have
   time order, yet random splitting allows later events into training and
   earlier events into holdout.
2. **Normal sampling happened before the split.** The global reservoir
   downsampled normal events across the complete timeline. This changed the
   holdout distribution and allowed future-period normal rows into training.
3. **No useful future labeled evaluation is available.** In an illustrative
   chronological split by 60/20/20 of the prepared timestamp span:

   | Period | Rows | Red-team | Normal |
   |---|---:|---:|---:|
   | train earliest 60 percent | 8,564 | 702 | 7,862 |
| validation next 20 percent | 3,071 | 0 | 3,071 |
| test latest 20 percent | 3,107 | 0 | 3,107 |

   The latest known positive predates the later authentication activity.
   Validation and test therefore have no positive labels. A future-period
   ROC-AUC, Average Precision, recall, or threshold objective is undefined.
   The available data cannot support the requested trustworthy temporal
   validation/test, regardless of model choice.
4. **Entity generalization cannot be measured from this CSV.** Source and
   destination users/computers were removed before persistence. Entity IDs are
   not model features, but their removal also prevents unseen-user, unseen-host,
   and unseen-pair analysis.
5. **Feature leakage is not evident in the retained feature definitions.**
   Retained model inputs are authentication type, logon type, orientation,
   success, hour of day, machine-account flag, and same-computer flag. Timestamp
   itself and entity identifiers are excluded from model inputs. The two
   boolean flags are transformations of source identity fields, so they encode
   limited identity structure; they are not future-looking. No event-history
   or rolling feature exists.
6. **Feature engineering is narrow.** These are raw event attributes plus a
   time-of-day transform and two source/destination-derived attributes. There
   are no user-, host-, relationship-, or historical-window aggregates.
   Adding them would require preserving IDs and maintaining strictly
   prior-event state; they have not been invented from the anonymized sample.
7. **Threshold and probability claims are unsupported.** The reported
   classification report uses 0.5 without an operational objective.
   Class-weighted logistic probabilities are not calibrated to natural
   prevalence. No Platt or isotonic calibration was performed.
8. **The old holdout is not a final test.** It was used to produce reported
   metrics and is sampled/random. Repeatedly tuning on it would further
   contaminate it.
9. **Repeated patterns and dataset artifacts remain uncertain.** The prepared
   file does not retain enough entity/event identity to identify repeated
   authentication relationships. The labeled red-team activity is a narrow
   benchmark simulation; predictive event types or logon patterns may be
   artifacts specific to that activity and need cross-period and cross-entity
   checks before interpretation.

## Potential Dataset-Specific Artifacts

The exploratory logistic coefficients rank authentication type (`NTLM`,
`Kerberos`) and the `same_computer` indicator among the strongest signals.
These may reflect how this particular red-team simulation generated events,
how the background LANL population is represented, or a real behavioral
difference. This dataset alone cannot distinguish those explanations. In
particular, authentication protocol and event orientation may not remain
predictive against a different attacker or environment; do not treat their
coefficients as portable detection logic. No feature is established as
leakage, but none has passed a future-period or unseen-entity test.

## Feature inventory

| Feature | Category | Available before event? | Caveat |
|---|---|---|---|
| authentication_type, logon_type, orientation, success | Raw event | Yes | Benchmark-specific categorical semantics |
| hour_of_day | Temporal | Yes | Derived from event timestamp; no date fed to model |
| machine_account | Source-user attribute | Yes | Limited identity-derived signal |
| same_computer | Source/destination relation | Yes | Can encode relationship structure; no history |
| source/destination IDs | Entity | No (removed) | Blocks group-generalization audit |
| rolling/history aggregates | Historical/aggregated | Not implemented | No future information used, but no behavior baseline |

## Baselines and final-model status

Only the historical logistic-regression baseline was run. Random forest,
gradient boosting, and anomaly-detection comparisons are intentionally not
presented as model-selection evidence: there is no positive-labeled temporal
validation/test set on which to compare them validly. Hyperparameter tuning,
calibration, validation threshold selection, bootstrap intervals, and a
precision-recall operating curve are not claimed. The CSV can reproduce the
legacy random-split diagnostic, but not the requested scientifically valid
final evaluation.

The coefficient export is exploratory only. It reflects a model fit to the
biased random-split training fold and is not causal or validated importance.
See `feature_importance.csv`.

## Required next data step

Rebuild the data from raw auth logs while retaining event timestamp and
temporary user/host keys for split and group diagnostics. Assign contiguous
time periods **before** sampling. Sample negatives only from training; retain
the complete validation/test periods for scoring, preferably in a streamed
format. Obtain ground-truth positives in the later validation and final test
periods. Until those labels exist, no operational threshold or final temporal
metric is defensible.
