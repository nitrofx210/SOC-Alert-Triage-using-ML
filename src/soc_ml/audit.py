"""Reproducible audit of the sampled LANL experiment and its evaluation limits."""

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from soc_ml.lanl import LANL_FEATURE_COLUMNS, LANL_TARGET_COLUMN
from soc_ml.lanl_model import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    validate_stratified_holdout,
)


def audit_lanl_sample(
    data_path: Path,
    report_dir: Path,
    *,
    auth_events_scanned: int,
    redteam_events_matched: int,
) -> dict[str, object]:
    """Reproduce the legacy random holdout and write an evidence-based audit.

    The random holdout is reported only to explain the old result; the audit
    explicitly does not designate it as a valid final evaluation.
    """
    if auth_events_scanned < 1:
        raise ValueError("auth_events_scanned must be at least 1")
    if not 0 < redteam_events_matched < auth_events_scanned:
        raise ValueError(
            "redteam_events_matched must be positive and below auth_events_scanned"
        )

    data = pd.read_csv(data_path)
    required = set(LANL_FEATURE_COLUMNS) | {LANL_TARGET_COLUMN, "time"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required LANL audit columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("LANL audit data must contain at least one row")
    if data.loc[:, LANL_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("LANL audit features must not contain missing values")
    if data[LANL_TARGET_COLUMN].isnull().any():
        raise ValueError("LANL audit target must not contain missing values")
    if set(data[LANL_TARGET_COLUMN].unique()) != {"normal", "redteam"}:
        raise ValueError("LANL audit data must contain normal and redteam classes")
    sample_redteam = int(data[LANL_TARGET_COLUMN].eq("redteam").sum())
    if sample_redteam != redteam_events_matched:
        raise ValueError(
            "redteam_events_matched must equal the red-team rows in the prepared data"
        )
    if auth_events_scanned < len(data):
        raise ValueError("auth_events_scanned cannot be below prepared sample rows")
    validate_stratified_holdout(data[LANL_TARGET_COLUMN])

    event_time = pd.to_numeric(data["time"], errors="coerce")
    if event_time.isnull().any() or not event_time.map(math.isfinite).all():
        raise ValueError("LANL audit timestamps must be finite numeric values")

    x_train, x_test, y_train, y_test = train_test_split(
        data.loc[:, LANL_FEATURE_COLUMNS],
        data[LANL_TARGET_COLUMN],
        test_size=0.25,
        random_state=42,
        stratify=data[LANL_TARGET_COLUMN],
    )
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    redteam_index = list(model.classes_).index("redteam")
    scores = model.predict_proba(x_test)[:, redteam_index]
    predictions = model.predict(x_test)
    binary_targets = y_test.eq("redteam").astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=["normal", "redteam"]
    ).ravel()
    sampled_precision = precision_score(
        y_test, predictions, pos_label="redteam", zero_division=0
    )
    sampled_recall = recall_score(
        y_test, predictions, pos_label="redteam", zero_division=0
    )
    sampled_f1 = f1_score(y_test, predictions, pos_label="redteam", zero_division=0)
    sampled_fpr = fp / (fp + tn) if fp + tn else 0.0
    original_prevalence = redteam_events_matched / auth_events_scanned
    adjusted_precision = _precision_at_prevalence(
        sampled_recall, sampled_fpr, original_prevalence
    )
    roc_auc = roc_auc_score(binary_targets, scores)
    average_precision = average_precision_score(binary_targets, scores)

    min_time = int(event_time.min())
    max_time = int(event_time.max())
    total_span = max_time - min_time
    split_bounds = (min_time + total_span * 0.60, min_time + total_span * 0.80)
    temporal_counts = _time_split_counts(event_time, data[LANL_TARGET_COLUMN], split_bounds)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_feature_importance(model, report_dir / "feature_importance.csv")
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_reports(
        report_dir,
        data=data,
        auth_events_scanned=auth_events_scanned,
        redteam_events_matched=redteam_events_matched,
        original_prevalence=original_prevalence,
        min_time=min_time,
        max_time=max_time,
        temporal_counts=temporal_counts,
        random_metrics={
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "precision": sampled_precision,
            "recall": sampled_recall,
            "f1": sampled_f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "false_positive_rate": sampled_fpr,
            "adjusted_precision": adjusted_precision,
        },
    )

    config_dir = report_dir.parent / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "model_config.json").write_text(
        json.dumps(
            {
                "experiment": "legacy_lanl_logistic_diagnostic",
                "random_seed": 42,
                "split": "stratified random 75/25; diagnostic only",
                "sampled_negative_to_positive_ratio": (
                    int(data[LANL_TARGET_COLUMN].eq("normal").sum()) / sample_redteam
                ),
                "sampling_note": (
                    "Sampling was performed upstream before splitting; the "
                    "prepared sample is not a representative deployment stream."
                ),
                "features": list(LANL_FEATURE_COLUMNS),
                "model": "LogisticRegression(max_iter=1000, class_weight='balanced')",
                "threshold": 0.5,
                "threshold_status": "legacy default; not validated for operations",
                "calibration": "none",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_dir = report_dir.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": None,
                "status": "not_selected",
                "reason": (
                    "No positive-labeled chronological validation set exists "
                    "after the available red-team label period."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "sample_rows": len(data),
        "sample_redteam": int(data[LANL_TARGET_COLUMN].eq("redteam").sum()),
        "sample_normal": int(data[LANL_TARGET_COLUMN].eq("normal").sum()),
        "original_prevalence": original_prevalence,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "sample_precision": sampled_precision,
        "sample_recall": sampled_recall,
        "adjusted_precision": adjusted_precision,
        "temporal_counts": temporal_counts,
    }


def _precision_at_prevalence(recall: float, false_positive_rate: float, prevalence: float) -> float:
    true_positive_rate_mass = recall * prevalence
    false_positive_mass = false_positive_rate * (1 - prevalence)
    denominator = true_positive_rate_mass + false_positive_mass
    return true_positive_rate_mass / denominator if denominator else 0.0


def _time_split_counts(
    event_time: pd.Series, target: pd.Series, boundaries: tuple[float, float]
) -> dict[str, dict[str, int]]:
    first, second = boundaries
    masks = {
        "train_earliest_60_percent": event_time <= first,
        "validation_next_20_percent": (event_time > first) & (event_time <= second),
        "test_latest_20_percent": event_time > second,
    }
    return {
        split: {
            "rows": int(mask.sum()),
            "redteam": int((mask & target.eq("redteam")).sum()),
            "normal": int((mask & target.eq("normal")).sum()),
        }
        for split, mask in masks.items()
    }


def _write_feature_importance(model: Pipeline, path: Path) -> None:
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    importance["interpretation"] = (
        "Exploratory coefficient from the biased random-split diagnostic; "
        "not a validated or causal importance."
    )
    importance.to_csv(path, index=False)


def _write_reports(
    report_dir: Path,
    *,
    data: pd.DataFrame,
    auth_events_scanned: int,
    redteam_events_matched: int,
    original_prevalence: float,
    min_time: int,
    max_time: int,
    temporal_counts: dict[str, dict[str, int]],
    random_metrics: dict[str, float | int],
) -> None:
    sample_positive = int(data[LANL_TARGET_COLUMN].eq("redteam").sum())
    sample_negative = int(data[LANL_TARGET_COLUMN].eq("normal").sum())
    sample_prevalence = sample_positive / len(data)
    latest_positive = int(
        pd.to_numeric(
            data.loc[data[LANL_TARGET_COLUMN].eq("redteam"), "time"]
        ).max()
    )
    duplicate_rows = int(data.duplicated().sum())
    temporal_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {counts['rows']:,} | {counts['redteam']:,} | "
        f"{counts['normal']:,} |"
        for name, counts in temporal_counts.items()
    )

    audit = f"""# LANL ML Pipeline Audit

## Scope and data

The prepared file contains {len(data):,} rows: {sample_positive:,} red-team and
{sample_negative:,} normal. Its sampled red-team prevalence is
{sample_prevalence:.4%}. The preprocessing run scanned {auth_events_scanned:,}
auth events and matched {redteam_events_matched:,} red-team events, for an
observed benchmark prevalence of {original_prevalence:.8%}. Normal events in
the prepared file were selected by a global reservoir, giving a prepared
negative-to-positive ratio of {sample_negative / sample_positive:.2f}:1. That
is a strong case-control sample, not a representative
alert stream.

The saved CSV spans timestamps {min_time:,} through {max_time:,}; the latest
matched positive is at {latest_positive:,}. There are {duplicate_rows:,}
byte-for-byte duplicate prepared rows. Because the preparer discarded source
user and computer IDs, those rows cannot be safely deduplicated or attributed
to distinct entities from this CSV alone.

## How the old metrics were produced

`train_lanl_model` used a stratified, random 75/25 event-row split
(`random_state=42`). A one-hot encoder and scaler were fitted inside a sklearn
pipeline on the training fold only (good); logistic regression used
`class_weight='balanced'`. The holdout report used the classifier's default
0.5 threshold. ROC-AUC and Average Precision were computed on the random
holdout, where prevalence remained {sample_prevalence:.4%}, not at the observed benchmark
prevalence. There was no threshold selection, calibration, hyperparameter
search, temporal split, group split, or independent test set.

Reproduction on this prepared sample:

| Metric | Random sampled holdout |
|---|---:|
| ROC-AUC | {random_metrics['roc_auc']:.6f} |
| Average Precision | {random_metrics['average_precision']:.6f} |
| Precision at legacy 0.5 | {random_metrics['precision']:.6f} |
| Recall at legacy 0.5 | {random_metrics['recall']:.6f} |
| F1 at legacy 0.5 | {random_metrics['f1']:.6f} |
| TP / FP / FN / TN | {random_metrics['true_positives']} / {random_metrics['false_positives']} / {random_metrics['false_negatives']} / {random_metrics['true_negatives']} |
| Sampled-holdout FPR | {random_metrics['false_positive_rate']:.6%} |

Applying that sampled FPR and recall to the observed prevalence gives an
illustrative prior-adjusted precision of
{random_metrics['adjusted_precision']:.8%}. This is an extrapolation from only
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
   {temporal_lines}

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

The historical logistic-regression baseline and a separate chronological
threshold experiment are reported. Random forest, gradient boosting, and
anomaly-detection comparisons are intentionally not presented as model
selection evidence. The chronological experiment selects a cutoff on one
labeled interval and evaluates it on a later interval, but both are within
the limited ground-truth period and the negative class was sampled globally.
It does not establish prospective performance, natural-prevalence precision,
or calibration. Bootstrap intervals and a production operating point are not
claimed. The CSV cannot establish performance after the last labeled
red-team event.

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
"""
    (report_dir / "ml_pipeline_audit.md").write_text(audit, encoding="utf-8")

    evaluation = f"""# LANL Model Evaluation Status

## Status: no validated final model

The prior model's random sampled-holdout results are reproducible but are not
a temporal or production-distribution evaluation. They are retained below as
the **before** diagnostic; no "after" performance improvement is claimed
because the available data cannot support a positive-labeled future test.

## Dataset

- Auth events scanned: {auth_events_scanned:,}
- Matched red-team events: {redteam_events_matched:,}
- Observed benchmark prevalence: {original_prevalence:.8%}
- Prepared sample: {sample_positive:,} red-team, {sample_negative:,} sampled
  normal ({sample_prevalence:.4%} positive)
- Sampling: all matched positives and a global reservoir of up to 20 normal
  events per positive
- Event-time range in prepared data: {min_time:,}–{max_time:,}
- Latest known red-team timestamp: {latest_positive:,}

## Legacy diagnostic ("before")

- Model: class-weighted logistic regression with one-hot categoricals and
  standardized numeric features.
- Split: stratified random 75/25, seed 42.
- Threshold: default 0.5, not selected on validation.
- ROC-AUC: {random_metrics['roc_auc']:.6f}
- Average Precision: {random_metrics['average_precision']:.6f}
- Sampled precision / recall / F1: {random_metrics['precision']:.6f} /
  {random_metrics['recall']:.6f} / {random_metrics['f1']:.6f}
- Confusion matrix (TN, FP, FN, TP): {random_metrics['true_negatives']},
  {random_metrics['false_positives']}, {random_metrics['false_negatives']},
  {random_metrics['true_positives']}
- Sampled false-positive rate: {random_metrics['false_positive_rate']:.6%}
- Illustrative precision at observed prevalence using that FPR:
  {random_metrics['adjusted_precision']:.8%}; extrapolated, not measured.

## Before versus after

| Evaluation | ROC-AUC | Average Precision | Precision / Recall | Status |
|---|---:|---:|---:|---|
| Before: random, downsampled holdout | {random_metrics['roc_auc']:.6f} | {random_metrics['average_precision']:.6f} | {random_metrics['precision']:.6f} / {random_metrics['recall']:.6f} | Reproduced; not trustworthy for future activity |
| After: temporal, natural-prevalence test | Not computed | Not computed | Not computed | No positive labels in future periods |

## Temporal, calibration, and operating-point assessment

A chronological 60/20/20 timestamp split has the following label coverage:

| Period | Rows | Red-team | Normal |
|---|---:|---:|---:|
{temporal_lines}

The latest available red-team event is at timestamp {latest_positive:,}.
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
python -m soc_ml.cli audit-lanl --data data\\lanl_auth_sample.csv --auth-events-scanned {auth_events_scanned} --redteam-events-matched {redteam_events_matched} --report-dir reports
```

The audit is deterministic (seed 42). For inference with the baseline model,
score a raw auth file to obtain an uncalibrated ranking score:

```powershell
python -m soc_ml.cli score-lanl-auth --auth data\\new_auth.txt.gz --model models\\lanl_redteam.joblib --output data\\new_auth_scores.csv.gz
```

The score is uncalibrated and no decision threshold is implicit. To use the
separately validation-selected historical threshold, first create the
`lanl_redteam_temporal.joblib` and `lanl_temporal_threshold.json` artifacts
with `tune-lanl-threshold`, then pass that JSON with `--threshold-file`. This
threshold is not production-validated.
"""
    (report_dir / "final_model_evaluation.md").write_text(
        evaluation, encoding="utf-8"
    )
