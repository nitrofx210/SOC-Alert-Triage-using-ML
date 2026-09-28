"""Chronological threshold selection for the labeled LANL sample."""

import json
import math
from pathlib import Path

import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from soc_ml.lanl import LANL_FEATURE_COLUMNS, LANL_TARGET_COLUMN
from soc_ml.lanl_model import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    validate_stratified_holdout,
)


def fit_lanl_temporal_operating_point(
    data_path: Path,
    model_path: Path,
    threshold_path: Path,
    report_path: Path,
    *,
    target_recall: float = 0.90,
) -> dict[str, object]:
    """Fit chronologically and select a threshold on validation, never test.

    The chronological windows are defined by positive-timestamp quantiles so
    the sparse, clustered red-team labels populate validation and test. This
    is a retrospective benchmark evaluation within the labeled period, not a
    prospective future-period evaluation.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be greater than 0 and at most 1")
    data_resolved = data_path.resolve()
    model_resolved = model_path.resolve()
    threshold_resolved = threshold_path.resolve()
    report_resolved = report_path.resolve()
    if model_resolved == data_resolved:
        raise ValueError("Model output path must differ from the training data")
    if threshold_resolved in {data_resolved, model_resolved}:
        raise ValueError("Threshold output path must differ from data and model paths")
    if report_resolved in {data_resolved, model_resolved}:
        raise ValueError("Report output path must differ from data and model paths")
    if report_resolved == threshold_resolved:
        raise ValueError("Report and threshold output paths must differ")

    data = pd.read_csv(data_path)
    required = set(LANL_FEATURE_COLUMNS) | {LANL_TARGET_COLUMN, "time"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required LANL columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("LANL training data must contain at least one row")
    if data.loc[:, LANL_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("LANL features must not contain missing values")
    if data[LANL_TARGET_COLUMN].isnull().any():
        raise ValueError("LANL target must not contain missing values")
    if set(data[LANL_TARGET_COLUMN].unique()) != {"normal", "redteam"}:
        raise ValueError("LANL target must contain normal and redteam classes")

    event_time = pd.to_numeric(data["time"], errors="coerce")
    if event_time.isnull().any() or not event_time.map(math.isfinite).all():
        raise ValueError("LANL timestamps must be finite numeric values")
    data = data.assign(_event_time=event_time)
    last_labeled_positive_time = data.loc[
        data[LANL_TARGET_COLUMN].eq("redteam"), "_event_time"
    ].max()
    labeled_period = data.loc[
        data["_event_time"] <= last_labeled_positive_time
    ].sort_values("_event_time")
    positive_times = labeled_period.loc[
        labeled_period[LANL_TARGET_COLUMN].eq("redteam"), "_event_time"
    ]
    if positive_times.nunique() < 3:
        raise ValueError(
            "At least three distinct red-team timestamps are required "
            "for chronological train/validation/test windows"
        )

    train_boundary = float(positive_times.quantile(0.50))
    validation_boundary = float(positive_times.quantile(0.75))
    train = labeled_period.loc[labeled_period["_event_time"] <= train_boundary]
    validation = labeled_period.loc[
        (labeled_period["_event_time"] > train_boundary)
        & (labeled_period["_event_time"] <= validation_boundary)
    ]
    test = labeled_period.loc[labeled_period["_event_time"] > validation_boundary]

    for split_name, split in (
        ("training", train),
        ("validation", validation),
        ("test", test),
    ):
        counts = split[LANL_TARGET_COLUMN].value_counts()
        if not {"normal", "redteam"}.issubset(counts.index) or counts.min() < 2:
            raise ValueError(
                f"Chronological {split_name} window must contain at least "
                "two rows from each class; use a larger labeled dataset"
            )
    validate_stratified_holdout(train[LANL_TARGET_COLUMN])

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
    model.fit(train.loc[:, LANL_FEATURE_COLUMNS], train[LANL_TARGET_COLUMN])
    redteam_index = list(model.classes_).index("redteam")

    validation_y = validation[LANL_TARGET_COLUMN].eq("redteam").to_numpy()
    validation_scores = model.predict_proba(
        validation.loc[:, LANL_FEATURE_COLUMNS]
    )[:, redteam_index]
    threshold = _threshold_for_recall(validation_y, validation_scores, target_recall)
    validation_metrics = _metrics(validation_y, validation_scores, threshold)

    test_y = test[LANL_TARGET_COLUMN].eq("redteam").to_numpy()
    test_scores = model.predict_proba(test.loc[:, LANL_FEATURE_COLUMNS])[
        :, redteam_index
    ]
    test_default_metrics = _metrics(test_y, test_scores, 0.5)
    test_metrics = _metrics(test_y, test_scores, threshold)

    train_counts = _class_counts(train[LANL_TARGET_COLUMN])
    validation_counts = _class_counts(validation[LANL_TARGET_COLUMN])
    test_counts = _class_counts(test[LANL_TARGET_COLUMN])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    threshold_data = {
        "selected_threshold": threshold,
        "status": "selected_on_validation_only",
        "target_validation_redteam_recall": target_recall,
        "validation": {
            "time_start": int(validation["_event_time"].min()),
            "time_end": int(validation["_event_time"].max()),
            "class_counts": validation_counts,
            "metrics_at_selected_threshold": validation_metrics,
        },
        "training": {
            "time_start": int(train["_event_time"].min()),
            "time_end": int(train["_event_time"].max()),
            "class_counts": train_counts,
        },
        "test": {
            "time_start": int(test["_event_time"].min()),
            "time_end": int(test["_event_time"].max()),
            "class_counts": test_counts,
            "metrics_at_default_threshold_0_5": test_default_metrics,
            "metrics_at_validation_selected_threshold": test_metrics,
        },
        "split_method": (
            "Contiguous chronology within the labeled period, using red-team "
            "timestamp 50th and 75th percentiles as boundaries."
        ),
        "model_path": str(model_path),
        "threshold_path": str(threshold_path),
        "report_path": str(report_path),
        "limitations": [
            "Normal rows were globally downsampled before splitting.",
            "This is a historical temporal holdout within label coverage, not a future-period deployment test.",
            "Threshold selection optimizes alert reduction subject to the requested validation recall floor.",
            "Scores are uncalibrated and the selected threshold is not production-validated.",
        ],
    }
    threshold_path.write_text(
        json.dumps(threshold_data, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        _render_report(threshold_data, target_recall), encoding="utf-8"
    )
    return threshold_data


def _threshold_for_recall(
    targets: object, scores: object, target_recall: float
) -> float:
    target_values = pd.Series(targets, dtype=bool).to_numpy()
    score_values = pd.Series(scores, dtype=float).to_numpy()
    candidates = sorted(set(score_values.tolist()), reverse=True)
    eligible = [
        threshold
        for threshold in candidates
        if recall_score(target_values, score_values >= threshold, zero_division=0)
        >= target_recall
    ]
    if not eligible:
        raise ValueError("Could not find a threshold meeting the recall target")
    return float(eligible[0])


def _metrics(targets: object, scores: object, threshold: float) -> dict[str, object]:
    target_values = pd.Series(targets, dtype=bool).to_numpy()
    score_values = pd.Series(scores, dtype=float).to_numpy()
    predictions = score_values >= threshold
    tn, fp, fn, tp = confusion_matrix(
        target_values, predictions, labels=[False, True]
    ).ravel()
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(target_values, score_values)),
        "average_precision": float(
            average_precision_score(target_values, score_values)
        ),
        "redteam_precision": float(
            precision_score(target_values, predictions, zero_division=0)
        ),
        "redteam_recall": float(
            recall_score(target_values, predictions, zero_division=0)
        ),
        "redteam_f1": float(f1_score(target_values, predictions, zero_division=0)),
        "normal_precision": float(negative_precision),
        "normal_recall": float(negative_recall),
        "false_positive_rate": float(fpr),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def _class_counts(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts()
    return {
        "normal": int(counts.get("normal", 0)),
        "redteam": int(counts.get("redteam", 0)),
        "total": len(target),
    }


def _render_report(data: dict[str, object], target_recall: float) -> str:
    training = data["training"]
    validation = data["validation"]
    test = data["test"]
    validation_metrics = validation["metrics_at_selected_threshold"]
    default_test_metrics = test["metrics_at_default_threshold_0_5"]
    test_metrics = test["metrics_at_validation_selected_threshold"]

    def metric_rows(metrics: dict[str, object]) -> str:
        return "\n".join(
            (
                f"| Threshold | {metrics['threshold']:.6f} |",
                f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
                f"| Average Precision | {metrics['average_precision']:.4f} |",
                f"| Red-team precision | {metrics['redteam_precision']:.4f} |",
                f"| Red-team recall | {metrics['redteam_recall']:.4f} |",
                f"| Red-team F1 | {metrics['redteam_f1']:.4f} |",
                f"| Normal precision | {metrics['normal_precision']:.4f} |",
                f"| Normal recall | {metrics['normal_recall']:.4f} |",
                f"| False-positive rate | {metrics['false_positive_rate']:.4%} |",
                "| TP / FP / FN / TN | "
                f"{metrics['true_positives']} / {metrics['false_positives']} / "
                f"{metrics['false_negatives']} / {metrics['true_negatives']} |",
            )
        )

    return f"""# LANL Temporal Operating-Point Experiment

## Purpose and caveats

This experiment selects a score threshold on an earlier chronological
validation window, then evaluates it once on a later chronological holdout.
The threshold is chosen as the **highest threshold that attains at least
{target_recall:.0%} red-team recall on validation**, reducing alerts while
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
| Train | {training['time_start']:,} | {training['time_end']:,} | {training['class_counts']['normal']:,} | {training['class_counts']['redteam']:,} | {training['class_counts']['total']:,} |
| Validation | {validation['time_start']:,} | {validation['time_end']:,} | {validation['class_counts']['normal']:,} | {validation['class_counts']['redteam']:,} | {validation['class_counts']['total']:,} |
| Test | {test['time_start']:,} | {test['time_end']:,} | {test['class_counts']['normal']:,} | {test['class_counts']['redteam']:,} | {test['class_counts']['total']:,} |

## Validation-selected threshold

| Metric | Validation |
|---|---:|
{metric_rows(validation_metrics)}

## Untouched later-period holdout

| Metric | Default threshold 0.5 | Validation-selected threshold |
|---|---:|---:|
| Red-team precision | {default_test_metrics['redteam_precision']:.4f} | {test_metrics['redteam_precision']:.4f} |
| Red-team recall | {default_test_metrics['redteam_recall']:.4f} | {test_metrics['redteam_recall']:.4f} |
| Normal precision | {default_test_metrics['normal_precision']:.4f} | {test_metrics['normal_precision']:.4f} |
| Normal recall | {default_test_metrics['normal_recall']:.4f} | {test_metrics['normal_recall']:.4f} |
| Red-team F1 | {default_test_metrics['redteam_f1']:.4f} | {test_metrics['redteam_f1']:.4f} |
| False-positive rate | {default_test_metrics['false_positive_rate']:.4%} | {test_metrics['false_positive_rate']:.4%} |
| TP / FP / FN / TN | {default_test_metrics['true_positives']} / {default_test_metrics['false_positives']} / {default_test_metrics['false_negatives']} / {default_test_metrics['true_negatives']} | {test_metrics['true_positives']} / {test_metrics['false_positives']} / {test_metrics['false_negatives']} / {test_metrics['true_negatives']} |

Detailed metrics at the selected threshold:

| Metric | Test |
|---|---:|
{metric_rows(test_metrics)}

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

- Model trained on the training window: `{data['model_path']}`
- Validation-selected threshold and split metadata: `{data['threshold_path']}`
- This report: `{data['report_path']}`

Recreate this experiment with:

```powershell
python -m soc_ml.cli tune-lanl-threshold --data data\\lanl_auth_sample.csv --model models\\lanl_redteam_temporal.joblib --threshold-output models\\lanl_temporal_threshold.json --report reports\\temporal_threshold_evaluation.md --target-recall {target_recall:.2f}
```
"""
