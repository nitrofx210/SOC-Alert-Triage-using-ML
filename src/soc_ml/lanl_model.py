"""Model training for the LANL Cyber1 red-team authentication experiment."""

import gzip
import math
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from soc_ml.lanl import (
    AUTH_COLUMNS,
    LANL_FEATURE_COLUMNS,
    LANL_TARGET_COLUMN,
    OUTPUT_COLUMNS,
    make_lanl_record,
)

CATEGORICAL_COLUMNS = (
    "authentication_type",
    "logon_type",
    "orientation",
    "success",
)
NUMERIC_COLUMNS = ("hour_of_day", "machine_account", "same_computer")


def validate_stratified_holdout(target: pd.Series, test_fraction: float = 0.25) -> None:
    """Reject class counts that cannot populate both stratified partitions."""
    class_count = target.nunique()
    if target.value_counts().min() < 2:
        raise ValueError("Each LANL class must contain at least two events")
    test_rows = math.ceil(len(target) * test_fraction)
    train_rows = len(target) - test_rows
    if test_rows < class_count or train_rows < class_count:
        raise ValueError(
            "LANL training data is too small for a stratified train/holdout split"
        )


def _ensure_distinct_model_path(data_path: Path, model_path: Path) -> None:
    if data_path.resolve() == model_path.resolve():
        raise ValueError("Model output path must differ from the training data path")


def train_lanl_model(
    data_path: Path, model_path: Path
) -> tuple[Pipeline, str, float, float]:
    """Fit the legacy random-split LANL diagnostic model.

    Its reported holdout metrics are not a valid temporal or deployment
    evaluation when the input was sampled before splitting.
    """
    _ensure_distinct_model_path(data_path, model_path)
    data = pd.read_csv(data_path)
    missing = sorted((set(LANL_FEATURE_COLUMNS) | {LANL_TARGET_COLUMN}) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required LANL columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("LANL training data must contain at least one row")
    if data.loc[:, LANL_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("LANL features must not contain missing values")
    if data[LANL_TARGET_COLUMN].isnull().any():
        raise ValueError("LANL target must not contain missing values")
    if set(data[LANL_TARGET_COLUMN].unique()) != {"normal", "redteam"}:
        raise ValueError("LANL target must contain both 'normal' and 'redteam' classes")
    validate_stratified_holdout(data[LANL_TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        data.loc[:, LANL_FEATURE_COLUMNS],
        data[LANL_TARGET_COLUMN],
        test_size=0.25,
        random_state=42,
        stratify=data[LANL_TARGET_COLUMN],
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
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

    predicted = model.predict(x_test)
    redteam_index = list(model.classes_).index("redteam")
    redteam_probabilities = model.predict_proba(x_test)[:, redteam_index]
    report = classification_report(y_test, predicted, zero_division=0)
    binary_targets = (y_test == "redteam").astype(int)
    roc_auc = roc_auc_score(binary_targets, redteam_probabilities)
    pr_auc = average_precision_score(binary_targets, redteam_probabilities)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, report, roc_auc, pr_auc


def score_lanl_events(data_path: Path, model_path: Path) -> pd.DataFrame:
    """Return an uncalibrated red-team ranking score for LANL event rows."""
    if data_path.resolve() == model_path.resolve():
        raise ValueError("Scoring input path must differ from the model path")
    data = pd.read_csv(data_path)
    missing = sorted(set(LANL_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required LANL columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("LANL scoring data must contain at least one row")
    if data.loc[:, LANL_FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("LANL scoring features must not contain missing values")

    model: Pipeline = joblib.load(model_path)
    classes = list(model.classes_)
    if set(classes) != {"normal", "redteam"}:
        raise ValueError("Saved LANL model must contain normal and redteam classes")
    redteam_index = classes.index("redteam")
    scores = model.predict_proba(data.loc[:, LANL_FEATURE_COLUMNS])[:, redteam_index]
    scored = data.copy()
    scored["redteam_score"] = scores
    return scored


def score_lanl_auth(
    auth_path: Path,
    model_path: Path,
    output_path: Path,
    *,
    chunksize: int = 100_000,
    threshold: float | None = None,
) -> int:
    """Score raw auth rows in chunks without requiring red-team labels."""
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")
    if output_path.resolve() in {auth_path.resolve(), model_path.resolve()}:
        raise ValueError("Output path must differ from the auth input and model")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a finite number between 0 and 1")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number between 0 and 1")

    model: Pipeline = joblib.load(model_path)
    classes = list(model.classes_)
    if set(classes) != {"normal", "redteam"}:
        raise ValueError("Saved LANL model must contain normal and redteam classes")
    redteam_index = classes.index("redteam")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    output_context = (
        gzip.open(output_path, "wt", newline="", encoding="utf-8")
        if output_path.suffix.lower() == ".gz"
        else output_path.open("w", newline="", encoding="utf-8")
    )
    output_columns = ["time", *LANL_FEATURE_COLUMNS, "redteam_score"]
    if threshold is not None:
        output_columns.append("predicted_label")
    with output_context as output_file:
        pd.DataFrame(columns=output_columns).to_csv(output_file, index=False)
        for chunk in pd.read_csv(
            auth_path,
            header=None,
            names=AUTH_COLUMNS,
            dtype=str,
            chunksize=chunksize,
        ):
            if chunk.isnull().any().any():
                raise ValueError("auth.txt contains missing fields")
            records = []
            for row in chunk.itertuples(index=False, name=None):
                (
                    time_value,
                    source_user,
                    _,
                    source_computer,
                    destination_computer,
                    auth_type,
                    logon_type,
                    orientation,
                    success,
                ) = row
                record = make_lanl_record(
                    time_value,
                    source_user,
                    source_computer,
                    destination_computer,
                    auth_type,
                    logon_type,
                    orientation,
                    success,
                    False,
                )
                records.append(record[:-1])

            features = pd.DataFrame(
                records, columns=OUTPUT_COLUMNS[:-1]
            ).loc[:, LANL_FEATURE_COLUMNS]
            scores = model.predict_proba(features)[:, redteam_index]
            scored = features.copy()
            scored.insert(0, "time", [record[0] for record in records])
            scored["redteam_score"] = scores
            if threshold is not None:
                scored["predicted_label"] = [
                    "redteam" if score >= threshold else "normal"
                    for score in scores
                ]
            scored.to_csv(output_file, index=False, header=False)
            rows_written += len(scored)

    return rows_written
