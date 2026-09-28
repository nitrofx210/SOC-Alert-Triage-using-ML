"""Training and scoring routines for the alert triage classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from soc_ml.data import FEATURE_COLUMNS, TARGET_COLUMN

CATEGORICAL_COLUMNS = (
    "alert_type",
    "severity",
    "asset_criticality",
    "source_reputation",
    "mitre_tactic",
)
NUMERIC_COLUMNS = (
    "failed_logins",
    "unusual_hour",
    "geo_anomaly",
    "known_ioc",
)


def _validate_features(data: pd.DataFrame) -> None:
    missing = sorted(set(FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required alert columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Alert data must contain at least one row")
    if data.loc[:, FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Alert features must not contain missing values")


def train_model(
    data_path: Path, model_path: Path
) -> tuple[Pipeline, str, float]:
    """Fit and persist a classifier; return the fitted model, report, and ROC AUC."""
    if data_path.resolve() == model_path.resolve():
        raise ValueError("Model output path must differ from the training data path")
    data = pd.read_csv(data_path)
    _validate_features(data)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training data must contain target column: {TARGET_COLUMN}")
    if data[TARGET_COLUMN].isnull().any():
        raise ValueError("Training target must not contain missing values")
    if set(data[TARGET_COLUMN].unique()) != {"actionable", "benign"}:
        raise ValueError(
            "Training target must contain exactly 'actionable' and 'benign' classes"
        )
    class_counts = data[TARGET_COLUMN].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each training class must contain at least two alerts")
    if len(data) < 5:
        raise ValueError(
            "Training data must contain at least five alerts for a stratified split"
        )

    features = data.loc[:, FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
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
    actionable_index = list(model.classes_).index("actionable")
    probabilities = model.predict_proba(x_test)[:, actionable_index]
    report = classification_report(y_test, predicted, zero_division=0)
    auc = roc_auc_score((y_test == "actionable").astype(int), probabilities)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, report, auc


def score_alerts(data_path: Path, model_path: Path) -> pd.DataFrame:
    """Score alert rows without using or requiring their disposition label."""
    data = pd.read_csv(data_path)
    _validate_features(data)
    model: Pipeline = joblib.load(model_path)
    actionable_index = list(model.classes_).index("actionable")
    probabilities = model.predict_proba(data.loc[:, FEATURE_COLUMNS])[:, actionable_index]

    scored = data.copy()
    scored["actionable_probability"] = probabilities
    scored["triage_recommendation"] = [
        _recommendation(probability) for probability in probabilities
    ]
    return scored


def _recommendation(probability: float) -> str:
    if probability >= 0.80:
        return "Escalate for analyst review"
    if probability >= 0.50:
        return "Investigate"
    return "Lower priority; retain for analyst review"
