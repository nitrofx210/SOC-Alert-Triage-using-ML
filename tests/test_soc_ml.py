import pandas as pd
import pytest

from soc_ml.data import FEATURE_COLUMNS, generate_synthetic_alerts
from soc_ml.audit import audit_lanl_sample
from soc_ml.lanl import (
    LANL_TARGET_COLUMN,
    OUTPUT_COLUMNS,
    prepare_lanl_auth,
)
from soc_ml.lanl_model import score_lanl_auth, score_lanl_events, train_lanl_model
from soc_ml.model import score_alerts, train_model
from soc_ml.operating_point import fit_lanl_temporal_operating_point


def test_synthetic_data_is_reproducible_and_labeled():
    first = generate_synthetic_alerts(rows=100, seed=7)
    second = generate_synthetic_alerts(rows=100, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["disposition"]) == {"actionable", "benign"}
    assert list(first.columns) == [*FEATURE_COLUMNS, "disposition"]


def test_training_and_scoring_round_trip(tmp_path):
    data_path = tmp_path / "alerts.csv"
    model_path = tmp_path / "models" / "triage.joblib"
    generate_synthetic_alerts(rows=300, seed=8).to_csv(data_path, index=False)

    _, report, auc = train_model(data_path, model_path)
    assert model_path.exists()
    assert "actionable" in report
    assert 0 <= auc <= 1

    input_path = tmp_path / "unlabeled.csv"
    alerts = (
        generate_synthetic_alerts(rows=20, seed=9)
        .drop(columns="disposition")
        .head(10)
    )
    alerts.to_csv(input_path, index=False)
    scored = score_alerts(input_path, model_path)

    assert len(scored) == 10
    assert scored["actionable_probability"].between(0, 1).all()
    assert scored["triage_recommendation"].notna().all()


def test_training_rejects_missing_features(tmp_path):
    data_path = tmp_path / "invalid.csv"
    pd.DataFrame({"disposition": ["benign", "actionable"]}).to_csv(
        data_path, index=False
    )

    with pytest.raises(ValueError, match="Missing required alert columns"):
        train_model(data_path, tmp_path / "model.joblib")


def test_training_rejects_unexpected_target_classes(tmp_path):
    data_path = tmp_path / "invalid_target.csv"
    alerts = generate_synthetic_alerts(rows=20, seed=10)
    alerts["disposition"] = ["urgent" if i % 2 else "benign" for i in range(20)]
    alerts.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="exactly 'actionable' and 'benign'"):
        train_model(data_path, tmp_path / "model.joblib")


def test_training_rejects_classes_too_small_for_stratification(tmp_path):
    data_path = tmp_path / "small_class.csv"
    alerts = generate_synthetic_alerts(rows=20, seed=11)
    alerts["disposition"] = ["actionable"] + ["benign"] * 19
    alerts.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="Each training class must contain"):
        train_model(data_path, tmp_path / "model.joblib")


def test_training_rejects_too_few_rows_for_stratified_split(tmp_path):
    data_path = tmp_path / "too_few_rows.csv"
    alerts = generate_synthetic_alerts(rows=20, seed=12).head(4).copy()
    alerts["disposition"] = ["actionable", "actionable", "benign", "benign"]
    alerts.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="at least five alerts"):
        train_model(data_path, tmp_path / "model.joblib")


def test_generator_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_alerts(rows=10)


def test_lanl_preparation_matches_redteam_and_samples_normals(tmp_path):
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    output_path = tmp_path / "prepared.csv"
    auth_lines = []
    redteam_lines = []
    for i in range(8):
        timestamp = str(1000 + i * 3600)
        user = f"U{i}@DOM1"
        source = f"C{i}"
        destination = f"C{i + 10}"
        auth_lines.append(
            f"{timestamp},{user},SYSTEM@{destination},{source},{destination},Negotiate,Network,LogOn,Success"
        )
        redteam_lines.append(f"{timestamp},{user},{source},{destination}")
    for i in range(80):
        auth_lines.append(
            f"{50000 + i},U99@DOM1,SYSTEM@C99,C99,C99,Negotiate,Service,LogOn,Success"
        )
    auth_path.write_text("\n".join(auth_lines), encoding="utf-8")
    redteam_path.write_text("\n".join(redteam_lines), encoding="utf-8")

    stats = prepare_lanl_auth(
        auth_path,
        redteam_path,
        output_path,
        chunksize=7,
        negative_ratio=2,
        max_negatives=100,
    )
    prepared = pd.read_csv(output_path)

    assert list(prepared.columns) == list(OUTPUT_COLUMNS)
    assert stats["redteam_events_matched"] == 8
    assert stats["normal_events_sampled"] == 16
    assert prepared["redteam"].value_counts().to_dict() == {"normal": 16, "redteam": 8}
    assert "source_user" not in prepared.columns


def test_lanl_train_pipeline_and_no_redteam_match(tmp_path):
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    prepared_path = tmp_path / "prepared.csv"
    model_path = tmp_path / "lanl.joblib"
    auth_lines = []
    redteam_lines = []
    for i in range(16):
        timestamp = str(1000 + i * 3600)
        user = f"U{i}@DOM1"
        source = f"C{i}"
        destination = f"C{i + 10}"
        auth_lines.append(
            f"{timestamp},{user},SYSTEM@{destination},{source},{destination},Negotiate,Network,LogOn,Success"
        )
        if i < 8:
            redteam_lines.append(f"{timestamp},{user},{source},{destination}")
    for i in range(80):
        auth_lines.append(
            f"{50000 + i},U99@DOM1,SYSTEM@C99,C99,C99,Negotiate,Service,LogOn,Success"
        )
    auth_path.write_text("\n".join(auth_lines), encoding="utf-8")
    redteam_path.write_text("\n".join(redteam_lines), encoding="utf-8")
    prepare_lanl_auth(auth_path, redteam_path, prepared_path, negative_ratio=4)

    _, report, roc_auc, average_precision = train_lanl_model(
        prepared_path, model_path
    )

    assert model_path.exists()
    assert "redteam" in report
    assert 0 <= roc_auc <= 1
    assert 0 <= average_precision <= 1

    no_match_auth = tmp_path / "no_match_auth.txt"
    no_match_auth.write_text(
        "999,U99@DOM1,SYSTEM@C99,C99,C99,Negotiate,Service,LogOn,Success",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No auth events matched"):
        prepare_lanl_auth(no_match_auth, redteam_path, tmp_path / "unused.csv")


def test_lanl_training_rejects_unusable_stratified_holdout(tmp_path):
    data_path = tmp_path / "too_small.csv"
    rows = []
    for target in ("normal", "normal", "redteam", "redteam"):
        rows.append(
            {
                "authentication_type": "Kerberos",
                "logon_type": "Network",
                "orientation": "LogOn",
                "success": "Success",
                "hour_of_day": 1,
                "machine_account": False,
                "same_computer": False,
                "redteam": target,
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="too small for a stratified"):
        train_lanl_model(data_path, tmp_path / "model.joblib")


def test_lanl_training_rejects_model_path_overwriting_data(tmp_path):
    data_path = tmp_path / "training.csv"
    rows = []
    for i in range(40):
        redteam = i % 2 == 0
        rows.append(
            {
                "authentication_type": "Kerberos" if redteam else "NTLM",
                "logon_type": "Network" if redteam else "Service",
                "orientation": "LogOn" if redteam else "LogOff",
                "success": "Failure" if redteam else "Success",
                "hour_of_day": i % 24,
                "machine_account": redteam,
                "same_computer": not redteam,
                "redteam": "redteam" if redteam else "normal",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)
    original = data_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="differ from the training data"):
        train_lanl_model(data_path, data_path)

    assert data_path.read_text(encoding="utf-8") == original


def test_lanl_scoring_returns_uncalibrated_ranking_score(tmp_path):
    data_path = tmp_path / "training.csv"
    model_path = tmp_path / "lanl.joblib"
    rows = []
    for i in range(40):
        redteam = i % 2 == 0
        rows.append(
            {
                "authentication_type": "Kerberos" if redteam else "NTLM",
                "logon_type": "Network" if redteam else "Service",
                "orientation": "LogOn" if redteam else "LogOff",
                "success": "Failure" if redteam else "Success",
                "hour_of_day": i % 24,
                "machine_account": redteam,
                "same_computer": not redteam,
                "redteam": "redteam" if redteam else "normal",
            }
        )
    training = pd.DataFrame(rows)
    training.to_csv(data_path, index=False)
    train_lanl_model(data_path, model_path)

    input_path = tmp_path / "unlabeled.csv"
    training.drop(columns=LANL_TARGET_COLUMN).head(6).assign(time=range(6)).to_csv(
        input_path, index=False
    )
    scored = score_lanl_events(input_path, model_path)

    assert len(scored) == 6
    assert scored["redteam_score"].between(0, 1).all()
    assert "redteam" not in scored.columns


def test_raw_lanl_auth_scoring_streams_without_labels_or_entity_ids(tmp_path):
    training_path = tmp_path / "training.csv"
    model_path = tmp_path / "lanl.joblib"
    rows = []
    for i in range(40):
        redteam = i % 2 == 0
        rows.append(
            {
                "authentication_type": "Kerberos" if redteam else "NTLM",
                "logon_type": "Network" if redteam else "Service",
                "orientation": "LogOn" if redteam else "LogOff",
                "success": "Failure" if redteam else "Success",
                "hour_of_day": i % 24,
                "machine_account": redteam,
                "same_computer": not redteam,
                "redteam": "redteam" if redteam else "normal",
            }
        )
    pd.DataFrame(rows).to_csv(training_path, index=False)
    train_lanl_model(training_path, model_path)

    auth_path = tmp_path / "auth.txt"
    auth_path.write_text(
        "100,U1@DOM,SYSTEM@C2,C1,C2,Kerberos,Network,LogOn,Failure\n"
        "200,U2@DOM,SYSTEM@C3,C3,C3,NTLM,Service,LogOff,Success\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "nested" / "scores.csv.gz"
    rows_written = score_lanl_auth(
        auth_path, model_path, output_path, chunksize=1, threshold=0.5
    )
    scored = pd.read_csv(output_path)

    assert rows_written == 2
    assert len(scored) == 2
    assert scored["redteam_score"].between(0, 1).all()
    assert set(scored["predicted_label"]).issubset({"normal", "redteam"})
    assert "source_user" not in scored.columns
    assert "redteam" not in scored.columns

    original_auth = auth_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Output path must differ"):
        score_lanl_auth(auth_path, model_path, auth_path)
    assert auth_path.read_text(encoding="utf-8") == original_auth


def test_temporal_threshold_selection_writes_validation_and_test_results(tmp_path):
    data_path = tmp_path / "prepared.csv"
    rows = []
    for i in range(120):
        redteam = i % 4 == 0
        rows.append(
            {
                "time": i * 100,
                "authentication_type": "Kerberos" if redteam else "NTLM",
                "logon_type": "Network" if redteam else "Service",
                "orientation": "LogOn" if redteam else "LogOff",
                "success": "Failure" if redteam else "Success",
                "hour_of_day": i % 24,
                "machine_account": redteam,
                "same_computer": not redteam,
                "redteam": "redteam" if redteam else "normal",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)
    model_path = tmp_path / "models" / "temporal.joblib"
    threshold_path = tmp_path / "models" / "temporal_threshold.json"
    report_path = tmp_path / "reports" / "temporal.md"

    result = fit_lanl_temporal_operating_point(
        data_path, model_path, threshold_path, report_path
    )

    assert model_path.exists()
    assert threshold_path.exists()
    assert report_path.exists()
    assert result["status"] == "selected_on_validation_only"
    assert 0 <= result["selected_threshold"] <= 1
    assert (
        result["validation"]["metrics_at_selected_threshold"]["redteam_recall"]
        >= 0.90
    )
    assert "Untouched later-period holdout" in report_path.read_text(
        encoding="utf-8"
    )


def test_lanl_audit_writes_reports_and_does_not_select_threshold(tmp_path):
    data_path = tmp_path / "prepared.csv"
    report_dir = tmp_path / "reports"
    rows = []
    for i in range(120):
        redteam = i % 3 == 0
        rows.append(
            {
                "time": i * 100,
                "authentication_type": "Kerberos" if redteam else "NTLM",
                "logon_type": "Network" if redteam else "Service",
                "orientation": "LogOn" if redteam else "LogOff",
                "success": "Failure" if redteam else "Success",
                "hour_of_day": i % 24,
                "machine_account": redteam,
                "same_computer": not redteam,
                "redteam": "redteam" if redteam else "normal",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    result = audit_lanl_sample(
        data_path,
        report_dir,
        auth_events_scanned=1_000_000,
        redteam_events_matched=40,
    )

    assert 0 <= result["roc_auc"] <= 1
    assert (report_dir / "ml_pipeline_audit.md").exists()
    assert (report_dir / "final_model_evaluation.md").exists()
    assert (report_dir / "feature_importance.csv").exists()
    assert (tmp_path / "configs" / "model_config.json").exists()
    threshold = (tmp_path / "models" / "threshold.json").read_text(
        encoding="utf-8"
    )
    assert '"selected_threshold": null' in threshold
