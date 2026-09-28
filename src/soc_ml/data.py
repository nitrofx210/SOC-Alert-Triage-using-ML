"""Synthetic SOC alert data for local experiments and demos."""

import random

import pandas as pd

ALERT_TYPES = (
    "login_anomaly",
    "malware_detection",
    "phishing_report",
    "network_connection",
    "privilege_change",
)
MITRE_TACTICS = (
    "initial_access",
    "execution",
    "persistence",
    "credential_access",
    "discovery",
    "command_and_control",
)
FEATURE_COLUMNS = (
    "alert_type",
    "severity",
    "asset_criticality",
    "source_reputation",
    "failed_logins",
    "unusual_hour",
    "geo_anomaly",
    "known_ioc",
    "mitre_tactic",
)
TARGET_COLUMN = "disposition"


def generate_synthetic_alerts(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible, illustrative alerts with a noisy synthetic target."""
    if rows < 20:
        raise ValueError("rows must be at least 20 to support a train/test split")

    rng = random.Random(seed)
    records = []
    for _ in range(rows):
        threat_signal = rng.random() < 0.35
        severity = rng.choices(
            ("low", "medium", "high"),
            weights=(0.25, 0.45, 0.30) if threat_signal else (0.50, 0.40, 0.10),
        )[0]
        source_reputation = rng.choices(
            ("trusted", "unknown", "suspicious"),
            weights=(0.08, 0.32, 0.60) if threat_signal else (0.58, 0.37, 0.05),
        )[0]
        known_ioc = rng.random() < (0.40 if threat_signal else 0.015)
        geo_anomaly = rng.random() < (0.55 if threat_signal else 0.12)
        unusual_hour = rng.random() < (0.52 if threat_signal else 0.20)
        failed_logins = (
            min(30, int(rng.expovariate(1 / 8)) + 1)
            if threat_signal
            else min(30, int(rng.expovariate(1 / 2)))
        )
        asset_criticality = rng.choices(
            ("low", "medium", "high"), weights=(0.25, 0.40, 0.35)
        )[0]
        alert_type = rng.choice(ALERT_TYPES)
        mitre_tactic = rng.choice(MITRE_TACTICS)

        risk = (
            (0.28 if severity == "high" else 0.12 if severity == "medium" else 0)
            + (0.27 if source_reputation == "suspicious" else 0)
            + (0.45 if known_ioc else 0)
            + (0.14 if geo_anomaly else 0)
            + (0.10 if unusual_hour else 0)
            + min(failed_logins, 10) * 0.025
            + (0.08 if asset_criticality == "high" else 0)
            + (0.12 if alert_type == "malware_detection" else 0)
        )
        actionable = rng.random() < min(0.97, max(0.03, risk))
        records.append(
            {
                "alert_type": alert_type,
                "severity": severity,
                "asset_criticality": asset_criticality,
                "source_reputation": source_reputation,
                "failed_logins": failed_logins,
                "unusual_hour": unusual_hour,
                "geo_anomaly": geo_anomaly,
                "known_ioc": known_ioc,
                "mitre_tactic": mitre_tactic,
                TARGET_COLUMN: "actionable" if actionable else "benign",
            }
        )

    return pd.DataFrame.from_records(records, columns=[*FEATURE_COLUMNS, TARGET_COLUMN])
