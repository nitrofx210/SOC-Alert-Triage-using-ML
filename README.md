# SOC Alert Triage and LANL Threat Hunting

A small, reproducible Python project for exploring two security-analytics
workflows:

- **Synthetic SOC alert triage:** generate toy labeled alerts, train a
  scikit-learn pipeline, and produce human-review suggestions.
- **LANL Cyber1 authentication experiment:** prepare a sampled authentication
  dataset, reproduce a historical red-team classification baseline, audit its
  evaluation limitations, and score authentication logs in chunks.

> **Research/learning project only.** This repository is not a production
> detection system. Model scores are uncalibrated ranking signals, not
> probabilities of compromise. Nothing here automatically suppresses, closes,
> or responds to alerts.

## Requirements

- Python 3.10 or later
- For the LANL workflow, local copies of `auth.txt.gz` and `redteam.txt.gz`
  from the [LANL Cyber1 dataset](https://csr.lanl.gov/data/cyber1/)
- Enough free disk space for the input, prepared sample, and any scored output

Install in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Commands below use `python -m soc_ml.cli`, which works from the repository
after installation. To run without installing, set `PYTHONPATH=src` first.

## Synthetic alert demo

Generate example data, train the model, and score the rows:

```powershell
python -m soc_ml.cli generate-data --output data\synthetic_alerts.csv --rows 1000
python -m soc_ml.cli train --data data\synthetic_alerts.csv --model models\alert_triage.joblib
python -m soc_ml.cli score --input data\synthetic_alerts.csv --model models\alert_triage.joblib --output data\scored_alerts.csv
```

The score output preserves input columns and adds:

- `actionable_probability` — model output; it is not a calibrated real-world
  risk probability.
- `triage_recommendation` — a fixed illustrative suggestion for analyst
  review.

The synthetic `disposition` labels and resulting evaluation metrics are
artificial and are useful only for demonstrating the code path.

### Synthetic alert fields

| Field | Meaning |
|---|---|
| `alert_type` | Illustrative alert family |
| `severity` | Detector-provided low/medium/high severity |
| `asset_criticality` | Illustrative asset importance |
| `source_reputation` | Trusted, unknown, or suspicious source context |
| `failed_logins` | Associated failed-login count |
| `unusual_hour` | Whether activity occurred at an unusual hour |
| `geo_anomaly` | Whether location differed from the expected pattern |
| `known_ioc` | Whether an indicator matched a known IOC |
| `mitre_tactic` | Illustrative ATT&CK tactic |
| `disposition` | Training target (`actionable` or `benign`); not required for scoring |

## LANL Cyber1 workflow

### Prepare a local sample

Download only the two required files from the LANL dataset page. Keep them
outside the repository and follow the dataset's terms of use. Then prepare a
sample (the defaults are 20 normal events per matched red-team event, capped at
100,000 normal events). Processing the full LANL authentication file can take
hours; it is streamed in chunks rather than loaded into memory all at once.

```powershell
python -m soc_ml.cli prepare-lanl --auth "C:\path\to\auth.txt.gz" --redteam "C:\path\to\redteam.txt.gz" --output data\lanl_auth_sample.csv
```

The command prints the scanned-event and matched-label counts. The output
retains event time and selected, non-identifier features, but removes user and
computer identifiers. It globally reservoir-samples normal rows; it is a
learning sample, not a natural-prevalence dataset.

### Reproduce the historical diagnostic and create the audit

`train-lanl` reproduces the original stratified random-split baseline. It is
provided for reference only—not as a valid future-period model evaluation:

```powershell
python -m soc_ml.cli train-lanl --data data\lanl_auth_sample.csv --model models\lanl_redteam.joblib
```

Use the counts printed by `prepare-lanl` with `audit-lanl`:

```powershell
python -m soc_ml.cli audit-lanl --data data\lanl_auth_sample.csv --auth-events-scanned <AUTH_EVENTS_SCANNED> --redteam-events-matched <REDTEAM_EVENTS_MATCHED> --report-dir reports
```

Replace the angle-bracketed values with the corresponding integer counts
printed by the preparation command; do not include the angle brackets. The
audit reproduces the legacy sampled-holdout metrics and writes:

- `reports\ml_pipeline_audit.md`
- `reports\final_model_evaluation.md`
- `reports\feature_importance.csv` (exploratory coefficients only)
- `configs\model_config.json`
- `models\threshold.json`

The threshold file intentionally records that no operating threshold was
selected when a positive-labeled chronological validation period is
unavailable.

### Score new authentication logs

The raw-log inference command reads compressed or plain LANL auth input in
chunks and writes event time, non-identifier features, and an uncalibrated
`redteam_score`. A `.gz` output filename enables gzip compression:

```powershell
python -m soc_ml.cli score-lanl-auth --auth "C:\path\to\new_auth.txt.gz" --model models\lanl_redteam.joblib --output data\new_auth_scores.csv.gz
```

The score can be used to rank rows for offline analysis only. It is not a
calibrated probability or a recommended automated decision.

## LANL evaluation caveats

The red-team target marks authentication rows matching the provided LANL
ground truth. This is a narrow benchmark experiment, not a general-purpose
threat detector. The old baseline uses a stratified random split after normal
events have already been globally downsampled. As a result:

- Holdout precision is not production alert precision.
- Random splitting mixes earlier and later activity across folds.
- Removed entity IDs prevent unseen-user/host generalization analysis.
- In the provided files, red-team labels end before the authentication log
  ends. Later chronological periods contain no positive labels, so future
  recall, PR-AUC, calibration, and a validation-selected threshold cannot be
  measured.

The included reports explain these limits. No final temporal test score,
calibrated probability, or operational threshold is claimed. A defensible
future-period evaluation requires positive ground-truth labels in later time
periods and rebuilding the data split from raw logs before sampling.

### Historical sample result

For transparency, the current prepared sample produced ROC-AUC **0.986**,
Average Precision **0.641**, and precision/recall **0.559/1.000** at the
baseline's default 0.5 threshold. These are metrics from a **stratified random
holdout of the downsampled sample**, not from a future-period or
natural-prevalence test. They should not be used to estimate production alert
quality. See [the evaluation-status report](reports/final_model_evaluation.md)
for the complete confusion matrix and limitations.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

## Repository contents

```text
configs/       Reproducible experiment configuration
models/        Locally generated model and threshold-status artifacts
reports/       Pipeline audit and evaluation-status reports
src/soc_ml/    CLI, data preparation, models, and audit implementation
tests/         Unit and workflow tests
```

Do not commit downloaded raw datasets or derived data/model artifacts unless
the dataset terms and your organization's data-handling policy allow it.
Before publishing, review all staged files and remove local datasets, scored
logs, and model artifacts that should not be distributed.
