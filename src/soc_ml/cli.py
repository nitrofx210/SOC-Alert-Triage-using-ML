"""Command-line interface for generating, training, and scoring SOC alerts."""

import argparse
from pathlib import Path

from soc_ml.audit import audit_lanl_sample
from soc_ml.data import generate_synthetic_alerts
from soc_ml.lanl import prepare_lanl_auth
from soc_ml.lanl_model import score_lanl_auth, score_lanl_events, train_lanl_model
from soc_ml.model import score_alerts, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc-triage",
        description="Train and score a human-in-the-loop SOC alert triage model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-data", help="create synthetic labeled alerts")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--rows", type=int, default=1000)
    generate.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="train a model and report test metrics")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)

    score = commands.add_parser("score", help="score an input alert CSV")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare-lanl", help="stream LANL auth events into a labeled training CSV"
    )
    prepare.add_argument("--auth", type=Path, required=True)
    prepare.add_argument("--redteam", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--chunksize", type=int, default=100_000)
    prepare.add_argument("--negative-ratio", type=int, default=20)
    prepare.add_argument("--max-negatives", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=42)

    lanl_train = commands.add_parser(
        "train-lanl",
        help="reproduce the legacy random-split LANL diagnostic model",
    )
    lanl_train.add_argument("--data", type=Path, required=True)
    lanl_train.add_argument("--model", type=Path, required=True)

    audit = commands.add_parser(
        "audit-lanl",
        help="reproduce the sampled LANL diagnostic and write methodology reports",
    )
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--auth-events-scanned", type=int, required=True)
    audit.add_argument("--redteam-events-matched", type=int, required=True)
    audit.add_argument("--report-dir", type=Path, default=Path("reports"))

    lanl_score = commands.add_parser(
        "score-lanl", help="rank LANL authentication events with a saved model"
    )
    lanl_score.add_argument("--input", type=Path, required=True)
    lanl_score.add_argument("--model", type=Path, required=True)
    lanl_score.add_argument("--output", type=Path, required=True)

    lanl_auth_score = commands.add_parser(
        "score-lanl-auth",
        help="stream and score raw LANL auth events without red-team labels",
    )
    lanl_auth_score.add_argument("--auth", type=Path, required=True)
    lanl_auth_score.add_argument("--model", type=Path, required=True)
    lanl_auth_score.add_argument("--output", type=Path, required=True)
    lanl_auth_score.add_argument("--chunksize", type=int, default=100_000)
    return parser


def _ensure_distinct_output(output: Path, *inputs: Path) -> None:
    output_path = output.resolve()
    if any(output_path == input_path.resolve() for input_path in inputs):
        raise ValueError("Output path must differ from the input and model paths")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate-data":
        data = generate_synthetic_alerts(rows=args.rows, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False)
        print(f"Wrote {len(data)} synthetic alerts to {args.output}")
    elif args.command == "train":
        _, report, auc = train_model(args.data, args.model)
        print(f"Saved model to {args.model}")
        print(f"Test ROC AUC: {auc:.3f}")
        print(report)
    elif args.command == "score":
        _ensure_distinct_output(args.output, args.input, args.model)
        scored = score_alerts(args.input, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"Scored {len(scored)} alerts to {args.output}")
    elif args.command == "prepare-lanl":
        stats = prepare_lanl_auth(
            args.auth,
            args.redteam,
            args.output,
            chunksize=args.chunksize,
            negative_ratio=args.negative_ratio,
            max_negatives=args.max_negatives,
            seed=args.seed,
        )
        print(f"Wrote labeled LANL events to {args.output}")
        for name, value in stats.items():
            print(f"{name.replace('_', ' ').title()}: {value:,}")
        print(
            "Normal events are downsampled; precision from this sample does not "
            "estimate production alert precision."
        )
    elif args.command == "train-lanl":
        _, report, roc_auc, pr_auc = train_lanl_model(args.data, args.model)
        print(f"Saved diagnostic model to {args.model}")
        print(f"Holdout ROC AUC: {roc_auc:.3f}")
        print(f"Holdout average precision: {pr_auc:.3f}")
        print(report)
        print(
            "Metrics reflect the prepared, downsampled dataset and are not "
            "production prevalence estimates. The random split is diagnostic "
            "only, not a trustworthy future-period evaluation."
        )
    elif args.command == "audit-lanl":
        results = audit_lanl_sample(
            args.data,
            args.report_dir,
            auth_events_scanned=args.auth_events_scanned,
            redteam_events_matched=args.redteam_events_matched,
        )
        print(f"Wrote audit reports to {args.report_dir}")
        print(f"Random sampled-holdout ROC AUC: {results['roc_auc']:.3f}")
        print(f"Random sampled-holdout average precision: {results['average_precision']:.3f}")
        print(
            "No threshold selected: the available data has no positive-labeled "
            "future validation period."
        )
    elif args.command == "score-lanl":
        _ensure_distinct_output(args.output, args.input, args.model)
        scored = score_lanl_events(args.input, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"Wrote uncalibrated red-team ranking scores for {len(scored)} events to {args.output}")
    elif args.command == "score-lanl-auth":
        rows = score_lanl_auth(
            args.auth, args.model, args.output, chunksize=args.chunksize
        )
        print(
            f"Wrote uncalibrated red-team ranking scores for {rows:,} raw "
            f"auth events to {args.output}"
        )


if __name__ == "__main__":
    main()
