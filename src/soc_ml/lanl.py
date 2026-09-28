"""Streaming preparation for the LANL Cyber1 authentication and red-team data."""

import csv
import random
from pathlib import Path

import pandas as pd

AUTH_COLUMNS = (
    "time",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "authentication_type",
    "logon_type",
    "orientation",
    "success",
)
REDTEAM_COLUMNS = ("time", "user", "source_computer", "destination_computer")
LANL_FEATURE_COLUMNS = (
    "authentication_type",
    "logon_type",
    "orientation",
    "success",
    "hour_of_day",
    "machine_account",
    "same_computer",
)
LANL_TARGET_COLUMN = "redteam"
OUTPUT_COLUMNS = ("time", *LANL_FEATURE_COLUMNS, LANL_TARGET_COLUMN)


def prepare_lanl_auth(
    auth_path: Path,
    redteam_path: Path,
    output_path: Path,
    *,
    chunksize: int = 100_000,
    negative_ratio: int = 20,
    max_negatives: int = 100_000,
    seed: int = 42,
) -> dict[str, int]:
    """Stream auth events, label red-team matches, and sample normal events.

    Entity identifiers are used only to match ground truth and are excluded
    from model features. A uniform reservoir limits memory used for negatives.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1")
    if max_negatives < 1:
        raise ValueError("max_negatives must be at least 1")

    redteam = _load_redteam_keys(redteam_path)
    rng = random.Random(seed)
    positives: list[tuple[object, ...]] = []
    negative_reservoir: list[tuple[object, ...]] = []
    negative_count = 0

    for chunk in pd.read_csv(
        auth_path,
        header=None,
        names=AUTH_COLUMNS,
        dtype=str,
        chunksize=chunksize,
    ):
        if chunk.isnull().any().any():
            raise ValueError("auth.txt contains missing fields")
        for row in chunk.itertuples(index=False, name=None):
            time_value, source_user, _, source_computer, destination_computer, auth_type, logon_type, orientation, success = row
            key = (time_value, source_user, source_computer, destination_computer)
            is_redteam = key in redteam
            record = make_lanl_record(
                time_value,
                source_user,
                source_computer,
                destination_computer,
                auth_type,
                logon_type,
                orientation,
                success,
                is_redteam,
            )
            if is_redteam:
                positives.append(record)
            else:
                negative_count += 1
                if len(negative_reservoir) < max_negatives:
                    negative_reservoir.append(record)
                else:
                    replacement = rng.randrange(negative_count)
                    if replacement < max_negatives:
                        negative_reservoir[replacement] = record

    if not positives:
        raise ValueError(
            "No auth events matched redteam.txt; check that both files are from "
            "the same LANL Cyber1 release and use the expected columns"
        )

    selected_negative_count = min(
        negative_count, len(positives) * negative_ratio, max_negatives
    )
    records = positives + negative_reservoir[:selected_negative_count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(records)

    return {
        "auth_events_scanned": len(positives) + negative_count,
        "redteam_events_matched": len(positives),
        "normal_events_seen": negative_count,
        "normal_events_sampled": selected_negative_count,
        "rows_written": len(records),
    }


def _load_redteam_keys(path: Path) -> set[tuple[str, str, str, str]]:
    data = pd.read_csv(path, header=None, names=REDTEAM_COLUMNS, dtype=str)
    if data.empty:
        raise ValueError("redteam.txt must contain at least one labeled event")
    if data.isnull().any().any():
        raise ValueError("redteam.txt contains missing values")
    return set(data.itertuples(index=False, name=None))


def make_lanl_record(
    time_value: str,
    source_user: str,
    source_computer: str,
    destination_computer: str,
    auth_type: str,
    logon_type: str,
    orientation: str,
    success: str,
    is_redteam: bool,
) -> tuple[object, ...]:
    try:
        hour_of_day = (int(time_value) // 3600) % 24
    except ValueError as error:
        raise ValueError(f"Invalid LANL event timestamp: {time_value!r}") from error
    return (
        time_value,
        auth_type,
        logon_type,
        orientation,
        success,
        hour_of_day,
        source_user.endswith("$"),
        source_computer == destination_computer,
        "redteam" if is_redteam else "normal",
    )
