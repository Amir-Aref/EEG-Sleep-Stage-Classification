"""Build validated 30-second epoch metadata from Sleep-EDF files.

This stage does not load complete EEG signals into memory and does not
perform feature scaling. It creates the authoritative epoch index used
by later signal-processing and modeling stages.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import mne
import numpy as np
import pandas as pd

try:
    from .config import (
        EPOCH_DURATION_SECONDS,
        EPOCH_METADATA_PATH,
        EPOCH_SUMMARY_PATH,
        IGNORED_SLEEP_EDF_ANNOTATIONS,
        SLEEP_EDF_ANNOTATION_MAPPING,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_RAW_DIR,
        SLEEP_STAGE_MAPPING,
        WAKE_TRIM_PADDING_MINUTES,
    )
except ImportError:
    from config import (
        EPOCH_DURATION_SECONDS,
        EPOCH_METADATA_PATH,
        EPOCH_SUMMARY_PATH,
        IGNORED_SLEEP_EDF_ANNOTATIONS,
        SLEEP_EDF_ANNOTATION_MAPPING,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_RAW_DIR,
        SLEEP_STAGE_MAPPING,
        WAKE_TRIM_PADDING_MINUTES,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TIME_TOLERANCE_SECONDS: Final[float] = 1e-6

VALID_FILE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "downloaded_verified",
        "verified_existing",
    }
)


def load_inventory() -> pd.DataFrame:
    """Load verified PSG/Hypnogram download records."""

    if not SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            "Download inventory does not exist. "
            "Run download_sleep_edfx.py first."
        )

    inventory = pd.read_csv(
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH
    )

    required_columns = {
        "recording_id",
        "subject_id",
        "night",
        "file_type",
        "filename",
        "status",
    }

    missing_columns = sorted(
        required_columns.difference(inventory.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Inventory is missing columns: {missing_columns}"
        )

    inventory = inventory[
        inventory["status"].isin(VALID_FILE_STATUSES)
    ].copy()

    if inventory.empty:
        raise ValueError(
            "No verified recordings were found in inventory."
        )

    return inventory


def resolve_recording_pairs(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per complete PSG/Hypnogram pair."""

    rows: list[dict[str, object]] = []

    for recording_id, group in inventory.groupby(
        "recording_id",
        sort=True,
    ):
        psg_rows = group[group["file_type"] == "psg"]
        hypnogram_rows = group[
            group["file_type"] == "hypnogram"
        ]

        if len(psg_rows) != 1 or len(hypnogram_rows) != 1:
            raise ValueError(
                f"Incomplete or duplicate pair for {recording_id}."
            )

        psg_row = psg_rows.iloc[0]
        hypnogram_row = hypnogram_rows.iloc[0]

        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": int(psg_row["subject_id"]),
                "night": int(psg_row["night"]),
                "psg_path": (
                    SLEEP_EDFX_RAW_DIR
                    / str(psg_row["filename"])
                ),
                "hypnogram_path": (
                    SLEEP_EDFX_RAW_DIR
                    / str(hypnogram_row["filename"])
                ),
            }
        )

    pairs = pd.DataFrame(rows).sort_values(
        ["subject_id", "night", "recording_id"],
        kind="stable",
    ).reset_index(drop=True)

    for column in ("psg_path", "hypnogram_path"):
        missing_paths = [
            str(path)
            for path in pairs[column]
            if not Path(path).exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"Missing local files in {column}: {missing_paths}"
            )

    return pairs


def datetime_difference_seconds(
    left: object,
    right: object,
) -> float:
    """Return an absolute-time difference or NaN when unavailable."""

    if left is None or right is None:
        return float("nan")

    if not isinstance(left, datetime):
        return float("nan")

    if not isinstance(right, datetime):
        return float("nan")

    return float((left - right).total_seconds())


def validate_epoch_grid_time(value: float) -> None:
    """Ensure an annotation onset lies on the 30-second grid."""

    nearest_grid_value = (
        round(value / EPOCH_DURATION_SECONDS)
        * EPOCH_DURATION_SECONDS
    )

    if not math.isclose(
        value,
        nearest_grid_value,
        rel_tol=0.0,
        abs_tol=TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError(
            "Annotation onset is not aligned to a "
            f"{EPOCH_DURATION_SECONDS}-second grid: {value}"
        )


def expand_annotations(
    recording_id: str,
    subject_id: int,
    night: int,
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
) -> pd.DataFrame:
    """Expand scored annotation segments into complete 30-second epochs."""

    sampling_frequency = float(raw.info["sfreq"])
    raw_duration_seconds = float(
        raw.n_times / sampling_frequency
    )

    records: list[dict[str, object]] = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        strict=True,
    ):
        raw_label = str(description)

        if raw_label in IGNORED_SLEEP_EDF_ANNOTATIONS:
            continue

        mapped_label = SLEEP_EDF_ANNOTATION_MAPPING.get(
            raw_label
        )

        if mapped_label is None:
            logger.warning(
                "Ignoring unsupported annotation %r in %s.",
                raw_label,
                recording_id,
            )
            continue

        annotation_start = max(float(onset), 0.0)
        annotation_end = min(
            float(onset + duration),
            raw_duration_seconds,
        )

        if (
            annotation_end - annotation_start
            < EPOCH_DURATION_SECONDS
            - TIME_TOLERANCE_SECONDS
        ):
            continue

        validate_epoch_grid_time(annotation_start)

        available_duration = (
            annotation_end - annotation_start
        )

        full_epoch_count = int(
            math.floor(
                (
                    available_duration
                    + TIME_TOLERANCE_SECONDS
                )
                / EPOCH_DURATION_SECONDS
            )
        )

        for local_epoch_index in range(full_epoch_count):
            start_time_seconds = (
                annotation_start
                + local_epoch_index
                * EPOCH_DURATION_SECONDS
            )

            end_time_seconds = (
                start_time_seconds
                + EPOCH_DURATION_SECONDS
            )

            if (
                end_time_seconds
                > raw_duration_seconds
                + TIME_TOLERANCE_SECONDS
            ):
                continue

            start_sample = int(
                round(
                    start_time_seconds
                    * sampling_frequency
                )
            )

            stop_sample = int(
                round(
                    end_time_seconds
                    * sampling_frequency
                )
            )

            if stop_sample > raw.n_times:
                continue

            records.append(
                {
                    "subject_id": subject_id,
                    "night": night,
                    "recording_id": recording_id,
                    "start_time_sec": start_time_seconds,
                    "end_time_sec": end_time_seconds,
                    "start_sample": start_sample,
                    "stop_sample": stop_sample,
                    "sampling_frequency_hz": (
                        sampling_frequency
                    ),
                    "epoch_duration_sec": (
                        EPOCH_DURATION_SECONDS
                    ),
                    "sleep_stage_raw": raw_label,
                    "sleep_stage": mapped_label,
                    "sleep_stage_encoded": int(
                        SLEEP_STAGE_MAPPING[mapped_label]
                    ),
                }
            )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError(
            f"No valid scored epochs generated for {recording_id}."
        )

    dataframe = dataframe.sort_values(
        ["start_time_sec", "end_time_sec"],
        kind="stable",
    ).reset_index(drop=True)

    duplicated_starts = dataframe[
        "start_time_sec"
    ].duplicated()

    if duplicated_starts.any():
        duplicate_times = dataframe.loc[
            duplicated_starts,
            "start_time_sec",
        ].tolist()

        raise ValueError(
            "Overlapping annotations produced duplicate epochs "
            f"for {recording_id}: {duplicate_times[:10]}"
        )

    return dataframe


def trim_long_wake_periods(
    epochs: pd.DataFrame,
    raw_duration_seconds: float,
) -> tuple[pd.DataFrame, float, float]:
    """Keep sleep plus limited Wake context at both recording ends."""

    sleep_epochs = epochs[
        epochs["sleep_stage"] != "Wake"
    ]

    if sleep_epochs.empty:
        raise ValueError(
            "No non-Wake epochs are available for trimming."
        )

    padding_seconds = (
        WAKE_TRIM_PADDING_MINUTES * 60
    )

    first_sleep_start = float(
        sleep_epochs["start_time_sec"].min()
    )

    last_sleep_end = float(
        sleep_epochs["end_time_sec"].max()
    )

    trim_start_seconds = max(
        0.0,
        first_sleep_start - padding_seconds,
    )

    trim_end_seconds = min(
        raw_duration_seconds,
        last_sleep_end + padding_seconds,
    )

    trimmed = epochs[
        (epochs["start_time_sec"] >= trim_start_seconds)
        & (epochs["end_time_sec"] <= trim_end_seconds)
    ].copy()

    trimmed = trimmed.reset_index(drop=True)
    trimmed.insert(
        3,
        "epoch_id",
        np.arange(len(trimmed), dtype=np.int64),
    )

    return (
        trimmed,
        trim_start_seconds,
        trim_end_seconds,
    )


def process_recording(
    row: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build epoch metadata and summary for one recording."""

    recording_id = str(row.recording_id)
    psg_path = Path(row.psg_path)
    hypnogram_path = Path(row.hypnogram_path)

    logger.info(
        "Building epoch metadata for %s",
        recording_id,
    )

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False,
        verbose="ERROR",
    )

    annotations = mne.read_annotations(
        hypnogram_path
    )

    sampling_frequency = float(raw.info["sfreq"])
    raw_duration_seconds = float(
        raw.n_times / sampling_frequency
    )

    expanded_epochs = expand_annotations(
        recording_id=recording_id,
        subject_id=int(row.subject_id),
        night=int(row.night),
        raw=raw,
        annotations=annotations,
    )

    (
        trimmed_epochs,
        trim_start_seconds,
        trim_end_seconds,
    ) = trim_long_wake_periods(
        expanded_epochs,
        raw_duration_seconds,
    )

    raw_measurement_date = raw.info.get(
        "meas_date"
    )

    annotation_origin_time = annotations.orig_time

    summary: dict[str, object] = {
        "recording_id": recording_id,
        "subject_id": int(row.subject_id),
        "night": int(row.night),
        "sampling_frequency_hz": sampling_frequency,
        "raw_duration_seconds": raw_duration_seconds,
        "annotation_end_seconds": float(
            (
                annotations.onset
                + annotations.duration
            ).max()
        ),
        "raw_annotation_start_difference_seconds": (
            datetime_difference_seconds(
                annotation_origin_time,
                raw_measurement_date,
            )
        ),
        "epochs_before_wake_trim": int(
            len(expanded_epochs)
        ),
        "epochs_after_wake_trim": int(
            len(trimmed_epochs)
        ),
        "epochs_removed_by_wake_trim": int(
            len(expanded_epochs) - len(trimmed_epochs)
        ),
        "trim_start_seconds": trim_start_seconds,
        "trim_end_seconds": trim_end_seconds,
    }

    class_counts = (
        trimmed_epochs["sleep_stage"]
        .value_counts()
        .to_dict()
    )

    for stage in SLEEP_STAGE_MAPPING:
        summary[f"count_{stage.lower()}"] = int(
            class_counts.get(stage, 0)
        )

    return trimmed_epochs, summary


def validate_combined_epochs(
    epochs: pd.DataFrame,
) -> None:
    """Validate structural invariants of the combined epoch table."""

    if epochs.empty:
        raise ValueError(
            "Combined epoch metadata is empty."
        )

    expected_duration = float(
        EPOCH_DURATION_SECONDS
    )

    actual_duration = (
        epochs["end_time_sec"]
        - epochs["start_time_sec"]
    )

    if not np.allclose(
        actual_duration,
        expected_duration,
        rtol=0.0,
        atol=TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError(
            "Non-30-second epochs detected."
        )

    if not epochs["sleep_stage"].isin(
        SLEEP_STAGE_MAPPING
    ).all():
        raise ValueError(
            "Unexpected normalized sleep-stage labels detected."
        )

    if epochs[
        ["recording_id", "epoch_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate recording/epoch identifiers detected."
        )

    if not (
        epochs["stop_sample"]
        > epochs["start_sample"]
    ).all():
        raise ValueError(
            "Invalid sample boundaries detected."
        )

    grid_position = (
        epochs["start_time_sec"]
        / EPOCH_DURATION_SECONDS
    )

    if not np.allclose(
        grid_position,
        np.round(grid_position),
        rtol=0.0,
        atol=TIME_TOLERANCE_SECONDS,
    ):
        raise ValueError(
            "Epoch starts are not aligned to the 30-second grid."
        )


def main() -> None:
    """Build metadata for all currently verified recordings."""

    inventory = load_inventory()
    pairs = resolve_recording_pairs(inventory)

    all_epochs: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for row in pairs.itertuples(index=False):
        epochs, summary = process_recording(row)
        all_epochs.append(epochs)
        summaries.append(summary)

    combined_epochs = pd.concat(
        all_epochs,
        ignore_index=True,
    )

    combined_epochs = combined_epochs.sort_values(
        ["subject_id", "night", "recording_id", "epoch_id"],
        kind="stable",
    ).reset_index(drop=True)

    summary_dataframe = pd.DataFrame(summaries).sort_values(
        ["subject_id", "night", "recording_id"],
        kind="stable",
    ).reset_index(drop=True)

    validate_combined_epochs(combined_epochs)

    EPOCH_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EPOCH_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_epochs.to_csv(
        EPOCH_METADATA_PATH,
        index=False,
        lineterminator="\n",
    )

    summary_dataframe.to_csv(
        EPOCH_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    print("\n=== EPOCH SUMMARY ===")
    print(
        summary_dataframe.to_string(index=False)
    )

    print("\n=== COMBINED CLASS DISTRIBUTION ===")
    print(
        combined_epochs["sleep_stage"]
        .value_counts()
        .reindex(SLEEP_STAGE_MAPPING.keys())
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nEpoch metadata validation: PASS")
    print("Recordings:", combined_epochs["recording_id"].nunique())
    print("Subjects:", combined_epochs["subject_id"].nunique())
    print("Epochs:", len(combined_epochs))
    print("Metadata saved to:", EPOCH_METADATA_PATH)
    print("Summary saved to:", EPOCH_SUMMARY_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "Epoch metadata generation failed."
        )
        sys.exit(1)
