"""Inspect Sleep-EDF PSG and Hypnogram pairs before preprocessing."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import mne
import pandas as pd

try:
    from .config import (
        DEFAULT_EEG_CHANNEL,
        EDF_INSPECTION_REPORT_PATH,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_RAW_DIR,
    )
except ImportError:
    from config import (
        DEFAULT_EEG_CHANNEL,
        EDF_INSPECTION_REPORT_PATH,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_RAW_DIR,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_RECORDINGS: Final[tuple[str, ...]] = (
    "SC4001",
    "SC4011",
    "SC4021",
    "SC4031",
)

EXPECTED_SLEEP_LABEL_PREFIX: Final[str] = "Sleep stage"


def serialize_datetime(value: object) -> str:
    """Convert MNE datetime metadata to a portable string."""

    if value is None:
        return ""

    isoformat = getattr(value, "isoformat", None)

    if callable(isoformat):
        return str(isoformat())

    return str(value)


def load_verified_inventory() -> pd.DataFrame:
    """Load the verified downloader inventory."""

    if not SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            "Download inventory not found. "
            "Run the verified downloader first."
        )

    inventory = pd.read_csv(
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH
    )

    selected = inventory[
        inventory["recording_id"].isin(
            EXPECTED_RECORDINGS
        )
    ].copy()

    if len(selected) != 8:
        raise ValueError(
            "Expected eight verified files for four recordings, "
            f"but found {len(selected)}."
        )

    if not selected["status"].eq(
        "verified_existing"
    ).all():
        raise ValueError(
            "All selected files must have verified_existing status."
        )

    return selected


def resolve_pair(
    inventory: pd.DataFrame,
    recording_id: str,
) -> tuple[Path, Path]:
    """Resolve one verified PSG/Hypnogram pair."""

    rows = inventory[
        inventory["recording_id"] == recording_id
    ]

    psg_rows = rows[rows["file_type"] == "psg"]
    hypnogram_rows = rows[
        rows["file_type"] == "hypnogram"
    ]

    if len(psg_rows) != 1 or len(hypnogram_rows) != 1:
        raise ValueError(
            f"Invalid file pair for {recording_id}."
        )

    psg_path = SLEEP_EDFX_RAW_DIR / psg_rows.iloc[0][
        "filename"
    ]

    hypnogram_path = (
        SLEEP_EDFX_RAW_DIR
        / hypnogram_rows.iloc[0]["filename"]
    )

    return psg_path, hypnogram_path


def inspect_recording(
    recording_id: str,
    psg_path: Path,
    hypnogram_path: Path,
) -> dict[str, object]:
    """Inspect one PSG/Hypnogram pair using MNE."""

    logger.info("Inspecting recording: %s", recording_id)

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False,
        verbose="ERROR",
    )

    annotations = mne.read_annotations(
        hypnogram_path,
    )

    sampling_frequency = float(raw.info["sfreq"])
    duration_seconds = float(
        raw.n_times / sampling_frequency
    )

    annotation_start = (
        float(annotations.onset.min())
        if len(annotations)
        else 0.0
    )

    annotation_end = (
        float(
            (
                annotations.onset
                + annotations.duration
            ).max()
        )
        if len(annotations)
        else 0.0
    )

    annotation_duration_seconds = float(
        annotations.duration.sum()
    )

    unique_labels = sorted(
        set(str(label) for label in annotations.description)
    )

    sleep_labels = [
        label
        for label in unique_labels
        if label.startswith(EXPECTED_SLEEP_LABEL_PREFIX)
    ]

    channel_names = list(raw.ch_names)

    alignment_difference_seconds = float(
        annotation_end - duration_seconds
    )

    return {
        "recording_id": recording_id,
        "psg_filename": psg_path.name,
        "hypnogram_filename": hypnogram_path.name,
        "sampling_frequency_hz": sampling_frequency,
        "n_channels": int(len(channel_names)),
        "channel_names_json": json.dumps(
            channel_names,
            ensure_ascii=False,
        ),
        "default_channel_present": (
            DEFAULT_EEG_CHANNEL in channel_names
        ),
        "n_samples": int(raw.n_times),
        "psg_duration_seconds": duration_seconds,
        "psg_duration_hours": duration_seconds / 3600.0,
        "measurement_date": serialize_datetime(
            raw.info.get("meas_date")
        ),
        "annotation_count": int(len(annotations)),
        "annotation_start_seconds": annotation_start,
        "annotation_end_seconds": annotation_end,
        "annotation_duration_sum_seconds": (
            annotation_duration_seconds
        ),
        "alignment_difference_seconds": (
            alignment_difference_seconds
        ),
        "unique_annotation_labels_json": json.dumps(
            unique_labels,
            ensure_ascii=False,
        ),
        "sleep_labels_json": json.dumps(
            sleep_labels,
            ensure_ascii=False,
        ),
    }


def validate_report(report: pd.DataFrame) -> None:
    """Validate key structural properties of inspected recordings."""

    if len(report) != len(EXPECTED_RECORDINGS):
        raise ValueError(
            "Unexpected inspection report length."
        )

    if report["recording_id"].nunique() != len(
        EXPECTED_RECORDINGS
    ):
        raise ValueError(
            "Duplicate or missing recording IDs."
        )

    if not report["default_channel_present"].all():
        missing = report.loc[
            ~report["default_channel_present"],
            "recording_id",
        ].tolist()

        raise ValueError(
            f"Default EEG channel missing in: {missing}"
        )

    if not report[
        "sampling_frequency_hz"
    ].gt(0).all():
        raise ValueError(
            "Invalid sampling frequency detected."
        )

    if not report["psg_duration_seconds"].gt(0).all():
        raise ValueError(
            "Invalid PSG duration detected."
        )

    if not report["annotation_count"].gt(0).all():
        raise ValueError(
            "Missing Hypnogram annotations detected."
        )

    if not report["sleep_labels_json"].str.contains(
        "Sleep stage"
    ).all():
        raise ValueError(
            "Sleep-stage annotations were not detected."
        )


def main() -> None:
    """Inspect all verified sample recordings."""

    inventory = load_verified_inventory()

    records: list[dict[str, object]] = []

    for recording_id in EXPECTED_RECORDINGS:
        psg_path, hypnogram_path = resolve_pair(
            inventory,
            recording_id,
        )

        records.append(
            inspect_recording(
                recording_id,
                psg_path,
                hypnogram_path,
            )
        )

    report = pd.DataFrame(records)
    validate_report(report)

    EDF_INSPECTION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        EDF_INSPECTION_REPORT_PATH,
        index=False,
        lineterminator="\n",
    )

    display_columns = [
        "recording_id",
        "sampling_frequency_hz",
        "n_channels",
        "default_channel_present",
        "psg_duration_hours",
        "annotation_count",
        "annotation_start_seconds",
        "annotation_end_seconds",
        "alignment_difference_seconds",
    ]

    print("\n=== EDF INSPECTION REPORT ===")
    print(
        report[display_columns].to_string(
            index=False
        )
    )

    print("\nChannel lists:")

    for row in report.itertuples(index=False):
        print(
            f"{row.recording_id}: "
            f"{row.channel_names_json}"
        )

    print("\nAnnotation labels:")

    for row in report.itertuples(index=False):
        print(
            f"{row.recording_id}: "
            f"{row.unique_annotation_labels_json}"
        )

    print(
        "\nEDF inspection validation: PASS"
    )
    print(
        "Report saved to:",
        EDF_INSPECTION_REPORT_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "Sleep-EDF inspection failed."
        )
        sys.exit(1)
