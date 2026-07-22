"""Secure downloader for the Sleep-EDF Sleep Cassette subset.

Features:
- deterministic subject-level selection
- resumable downloads using HTTP Range requests
- SHA-256 verification against PhysioNet checksums
- atomic promotion from .part to final file
- download inventory generation
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    from .config import (
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        PROJECT_ROOT,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_RAW_DIR,
        ensure_runtime_directories,
    )
except ImportError:
    from config import (
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        PROJECT_ROOT,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_RAW_DIR,
        ensure_runtime_directories,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

USER_AGENT: Final[str] = (
    "EEG-Sleep-Stage-Classification/"
    "1.0 academic-project"
)

CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024

INVENTORY_COLUMNS: Final[tuple[str, ...]] = (
    "recording_id",
    "subject_id",
    "night",
    "file_type",
    "filename",
    "local_path",
    "size_bytes",
    "expected_sha256",
    "actual_sha256",
    "status",
    "checked_at_utc",
)


def parse_arguments() -> argparse.Namespace:
    """Parse downloader command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and verify Sleep-EDF Sleep Cassette files."
        )
    )

    parser.add_argument(
        "--subject-count",
        type=int,
        default=4,
        help=(
            "Number of independent subjects to select when explicit "
            "recording IDs are not supplied. Default: 4."
        ),
    )

    parser.add_argument(
        "--nights-per-subject",
        type=int,
        default=1,
        choices=(1, 2),
        help="Maximum nights selected per subject. Default: 1.",
    )

    parser.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help=(
            "Explicit recording ID such as SC4001. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected files without downloading them.",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify selected local files without downloading "
            "missing files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing local and partial files.",
    )

    return parser.parse_args()


def load_manifest() -> pd.DataFrame:
    """Load and minimally validate the official source manifest."""

    if not SLEEP_EDFX_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Manifest not found. Run "
            "`python scripts/build_dataset_manifest.py` first."
        )

    manifest = pd.read_csv(
        SLEEP_EDFX_MANIFEST_PATH,
        dtype={"subject_code": "string"},
    )

    required_columns = {
        "recording_id",
        "subject_id",
        "night",
        "psg_filename",
        "hypnogram_filename",
        "psg_url",
        "hypnogram_url",
        "psg_sha256",
        "hypnogram_sha256",
    }

    missing_columns = sorted(
        required_columns.difference(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {missing_columns}"
        )

    return manifest


def select_recordings(
    manifest: pd.DataFrame,
    recording_ids: list[str],
    subject_count: int,
    nights_per_subject: int,
) -> pd.DataFrame:
    """Select explicit recordings or deterministic independent subjects."""

    if recording_ids:
        normalized_ids = [
            recording_id.strip().upper()
            for recording_id in recording_ids
        ]

        missing_ids = sorted(
            set(normalized_ids)
            - set(manifest["recording_id"])
        )

        if missing_ids:
            raise ValueError(
                f"Recording IDs not found in manifest: {missing_ids}"
            )

        selected = manifest[
            manifest["recording_id"].isin(normalized_ids)
        ].copy()

        order = {
            recording_id: index
            for index, recording_id in enumerate(normalized_ids)
        }

        selected["_selection_order"] = selected[
            "recording_id"
        ].map(order)

        return (
            selected
            .sort_values("_selection_order", kind="stable")
            .drop(columns="_selection_order")
            .reset_index(drop=True)
        )

    if subject_count <= 0:
        raise ValueError("subject-count must be greater than zero.")

    ordered = manifest.sort_values(
        ["subject_id", "night", "recording_id"],
        kind="stable",
    )

    selected_subjects = (
        ordered["subject_id"]
        .drop_duplicates()
        .head(subject_count)
        .tolist()
    )

    selected = (
        ordered[
            ordered["subject_id"].isin(selected_subjects)
        ]
        .groupby(
            "subject_id",
            sort=False,
            group_keys=False,
        )
        .head(nights_per_subject)
        .reset_index(drop=True)
    )

    if selected["subject_id"].nunique() != subject_count:
        raise ValueError(
            "Could not select the requested number of subjects."
        )

    return selected


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a local file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while chunk := file_handle.read(CHUNK_SIZE_BYTES):
            digest.update(chunk)

    return digest.hexdigest()


def local_file_is_valid(
    path: Path,
    expected_sha256: str,
) -> tuple[bool, str | None]:
    """Return whether a local file exists and matches its checksum."""

    if not path.exists():
        return False, None

    actual_sha256 = sha256_file(path)

    return (
        actual_sha256.lower() == expected_sha256.lower(),
        actual_sha256,
    )


def download_with_resume(
    url: str,
    destination: Path,
    expected_sha256: str,
    overwrite: bool,
) -> tuple[str, int, str]:
    """Download one file with resume support and checksum validation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    if overwrite:
        destination.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)

    is_valid, actual_sha256 = local_file_is_valid(
        destination,
        expected_sha256,
    )

    if is_valid and actual_sha256 is not None:
        logger.info(
            "Already downloaded and verified: %s",
            destination.name,
        )

        return (
            actual_sha256,
            destination.stat().st_size,
            "verified_existing",
        )

    if destination.exists():
        logger.warning(
            "Existing final file failed verification; "
            "moving it back to partial state: %s",
            destination.name,
        )

        partial_path.unlink(missing_ok=True)
        destination.replace(partial_path)

    existing_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    }

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    request = Request(url, headers=headers)

    logger.info(
        "%s download: %s",
        "Resuming" if existing_size else "Starting",
        destination.name,
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_status = getattr(response, "status", None)

            if existing_size > 0 and response_status == 206:
                file_mode = "ab"
            else:
                if existing_size > 0:
                    logger.warning(
                        "Server did not honor Range request; "
                        "restarting %s from zero.",
                        destination.name,
                    )

                file_mode = "wb"
                existing_size = 0

            with partial_path.open(file_mode) as file_handle:
                while True:
                    chunk = response.read(CHUNK_SIZE_BYTES)

                    if not chunk:
                        break

                    file_handle.write(chunk)

    except (HTTPError, URLError, TimeoutError):
        logger.exception(
            "Download interrupted for %s. Partial file retained.",
            destination.name,
        )
        raise

    actual_sha256 = sha256_file(partial_path)

    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "SHA-256 verification failed for "
            f"{destination.name}. "
            f"Expected {expected_sha256}, got {actual_sha256}. "
            f"Partial file retained at {partial_path}."
        )

    os.replace(partial_path, destination)

    logger.info(
        "Downloaded and verified: %s",
        destination.name,
    )

    return (
        actual_sha256,
        destination.stat().st_size,
        "downloaded_verified",
    )


def create_file_tasks(
    selected: pd.DataFrame,
) -> list[dict[str, object]]:
    """Expand selected recording rows into PSG and Hypnogram tasks."""

    tasks: list[dict[str, object]] = []

    for row in selected.itertuples(index=False):
        for file_type in ("psg", "hypnogram"):
            filename = getattr(row, f"{file_type}_filename")
            url = getattr(row, f"{file_type}_url")
            expected_sha256 = getattr(
                row,
                f"{file_type}_sha256",
            )

            tasks.append(
                {
                    "recording_id": row.recording_id,
                    "subject_id": int(row.subject_id),
                    "night": int(row.night),
                    "file_type": file_type,
                    "filename": filename,
                    "url": url,
                    "expected_sha256": expected_sha256,
                    "destination": SLEEP_EDFX_RAW_DIR / filename,
                }
            )

    return tasks


def save_inventory(records: list[dict[str, object]]) -> None:
    """Merge current verification records into the download inventory."""

    new_inventory = pd.DataFrame(
        records,
        columns=INVENTORY_COLUMNS,
    )

    if SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH.exists():
        existing_inventory = pd.read_csv(
            SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH
        )

        combined = pd.concat(
            [existing_inventory, new_inventory],
            ignore_index=True,
        )
    else:
        combined = new_inventory

    combined = (
        combined
        .sort_values(
            ["recording_id", "file_type", "checked_at_utc"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["recording_id", "file_type"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        index=False,
        lineterminator="\n",
    )

    logger.info(
        "Download inventory saved to: %s",
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
    )


def main() -> None:
    """Select, download and verify Sleep-EDF recording pairs."""

    arguments = parse_arguments()
    ensure_runtime_directories()

    manifest = load_manifest()

    selected = select_recordings(
        manifest=manifest,
        recording_ids=arguments.recording_id,
        subject_count=arguments.subject_count,
        nights_per_subject=arguments.nights_per_subject,
    )

    tasks = create_file_tasks(selected)

    print("\n=== SELECTED RECORDINGS ===")
    print(
        selected[
            [
                "recording_id",
                "subject_id",
                "night",
                "psg_filename",
                "hypnogram_filename",
            ]
        ].to_string(index=False)
    )

    print(f"\nFiles selected: {len(tasks)}")
    print(f"Destination: {SLEEP_EDFX_RAW_DIR}")

    if arguments.dry_run:
        print("\nDry run complete. No files were downloaded.")
        return

    inventory_records: list[dict[str, object]] = []

    for task in tasks:
        destination = Path(task["destination"])
        expected_sha256 = str(task["expected_sha256"])

        checked_at = datetime.now(timezone.utc).isoformat()

        if arguments.verify_only:
            is_valid, actual_sha256 = local_file_is_valid(
                destination,
                expected_sha256,
            )

            status = (
                "verified_existing"
                if is_valid
                else "missing_or_invalid"
            )

            size_bytes = (
                destination.stat().st_size
                if destination.exists()
                else 0
            )
        else:
            actual_sha256, size_bytes, status = (
                download_with_resume(
                    url=str(task["url"]),
                    destination=destination,
                    expected_sha256=expected_sha256,
                    overwrite=arguments.overwrite,
                )
            )

        inventory_records.append(
            {
                "recording_id": task["recording_id"],
                "subject_id": task["subject_id"],
                "night": task["night"],
                "file_type": task["file_type"],
                "filename": task["filename"],
                "local_path": destination.resolve().relative_to(
                    PROJECT_ROOT.resolve()
                ).as_posix(),
                "size_bytes": int(size_bytes),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256 or "",
                "status": status,
                "checked_at_utc": checked_at,
            }
        )

        save_inventory(inventory_records)

    invalid_statuses = {
        "missing_or_invalid",
    }

    failed_records = [
        record
        for record in inventory_records
        if record["status"] in invalid_statuses
    ]

    if failed_records:
        raise RuntimeError(
            f"{len(failed_records)} selected files are missing "
            "or invalid."
        )

    print("\nDownload/verification completed successfully.")
    print(
        pd.DataFrame(inventory_records)[
            [
                "recording_id",
                "file_type",
                "filename",
                "size_bytes",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Sleep-EDF download process failed.")
        sys.exit(1)
