"""Build a reproducible manifest for Sleep-EDF Sleep Cassette files."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

try:
    from .config import (
        DATA_METADATA_DIR,
        EXPECTED_SLEEP_CASSETTE_RECORDINGS,
        SLEEP_EDFX_CHECKSUMS_PATH,
        SLEEP_EDFX_CHECKSUMS_URL,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_SLEEP_CASSETTE_URL,
        SLEEP_EDFX_VERSION,
        ensure_runtime_directories,
    )
except ImportError:
    from config import (
        DATA_METADATA_DIR,
        EXPECTED_SLEEP_CASSETTE_RECORDINGS,
        SLEEP_EDFX_CHECKSUMS_PATH,
        SLEEP_EDFX_CHECKSUMS_URL,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_SLEEP_CASSETTE_URL,
        SLEEP_EDFX_VERSION,
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

PSG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^SC\d{4}[A-Z]\d-PSG\.edf$"
)

HYPNOGRAM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^SC\d{4}[A-Z]{2}-Hypnogram\.edf$"
)


class LinkCollector(HTMLParser):
    """Collect href values from an HTML directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return

        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.add(value)


def download_text(url: str) -> str:
    """Download UTF-8 text from an official source URL."""

    request = Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    with urlopen(request, timeout=60) as response:
        content = response.read()

    return content.decode("utf-8", errors="strict")


def extract_filenames(index_html: str) -> set[str]:
    """Extract EDF filenames from the PhysioNet directory page."""

    parser = LinkCollector()
    parser.feed(index_html)

    filenames: set[str] = set()

    for href in parser.links:
        path = unquote(urlsplit(href).path)
        filename = Path(path).name

        if filename:
            filenames.add(filename)

    return filenames


def parse_checksums(text: str) -> dict[str, str]:
    """Parse PhysioNet SHA256SUMS into path-to-digest mapping."""

    checksums: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        digest, relative_path = parts
        relative_path = relative_path.lstrip("*").replace("\\", "/")

        if re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            checksums[relative_path] = digest.lower()

    return checksums


def create_manifest() -> pd.DataFrame:
    """Create and validate the Sleep Cassette PSG/Hypnogram manifest."""

    logger.info(
        "Reading official Sleep Cassette directory: %s",
        SLEEP_EDFX_SLEEP_CASSETTE_URL,
    )

    index_html = download_text(
        f"{SLEEP_EDFX_SLEEP_CASSETTE_URL}/"
    )

    checksum_text = download_text(
        SLEEP_EDFX_CHECKSUMS_URL
    )

    filenames = extract_filenames(index_html)
    checksums = parse_checksums(checksum_text)

    psg_by_recording = {
        filename[:6]: filename
        for filename in filenames
        if PSG_PATTERN.fullmatch(filename)
    }

    hypnogram_by_recording = {
        filename[:6]: filename
        for filename in filenames
        if HYPNOGRAM_PATTERN.fullmatch(filename)
    }

    psg_keys = set(psg_by_recording)
    hypnogram_keys = set(hypnogram_by_recording)

    missing_hypnograms = sorted(psg_keys - hypnogram_keys)
    missing_psgs = sorted(hypnogram_keys - psg_keys)

    if missing_hypnograms or missing_psgs:
        raise ValueError(
            "Incomplete PSG/Hypnogram pairs. "
            f"Missing hypnograms: {missing_hypnograms}; "
            f"Missing PSGs: {missing_psgs}"
        )

    rows: list[dict[str, object]] = []

    for recording_id in sorted(psg_keys):
        psg_filename = psg_by_recording[recording_id]
        hypnogram_filename = hypnogram_by_recording[
            recording_id
        ]

        subject_code = recording_id[3:5]
        night = int(recording_id[5])

        psg_relative_path = (
            f"sleep-cassette/{psg_filename}"
        )
        hypnogram_relative_path = (
            f"sleep-cassette/{hypnogram_filename}"
        )

        psg_sha256 = checksums.get(psg_relative_path)
        hypnogram_sha256 = checksums.get(
            hypnogram_relative_path
        )

        if not psg_sha256 or not hypnogram_sha256:
            raise ValueError(
                "Missing official SHA-256 checksum for "
                f"{recording_id}"
            )

        rows.append(
            {
                "dataset_name": "Sleep-EDF Expanded",
                "dataset_version": SLEEP_EDFX_VERSION,
                "subset": "sleep-cassette",
                "subject_id": int(subject_code),
                "subject_code": subject_code,
                "night": night,
                "recording_id": recording_id,
                "psg_filename": psg_filename,
                "hypnogram_filename": hypnogram_filename,
                "psg_url": urljoin(
                    f"{SLEEP_EDFX_SLEEP_CASSETTE_URL}/",
                    psg_filename,
                ),
                "hypnogram_url": urljoin(
                    f"{SLEEP_EDFX_SLEEP_CASSETTE_URL}/",
                    hypnogram_filename,
                ),
                "psg_sha256": psg_sha256,
                "hypnogram_sha256": hypnogram_sha256,
            }
        )

    manifest = pd.DataFrame(rows).sort_values(
        ["subject_id", "night"],
        kind="stable",
    ).reset_index(drop=True)

    if len(manifest) != EXPECTED_SLEEP_CASSETTE_RECORDINGS:
        raise ValueError(
            "Unexpected Sleep Cassette recording count: "
            f"expected {EXPECTED_SLEEP_CASSETTE_RECORDINGS}, "
            f"found {len(manifest)}"
        )

    if manifest["recording_id"].duplicated().any():
        raise ValueError(
            "Duplicate recording IDs found in manifest."
        )

    return manifest


def save_manifest(manifest: pd.DataFrame) -> None:
    """Save manifest, checksum source and deterministic summary."""

    DATA_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    checksum_text = download_text(
        SLEEP_EDFX_CHECKSUMS_URL
    )

    SLEEP_EDFX_CHECKSUMS_PATH.write_text(
        checksum_text,
        encoding="utf-8",
        newline="\n",
    )

    manifest.to_csv(
        SLEEP_EDFX_MANIFEST_PATH,
        index=False,
        lineterminator="\n",
    )

    manifest_bytes = SLEEP_EDFX_MANIFEST_PATH.read_bytes()

    summary = {
        "dataset_name": "Sleep-EDF Expanded",
        "dataset_version": SLEEP_EDFX_VERSION,
        "subset": "sleep-cassette",
        "recording_count": int(len(manifest)),
        "subject_count": int(
            manifest["subject_id"].nunique()
        ),
        "night_counts": {
            str(key): int(value)
            for key, value in (
                manifest["night"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "source_directory": (
            SLEEP_EDFX_SLEEP_CASSETTE_URL
        ),
        "checksum_source": SLEEP_EDFX_CHECKSUMS_URL,
    }

    summary_path = SLEEP_EDFX_MANIFEST_PATH.with_suffix(
        ".summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    logger.info(
        "Manifest saved to: %s",
        SLEEP_EDFX_MANIFEST_PATH,
    )
    logger.info(
        "Summary saved to: %s",
        summary_path,
    )
    logger.info(
        "Complete recording pairs: %d",
        len(manifest),
    )
    logger.info(
        "Unique subjects: %d",
        manifest["subject_id"].nunique(),
    )


def main() -> None:
    """Build the versioned Sleep-EDF source manifest."""

    ensure_runtime_directories()

    manifest = create_manifest()
    save_manifest(manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "Failed to build Sleep-EDF manifest."
        )
        raise
