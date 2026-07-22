"""Central configuration for the EEG sleep-stage project.

All project paths and shared domain constants must be defined here.
Scripts should import values from this module instead of recreating
paths or label mappings independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Final[Path] = DATA_DIR / "sample"
DATA_METADATA_DIR: Final[Path] = DATA_DIR / "metadata"

DATABASE_DIR: Final[Path] = PROJECT_ROOT / "database"
DATABASE_PATH: Final[Path] = DATABASE_DIR / "sleep_eeg.db"

OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
SQL_QUERY_OUTPUTS_DIR: Final[Path] = DOCS_DIR / "sql_query_outputs"



# ---------------------------------------------------------------------
# External dataset source
# ---------------------------------------------------------------------

SLEEP_EDFX_VERSION: Final[str] = "1.0.0"

SLEEP_EDFX_BASE_URL: Final[str] = (
    "https://physionet.org/files/sleep-edfx/"
    f"{SLEEP_EDFX_VERSION}"
)

SLEEP_EDFX_SLEEP_CASSETTE_URL: Final[str] = (
    f"{SLEEP_EDFX_BASE_URL}/sleep-cassette"
)

SLEEP_EDFX_CHECKSUMS_URL: Final[str] = (
    f"{SLEEP_EDFX_BASE_URL}/SHA256SUMS.txt"
)

SLEEP_EDFX_RAW_DIR: Final[Path] = (
    RAW_DATA_DIR
    / "sleep-edfx"
    / SLEEP_EDFX_VERSION
    / "sleep-cassette"
)

SLEEP_EDFX_MANIFEST_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_sleep_cassette_manifest.csv"
)

SLEEP_EDFX_CHECKSUMS_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_sha256sums.txt"
)

SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_download_inventory.csv"
)

EDF_INSPECTION_REPORT_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_edf_inspection.csv"
)

EPOCH_METADATA_PATH: Final[Path] = (
    INTERIM_DATA_DIR
    / "sleep_edfx_epoch_metadata.csv"
)

EPOCH_SUMMARY_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_epoch_summary.csv"
)

FEATURE_PARTS_DIR: Final[Path] = (
    INTERIM_DATA_DIR
    / "features_by_recording"
)

EPOCH_FEATURES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR
    / "sleep_edfx_epoch_features.csv"
)

FEATURE_SCHEMA_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_feature_schema.json"
)

FEATURE_EXTRACTION_SUMMARY_PATH: Final[Path] = (
    DATA_METADATA_DIR
    / "sleep_edfx_feature_extraction_summary.csv"
)

EEG_FILTER_LOW_HZ: Final[float] = 0.3
EEG_FILTER_HIGH_HZ: Final[float] = 35.0
WELCH_WINDOW_SECONDS: Final[float] = 4.0
WELCH_OVERLAP_FRACTION: Final[float] = 0.5

EEG_FREQUENCY_BANDS: Final[dict[str, tuple[float, float]]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

FLATLINE_STD_THRESHOLD_UV: Final[float] = 0.1
AMPLITUDE_ARTIFACT_THRESHOLD_UV: Final[float] = 500.0

WAKE_TRIM_PADDING_MINUTES: Final[int] = 30

SLEEP_EDF_ANNOTATION_MAPPING: Final[dict[str, str]] = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

IGNORED_SLEEP_EDF_ANNOTATIONS: Final[tuple[str, ...]] = (
    "Sleep stage ?",
    "Movement time",
)

EXPECTED_SLEEP_CASSETTE_RECORDINGS: Final[int] = 153

# ---------------------------------------------------------------------
# Current Phase 2 compatibility paths
# ---------------------------------------------------------------------

LEGACY_PROCESSED_FEATURES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR
    / "sleep_edf_sample_features_subject0.csv"
)

PREPROCESSED_FEATURES_PATH: Final[Path] = (
    OUTPUTS_DIR
    / "preprocessed_features.csv"
)

MODEL_READY_DATASET_PATH: Final[Path] = (
    OUTPUTS_DIR
    / "model_ready_dataset.csv"
)


# ---------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------

DATABASE_IMPORT_SCRIPT: Final[str] = "scripts/import_to_db.py"

PHASE2_PIPELINE_STEPS: Final[tuple[str, ...]] = (
    "scripts/load_data.py",
    "scripts/preprocess.py",
    "scripts/feature_engineering.py",
)


# ---------------------------------------------------------------------
# EEG and sleep-stage domain constants
# ---------------------------------------------------------------------

EPOCH_DURATION_SECONDS: Final[int] = 30
DEFAULT_SAMPLING_FREQUENCY_HZ: Final[float] = 100.0
DEFAULT_EEG_CHANNEL: Final[str] = "EEG Fpz-Cz"

VALID_SLEEP_STAGES: Final[tuple[str, ...]] = (
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
)

SLEEP_STAGE_MAPPING: Final[dict[str, int]] = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}

POWER_COLUMNS: Final[tuple[str, ...]] = (
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
)


# ---------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------

RUNTIME_DIRECTORIES: Final[tuple[Path, ...]] = (
    RAW_DATA_DIR,
    DATA_METADATA_DIR,
    SLEEP_EDFX_RAW_DIR,
    FEATURE_PARTS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SQL_QUERY_OUTPUTS_DIR,
)


def ensure_runtime_directories() -> None:
    """Create all directories required during pipeline execution."""

    for directory in RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
