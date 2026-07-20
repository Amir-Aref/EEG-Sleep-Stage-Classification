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

DATABASE_DIR: Final[Path] = PROJECT_ROOT / "database"
DATABASE_PATH: Final[Path] = DATABASE_DIR / "sleep_eeg.db"

OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
FIGURES_DIR: Final[Path] = OUTPUTS_DIR / "figures"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
SQL_QUERY_OUTPUTS_DIR: Final[Path] = DOCS_DIR / "sql_query_outputs"


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
