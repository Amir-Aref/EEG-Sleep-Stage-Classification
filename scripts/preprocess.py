"""Phase 2 compatibility preprocessing for EEG epoch features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Final

import pandas as pd

try:
    from .config import (
        POWER_COLUMNS,
        PREPROCESSED_FEATURES_PATH,
        VALID_SLEEP_STAGES,
    )
    from .database_connection import get_connection
except ImportError:
    from config import (
        POWER_COLUMNS,
        PREPROCESSED_FEATURES_PATH,
        VALID_SLEEP_STAGES,
    )
    from database_connection import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH: Final[Path] = PREPROCESSED_FEATURES_PATH

VALID_STAGE_SET: Final[frozenset[str]] = frozenset(
    VALID_SLEEP_STAGES
)

NON_NEGATIVE_COLUMNS: Final[tuple[str, ...]] = (
    "signal_energy",
    *POWER_COLUMNS,
)


def load_data_from_db() -> pd.DataFrame:
    """Load all EEG epoch records from SQLite."""

    try:
        with get_connection() as connection:
            dataframe = pd.read_sql_query(
                "SELECT * FROM eeg_epochs;",
                connection,
            )

        logger.info(
            "Loaded dataframe from database. Shape: %s",
            dataframe.shape,
        )

        return dataframe

    except Exception:
        logger.exception("Failed to load data from database.")
        raise


def validate_sleep_stage(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows whose sleep-stage labels are not supported."""

    if "sleep_stage" not in dataframe.columns:
        raise ValueError(
            "Column 'sleep_stage' not found in dataset."
        )

    initial_length = len(dataframe)

    validated = dataframe[
        dataframe["sleep_stage"].isin(VALID_STAGE_SET)
    ].copy()

    dropped_rows = initial_length - len(validated)

    if dropped_rows:
        logger.warning(
            "Dropped %d rows with invalid sleep-stage labels.",
            dropped_rows,
        )

    return validated


def remove_physical_anomalies(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Remove rows containing negative energy or power values."""

    initial_length = len(dataframe)
    cleaned = dataframe

    for column in columns:
        if column in cleaned.columns:
            cleaned = cleaned[cleaned[column] >= 0]

    dropped_rows = initial_length - len(cleaned)

    if dropped_rows:
        logger.warning(
            "Dropped %d rows with negative physical values.",
            dropped_rows,
        )

    return cleaned.copy()


def clean_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the original Phase 2 cleaning behavior."""

    initial_length = len(dataframe)

    cleaned = dataframe.drop_duplicates()
    cleaned = cleaned.dropna()
    cleaned = remove_physical_anomalies(
        cleaned,
        NON_NEGATIVE_COLUMNS,
    )

    logger.info(
        "Rows before cleaning: %d | Rows after cleaning: %d",
        initial_length,
        len(cleaned),
    )

    return cleaned


def save_output(
    dataframe: pd.DataFrame,
    path: Path = OUTPUT_PATH,
) -> None:
    """Save the preprocessed Phase 2 dataset."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

        logger.info(
            "Saved preprocessed dataset to: %s",
            path,
        )

    except Exception:
        logger.exception("Failed to save preprocessed data.")
        raise


def main() -> None:
    """Run Phase 2 compatibility preprocessing."""

    logger.info("Starting preprocessing pipeline.")

    dataframe = load_data_from_db()
    dataframe = validate_sleep_stage(dataframe)
    dataframe = clean_dataframe(dataframe)

    save_output(dataframe)

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
