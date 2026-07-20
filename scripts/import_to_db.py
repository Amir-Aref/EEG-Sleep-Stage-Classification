"""Import the current Phase 2 feature dataset into SQLite."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd

try:
    from .config import LEGACY_PROCESSED_FEATURES_PATH
    from .database_connection import get_connection
except ImportError:
    # Supports direct execution: python scripts/import_to_db.py
    from config import LEGACY_PROCESSED_FEATURES_PATH
    from database_connection import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_CSV: Final[Path] = LEGACY_PROCESSED_FEATURES_PATH

CREATE_SUBJECTS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    recording_id TEXT,
    notes TEXT
);
"""

CREATE_EPOCHS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS eeg_epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    start_time_sec REAL NOT NULL,
    eeg_channel TEXT NOT NULL,
    mean REAL,
    std REAL,
    min REAL,
    max REAL,
    signal_energy REAL,
    delta_power REAL,
    theta_power REAL,
    alpha_power REAL,
    beta_power REAL,
    sleep_stage TEXT NOT NULL,
    sleep_stage_raw TEXT,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    UNIQUE(subject_id, epoch_id, eeg_channel)
);
"""


def setup_database(connection: sqlite3.Connection) -> None:
    """Create the Phase 2 schema and reset its current contents."""

    connection.execute(CREATE_SUBJECTS_TABLE)
    connection.execute(CREATE_EPOCHS_TABLE)

    # Preserved temporarily for Phase 2 regression compatibility.
    connection.execute("DELETE FROM eeg_epochs;")
    connection.execute("DELETE FROM subjects;")


def import_data(csv_path: Path = INPUT_CSV) -> None:
    """Import the feature CSV into the Phase 2 database."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "subject_id",
        "epoch_id",
        "start_time_sec",
        "eeg_channel",
        "sleep_stage",
    }

    missing_columns = sorted(
        required_columns.difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{missing_columns}"
        )

    try:
        with get_connection() as connection:
            setup_database(connection)

            subject_ids = sorted(
                int(subject_id)
                for subject_id in dataframe["subject_id"].unique()
            )

            for subject_id in subject_ids:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO subjects (
                        subject_id,
                        source_dataset,
                        recording_id,
                        notes
                    )
                    VALUES (?, ?, ?, ?);
                    """,
                    (
                        subject_id,
                        "Sleep-EDF Database Expanded",
                        f"subject_{subject_id}",
                        "Imported from processed EEG feature CSV",
                    ),
                )

            dataframe.to_sql(
                "eeg_epochs",
                connection,
                if_exists="append",
                index=False,
            )

        logger.info("Database import completed successfully.")
        logger.info("Rows imported: %d", len(dataframe))

    except Exception:
        logger.exception("Database import failed.")
        raise


def main() -> None:
    """Run the Phase 2 database import."""

    import_data()


if __name__ == "__main__":
    main()
