import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List
from database_connection import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "sleep_edf_sample_features_subject0.csv"

CREATE_SUBJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY,
    source_dataset TEXT NOT NULL,
    recording_id TEXT,
    notes TEXT
);
"""

CREATE_EPOCHS_TABLE = """
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

def setup_database(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_SUBJECTS_TABLE)
    conn.execute(CREATE_EPOCHS_TABLE)
    conn.execute("DELETE FROM eeg_epochs;")
    conn.execute("DELETE FROM subjects;")

def import_data(csv_path: Path) -> None:
    if not csv_path.exists():
        logger.error(f"Input CSV not found at {csv_path}")
        raise FileNotFoundError(f"Missing file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        
        with get_connection() as conn:
            setup_database(conn)
            
            subject_ids: List[int] = sorted(df["subject_id"].unique())
            for sid in subject_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO subjects
                    (subject_id, source_dataset, recording_id, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        int(sid),
                        "Sleep-EDF Database Expanded",
                        f"subject_{sid}",
                        "Imported from processed EEG feature CSV"
                    )
                )
            
            df.to_sql("eeg_epochs", conn, if_exists="append", index=False)
            
        logger.info("Database import completed successfully.")
        logger.info(f"Rows imported: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error during database import: {e}")
        raise

def main() -> None:
    import_data(INPUT_CSV)

if __name__ == "__main__":
    main()