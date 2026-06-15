from pathlib import Path
import pandas as pd
from database_connection import get_connection

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

def main():
    df = pd.read_csv(INPUT_CSV)

    with get_connection() as conn:
        conn.execute(CREATE_SUBJECTS_TABLE)
        conn.execute(CREATE_EPOCHS_TABLE)

        subject_ids = sorted(df["subject_id"].unique())

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

    print("Database import completed successfully.")
    print(f"Rows imported: {len(df)}")

if __name__ == "__main__":
    main()