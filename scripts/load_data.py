from pathlib import Path
import pandas as pd
from database_connection import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "sql_query_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_query(query, output_name):
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)

    output_path = OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(df.head())

    return df

def main():
    run_query(
        """
        SELECT sleep_stage, COUNT(*) AS epoch_count
        FROM eeg_epochs
        GROUP BY sleep_stage
        ORDER BY epoch_count DESC;
        """,
        "01_stage_distribution.csv"
    )

    run_query(
        """
        SELECT sleep_stage,
               AVG(delta_power) AS avg_delta_power,
               AVG(theta_power) AS avg_theta_power,
               AVG(alpha_power) AS avg_alpha_power,
               AVG(beta_power) AS avg_beta_power
        FROM eeg_epochs
        GROUP BY sleep_stage;
        """,
        "02_band_power_by_stage.csv"
    )

    run_query(
        """
        SELECT subject_id, COUNT(*) AS epoch_count
        FROM eeg_epochs
        GROUP BY subject_id;
        """,
        "03_epoch_count_by_subject.csv"
    )

    run_query(
        """
        SELECT *
        FROM eeg_epochs
        LIMIT 10;
        """,
        "04_sample_rows.csv"
    )

    with get_connection() as conn:
        total_rows = pd.read_sql_query(
            "SELECT COUNT(*) AS total_rows FROM eeg_epochs;",
            conn
        )

    summary_path = OUTPUT_DIR / "05_total_rows.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(total_rows.to_string(index=False))

    print(f"Saved: {summary_path}")
    print(total_rows)

if __name__ == "__main__":
    main()