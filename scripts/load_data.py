import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from database_connection import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "sql_query_outputs"

def run_query(query: str, output_name: str) -> Optional[pd.DataFrame]:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        output_path = OUTPUT_DIR / output_name
        df.to_csv(output_path, index=False)

        logger.info(f"Saved query result to: {output_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to execute query or save output '{output_name}': {e}")
        raise

def save_total_rows_summary() -> None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with get_connection() as conn:
            total_rows = pd.read_sql_query("SELECT COUNT(*) AS total_rows FROM eeg_epochs;", conn)

        summary_path = OUTPUT_DIR / "05_total_rows.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(total_rows.to_string(index=False))

        logger.info(f"Saved database summary to: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save total rows summary: {e}")
        raise

def main() -> None:
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

    save_total_rows_summary()
    logger.info("All data loading and query executions completed successfully.")

if __name__ == "__main__":
    main()