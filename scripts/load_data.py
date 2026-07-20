"""Execute Phase 2 analytical SQL queries and save their results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .config import SQL_QUERY_OUTPUTS_DIR
    from .database_connection import get_connection
except ImportError:
    from config import SQL_QUERY_OUTPUTS_DIR
    from database_connection import get_connection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = SQL_QUERY_OUTPUTS_DIR


def run_query(
    query: str,
    output_name: str,
) -> Optional[pd.DataFrame]:
    """Execute a SQL query and save its result as CSV."""

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with get_connection() as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
            )

        output_path = OUTPUT_DIR / output_name
        dataframe.to_csv(output_path, index=False)

        logger.info("Saved query result to: %s", output_path)

        return dataframe

    except Exception:
        logger.exception(
            "Failed to execute query or save output %r.",
            output_name,
        )
        raise


def save_total_rows_summary() -> None:
    """Save the number of EEG epochs currently stored in SQLite."""

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with get_connection() as connection:
            total_rows = pd.read_sql_query(
                """
                SELECT COUNT(*) AS total_rows
                FROM eeg_epochs;
                """,
                connection,
            )

        summary_path = OUTPUT_DIR / "05_total_rows.txt"

        summary_path.write_text(
            total_rows.to_string(index=False),
            encoding="utf-8",
        )

        logger.info(
            "Saved database summary to: %s",
            summary_path,
        )

    except Exception:
        logger.exception("Failed to save total-row summary.")
        raise


def main() -> None:
    """Run all Phase 2 database queries."""

    run_query(
        """
        SELECT sleep_stage, COUNT(*) AS epoch_count
        FROM eeg_epochs
        GROUP BY sleep_stage
        ORDER BY epoch_count DESC;
        """,
        "01_stage_distribution.csv",
    )

    run_query(
        """
        SELECT
            sleep_stage,
            AVG(delta_power) AS avg_delta_power,
            AVG(theta_power) AS avg_theta_power,
            AVG(alpha_power) AS avg_alpha_power,
            AVG(beta_power) AS avg_beta_power
        FROM eeg_epochs
        GROUP BY sleep_stage;
        """,
        "02_band_power_by_stage.csv",
    )

    run_query(
        """
        SELECT subject_id, COUNT(*) AS epoch_count
        FROM eeg_epochs
        GROUP BY subject_id;
        """,
        "03_epoch_count_by_subject.csv",
    )

    run_query(
        """
        SELECT *
        FROM eeg_epochs
        LIMIT 10;
        """,
        "04_sample_rows.csv",
    )

    save_total_rows_summary()

    logger.info(
        "All data loading and query executions completed successfully."
    )


if __name__ == "__main__":
    main()
