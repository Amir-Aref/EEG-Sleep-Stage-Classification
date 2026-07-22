"""Phase 2 end-to-end EEG data pipeline."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.config import (
    DATABASE_IMPORT_SCRIPT,
    DATABASE_PATH,
    MODEL_READY_DATASET_PATH,
    PHASE2_PIPELINE_STEPS,
    PROJECT_ROOT,
    ensure_runtime_directories,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_script(script_path: str) -> None:
    """Execute one project script using the active Python interpreter."""

    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        logger.error("Script not found: %s", full_path)
        raise FileNotFoundError(f"Missing script: {full_path}")

    logger.info("Running: %s", script_path)

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout.strip():
        logger.info(
            "Output from %s:\n%s",
            script_path,
            result.stdout.strip(),
        )

    if result.returncode != 0:
        logger.error(
            "Execution failed for %s\nSTDOUT:\n%s\nSTDERR:\n%s",
            script_path,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(
            f"Pipeline step failed with exit code "
            f"{result.returncode}: {script_path}"
        )

    logger.info("Successfully completed: %s", script_path)


def main() -> None:
    """Run the complete Phase 2 pipeline."""

    logger.info("Starting Full EEG Sleep Data Pipeline...")

    ensure_runtime_directories()

    if DATABASE_PATH.exists():
        logger.info(
            "Database exists. Running import step to preserve "
            "the current Phase 2 behavior."
        )
    else:
        logger.info(
            "Database not found. Initializing it through the import step."
        )

    run_script(DATABASE_IMPORT_SCRIPT)

    for step in PHASE2_PIPELINE_STEPS:
        run_script(step)

    if not MODEL_READY_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Pipeline finished but final output is missing: "
            f"{MODEL_READY_DATASET_PATH}"
        )

    try:
        dataframe = pd.read_csv(MODEL_READY_DATASET_PATH)
    except Exception:
        logger.exception(
            "Pipeline finished but the final output could not be read."
        )
        raise

    logger.info("Pipeline completed successfully.")
    logger.info(
        "Final model-ready dataset saved at: %s",
        MODEL_READY_DATASET_PATH,
    )
    logger.info(
        "Total processed rows ready for Phase 3: %d",
        len(dataframe),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("EEG data pipeline failed.")
        sys.exit(1)
