import logging
import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

def run_script(script_path: str) -> None:
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        logger.error(f"Script not found: {full_path}")
        raise FileNotFoundError(f"Missing script: {full_path}")

    logger.info(f"Running: {script_path}")
    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Execution failed for {script_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        sys.exit(result.returncode)
        
    logger.info(f"Successfully completed: {script_path}")

def main() -> None:
    logger.info("Starting Full EEG Sleep Data Pipeline...")

    db_path = PROJECT_ROOT / "database" / "sleep_eeg.db"
    if not db_path.exists():
        logger.info("Database not found. Running import_to_db.py...")
        run_script("scripts/import_to_db.py")
    else:
        logger.info("Database exists. Running import_to_db.py to ensure fresh data state...")
        run_script("scripts/import_to_db.py")

    pipeline_steps: List[str] = [
        "scripts/load_data.py",
        "scripts/preprocess.py",
        "scripts/feature_engineering.py"
    ]

    for step in pipeline_steps:
        run_script(step)

    final_output = PROJECT_ROOT / "outputs" / "model_ready_dataset.csv"
    if final_output.exists():
        try:
            df = pd.read_csv(final_output)
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Final model-ready dataset saved at: {final_output}")
            logger.info(f"Total processed rows ready for Phase 3: {len(df)}")
        except Exception as e:
            logger.error(f"Pipeline finished but failed to read final output for row counting: {e}")
            sys.exit(1)
    else:
        logger.error(f"Pipeline finished but final output is missing at: {final_output}")
        sys.exit(1)

if __name__ == "__main__":
    main()