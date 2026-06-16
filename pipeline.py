import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def run_preprocess(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    preprocess_script = scripts_dir / "preprocess.py"

    if not preprocess_script.exists():
        raise FileNotFoundError(f"preprocess.py not found at {preprocess_script}")

    logger.info("Running preprocessing script: %s", preprocess_script)
    result = subprocess.run(
        [sys.executable, str(preprocess_script)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            "Preprocessing failed.\nSTDOUT:\n%s\nSTDERR:\n%s",
            result.stdout,
            result.stderr,
        )
        raise RuntimeError("Preprocessing step failed")

    logger.info("Preprocessing completed successfully.")


def run_feature_engineering(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    feature_script = scripts_dir / "feature_engineering.py"

    if not feature_script.exists():
        raise FileNotFoundError(f"feature_engineering.py not found at {feature_script}")

    logger.info("Running feature engineering script: %s", feature_script)
    result = subprocess.run(
        [sys.executable, str(feature_script)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            "Feature engineering failed.\nSTDOUT:\n%s\nSTDERR:\n%s",
            result.stdout,
            result.stderr,
        )
        raise RuntimeError("Feature engineering step failed")

    logger.info("Feature engineering completed successfully.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    logger.info("Starting EEG Sleep Data pipeline from project root: %s", project_root)

    input_path = project_root / "data" / "sample" / "sleep_edf_sample_features_subject0.csv"
    intermediate_path = project_root / "outputs" / "preprocessed_intermediate.csv"
    model_ready_path = project_root / "outputs" / "model_ready_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")

    run_preprocess(project_root)
    if not intermediate_path.exists():
        raise FileNotFoundError(f"Expected intermediate file not found at {intermediate_path}")

    run_feature_engineering(project_root)
    if not model_ready_path.exists():
        raise FileNotFoundError(f"Expected model-ready file not found at {model_ready_path}")

    logger.info("Pipeline finished successfully. Final output: %s", model_ready_path)


if __name__ == "__main__":
    main()
