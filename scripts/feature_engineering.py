import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EPSILON = 1e-8
REQUIRED_POWER_COLUMNS = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
]
SLEEP_STAGE_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def main() -> None:
    """Generate engineered EEG features and a model-ready dataset."""
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "outputs" / "preprocessed_intermediate.csv"
    features_output_path = project_root / "outputs" / "preprocessed_features.csv"
    model_ready_output_path = project_root / "outputs" / "model_ready_dataset.csv"

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.error("Run scripts/preprocess.py before feature engineering.")
        return

    logger.info("Loading intermediate dataset from: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded dataframe with shape: %s", df.shape)

    missing_power_columns = [
        column for column in REQUIRED_POWER_COLUMNS if column not in df.columns
    ]
    if missing_power_columns:
        logger.error(
            "Missing required power columns: %s",
            missing_power_columns,
        )
        return

    logger.info("Calculating engineered features...")

    df["total_power"] = df[REQUIRED_POWER_COLUMNS].sum(axis=1)
    df["relative_delta_power"] = df["delta_power"] / (df["total_power"] + EPSILON)
    df["relative_theta_power"] = df["theta_power"] / (df["total_power"] + EPSILON)
    df["relative_alpha_power"] = df["alpha_power"] / (df["total_power"] + EPSILON)
    df["relative_beta_power"] = df["beta_power"] / (df["total_power"] + EPSILON)
    df["delta_theta_ratio"] = df["delta_power"] / (df["theta_power"] + EPSILON)
    df["alpha_beta_ratio"] = df["alpha_power"] / (df["beta_power"] + EPSILON)

    if "signal_energy" in df.columns:
        df["log_signal_energy"] = np.log(df["signal_energy"] + EPSILON)
    else:
        logger.warning(
            "Column 'signal_energy' not found; skipping log_signal_energy."
        )

    if "sleep_stage" not in df.columns:
        logger.error("Column 'sleep_stage' is missing; cannot encode labels.")
        return

    logger.info("Encoding sleep_stage labels...")
    df["sleep_stage_encoded"] = df["sleep_stage"].map(SLEEP_STAGE_MAPPING)

    if df["sleep_stage_encoded"].isna().any():
        logger.error("Found unmapped sleep_stage labels after preprocessing.")
        return

    df["sleep_stage_encoded"] = df["sleep_stage_encoded"].astype(int)

    features_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving feature-level dataset to: %s", features_output_path)
    df.to_csv(features_output_path, index=False)

    model_drop_columns = [
        "sleep_stage",
        "sleep_stage_raw",
        "eeg_channel",
        "subject_id",
        "epoch_id",
        "start_time_sec",
    ]
    model_ready_df = df.drop(columns=model_drop_columns, errors="ignore")

    logger.info("Saving model-ready dataset to: %s", model_ready_output_path)
    model_ready_df.to_csv(model_ready_output_path, index=False)

    logger.info(
        "Feature engineering completed successfully. Model-ready shape: %s",
        model_ready_df.shape,
    )


if __name__ == "__main__":
    main()
