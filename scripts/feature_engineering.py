"""Phase 2 compatibility feature engineering.

The current formulas and scaling behavior are intentionally preserved
until the scientific feature-engineering rebuild is implemented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        MODEL_READY_DATASET_PATH,
        POWER_COLUMNS,
        PREPROCESSED_FEATURES_PATH,
        SLEEP_STAGE_MAPPING,
    )
except ImportError:
    from config import (
        MODEL_READY_DATASET_PATH,
        POWER_COLUMNS,
        PREPROCESSED_FEATURES_PATH,
        SLEEP_STAGE_MAPPING,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_PATH: Final[Path] = PREPROCESSED_FEATURES_PATH
OUTPUT_PATH: Final[Path] = MODEL_READY_DATASET_PATH

# Preserved temporarily for Phase 2 regression compatibility.
EPSILON: Final[float] = 1e-8

REQUIRED_POWER_COLUMNS: Final[tuple[str, ...]] = POWER_COLUMNS


def load_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the preprocessed epoch-level feature dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    dataframe = pd.read_csv(path)

    logger.info(
        "Loaded dataframe with shape: %s",
        dataframe.shape,
    )

    return dataframe


def generate_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the original Phase 2 engineered features."""

    dataframe = dataframe.copy()

    missing_columns = [
        column
        for column in REQUIRED_POWER_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dataframe["total_power"] = dataframe[
        list(REQUIRED_POWER_COLUMNS)
    ].sum(axis=1)

    dataframe["relative_delta_power"] = (
        dataframe["delta_power"]
        / (dataframe["total_power"] + EPSILON)
    )

    dataframe["relative_theta_power"] = (
        dataframe["theta_power"]
        / (dataframe["total_power"] + EPSILON)
    )

    dataframe["relative_alpha_power"] = (
        dataframe["alpha_power"]
        / (dataframe["total_power"] + EPSILON)
    )

    dataframe["relative_beta_power"] = (
        dataframe["beta_power"]
        / (dataframe["total_power"] + EPSILON)
    )

    dataframe["delta_theta_ratio"] = (
        dataframe["delta_power"]
        / (dataframe["theta_power"] + EPSILON)
    )

    dataframe["alpha_beta_ratio"] = (
        dataframe["alpha_power"]
        / (dataframe["beta_power"] + EPSILON)
    )

    if "signal_energy" in dataframe.columns:
        dataframe["log_signal_energy"] = np.log1p(
            dataframe["signal_energy"]
        )
    else:
        logger.warning(
            "Column 'signal_energy' was not found; "
            "skipping log_signal_energy."
        )

    if "sleep_stage" not in dataframe.columns:
        raise ValueError(
            "Column 'sleep_stage' is missing; "
            "labels cannot be encoded."
        )

    dataframe["sleep_stage_encoded"] = dataframe[
        "sleep_stage"
    ].map(SLEEP_STAGE_MAPPING)

    if dataframe["sleep_stage_encoded"].isna().any():
        raise ValueError(
            "Found unmapped sleep-stage labels after preprocessing."
        )

    dataframe["sleep_stage_encoded"] = dataframe[
        "sleep_stage_encoded"
    ].astype(int)

    numeric_features = [
        "mean",
        "std",
        "min",
        "max",
        "signal_energy",
        "log_signal_energy",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "total_power",
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_beta_power",
        "delta_theta_ratio",
        "alpha_beta_ratio",
    ]

    columns_to_scale = [
        column
        for column in numeric_features
        if column in dataframe.columns
    ]

    if columns_to_scale:
        scaler = StandardScaler()

        dataframe[columns_to_scale] = scaler.fit_transform(
            dataframe[columns_to_scale]
        )

        logger.info(
            "Applied StandardScaler to numeric features."
        )

    expected_columns = [
        "subject_id",
        "epoch_id",
        "start_time_sec",
        "eeg_channel",
        "mean",
        "std",
        "min",
        "max",
        "signal_energy",
        "log_signal_energy",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "total_power",
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_beta_power",
        "delta_theta_ratio",
        "alpha_beta_ratio",
        "sleep_stage",
        "sleep_stage_encoded",
    ]

    final_columns = [
        column
        for column in expected_columns
        if column in dataframe.columns
    ]

    extra_columns = [
        column
        for column in dataframe.columns
        if column not in expected_columns
    ]

    return dataframe[final_columns + extra_columns]


def save_data(
    dataframe: pd.DataFrame,
    path: Path = OUTPUT_PATH,
) -> None:
    """Save the Phase 2 model-ready dataset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)

    logger.info(
        "Saved model-ready dataset to: %s",
        path,
    )


def main() -> None:
    """Run Phase 2 compatibility feature engineering."""

    logger.info("Starting feature engineering.")

    try:
        dataframe = load_data()
        featured_dataframe = generate_features(dataframe)

        save_data(featured_dataframe)

        logger.info(
            "Feature engineering completed successfully. "
            "Model-ready shape: %s",
            featured_dataframe.shape,
        )

    except Exception:
        logger.exception("Feature engineering failed.")
        raise


if __name__ == "__main__":
    main()
