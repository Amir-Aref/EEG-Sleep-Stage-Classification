import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "preprocessed_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "model_ready_dataset.csv"

EPSILON = 1e-8
REQUIRED_POWER_COLUMNS = ["delta_power", "theta_power", "alpha_power", "beta_power"]
SLEEP_STAGE_MAPPING = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataframe with shape: {df.shape}")
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in REQUIRED_POWER_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    df["total_power"] = df[REQUIRED_POWER_COLUMNS].sum(axis=1)
    df["relative_delta_power"] = df["delta_power"] / (df["total_power"] + EPSILON)
    df["relative_theta_power"] = df["theta_power"] / (df["total_power"] + EPSILON)
    df["relative_alpha_power"] = df["alpha_power"] / (df["total_power"] + EPSILON)
    df["relative_beta_power"] = df["beta_power"] / (df["total_power"] + EPSILON)
    df["delta_theta_ratio"] = df["delta_power"] / (df["theta_power"] + EPSILON)
    df["alpha_beta_ratio"] = df["alpha_power"] / (df["beta_power"] + EPSILON)

    if "signal_energy" in df.columns:
        df["log_signal_energy"] = np.log1p(df["signal_energy"])
    else:
        logger.warning("Column 'signal_energy' not found; skipping log_signal_energy.")

    if "sleep_stage" not in df.columns:
        raise ValueError("Column 'sleep_stage' is missing; cannot encode labels.")

    df["sleep_stage_encoded"] = df["sleep_stage"].map(SLEEP_STAGE_MAPPING)
    
    if df["sleep_stage_encoded"].isna().any():
        raise ValueError("Found unmapped sleep_stage labels after preprocessing.")
        
    df["sleep_stage_encoded"] = df["sleep_stage_encoded"].astype(int)
    
    numeric_features = [
        "mean", "std", "min", "max", "signal_energy", "log_signal_energy",
        "delta_power", "theta_power", "alpha_power", "beta_power", "total_power",
        "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power",
        "delta_theta_ratio", "alpha_beta_ratio"
    ]
    
    cols_to_scale = [col for col in numeric_features if col in df.columns]
    
    if cols_to_scale:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info("Applied StandardScaler to numeric features.")
    
    expected_columns = [
        "subject_id", "epoch_id", "start_time_sec", "eeg_channel",
        "mean", "std", "min", "max", "signal_energy", "log_signal_energy",
        "delta_power", "theta_power", "alpha_power", "beta_power", "total_power",
        "relative_delta_power", "relative_theta_power",
        "relative_alpha_power", "relative_beta_power",
        "delta_theta_ratio", "alpha_beta_ratio",
        "sleep_stage", "sleep_stage_encoded"
    ]
    
    final_columns = [col for col in expected_columns if col in df.columns]
    extra_columns = [col for col in df.columns if col not in expected_columns]
    
    return df[final_columns + extra_columns]

def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved model-ready dataset to: {path}")

def main() -> None:
    logger.info("Starting feature engineering...")
    try:
        df = load_data(INPUT_PATH)
        df_featured = generate_features(df)
        save_data(df_featured, OUTPUT_PATH)
        logger.info(f"Feature engineering completed successfully. Model-ready shape: {df_featured.shape}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()