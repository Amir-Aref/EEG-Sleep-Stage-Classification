import logging
import pandas as pd
from pathlib import Path
from typing import List
from database_connection import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "preprocessed_features.csv"

VALID_STAGES = {"Wake", "N1", "N2", "N3", "REM"}
POWER_COLUMNS = ["signal_energy", "delta_power", "theta_power", "alpha_power", "beta_power"]

def load_data_from_db() -> pd.DataFrame:
    try:
        with get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM eeg_epochs", conn)
        logger.info(f"Loaded dataframe from database. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        raise

def validate_sleep_stage(df: pd.DataFrame) -> pd.DataFrame:
    if "sleep_stage" not in df.columns:
        raise ValueError("Column 'sleep_stage' not found in dataset")

    initial_len = len(df)
    df = df[df["sleep_stage"].isin(VALID_STAGES)]
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid sleep_stage labels")

    return df

def remove_physical_anomalies(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    initial_len = len(df)
    
    for col in cols:
        if col in df.columns:
            df = df[df[col] >= 0]
            
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with negative physical values (e.g., power/energy)")
        
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)

    df = df.drop_duplicates()
    df = df.dropna()
    df = remove_physical_anomalies(df, POWER_COLUMNS)

    final_len = len(df)
    logger.info(f"Rows before cleaning: {initial_len} | Rows after cleaning: {final_len}")

    return df

def save_output(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved preprocessed dataset to: {path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed data: {e}")
        raise

def main() -> None:
    logger.info("Starting preprocessing pipeline")
    
    df = load_data_from_db()
    df = validate_sleep_stage(df)
    df = clean_dataframe(df)
    save_output(df, OUTPUT_PATH)
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()