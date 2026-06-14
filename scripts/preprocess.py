
import os
import logging
import pandas as pd

INPUT_PATH = os.path.join("data", "sample", "sleep_edf_sample_features_subject0.csv")
OUTPUT_PATH = os.path.join("outputs", "preprocessed_intermediate.csv")

VALID_STAGES = {"Wake", "N1", "N2", "N3", "REM"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

def load_data(path):
    logging.info(f"Loading data from {os.path.abspath(path)}")
    df = pd.read_csv(path)
    logging.info(f"Loaded dataframe shape: {df.shape}")
    return df


def validate_sleep_stage(df):
    if "sleep_stage" not in df.columns:
        raise ValueError("Column 'sleep_stage' not found in dataset")

    before = len(df)
    df = df[df["sleep_stage"].isin(VALID_STAGES)]
    dropped = before - len(df)

    if dropped > 0:
        logging.warning(f"Dropped {dropped} rows with invalid sleep_stage labels")

    return df


def clean_dataframe(df):

    before = len(df)

    df = df.drop_duplicates()
    df = df.dropna()

    after = len(df)

    logging.info(f"Rows before cleaning: {before}")
    logging.info(f"Rows after cleaning: {after}")

    return df


def save_output(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.info(f"Saving cleaned dataset to {os.path.abspath(path)}")
    df.to_csv(path, index=False)


def main():

    logging.info("Starting preprocessing pipeline")

    df = load_data(INPUT_PATH)

    df = validate_sleep_stage(df)

    df = clean_dataframe(df)

    save_output(df, OUTPUT_PATH)

    logging.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()
