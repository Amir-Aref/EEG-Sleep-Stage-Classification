"""Generate reproducible EDA reports for the EEG model-input dataset.

This script performs descriptive analysis only. It does not scale,
split, remove outliers, or modify the model-input dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import (
        EDA_OUTLIER_SUMMARY_PATH,
        EDA_OUTPUT_DIR,
        EDA_STAGE_SUMMARY_PATH,
        EDA_SUBJECT_SUMMARY_PATH,
        EDA_SUMMARY_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
    )
except ImportError:
    from config import (
        EDA_OUTLIER_SUMMARY_PATH,
        EDA_OUTPUT_DIR,
        EDA_STAGE_SUMMARY_PATH,
        EDA_SUBJECT_SUMMARY_PATH,
        EDA_SUMMARY_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGE_ORDER: Final[tuple[str, ...]] = (
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
)

RELATIVE_POWER_COLUMNS: Final[tuple[str, ...]] = (
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_sigma_power",
)

BOXPLOT_FEATURES: Final[tuple[str, ...]] = (
    "relative_delta_power",
    "relative_sigma_power",
    "spectral_entropy",
    "zero_crossing_rate",
)

SUBJECT_SHIFT_FEATURES: Final[tuple[str, ...]] = (
    "std_uv",
    "spectral_centroid_hz",
)

HYPNOGRAM_MAPPING: Final[dict[str, int]] = {
    "N3": 0,
    "N2": 1,
    "N1": 2,
    "REM": 3,
    "Wake": 4,
}


def load_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate the model-input table and its schema."""

    if not MODEL_INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Model-input dataset does not exist. "
            "Run build_model_input.py first."
        )

    if not MODEL_FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "Model feature schema does not exist."
        )

    dataframe = pd.read_csv(
        MODEL_INPUT_DATASET_PATH
    )

    schema = json.loads(
        MODEL_FEATURE_SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    feature_columns = list(
        schema["selected_features"]
    )

    missing_columns = sorted(
        set(
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
                "sleep_stage",
                "sleep_stage_encoded",
                *feature_columns,
            ]
        ).difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"EDA input is missing columns: {missing_columns}"
        )

    numeric_values = dataframe[
        feature_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "EDA input contains non-finite feature values."
        )

    actual_stages = set(
        dataframe["sleep_stage"].unique()
    )

    if actual_stages != set(STAGE_ORDER):
        raise ValueError(
            f"Unexpected sleep stages: {actual_stages}"
        )

    return dataframe, schema


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> Path:
    """Save and close one Matplotlib figure."""

    EDA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = EDA_OUTPUT_DIR / filename

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    logger.info("Saved figure: %s", output_path)

    return output_path


def build_class_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build class count and percentage table."""

    counts = (
        dataframe["sleep_stage"]
        .value_counts()
        .reindex(STAGE_ORDER)
        .fillna(0)
        .astype(int)
    )

    distribution = pd.DataFrame(
        {
            "sleep_stage": list(STAGE_ORDER),
            "count": counts.to_numpy(),
        }
    )

    distribution["percentage"] = (
        distribution["count"]
        / distribution["count"].sum()
        * 100.0
    )

    return distribution


def build_stage_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build robust per-stage summaries for all model features."""

    rows: list[dict[str, object]] = []

    for stage in STAGE_ORDER:
        stage_data = dataframe[
            dataframe["sleep_stage"] == stage
        ]

        for feature in feature_columns:
            series = stage_data[feature]

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))

            rows.append(
                {
                    "sleep_stage": stage,
                    "feature": feature,
                    "count": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "median": float(series.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                }
            )

    return pd.DataFrame(rows)


def build_subject_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build median feature values for each subject/recording."""

    rows: list[dict[str, object]] = []

    grouped = dataframe.groupby(
        ["subject_id", "recording_id", "night"],
        sort=True,
    )

    for (
        subject_id,
        recording_id,
        night,
    ), group in grouped:
        record: dict[str, object] = {
            "subject_id": int(subject_id),
            "recording_id": str(recording_id),
            "night": int(night),
            "epoch_count": int(len(group)),
        }

        for feature in feature_columns:
            record[f"median_{feature}"] = float(
                group[feature].median()
            )

        rows.append(record)

    return pd.DataFrame(rows)


def build_outlier_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Flag global IQR outliers without removing any rows."""

    records: list[dict[str, object]] = []

    for feature in feature_columns:
        series = dataframe[feature]

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        records.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(
                    outlier_mask.sum()
                ),
                "outlier_percentage": float(
                    outlier_mask.mean() * 100.0
                ),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "outlier_percentage",
                "feature",
            ],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def plot_class_distribution(
    distribution: pd.DataFrame,
) -> Path:
    """Plot class counts with percentages."""

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        distribution["sleep_stage"],
        distribution["count"],
    )

    axis.set_title(
        "Sleep-stage class distribution"
    )
    axis.set_xlabel("Sleep stage")
    axis.set_ylabel("Epoch count")

    for bar, percentage in zip(
        bars,
        distribution["percentage"],
        strict=True,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    return save_figure(
        figure,
        "class_distribution.png",
    )


def plot_relative_band_power(
    dataframe: pd.DataFrame,
) -> Path:
    """Plot median relative powers by sleep stage."""

    plot_source = dataframe[
        [
            "sleep_stage",
            *RELATIVE_POWER_COLUMNS,
        ]
    ].copy()

    plot_source["relative_beta_power_derived"] = (
        1.0
        - plot_source[
            list(RELATIVE_POWER_COLUMNS)
        ].sum(axis=1)
    )

    beta_values = plot_source[
        "relative_beta_power_derived"
    ].to_numpy(dtype=float)

    if (
        (beta_values < -1e-10).any()
        or (beta_values > 1.0 + 1e-10).any()
    ):
        raise ValueError(
            "Derived relative Beta power is outside [0, 1]."
        )

    plot_source[
        "relative_beta_power_derived"
    ] = plot_source[
        "relative_beta_power_derived"
    ].clip(lower=0.0, upper=1.0)

    plot_columns = [
        *RELATIVE_POWER_COLUMNS,
        "relative_beta_power_derived",
    ]

    plot_data = (
        plot_source.groupby(
            "sleep_stage"
        )[plot_columns]
        .median()
        .reindex(STAGE_ORDER)
    )

    plot_data.columns = [
        "Delta",
        "Theta",
        "Alpha",
        "Sigma",
        "Beta (derived per epoch)",
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    plot_data.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Median relative EEG band power by sleep stage"
    )
    axis.set_xlabel("Sleep stage")
    axis.set_ylabel("Median relative power")
    axis.set_ylim(bottom=0)
    axis.tick_params(
        axis="x",
        rotation=0,
    )
    axis.legend(
        title="Frequency band"
    )

    return save_figure(
        figure,
        "relative_band_power_by_stage.png",
    )


def plot_stage_boxplot(
    dataframe: pd.DataFrame,
    feature: str,
) -> Path:
    """Plot one robust stage-wise feature boxplot."""

    values = [
        dataframe.loc[
            dataframe["sleep_stage"] == stage,
            feature,
        ].to_numpy(dtype=float)
        for stage in STAGE_ORDER
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.boxplot(
        values,
        tick_labels=STAGE_ORDER,
        showfliers=False,
    )

    axis.set_title(
        f"{feature} by sleep stage"
    )
    axis.set_xlabel("Sleep stage")
    axis.set_ylabel(feature)

    return save_figure(
        figure,
        f"boxplot_{feature}.png",
    )


def plot_subject_shift(
    subject_summary: pd.DataFrame,
    feature: str,
) -> Path:
    """Plot one subject-level median feature."""

    column = f"median_{feature}"

    labels = (
        "S"
        + subject_summary["subject_id"].astype(str)
        + "\n"
        + subject_summary["recording_id"].astype(str)
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        labels,
        subject_summary[column],
    )

    axis.set_title(
        f"Subject shift: median {feature}"
    )
    axis.set_xlabel("Subject / recording")
    axis.set_ylabel(f"Median {feature}")

    return save_figure(
        figure,
        f"subject_shift_{feature}.png",
    )


def plot_hypnogram(
    dataframe: pd.DataFrame,
    recording_id: str,
) -> Path:
    """Plot the stage sequence of one recording."""

    recording = (
        dataframe[
            dataframe["recording_id"] == recording_id
        ]
        .sort_values(
            "epoch_id",
            kind="stable",
        )
        .copy()
    )

    stage_values = recording[
        "sleep_stage"
    ].map(HYPNOGRAM_MAPPING)

    if stage_values.isna().any():
        raise ValueError(
            f"Unknown stage in {recording_id}."
        )

    elapsed_hours = (
        np.arange(len(recording))
        * 30.0
        / 3600.0
    )

    figure, axis = plt.subplots(
        figsize=(12, 4)
    )

    axis.step(
        elapsed_hours,
        stage_values,
        where="post",
    )

    axis.set_title(
        f"Sleep-stage sequence: {recording_id}"
    )
    axis.set_xlabel(
        "Hours from retained analysis window"
    )
    axis.set_ylabel("Sleep stage")
    axis.set_yticks(
        list(HYPNOGRAM_MAPPING.values()),
        labels=list(HYPNOGRAM_MAPPING.keys()),
    )
    axis.set_ylim(-0.25, 4.25)
    axis.grid(
        axis="x",
        alpha=0.25,
    )

    return save_figure(
        figure,
        f"hypnogram_{recording_id}.png",
    )


def main() -> None:
    """Run descriptive EEG analysis."""

    dataframe, schema = load_inputs()

    feature_columns = list(
        schema["selected_features"]
    )

    class_distribution = (
        build_class_distribution(dataframe)
    )

    stage_summary = build_stage_summary(
        dataframe,
        feature_columns,
    )

    subject_summary = build_subject_summary(
        dataframe,
        feature_columns,
    )

    outlier_summary = build_outlier_summary(
        dataframe,
        feature_columns,
    )

    EDA_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_summary.to_csv(
        EDA_STAGE_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    subject_summary.to_csv(
        EDA_SUBJECT_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    outlier_summary.to_csv(
        EDA_OUTLIER_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    generated_figures: list[Path] = []

    generated_figures.append(
        plot_class_distribution(
            class_distribution
        )
    )

    generated_figures.append(
        plot_relative_band_power(dataframe)
    )

    for feature in BOXPLOT_FEATURES:
        generated_figures.append(
            plot_stage_boxplot(
                dataframe,
                feature,
            )
        )

    for feature in SUBJECT_SHIFT_FEATURES:
        generated_figures.append(
            plot_subject_shift(
                subject_summary,
                feature,
            )
        )

    for recording_id in sorted(
        dataframe["recording_id"].unique()
    ):
        generated_figures.append(
            plot_hypnogram(
                dataframe,
                str(recording_id),
            )
        )

    summary = {
        "row_count": int(len(dataframe)),
        "subject_count": int(
            dataframe["subject_id"].nunique()
        ),
        "recording_count": int(
            dataframe["recording_id"].nunique()
        ),
        "feature_count": len(feature_columns),
        "class_distribution": (
            class_distribution.to_dict(
                orient="records"
            )
        ),
        "outlier_policy": {
            "method": "global Tukey IQR rule",
            "multiplier": 1.5,
            "rows_removed": 0,
            "purpose": (
                "Descriptive flagging only; no samples "
                "are removed during EDA."
            ),
        },
        "highest_outlier_features": (
            outlier_summary.head(10)[
                [
                    "feature",
                    "outlier_count",
                    "outlier_percentage",
                ]
            ].to_dict(orient="records")
        ),
        "generated_figures": [
            str(
                path.relative_to(
                    EDA_OUTPUT_DIR.parent.parent
                )
            ).replace("\\", "/")
            for path in generated_figures
        ],
        "dataset_scope": (
            "Four-subject local pipeline validation dataset. "
            "Final statistical conclusions require the larger "
            "Kaggle execution dataset."
        ),
    }

    EDA_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    print("\n=== EEG EDA SUMMARY ===")
    print("Rows:", len(dataframe))
    print(
        "Subjects:",
        dataframe["subject_id"].nunique(),
    )
    print(
        "Recordings:",
        dataframe["recording_id"].nunique(),
    )
    print("Features:", len(feature_columns))
    print(
        "Figures generated:",
        len(generated_figures),
    )

    print("\n=== CLASS DISTRIBUTION ===")
    print(
        class_distribution.to_string(
            index=False,
            formatters={
                "percentage": lambda value: (
                    f"{value:.2f}%"
                )
            },
        )
    )

    print("\n=== TOP OUTLIER FEATURES ===")
    print(
        outlier_summary[
            [
                "feature",
                "outlier_count",
                "outlier_percentage",
                "lower_bound",
                "upper_bound",
            ]
        ].head(15).to_string(index=False)
    )

    print("\n=== SUBJECT SHIFT ===")
    print(
        subject_summary[
            [
                "subject_id",
                "recording_id",
                "epoch_count",
                "median_std_uv",
                "median_spectral_centroid_hz",
                "median_relative_delta_power",
                "median_relative_sigma_power",
            ]
        ].to_string(index=False)
    )

    print("\nOutlier rows removed: 0")
    print("Scaling applied: False")
    print("EEG EDA validation: PASS")
    print("Figures directory:", EDA_OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "EEG EDA failed."
        )
        sys.exit(1)
