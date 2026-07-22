"""Build a leakage-safe, unscaled EEG model-input table."""

from __future__ import annotations

import json
import logging
import sys
from typing import Final

import numpy as np
import pandas as pd

try:
    from .config import (
        EPOCH_FEATURES_PATH,
        FEATURE_SCHEMA_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
        SLEEP_STAGE_MAPPING,
    )
except ImportError:
    from config import (
        EPOCH_FEATURES_PATH,
        FEATURE_SCHEMA_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
        SLEEP_STAGE_MAPPING,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "subject_id",
    "recording_id",
    "night",
    "epoch_id",
)

TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "sleep_stage",
    "sleep_stage_encoded",
)

QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "quality_issue_flag",
)

SELECTED_FEATURES: Final[tuple[str, ...]] = (
    "mean_uv",
    "std_uv",
    "median_uv",
    "min_uv",
    "max_uv",
    "peak_to_peak_uv",
    "zero_crossing_rate",
    "line_length_uv",
    "skewness",
    "kurtosis_excess",
    "hjorth_mobility",
    "hjorth_complexity",
    "delta_power_uv2",
    "theta_power_uv2",
    "alpha_power_uv2",
    "sigma_power_uv2",
    "beta_power_uv2",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_sigma_power",
    "theta_alpha_ratio",
    "alpha_beta_ratio",
    "sigma_beta_ratio",
    "spectral_entropy",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_edge_frequency_95_hz",
)

DROPPED_FEATURE_REASONS: Final[dict[str, str]] = {
    "rms_uv": (
        "Redundant with mean_uv and std_uv because "
        "RMS² = mean² + variance."
    ),
    "mean_square_uv2": (
        "Deterministically equal to rms_uv squared."
    ),
    "signal_energy_uv2": (
        "Deterministically equal to mean square "
        "multiplied by sample count."
    ),
    "hjorth_activity": (
        "Deterministically equal to signal variance."
    ),
    "total_band_power_uv2": (
        "Deterministically equal to the sum of "
        "absolute band powers."
    ),
    "relative_beta_power": (
        "Linearly determined by the other four "
        "relative powers because their sum is one."
    ),
    "delta_theta_ratio": (
        "Removed after empirical Spearman correlation "
        "magnitude exceeded 0.995."
    ),
}


def load_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load complete extracted features and source schema."""

    if not EPOCH_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Complete feature dataset is missing."
        )

    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "Source feature schema is missing."
        )

    features = pd.read_csv(EPOCH_FEATURES_PATH)

    schema = json.loads(
        FEATURE_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    return features, schema


def validate_feature_selection(
    features: pd.DataFrame,
    source_schema: dict[str, object],
) -> None:
    """Validate selected and excluded feature roles."""

    source_features = set(
        source_schema["feature_columns"]
    )

    selected = set(SELECTED_FEATURES)
    dropped = set(DROPPED_FEATURE_REASONS)

    if selected.intersection(dropped):
        raise ValueError(
            "A feature cannot be both selected and dropped."
        )

    missing_selected = sorted(
        selected.difference(source_features)
    )

    if missing_selected:
        raise ValueError(
            f"Selected features are missing: {missing_selected}"
        )

    unclassified_features = sorted(
        source_features.difference(
            selected.union(dropped)
        )
    )

    if unclassified_features:
        raise ValueError(
            "Source features were neither selected nor dropped: "
            f"{unclassified_features}"
        )

    missing_columns = sorted(
        set(
            (
                *IDENTIFIER_COLUMNS,
                *TARGET_COLUMNS,
                *QUALITY_COLUMNS,
                *SELECTED_FEATURES,
            )
        ).difference(features.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )


def validate_model_input(
    model_input: pd.DataFrame,
) -> None:
    """Validate structural and numerical model-input invariants."""

    if model_input.empty:
        raise ValueError(
            "Model-input table is empty."
        )

    if model_input[
        ["recording_id", "epoch_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate recording/epoch IDs detected."
        )

    feature_values = model_input[
        list(SELECTED_FEATURES)
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Selected model features contain NaN or infinity."
        )

    if model_input["subject_id"].nunique() < 2:
        raise ValueError(
            "Subject-wise splitting requires multiple subjects."
        )

    expected_mapping = {
        str(stage): int(encoded)
        for stage, encoded in SLEEP_STAGE_MAPPING.items()
    }

    actual_mapping = (
        model_input[
            ["sleep_stage", "sleep_stage_encoded"]
        ]
        .drop_duplicates()
        .set_index("sleep_stage")[
            "sleep_stage_encoded"
        ]
        .to_dict()
    )

    if actual_mapping != expected_mapping:
        raise ValueError(
            "Target mapping does not match project configuration."
        )


def save_schema(
    subject_count: int,
    recording_count: int,
) -> None:
    """Save explicit feature, split and scaling policies."""

    schema = {
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "quality_columns": list(QUALITY_COLUMNS),
        "selected_feature_count": len(SELECTED_FEATURES),
        "selected_features": list(SELECTED_FEATURES),
        "dropped_feature_count": len(
            DROPPED_FEATURE_REASONS
        ),
        "dropped_features": DROPPED_FEATURE_REASONS,
        "feature_scaling_applied": False,
        "split_policy": {
            "group_column": "subject_id",
            "random_epoch_split_allowed": False,
            "reason": (
                "Epochs from the same subject are correlated "
                "and must not cross dataset partitions."
            ),
        },
        "scaling_policy": {
            "fit_on": "training partition only",
            "transform": [
                "training partition",
                "validation partition",
                "test partition",
            ],
            "fit_before_split_allowed": False,
        },
        "current_dataset_scope": {
            "subjects": int(subject_count),
            "recordings": int(recording_count),
            "purpose": (
                "Dataset scope is determined from the current "
                "pipeline execution. Local four-subject runs are "
                "for validation; larger Kaggle runs are used for "
                "final benchmarking."
            ),
        },
    }

    MODEL_FEATURE_SCHEMA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_FEATURE_SCHEMA_PATH.write_text(
        (
            json.dumps(
                schema,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Build and validate the unscaled model-input table."""

    features, source_schema = load_inputs()

    validate_feature_selection(
        features,
        source_schema,
    )

    output_columns = [
        *IDENTIFIER_COLUMNS,
        *TARGET_COLUMNS,
        *QUALITY_COLUMNS,
        *SELECTED_FEATURES,
    ]

    model_input = features[
        output_columns
    ].copy()

    model_input = model_input.sort_values(
        [
            "subject_id",
            "recording_id",
            "night",
            "epoch_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    validate_model_input(model_input)

    MODEL_INPUT_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_input.to_csv(
        MODEL_INPUT_DATASET_PATH,
        index=False,
        lineterminator="\n",
    )

    save_schema(
        subject_count=int(
            model_input["subject_id"].nunique()
        ),
        recording_count=int(
            model_input["recording_id"].nunique()
        ),
    )

    print("\n=== MODEL INPUT DATASET ===")
    print("Rows:", len(model_input))
    print("Columns:", len(model_input.columns))
    print("Subjects:", model_input["subject_id"].nunique())
    print(
        "Recordings:",
        model_input["recording_id"].nunique(),
    )
    print("Selected features:", len(SELECTED_FEATURES))
    print(
        "Dropped features:",
        len(DROPPED_FEATURE_REASONS),
    )
    print(
        "Quality issues:",
        int(model_input["quality_issue_flag"].sum()),
    )

    print("\nSelected feature names:")
    for index, feature in enumerate(
        SELECTED_FEATURES,
        start=1,
    ):
        print(f"{index:02d}. {feature}")

    print("\nDropped feature reasons:")
    for feature, reason in (
        DROPPED_FEATURE_REASONS.items()
    ):
        print(f"- {feature}: {reason}")

    print("\nClass distribution:")
    print(
        model_input["sleep_stage"]
        .value_counts()
        .reindex(SLEEP_STAGE_MAPPING.keys())
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nRows per subject:")
    print(
        model_input["subject_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nScaling applied: False")
    print("Random epoch split allowed: False")
    print("Model-input validation: PASS")
    print("Saved to:", MODEL_INPUT_DATASET_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "Model-input generation failed."
        )
        sys.exit(1)
