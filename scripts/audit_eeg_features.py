"""Audit extracted EEG features before EDA and modeling."""

from __future__ import annotations

import json
import logging
import sys
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from .config import (
        EPOCH_FEATURES_PATH,
        FEATURE_AUDIT_REPORT_PATH,
        FEATURE_SCHEMA_PATH,
        HIGH_CORRELATION_PAIRS_PATH,
        STAGE_FEATURE_SUMMARY_PATH,
        SUBJECT_FEATURE_SUMMARY_PATH,
    )
except ImportError:
    from config import (
        EPOCH_FEATURES_PATH,
        FEATURE_AUDIT_REPORT_PATH,
        FEATURE_SCHEMA_PATH,
        HIGH_CORRELATION_PAIRS_PATH,
        STAGE_FEATURE_SUMMARY_PATH,
        SUBJECT_FEATURE_SUMMARY_PATH,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HIGH_CORRELATION_THRESHOLD = 0.995
RELATION_TOLERANCE = 1e-9

STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]

KEY_FEATURES = [
    "std_uv",
    "rms_uv",
    "peak_to_peak_uv",
    "total_band_power_uv2",
    "zero_crossing_rate",
    "line_length_uv",
    "hjorth_mobility",
    "hjorth_complexity",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_sigma_power",
    "relative_beta_power",
    "spectral_entropy",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_edge_frequency_95_hz",
]


def load_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load extracted features and explicit feature schema."""

    if not EPOCH_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Complete feature dataset does not exist. "
            "Run extract_eeg_features.py --all first."
        )

    if not FEATURE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "Feature schema does not exist."
        )

    features = pd.read_csv(EPOCH_FEATURES_PATH)

    schema = json.loads(
        FEATURE_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    feature_columns = schema["feature_columns"]

    missing = sorted(
        set(feature_columns).difference(features.columns)
    )

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    values = features[feature_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values detected in feature matrix."
        )

    return features, schema


def relationship_error(
    observed: pd.Series,
    expected: pd.Series,
) -> float:
    """Calculate maximum normalized numerical error."""

    observed_values = observed.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)

    scale = np.maximum(
        np.maximum(
            np.abs(observed_values),
            np.abs(expected_values),
        ),
        1.0,
    )

    normalized_error = (
        np.abs(observed_values - expected_values)
        / scale
    )

    return float(np.max(normalized_error))


def audit_deterministic_relationships(
    features: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Check mathematically deterministic feature relationships."""

    absolute_power_columns = [
        "delta_power_uv2",
        "theta_power_uv2",
        "alpha_power_uv2",
        "sigma_power_uv2",
        "beta_power_uv2",
    ]

    relative_power_columns = [
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_sigma_power",
        "relative_beta_power",
    ]

    checks = {
        "rms_squared_equals_mean_square": relationship_error(
            features["rms_uv"] ** 2,
            features["mean_square_uv2"],
        ),
        "energy_equals_mean_square_times_samples": (
            relationship_error(
                features["signal_energy_uv2"],
                features["mean_square_uv2"]
                * features["sample_count"],
            )
        ),
        "hjorth_activity_equals_std_squared": (
            relationship_error(
                features["hjorth_activity"],
                features["std_uv"] ** 2,
            )
        ),
        "total_power_equals_band_sum": (
            relationship_error(
                features["total_band_power_uv2"],
                features[absolute_power_columns].sum(axis=1),
            )
        ),
        "relative_power_sum_equals_one": float(
            np.max(
                np.abs(
                    features[relative_power_columns]
                    .sum(axis=1)
                    .to_numpy()
                    - 1.0
                )
            )
        ),
    }

    return {
        name: {
            "maximum_normalized_error": error,
            "passes": bool(
                error <= RELATION_TOLERANCE
            ),
        }
        for name, error in checks.items()
    }


def find_high_correlation_pairs(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Find highly correlated feature pairs using Spearman correlation."""

    correlation = features[feature_columns].corr(
        method="spearman"
    )

    records: list[dict[str, object]] = []

    for left, right in combinations(feature_columns, 2):
        value = float(correlation.loc[left, right])

        if (
            np.isfinite(value)
            and abs(value) >= HIGH_CORRELATION_THRESHOLD
        ):
            records.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "spearman_correlation": value,
                    "absolute_correlation": abs(value),
                }
            )

    dataframe = pd.DataFrame(records)

    if not dataframe.empty:
        dataframe = dataframe.sort_values(
            [
                "absolute_correlation",
                "feature_left",
                "feature_right",
            ],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)

    return dataframe


def build_stage_summary(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build median and IQR summaries by sleep stage."""

    rows: list[dict[str, object]] = []

    for stage in STAGE_ORDER:
        group = features[
            features["sleep_stage"] == stage
        ]

        for feature in KEY_FEATURES:
            rows.append(
                {
                    "sleep_stage": stage,
                    "feature": feature,
                    "count": int(len(group)),
                    "median": float(
                        group[feature].median()
                    ),
                    "q1": float(
                        group[feature].quantile(0.25)
                    ),
                    "q3": float(
                        group[feature].quantile(0.75)
                    ),
                    "iqr": float(
                        group[feature].quantile(0.75)
                        - group[feature].quantile(0.25)
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_subject_summary(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build robust summaries for identifying subject-level shift."""

    rows: list[dict[str, object]] = []

    for (
        subject_id,
        recording_id,
    ), group in features.groupby(
        ["subject_id", "recording_id"],
        sort=True,
    ):
        row: dict[str, object] = {
            "subject_id": int(subject_id),
            "recording_id": str(recording_id),
            "epoch_count": int(len(group)),
        }

        for feature in KEY_FEATURES:
            row[f"median_{feature}"] = float(
                group[feature].median()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run feature audit and save reports."""

    features, schema = load_inputs()
    feature_columns = list(schema["feature_columns"])

    constant_features = [
        feature
        for feature in feature_columns
        if features[feature].nunique(
            dropna=False
        ) <= 1
    ]

    deterministic_relationships = (
        audit_deterministic_relationships(features)
    )

    high_correlation_pairs = (
        find_high_correlation_pairs(
            features,
            feature_columns,
        )
    )

    stage_summary = build_stage_summary(features)
    subject_summary = build_subject_summary(features)

    report = {
        "row_count": int(len(features)),
        "column_count": int(len(features.columns)),
        "feature_count": int(len(feature_columns)),
        "recording_count": int(
            features["recording_id"].nunique()
        ),
        "subject_count": int(
            features["subject_id"].nunique()
        ),
        "constant_features": constant_features,
        "high_correlation_threshold": (
            HIGH_CORRELATION_THRESHOLD
        ),
        "high_correlation_pair_count": int(
            len(high_correlation_pairs)
        ),
        "deterministic_relationships": (
            deterministic_relationships
        ),
        "class_distribution": {
            str(key): int(value)
            for key, value in (
                features["sleep_stage"]
                .value_counts()
                .reindex(STAGE_ORDER)
                .fillna(0)
                .items()
            )
        },
        "quality_issue_count": int(
            features["quality_issue_flag"].sum()
        ),
    }

    FEATURE_AUDIT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_AUDIT_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    high_correlation_pairs.to_csv(
        HIGH_CORRELATION_PAIRS_PATH,
        index=False,
        lineterminator="\n",
    )

    stage_summary.to_csv(
        STAGE_FEATURE_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    subject_summary.to_csv(
        SUBJECT_FEATURE_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    print("\n=== FEATURE AUDIT ===")
    print("Rows:", len(features))
    print("Features:", len(feature_columns))
    print("Subjects:", features["subject_id"].nunique())
    print("Constant features:", constant_features)
    print(
        "High-correlation pairs:",
        len(high_correlation_pairs),
    )

    print("\n=== DETERMINISTIC RELATIONSHIPS ===")
    for name, result in deterministic_relationships.items():
        print(
            f"{name}: "
            f"error={result['maximum_normalized_error']:.3e}, "
            f"passes={result['passes']}"
        )

    print("\n=== TOP HIGH-CORRELATION PAIRS ===")
    if high_correlation_pairs.empty:
        print("None")
    else:
        print(
            high_correlation_pairs.head(20).to_string(
                index=False
            )
        )

    print("\n=== MEDIAN RELATIVE POWER BY STAGE ===")
    relative_features = [
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_sigma_power",
        "relative_beta_power",
    ]

    relative_summary = (
        features.groupby("sleep_stage")[
            relative_features
        ]
        .median()
        .reindex(STAGE_ORDER)
    )

    print(relative_summary.to_string())

    print("\n=== SUBJECT SHIFT CHECK ===")
    print(
        subject_summary[
            [
                "subject_id",
                "recording_id",
                "epoch_count",
                "median_std_uv",
                "median_total_band_power_uv2",
                "median_spectral_entropy",
                "median_spectral_centroid_hz",
            ]
        ].to_string(index=False)
    )

    print("\nEEG feature audit: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "EEG feature audit failed."
        )
        sys.exit(1)
