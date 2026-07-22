"""Automated contracts for the rebuilt Phase 2 EEG pipeline.

Unit tests do not require raw EDF files.
Generated-artifact tests are skipped when local artifacts are absent,
which keeps the suite usable in clean CI environments.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_model_input import (
    DROPPED_FEATURE_REASONS,
    SELECTED_FEATURES,
)
from scripts.config import (
    EDA_SUMMARY_PATH,
    EPOCH_FEATURES_PATH,
    EPOCH_METADATA_PATH,
    FEATURE_SCHEMA_PATH,
    MODEL_FEATURE_SCHEMA_PATH,
    MODEL_INPUT_DATASET_PATH,
    PROJECT_ROOT,
    SLEEP_STAGE_MAPPING,
)
from scripts.extract_eeg_features import (
    calculate_hjorth,
    integrate_band_power,
    safe_ratio,
)


GENERATED_ARTIFACTS_AVAILABLE = all(
    path.exists()
    for path in (
        EPOCH_FEATURES_PATH,
        EPOCH_METADATA_PATH,
        MODEL_INPUT_DATASET_PATH,
    )
)


class FeatureMathTests(unittest.TestCase):
    """Test reusable mathematical feature functions."""

    def test_safe_ratio_valid_inputs(self) -> None:
        self.assertAlmostEqual(
            safe_ratio(10.0, 4.0),
            2.5,
        )

    def test_safe_ratio_rejects_zero_denominator(self) -> None:
        self.assertTrue(
            math.isnan(
                safe_ratio(1.0, 0.0)
            )
        )

    def test_hjorth_constant_signal(self) -> None:
        signal = np.ones(
            3000,
            dtype=float,
        )

        activity, mobility, complexity = (
            calculate_hjorth(signal)
        )

        self.assertEqual(activity, 0.0)
        self.assertEqual(mobility, 0.0)
        self.assertEqual(complexity, 0.0)

    def test_hjorth_nonconstant_signal_is_finite(self) -> None:
        time = np.linspace(
            0.0,
            30.0,
            3000,
            endpoint=False,
        )

        signal = np.sin(
            2.0 * np.pi * 2.0 * time
        )

        activity, mobility, complexity = (
            calculate_hjorth(signal)
        )

        self.assertGreater(activity, 0.0)
        self.assertGreater(mobility, 0.0)
        self.assertGreater(complexity, 0.0)

        self.assertTrue(
            np.isfinite(
                [
                    activity,
                    mobility,
                    complexity,
                ]
            ).all()
        )

    def test_band_power_uses_frequency_resolution(self) -> None:
        frequencies = np.array(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            dtype=float,
        )

        psd = np.full(
            frequencies.shape,
            2.0,
            dtype=float,
        )

        power = integrate_band_power(
            frequencies=frequencies,
            psd=psd,
            low_hz=1.0,
            high_hz=4.0,
            include_high=False,
        )

        # Three bins × PSD 2 × resolution 1 Hz.
        self.assertAlmostEqual(power, 6.0)


class FeatureSchemaTests(unittest.TestCase):
    """Test explicit feature-role and leakage policies."""

    def test_all_source_features_are_classified(self) -> None:
        source_schema = json.loads(
            FEATURE_SCHEMA_PATH.read_text(
                encoding="utf-8"
            )
        )

        source_features = set(
            source_schema["feature_columns"]
        )

        selected_features = set(
            SELECTED_FEATURES
        )

        dropped_features = set(
            DROPPED_FEATURE_REASONS
        )

        self.assertFalse(
            selected_features.intersection(
                dropped_features
            )
        )

        self.assertEqual(
            source_features,
            selected_features.union(
                dropped_features
            ),
        )

        self.assertEqual(
            len(SELECTED_FEATURES),
            28,
        )

        self.assertEqual(
            len(DROPPED_FEATURE_REASONS),
            7,
        )

    def test_model_schema_prevents_leakage(self) -> None:
        model_schema = json.loads(
            MODEL_FEATURE_SCHEMA_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(
            model_schema[
                "feature_scaling_applied"
            ]
        )

        self.assertFalse(
            model_schema[
                "split_policy"
            ][
                "random_epoch_split_allowed"
            ]
        )

        self.assertEqual(
            model_schema[
                "split_policy"
            ][
                "group_column"
            ],
            "subject_id",
        )

        self.assertEqual(
            model_schema[
                "scaling_policy"
            ][
                "fit_on"
            ],
            "training partition only",
        )


@unittest.skipUnless(
    GENERATED_ARTIFACTS_AVAILABLE,
    "Generated local Phase 2 artifacts are unavailable.",
)
class LocalArtifactContractTests(unittest.TestCase):
    """Validate generated local artifacts without raw EDF access."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.features = pd.read_csv(
            EPOCH_FEATURES_PATH
        )

        cls.metadata = pd.read_csv(
            EPOCH_METADATA_PATH
        )

        cls.model_input = pd.read_csv(
            MODEL_INPUT_DATASET_PATH
        )

    def test_artifact_row_counts_and_identifiers(self) -> None:
        self.assertEqual(
            len(self.features),
            len(self.metadata),
        )

        self.assertEqual(
            len(self.model_input),
            len(self.metadata),
        )

        self.assertFalse(
            self.features[
                [
                    "recording_id",
                    "epoch_id",
                ]
            ].duplicated().any()
        )

        self.assertFalse(
            self.model_input[
                [
                    "recording_id",
                    "epoch_id",
                ]
            ].duplicated().any()
        )

    def test_metadata_matches_feature_rows(self) -> None:
        columns = [
            "recording_id",
            "epoch_id",
            "sleep_stage",
            "sleep_stage_encoded",
            "start_sample",
            "stop_sample",
        ]

        expected = (
            self.metadata[columns]
            .sort_values(
                [
                    "recording_id",
                    "epoch_id",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        actual = (
            self.features[columns]
            .sort_values(
                [
                    "recording_id",
                    "epoch_id",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(
            expected,
            actual,
        )

    def test_relative_powers_sum_to_one(self) -> None:
        relative_columns = [
            "relative_delta_power",
            "relative_theta_power",
            "relative_alpha_power",
            "relative_sigma_power",
            "relative_beta_power",
        ]

        relative_sum = self.features[
            relative_columns
        ].sum(axis=1)

        self.assertTrue(
            np.allclose(
                relative_sum.to_numpy(),
                1.0,
                rtol=0.0,
                atol=1e-10,
            )
        )

    def test_model_feature_matrix_is_finite(self) -> None:
        values = self.model_input[
            list(SELECTED_FEATURES)
        ].to_numpy(dtype=float)

        self.assertTrue(
            np.isfinite(values).all()
        )

        self.assertEqual(
            values.shape[1],
            28,
        )

        self.assertGreaterEqual(
            self.model_input[
                "subject_id"
            ].nunique(),
            2,
        )

    def test_target_mapping_matches_configuration(self) -> None:
        actual_mapping = (
            self.model_input[
                [
                    "sleep_stage",
                    "sleep_stage_encoded",
                ]
            ]
            .drop_duplicates()
            .set_index("sleep_stage")[
                "sleep_stage_encoded"
            ]
            .to_dict()
        )

        expected_mapping = {
            str(stage): int(encoded)
            for stage, encoded
            in SLEEP_STAGE_MAPPING.items()
        }

        self.assertEqual(
            actual_mapping,
            expected_mapping,
        )


@unittest.skipUnless(
    EDA_SUMMARY_PATH.exists(),
    "Generated EDA summary is unavailable.",
)
class EdaArtifactTests(unittest.TestCase):
    """Validate the generated EDA report inventory."""

    def test_all_declared_figures_exist(self) -> None:
        summary = json.loads(
            EDA_SUMMARY_PATH.read_text(
                encoding="utf-8"
            )
        )

        figures = summary[
            "generated_figures"
        ]

        expected_figure_count = (
            8
            + int(
                summary["recording_count"]
            )
        )

        self.assertEqual(
            len(figures),
            expected_figure_count,
        )

        for relative_path in figures:
            path = (
                PROJECT_ROOT
                / Path(relative_path)
            )

            self.assertTrue(
                path.exists(),
                path,
            )

            self.assertGreater(
                path.stat().st_size,
                10_000,
                path,
            )

        self.assertEqual(
            summary[
                "outlier_policy"
            ][
                "rows_removed"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
