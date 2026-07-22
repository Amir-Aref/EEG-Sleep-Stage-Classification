from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase3_metrics import (
    align_probability_columns,
    build_prediction_frame,
    calculate_classification_metrics,
    ordered_class_items,
    validate_prediction_contract,
    write_metrics_json,
    write_prediction_csv,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def synthetic_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    y_true = np.tile(
        np.arange(5),
        4,
    )

    y_pred = y_true.copy()

    y_pred[[1, 7, 13, 19]] = np.array(
        [2, 1, 4, 0]
    )

    probabilities = np.full(
        (len(y_pred), 5),
        0.025,
        dtype=float,
    )

    probabilities[
        np.arange(len(y_pred)),
        y_pred,
    ] = 0.9

    return y_true, y_pred, probabilities


def identifier_frame(
    row_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": np.arange(
                row_count
            ) // 5,
            "recording_id": [
                f"recording_{index // 5}"
                for index in range(
                    row_count
                )
            ],
            "night": 1,
            "epoch_id": np.arange(
                row_count
            ),
        }
    )


class Phase3MetricsTests(unittest.TestCase):
    def test_class_order_is_encoded_order(
        self,
    ) -> None:
        self.assertEqual(
            ordered_class_items(
                {
                    "REM": 4,
                    "N3": 3,
                    "Wake": 0,
                    "N2": 2,
                    "N1": 1,
                }
            ),
            [
                ("Wake", 0),
                ("N1", 1),
                ("N2", 2),
                ("N3", 3),
                ("REM", 4),
            ],
        )

    def test_probability_columns_are_aligned(
        self,
    ) -> None:
        source_classes = np.array(
            [4, 2, 0, 3, 1]
        )

        source_probabilities = np.array(
            [
                [
                    0.05,
                    0.10,
                    0.60,
                    0.15,
                    0.10,
                ],
                [
                    0.70,
                    0.10,
                    0.05,
                    0.10,
                    0.05,
                ],
            ]
        )

        aligned = align_probability_columns(
            probabilities=source_probabilities,
            estimator_classes=source_classes,
            class_mapping=CLASS_MAPPING,
        )

        np.testing.assert_allclose(
            aligned[0],
            [
                0.60,
                0.10,
                0.10,
                0.15,
                0.05,
            ],
        )

        np.testing.assert_allclose(
            aligned[1],
            [
                0.05,
                0.05,
                0.10,
                0.10,
                0.70,
            ],
        )

    def test_invalid_probability_rows_are_rejected(
        self,
    ) -> None:
        y_true = np.array(
            [0, 1]
        )

        y_pred = np.array(
            [0, 1]
        )

        invalid_probabilities = np.array(
            [
                [
                    0.8,
                    0.1,
                    0.1,
                    0.1,
                    -0.1,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        with self.assertRaises(
            ValueError
        ):
            validate_prediction_contract(
                y_true=y_true,
                y_pred=y_pred,
                probabilities=(
                    invalid_probabilities
                ),
                class_mapping=CLASS_MAPPING,
            )

    def test_prediction_probability_mismatch_is_rejected(
        self,
    ) -> None:
        y_true = np.array(
            [0, 1]
        )

        y_pred = np.array(
            [1, 1]
        )

        probabilities = np.array(
            [
                [
                    0.9,
                    0.025,
                    0.025,
                    0.025,
                    0.025,
                ],
                [
                    0.025,
                    0.9,
                    0.025,
                    0.025,
                    0.025,
                ],
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "maximum-probability",
        ):
            validate_prediction_contract(
                y_true=y_true,
                y_pred=y_pred,
                probabilities=probabilities,
                class_mapping=CLASS_MAPPING,
            )

    def test_perfect_predictions_have_perfect_core_metrics(
        self,
    ) -> None:
        y_true = np.tile(
            np.arange(5),
            2,
        )

        y_pred = y_true.copy()

        probabilities = np.full(
            (len(y_true), 5),
            0.0125,
        )

        probabilities[
            np.arange(len(y_true)),
            y_true,
        ] = 0.95

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_mapping=CLASS_MAPPING,
        )

        self.assertEqual(
            metrics["accuracy"],
            1.0,
        )
        self.assertEqual(
            metrics["balanced_accuracy"],
            1.0,
        )
        self.assertEqual(
            metrics["macro_f1"],
            1.0,
        )
        self.assertEqual(
            metrics["weighted_f1"],
            1.0,
        )
        self.assertEqual(
            metrics["cohen_kappa"],
            1.0,
        )

    def test_metrics_include_every_class(
        self,
    ) -> None:
        (
            y_true,
            y_pred,
            probabilities,
        ) = synthetic_predictions()

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_mapping=CLASS_MAPPING,
        )

        self.assertEqual(
            metrics["sample_count"],
            20,
        )

        self.assertEqual(
            [
                item["class_name"]
                for item in metrics[
                    "per_class"
                ]
            ],
            [
                "Wake",
                "N1",
                "N2",
                "N3",
                "REM",
            ],
        )

        self.assertEqual(
            np.asarray(
                metrics[
                    "confusion_matrix"
                ]["raw"]
            ).shape,
            (5, 5),
        )

        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "cohen_kappa",
            "multiclass_log_loss",
            "mean_prediction_confidence",
            "mean_prediction_margin",
            "mean_normalized_entropy",
        ):
            self.assertTrue(
                np.isfinite(
                    metrics[metric_name]
                )
            )

    def test_prediction_frame_has_expected_contract(
        self,
    ) -> None:
        (
            y_true,
            y_pred,
            probabilities,
        ) = synthetic_predictions()

        identifiers = identifier_frame(
            len(y_true)
        )

        frame = build_prediction_frame(
            identifiers=identifiers,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_mapping=CLASS_MAPPING,
            run_metadata={
                "model_name": (
                    "logistic_regression"
                ),
                "outer_fold": 1,
                "partition": "test",
            },
        )

        self.assertEqual(
            len(frame),
            len(y_true),
        )

        expected_columns = {
            "model_name",
            "outer_fold",
            "partition",
            "subject_id",
            "recording_id",
            "night",
            "epoch_id",
            "true_label_encoded",
            "true_label",
            "predicted_label_encoded",
            "predicted_label",
            "is_correct",
            "prediction_confidence",
            "prediction_margin",
            "prediction_entropy",
            "prediction_normalized_entropy",
            "probability_wake",
            "probability_n1",
            "probability_n2",
            "probability_n3",
            "probability_rem",
        }

        self.assertTrue(
            expected_columns.issubset(
                frame.columns
            )
        )

        probability_columns = [
            "probability_wake",
            "probability_n1",
            "probability_n2",
            "probability_n3",
            "probability_rem",
        ]

        np.testing.assert_allclose(
            frame[
                probability_columns
            ].sum(axis=1),
            np.ones(len(frame)),
            atol=1e-12,
        )

        self.assertEqual(
            int(frame["is_correct"].sum()),
            int(
                np.sum(
                    y_true == y_pred
                )
            ),
        )

    def test_mismatched_identifier_count_is_rejected(
        self,
    ) -> None:
        (
            y_true,
            y_pred,
            probabilities,
        ) = synthetic_predictions()

        identifiers = identifier_frame(
            len(y_true) - 1
        )

        with self.assertRaisesRegex(
            ValueError,
            "Identifier row count",
        ):
            build_prediction_frame(
                identifiers=identifiers,
                y_true=y_true,
                y_pred=y_pred,
                probabilities=probabilities,
                class_mapping=CLASS_MAPPING,
                run_metadata={
                    "model_name": "test",
                },
            )

    def test_artifact_writes_are_byte_deterministic(
        self,
    ) -> None:
        (
            y_true,
            y_pred,
            probabilities,
        ) = synthetic_predictions()

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_mapping=CLASS_MAPPING,
        )

        predictions = build_prediction_frame(
            identifiers=identifier_frame(
                len(y_true)
            ),
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_mapping=CLASS_MAPPING,
            run_metadata={
                "model_name": "test_model",
                "outer_fold": 1,
                "partition": "test",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            metrics_path = (
                root / "metrics.json"
            )

            predictions_path = (
                root / "predictions.csv"
            )

            write_metrics_json(
                metrics=metrics,
                output_path=metrics_path,
            )

            write_prediction_csv(
                predictions=predictions,
                output_path=predictions_path,
            )

            first_metrics = (
                metrics_path.read_bytes()
            )

            first_predictions = (
                predictions_path.read_bytes()
            )

            write_metrics_json(
                metrics=metrics,
                output_path=metrics_path,
            )

            write_prediction_csv(
                predictions=predictions,
                output_path=predictions_path,
            )

            self.assertEqual(
                first_metrics,
                metrics_path.read_bytes(),
            )

            self.assertEqual(
                first_predictions,
                predictions_path.read_bytes(),
            )

            loaded_metrics = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded_metrics["sample_count"],
                len(y_true),
            )


if __name__ == "__main__":
    unittest.main()
