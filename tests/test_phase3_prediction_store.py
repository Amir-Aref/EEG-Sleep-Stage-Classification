from __future__ import annotations

import math
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.phase3_metrics import (
    write_prediction_csv,
)
from scripts.phase3_prediction_store import (
    audit_prediction_database,
    persist_prediction_run,
    prediction_frame_sha256,
    read_prediction_run,
    setup_prediction_database,
    sha256_bytes,
    synthetic_prediction_frame,
    validate_prediction_frame,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def valid_metadata() -> dict:
    return {
        "artifact_type": (
            "phase3_prediction_test"
        ),
        "model_file_path": (
            "artifacts/models/test.joblib"
        ),
        "model_file_sha256": (
            "b" * 64
        ),
        "model_name": "test_model",
        "candidate_id": (
            "test_model__candidate_001"
        ),
        "outer_fold": 1,
        "deployment_ready": False,
        "non_deployment_override": True,
        "input_scope": "unit_test",
        "feature_names": [
            "feature_a",
            "feature_b",
        ],
        "class_mapping": CLASS_MAPPING,
    }


class Phase3PredictionStoreTests(
    unittest.TestCase
):
    def test_schema_enables_foreign_keys(
        self,
    ) -> None:
        connection = sqlite3.connect(
            ":memory:"
        )

        try:
            setup_prediction_database(
                connection
            )

            enabled = connection.execute(
                "PRAGMA foreign_keys"
            ).fetchone()[0]

            self.assertEqual(
                enabled,
                1,
            )

            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
            }

            self.assertIn(
                "prediction_runs",
                tables,
            )

            self.assertIn(
                "prediction_rows",
                tables,
            )

        finally:
            connection.close()

    def test_valid_run_roundtrips(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "predictions.sqlite3"
            )

            predictions = (
                synthetic_prediction_frame()
            )

            result = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=predictions,
            )

            run, rows = read_prediction_run(
                database_path=database_path,
                run_id=result.run_id,
            )

        self.assertTrue(
            result.inserted
        )

        self.assertEqual(
            run["input_row_count"],
            2,
        )

        self.assertEqual(
            len(rows),
            2,
        )

        self.assertEqual(
            rows[
                "predicted_label"
            ].tolist(),
            ["Wake", "N2"],
        )

    def test_repeated_persistence_is_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "predictions.sqlite3"
            )

            predictions = (
                synthetic_prediction_frame()
            )

            first = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=predictions,
            )

            second = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=predictions,
            )

            audit = audit_prediction_database(
                database_path
            )

        self.assertTrue(
            first.inserted
        )

        self.assertFalse(
            second.inserted
        )

        self.assertEqual(
            first.run_id,
            second.run_id,
        )

        self.assertEqual(
            audit[
                "prediction_run_count"
            ],
            1,
        )

        self.assertEqual(
            audit[
                "prediction_row_count"
            ],
            2,
        )

    def test_non_deployment_model_requires_override(
        self,
    ) -> None:
        metadata = valid_metadata()

        metadata[
            "non_deployment_override"
        ] = False

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "explicit override",
            ):
                persist_prediction_run(
                    database_path=(
                        Path(directory)
                        / "predictions.sqlite3"
                    ),
                    run_metadata=metadata,
                    predictions=(
                        synthetic_prediction_frame()
                    ),
                )

    def test_invalid_probability_is_rejected(
        self,
    ) -> None:
        predictions = (
            synthetic_prediction_frame()
        )

        predictions.loc[
            0,
            "probability_wake",
        ] = 0.95

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "sum",
            ):
                persist_prediction_run(
                    database_path=(
                        Path(directory)
                        / "predictions.sqlite3"
                    ),
                    run_metadata=(
                        valid_metadata()
                    ),
                    predictions=predictions,
                )

    def test_duplicate_identifiers_are_rejected(
        self,
    ) -> None:
        predictions = (
            synthetic_prediction_frame()
        )

        predictions.loc[
            1,
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ],
        ] = predictions.loc[
            0,
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ],
        ].to_numpy()

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "not unique",
            ):
                persist_prediction_run(
                    database_path=(
                        Path(directory)
                        / "predictions.sqlite3"
                    ),
                    run_metadata=(
                        valid_metadata()
                    ),
                    predictions=predictions,
                )

    def test_ground_truth_columns_are_optional(
        self,
    ) -> None:
        predictions = (
            synthetic_prediction_frame()
        )

        predictions[
            "true_label_encoded"
        ] = np.array(
            [0, 0],
            dtype=int,
        )

        predictions[
            "true_label"
        ] = [
            "Wake",
            "Wake",
        ]

        predictions[
            "is_correct"
        ] = [
            True,
            False,
        ]

        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "predictions.sqlite3"
            )

            result = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=predictions,
            )

            _, rows = read_prediction_run(
                database_path=database_path,
                run_id=result.run_id,
            )

        self.assertEqual(
            rows["is_correct"].tolist(),
            [1, 0],
        )

    def test_signed_zero_is_canonicalized(
        self,
    ) -> None:
        predictions = (
            synthetic_prediction_frame()
            .iloc[[0]]
            .reset_index(drop=True)
        )

        probabilities = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        predictions.loc[
            0,
            [
                "probability_wake",
                "probability_n1",
                "probability_n2",
                "probability_n3",
                "probability_rem",
            ],
        ] = probabilities

        predictions.loc[
            0,
            "predicted_label_encoded",
        ] = 0

        predictions.loc[
            0,
            "predicted_label",
        ] = "Wake"

        predictions.loc[
            0,
            "probability_argmax_label_encoded",
        ] = 0

        predictions.loc[
            0,
            "probability_argmax_label",
        ] = "Wake"

        predictions.loc[
            0,
            "predict_probability_argmax_agree",
        ] = True

        predictions.loc[
            0,
            "prediction_confidence",
        ] = 1.0

        predictions.loc[
            0,
            "prediction_margin",
        ] = 1.0

        predictions.loc[
            0,
            "prediction_entropy",
        ] = -0.0

        predictions.loc[
            0,
            "prediction_normalized_entropy",
        ] = -0.0

        validated = validate_prediction_frame(
            predictions=predictions,
            class_mapping=CLASS_MAPPING,
        )

        for column in (
            "prediction_entropy",
            "prediction_normalized_entropy",
            "probability_n1",
            "probability_n2",
            "probability_n3",
            "probability_rem",
        ):
            value = float(
                validated.loc[
                    0,
                    column,
                ]
            )

            self.assertEqual(
                value,
                0.0,
            )

            self.assertFalse(
                np.signbit(value),
                msg=(
                    f"{column} retained "
                    "a negative-zero sign."
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "predictions.csv"
            )

            write_prediction_csv(
                predictions=validated,
                output_path=output_path,
            )

            csv_text = output_path.read_text(
                encoding="utf-8"
            )

        self.assertNotIn(
            ",-0,",
            csv_text,
        )

    def test_prediction_hash_uses_canonical_csv_serialization(
        self,
    ) -> None:
        first = synthetic_prediction_frame()

        second = first.copy()

        epsilon = np.finfo(float).eps

        second.loc[
            0,
            "probability_wake",
        ] += epsilon

        second.loc[
            0,
            "probability_n1",
        ] -= epsilon

        probability_columns = [
            "probability_wake",
            "probability_n1",
            "probability_n2",
            "probability_n3",
            "probability_rem",
        ]

        probabilities = second.loc[
            0,
            probability_columns,
        ].to_numpy(
            dtype=float
        )

        sorted_probabilities = np.sort(
            probabilities
        )

        entropy = -np.sum(
            probabilities
            * np.log(
                np.clip(
                    probabilities,
                    np.finfo(float).tiny,
                    1.0,
                )
            )
        )

        second.loc[
            0,
            "prediction_confidence",
        ] = sorted_probabilities[-1]

        second.loc[
            0,
            "prediction_margin",
        ] = (
            sorted_probabilities[-1]
            - sorted_probabilities[-2]
        )

        second.loc[
            0,
            "prediction_entropy",
        ] = entropy

        second.loc[
            0,
            "prediction_normalized_entropy",
        ] = entropy / math.log(5)

        material_change = first.copy()

        material_change.loc[
            0,
            "prediction_confidence",
        ] += 1e-6

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first_path = (
                root / "first.csv"
            )

            second_path = (
                root / "second.csv"
            )

            write_prediction_csv(
                predictions=first,
                output_path=first_path,
            )

            write_prediction_csv(
                predictions=second,
                output_path=second_path,
            )

            first_bytes = (
                first_path.read_bytes()
            )

            second_bytes = (
                second_path.read_bytes()
            )

        self.assertEqual(
            first_bytes,
            second_bytes,
        )

        first_hash = (
            prediction_frame_sha256(
                first
            )
        )

        second_hash = (
            prediction_frame_sha256(
                second
            )
        )

        self.assertEqual(
            first_hash,
            sha256_bytes(
                first_bytes
            ),
        )

        self.assertEqual(
            second_hash,
            sha256_bytes(
                second_bytes
            ),
        )

        self.assertEqual(
            first_hash,
            second_hash,
        )

        self.assertNotEqual(
            first_hash,
            prediction_frame_sha256(
                material_change
            ),
        )

    def test_maximum_entropy_respects_sql_boundary(
        self,
    ) -> None:
        predictions = (
            synthetic_prediction_frame()
            .iloc[[0]]
            .reset_index(drop=True)
        )

        probabilities = np.full(
            5,
            0.2,
            dtype=float,
        )

        entropy = -np.sum(
            probabilities
            * np.log(probabilities)
        )

        predictions.loc[
            0,
            [
                "probability_wake",
                "probability_n1",
                "probability_n2",
                "probability_n3",
                "probability_rem",
            ],
        ] = probabilities

        predictions.loc[
            0,
            "predicted_label_encoded",
        ] = 0

        predictions.loc[
            0,
            "predicted_label",
        ] = "Wake"

        predictions.loc[
            0,
            "probability_argmax_label_encoded",
        ] = 0

        predictions.loc[
            0,
            "probability_argmax_label",
        ] = "Wake"

        predictions.loc[
            0,
            "predict_probability_argmax_agree",
        ] = True

        predictions.loc[
            0,
            "prediction_confidence",
        ] = 0.2

        predictions.loc[
            0,
            "prediction_margin",
        ] = 0.0

        predictions.loc[
            0,
            "prediction_entropy",
        ] = entropy

        predictions.loc[
            0,
            "prediction_normalized_entropy",
        ] = entropy / math.log(5)

        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "predictions.sqlite3"
            )

            result = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=predictions,
            )

            _, rows = read_prediction_run(
                database_path=database_path,
                run_id=result.run_id,
            )

        self.assertTrue(
            result.inserted
        )

        self.assertLessEqual(
            rows.loc[
                0,
                "prediction_entropy",
            ],
            math.log(5),
        )

        self.assertLessEqual(
            rows.loc[
                0,
                "prediction_normalized_entropy",
            ],
            1.0,
        )

    def test_foreign_key_cascade_removes_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "predictions.sqlite3"
            )

            result = persist_prediction_run(
                database_path=database_path,
                run_metadata=valid_metadata(),
                predictions=(
                    synthetic_prediction_frame()
                ),
            )

            connection = sqlite3.connect(
                database_path
            )

            try:
                connection.execute(
                    "PRAGMA foreign_keys = ON"
                )

                with connection:
                    connection.execute(
                        """
                        DELETE FROM prediction_runs
                        WHERE run_id = ?
                        """,
                        (result.run_id,),
                    )

                count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM prediction_rows
                    """
                ).fetchone()[0]

            finally:
                connection.close()

        self.assertEqual(
            count,
            0,
        )

    def test_direct_script_entrypoint_imports(
        self,
    ) -> None:
        project_root = Path(
            __file__
        ).resolve().parents[1]

        script_path = (
            project_root
            / "scripts"
            / "phase3_prediction_store.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "stdout:\n"
                f"{result.stdout}\n"
                "stderr:\n"
                f"{result.stderr}"
            ),
        )

        self.assertIn(
            "Use --smoke-test.",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
