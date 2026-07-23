from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from scripts.phase3_model_artifacts import (
    sha256_file,
)
from scripts.phase3_prediction_pipeline import (
    build_prediction_output,
    execute_prediction_pipeline,
    load_prediction_model,
    prepare_prediction_input,
)
from scripts.phase3_prediction_store import (
    audit_prediction_database,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}

FEATURE_NAMES = (
    "feature_a",
    "feature_b",
)


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def create_model_fixture(
    root: Path,
) -> tuple[Path, Path]:
    X = pd.DataFrame(
        {
            "feature_a": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "feature_b": [
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
            ],
        }
    )

    y = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            2,
            3,
            4,
        ],
        dtype=int,
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                "passthrough",
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="prior"
                ),
            ),
        ]
    )

    pipeline.fit(
        X,
        y,
    )

    metadata = {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_trained_outer_pipeline"
        ),
        "intended_use": (
            "unit_test"
        ),
        "deployment_ready": False,
        "deployment_block_reason": (
            "Outer-fold test model."
        ),
        "outer_fold": 1,
        "training_scope": (
            "outer_development_subjects_only"
        ),
        "training_subjects": [
            1,
            2,
            3,
        ],
        "excluded_test_subjects": [0],
        "training_row_count": len(X),
        "test_feature_matrix_loaded": False,
        "test_predictions_loaded": False,
        "outer_test_metrics_loaded": False,
        "hyperparameter_search_performed": False,
        "model_name": "dummy_prior",
        "candidate_id": (
            "dummy_prior__candidate_001"
        ),
        "candidate_parameters": {},
        "selection_validation_summary": {},
        "feature_names": list(
            FEATURE_NAMES
        ),
        "feature_count": len(
            FEATURE_NAMES
        ),
        "class_mapping": CLASS_MAPPING,
        "group_column": "subject_id",
        "target_column": (
            "sleep_stage_encoded"
        ),
        "target_name_column": (
            "sleep_stage"
        ),
        "source": {},
        "runtime": {},
    }

    model_path = (
        root / "model.joblib"
    )

    joblib.dump(
        {
            "metadata": metadata,
            "pipeline": pipeline,
        },
        model_path,
        compress=3,
    )

    model_record = {
        "outer_fold": 1,
        "training_subjects": [
            1,
            2,
            3,
        ],
        "excluded_test_subjects": [0],
        "model_name": "dummy_prior",
        "candidate_id": (
            "dummy_prior__candidate_001"
        ),
        "candidate_parameters": {},
        "training_row_count": len(X),
        "feature_count": len(
            FEATURE_NAMES
        ),
        "model_file_path": str(
            model_path.resolve()
        ),
        "model_file_size_bytes": (
            model_path.stat().st_size
        ),
        "model_file_sha256": (
            sha256_file(model_path)
        ),
        "verification_row_count": 2,
        "reload_prediction_match": True,
        "reload_probability_match": True,
        "deployment_ready": False,
    }

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_local_trained_model_manifest"
        ),
        "intended_use": "unit_test",
        "complete_model_set": True,
        "model_count": 1,
        "evaluated_outer_folds": [1],
        "deployment": {
            "ready": False,
        },
        "models": [
            model_record
        ],
    }

    manifest_path = (
        root / "manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    return (
        manifest_path,
        model_path,
    )


def prediction_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [
                10,
                10,
                10,
            ],
            "recording_id": [
                "REC",
                "REC",
                "REC",
            ],
            "night": [
                1,
                1,
                1,
            ],
            "epoch_id": [
                0,
                1,
                2,
            ],
            "feature_b": [
                3.0,
                2.0,
                1.0,
            ],
            "feature_a": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )


class Phase3PredictionPipelineTests(
    unittest.TestCase
):
    def test_model_manifest_and_hash_are_validated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, model_path = (
                create_model_fixture(root)
            )

            loaded = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            self.assertEqual(
                loaded.model_path,
                model_path.resolve(),
            )

            self.assertEqual(
                loaded.metadata[
                    "candidate_id"
                ],
                "dummy_prior__candidate_001",
            )

    def test_git_lfs_pointer_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, model_path = (
                create_model_fixture(root)
            )

            model_path.write_text(
                (
                    "version "
                    "https://git-lfs.github.com/spec/v1\n"
                    "oid sha256:"
                    + "a" * 64
                    + "\n"
                    "size 100\n"
                ),
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Git LFS pointer",
            ):
                load_prediction_model(
                    manifest_path=(
                        manifest_path
                    ),
                    outer_fold=1,
                )

    def test_non_deployment_model_requires_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            prepared = prepare_prediction_input(
                frame=(
                    prediction_input_frame()
                ),
                model_metadata=(
                    model.metadata
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "not deployment-ready",
            ):
                build_prediction_output(
                    model=model,
                    prepared=prepared,
                    allow_non_deployment_model=False,
                )

    def test_targetless_prediction_reorders_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            execution = (
                execute_prediction_pipeline(
                    model=model,
                    input_frame=(
                        prediction_input_frame()
                    ),
                    input_scope="unit_test",
                    allow_non_deployment_model=True,
                )
            )

        self.assertEqual(
            list(
                execution.predictions[
                    [
                        "subject_id",
                        "recording_id",
                        "night",
                        "epoch_id",
                    ]
                ].columns
            ),
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ],
        )

        self.assertEqual(
            len(execution.predictions),
            3,
        )

        self.assertNotIn(
            "true_label_encoded",
            execution.predictions.columns,
        )

        self.assertEqual(
            execution.predictions[
                "source_row_index"
            ].tolist(),
            [0, 1, 2],
        )

    def test_optional_ground_truth_is_preserved(
        self,
    ) -> None:
        frame = prediction_input_frame()

        frame[
            "sleep_stage_encoded"
        ] = [
            0,
            1,
            2,
        ]

        frame["sleep_stage"] = [
            "Wake",
            "N1",
            "N2",
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            execution = (
                execute_prediction_pipeline(
                    model=model,
                    input_frame=frame,
                    input_scope="unit_test",
                    allow_non_deployment_model=True,
                )
            )

        self.assertIn(
            "true_label_encoded",
            execution.predictions.columns,
        )

        self.assertIn(
            "true_label",
            execution.predictions.columns,
        )

        self.assertIn(
            "is_correct",
            execution.predictions.columns,
        )

        expected_correct = (
            execution.predictions[
                "true_label_encoded"
            ].to_numpy()
            == execution.predictions[
                "predicted_label_encoded"
            ].to_numpy()
        )

        np.testing.assert_array_equal(
            execution.predictions[
                "is_correct"
            ].to_numpy(),
            expected_correct,
        )

    def test_missing_feature_is_rejected(
        self,
    ) -> None:
        frame = prediction_input_frame().drop(
            columns=["feature_a"]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Missing prediction input",
            ):
                prepare_prediction_input(
                    frame=frame,
                    model_metadata=(
                        model.metadata
                    ),
                )

    def test_unexpected_column_is_rejected(
        self,
    ) -> None:
        frame = prediction_input_frame()

        frame[
            "unexpected_leakage"
        ] = 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unexpected prediction input",
            ):
                prepare_prediction_input(
                    frame=frame,
                    model_metadata=(
                        model.metadata
                    ),
                )

    def test_duplicate_identifiers_are_rejected(
        self,
    ) -> None:
        frame = prediction_input_frame()

        frame.loc[
            1,
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ],
        ] = frame.loc[
            0,
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ],
        ].to_numpy()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            with self.assertRaisesRegex(
                ValueError,
                "not unique",
            ):
                prepare_prediction_input(
                    frame=frame,
                    model_metadata=(
                        model.metadata
                    ),
                )

    def test_database_and_csv_are_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            manifest_path, _ = (
                create_model_fixture(root)
            )

            model = load_prediction_model(
                manifest_path=manifest_path,
                outer_fold=1,
            )

            database_path = (
                root
                / "predictions.sqlite3"
            )

            csv_path = (
                root
                / "predictions.csv"
            )

            first = execute_prediction_pipeline(
                model=model,
                input_frame=(
                    prediction_input_frame()
                ),
                input_scope="unit_test",
                allow_non_deployment_model=True,
                database_path=database_path,
                output_csv_path=csv_path,
            )

            first_csv = (
                csv_path.read_bytes()
            )

            second = execute_prediction_pipeline(
                model=model,
                input_frame=(
                    prediction_input_frame()
                ),
                input_scope="unit_test",
                allow_non_deployment_model=True,
                database_path=database_path,
                output_csv_path=csv_path,
            )

            second_csv = (
                csv_path.read_bytes()
            )

            audit = audit_prediction_database(
                database_path
            )

        self.assertIsNotNone(
            first.store_result
        )

        self.assertIsNotNone(
            second.store_result
        )

        self.assertTrue(
            first.store_result.inserted
        )

        self.assertFalse(
            second.store_result.inserted
        )

        self.assertEqual(
            first.store_result.run_id,
            second.store_result.run_id,
        )

        self.assertEqual(
            first_csv,
            second_csv,
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
            3,
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
            / "phase3_prediction_pipeline.py"
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
            "Use --smoke-test",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
