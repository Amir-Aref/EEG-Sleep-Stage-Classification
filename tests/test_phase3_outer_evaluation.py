from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase3_dataset import (
    Phase3DatasetBundle,
)
from scripts.phase3_outer_evaluation import (
    build_outer_summary_frame,
    run_outer_evaluation,
    validate_outer_evaluation_inputs,
    write_outer_evaluation_artifacts,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256(
        path.read_bytes()
    )

    return digest.hexdigest()


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def synthetic_bundle(
    root: Path,
) -> Phase3DatasetBundle:
    rows = []

    encoded_to_name = {
        encoded: name
        for name, encoded
        in CLASS_MAPPING.items()
    }

    for subject_id in range(4):
        for encoded in range(5):
            for repetition in range(4):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "recording_id": (
                            f"recording_{subject_id}"
                        ),
                        "night": 1,
                        "epoch_id": (
                            encoded * 10
                            + repetition
                        ),
                        "encoded": encoded,
                        "stage": (
                            encoded_to_name[
                                encoded
                            ]
                        ),
                        "feature_1": (
                            encoded
                            + subject_id * 0.01
                            + repetition * 0.001
                        ),
                        "feature_2": (
                            encoded**2
                            + repetition * 0.01
                        ),
                        "feature_3": (
                            np.sin(encoded + 1)
                            + subject_id * 0.001
                        ),
                    }
                )

    frame = pd.DataFrame(rows)

    data_path = root / "input.csv"
    schema_path = root / "schema.json"
    protocol_path = root / "protocol.json"

    data_path.write_text(
        "synthetic-data\n",
        encoding="utf-8",
        newline="\n",
    )

    schema_path.write_text(
        '{"schema":1}\n',
        encoding="utf-8",
        newline="\n",
    )

    protocol_path.write_text(
        '{"protocol":1}\n',
        encoding="utf-8",
        newline="\n",
    )

    return Phase3DatasetBundle(
        X=frame[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].copy(),
        y=frame[
            "encoded"
        ].to_numpy(dtype=int),
        groups=frame[
            "subject_id"
        ].to_numpy(dtype=int),
        identifiers=frame[
            [
                "subject_id",
                "recording_id",
                "night",
                "epoch_id",
            ]
        ].copy(),
        quality=pd.DataFrame(
            {
                "quality_issue_flag": (
                    np.zeros(
                        len(frame),
                        dtype=bool,
                    )
                )
            }
        ),
        target_names=frame[
            "stage"
        ].to_numpy(dtype=object),
        row_indices=np.arange(
            len(frame),
            dtype=int,
        ),
        feature_names=(
            "feature_1",
            "feature_2",
            "feature_3",
        ),
        identifier_columns=(
            "subject_id",
            "recording_id",
            "night",
            "epoch_id",
        ),
        quality_columns=(
            "quality_issue_flag",
        ),
        class_mapping=dict(
            CLASS_MAPPING
        ),
        group_column="subject_id",
        target_column=(
            "sleep_stage_encoded"
        ),
        target_name_column="sleep_stage",
        source_column_count=10,
        data_path=data_path,
        schema_path=schema_path,
        protocol_path=protocol_path,
        data_sha256="data-hash",
        schema_sha256="schema-hash",
        protocol_sha256="protocol-hash",
    )


def synthetic_manifest(
    bundle: Phase3DatasetBundle,
) -> dict:
    subjects = [0, 1, 2, 3]
    splits = []

    for outer_fold, test_subject in enumerate(
        subjects,
        start=1,
    ):
        development = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        for inner_fold, validation_subject in enumerate(
            development,
            start=1,
        ):
            train_subjects = [
                subject
                for subject in development
                if subject
                != validation_subject
            ]

            splits.append(
                {
                    "split_id": (
                        f"outer_{outer_fold:02d}_"
                        f"inner_{inner_fold:02d}"
                    ),
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "outer_development_subjects": (
                        development
                    ),
                    "train_subjects": (
                        train_subjects
                    ),
                    "validation_subjects": [
                        validation_subject
                    ],
                    "test_subjects": [
                        test_subject
                    ],
                }
            )

    return {
        "schema_version": "1.0.0",
        "row_count": bundle.row_count,
        "subject_count": 4,
        "subjects": subjects,
        "outer_fold_count": 4,
        "inner_fold_count_per_outer": 3,
        "total_split_count": 12,
        "group_column": "subject_id",
        "target_column": (
            "sleep_stage_encoded"
        ),
        "target_name_column": (
            "sleep_stage"
        ),
        "class_mapping": dict(
            CLASS_MAPPING
        ),
        "source": {
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "protocol_sha256": (
                bundle.protocol_sha256
            ),
        },
        "splits": splits,
    }


def tiny_registry() -> dict:
    return {
        "primary_metric": "macro_f1",
        "models": {
            "logistic_regression": {
                "estimator": (
                    "sklearn.linear_model."
                    "LogisticRegression"
                ),
                "eligible_for_selection": True,
                "preprocessing": (
                    "standard_scaler"
                ),
                "complexity_rank": 1,
                "fixed_parameters": {
                    "solver": "lbfgs",
                    "l1_ratio": 0.0,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": 42,
                },
                "parameter_grid": [
                    {
                        "classifier__C": [
                            0.1
                        ]
                    }
                ],
            }
        },
    }


def candidate_result() -> dict:
    return {
        "model_name": (
            "logistic_regression"
        ),
        "candidate_index": 1,
        "candidate_id": (
            "logistic_regression__"
            "candidate_001"
        ),
        "candidate_parameters": {
            "classifier__C": 0.1
        },
        "eligible_for_selection": True,
        "complexity_rank": 1,
        "fold_count": 3,
        "fold_metrics": [],
        "aggregate": {
            "mean_macro_f1": 0.8,
            "std_macro_f1": 0.01,
            "mean_balanced_accuracy": 0.8,
            "std_balanced_accuracy": 0.01,
            "mean_weighted_f1": 0.8,
            "std_weighted_f1": 0.01,
            "mean_accuracy": 0.8,
            "std_accuracy": 0.01,
            "mean_cohen_kappa": 0.7,
            "std_cohen_kappa": 0.01,
            "mean_multiclass_log_loss": 0.5,
            "std_multiclass_log_loss": 0.01,
        },
    }


def synthetic_selection_artifact(
    bundle: Phase3DatasetBundle,
    manifest_path: Path,
    registry_path: Path,
) -> dict:
    outer_results = []

    subjects = [0, 1, 2, 3]

    for outer_fold, test_subject in enumerate(
        subjects,
        start=1,
    ):
        development = [
            subject
            for subject in subjects
            if subject != test_subject
        ]

        candidate = candidate_result()

        ranked = dict(candidate)
        ranked["selection_rank"] = 1

        outer_results.append(
            {
                "outer_fold": outer_fold,
                "test_subjects": [
                    test_subject
                ],
                "outer_development_subjects": (
                    development
                ),
                "evaluated_candidate_count": 1,
                "selected_candidate": {
                    "selection_rank": 1,
                    "model_name": (
                        candidate["model_name"]
                    ),
                    "candidate_index": 1,
                    "candidate_id": (
                        candidate[
                            "candidate_id"
                        ]
                    ),
                    "candidate_parameters": (
                        candidate[
                            "candidate_parameters"
                        ]
                    ),
                    "complexity_rank": 1,
                    "aggregate": (
                        candidate["aggregate"]
                    ),
                },
                "ranked_selectable_candidates": [
                    ranked
                ],
                "all_candidate_results": [
                    candidate
                ],
            }
        )

    return {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_inner_model_selection"
        ),
        "candidate_space_complete": True,
        "source": {
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_sha256": (
                bundle.schema_sha256
            ),
            "evaluation_protocol_sha256": (
                bundle.protocol_sha256
            ),
            "split_manifest_sha256": (
                sha256_file(manifest_path)
            ),
            "model_registry_sha256": (
                sha256_file(registry_path)
            ),
        },
        "selection_result": {
            "schema_version": "1.0.0",
            "primary_metric": "macro_f1",
            "selection_partition": (
                "validation"
            ),
            "test_metrics_included": False,
            "test_predictions_included": False,
            "test_feature_matrix_loaded": False,
            "candidate_space_complete": True,
            "evaluated_outer_folds": [
                1,
                2,
                3,
                4,
            ],
            "evaluated_models": [
                "logistic_regression"
            ],
            "outer_results": (
                outer_results
            ),
        },
    }


class Phase3OuterEvaluationTests(
    unittest.TestCase
):
    def fixture(self, root: Path):
        bundle = synthetic_bundle(root)

        manifest = synthetic_manifest(
            bundle
        )

        registry = tiny_registry()

        manifest_path = (
            root / "manifest.json"
        )

        registry_path = (
            root / "registry.json"
        )

        write_json(
            manifest_path,
            manifest,
        )

        write_json(
            registry_path,
            registry,
        )

        selection = (
            synthetic_selection_artifact(
                bundle=bundle,
                manifest_path=manifest_path,
                registry_path=registry_path,
            )
        )

        selection_path = (
            root / "selection.json"
        )

        write_json(
            selection_path,
            selection,
        )

        return {
            "bundle": bundle,
            "manifest": manifest,
            "registry": registry,
            "selection": selection,
            "manifest_path": manifest_path,
            "registry_path": registry_path,
            "selection_path": selection_path,
        }

    def run_fixture(
        self,
        root: Path,
        outer_folds=None,
    ):
        fixture = self.fixture(root)

        artifact, predictions = (
            run_outer_evaluation(
                bundle=fixture["bundle"],
                split_manifest=(
                    fixture["manifest"]
                ),
                selection_artifact=(
                    fixture["selection"]
                ),
                registry=fixture["registry"],
                split_manifest_path=(
                    fixture["manifest_path"]
                ),
                selection_artifact_path=(
                    fixture["selection_path"]
                ),
                registry_path=(
                    fixture["registry_path"]
                ),
                outer_folds=outer_folds,
            )
        )

        return (
            fixture,
            artifact,
            predictions,
        )

    def test_frozen_selection_contract_is_accepted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(
                Path(directory)
            )

            grouped, selected = (
                validate_outer_evaluation_inputs(
                    bundle=fixture["bundle"],
                    split_manifest=(
                        fixture["manifest"]
                    ),
                    selection_artifact=(
                        fixture["selection"]
                    ),
                    registry=(
                        fixture["registry"]
                    ),
                    split_manifest_path=(
                        fixture["manifest_path"]
                    ),
                    selection_artifact_path=(
                        fixture["selection_path"]
                    ),
                    registry_path=(
                        fixture["registry_path"]
                    ),
                )
            )

        self.assertEqual(
            list(grouped),
            [1, 2, 3, 4],
        )

        self.assertEqual(
            list(selected),
            [1, 2, 3, 4],
        )

    def test_source_hash_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(
                Path(directory)
            )

            fixture["selection"][
                "source"
            ]["model_registry_sha256"] = (
                "incorrect"
            )

            with self.assertRaisesRegex(
                ValueError,
                "model_registry_sha256",
            ):
                validate_outer_evaluation_inputs(
                    bundle=fixture["bundle"],
                    split_manifest=(
                        fixture["manifest"]
                    ),
                    selection_artifact=(
                        fixture["selection"]
                    ),
                    registry=(
                        fixture["registry"]
                    ),
                    split_manifest_path=(
                        fixture["manifest_path"]
                    ),
                    selection_artifact_path=(
                        fixture["selection_path"]
                    ),
                    registry_path=(
                        fixture["registry_path"]
                    ),
                )

    def test_outer_evaluation_uses_selected_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, artifact, _ = (
                self.run_fixture(
                    Path(directory),
                    outer_folds=[1],
                )
            )

        fold = artifact[
            "outer_results"
        ][0]

        self.assertEqual(
            fold["model_name"],
            "logistic_regression",
        )

        self.assertEqual(
            fold["candidate_id"],
            (
                "logistic_regression__"
                "candidate_001"
            ),
        )

        self.assertEqual(
            fold["candidate_parameters"],
            {
                "classifier__C": 0.1
            },
        )

    def test_development_and_test_subjects_are_disjoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, artifact, _ = (
                self.run_fixture(
                    Path(directory),
                    outer_folds=[1],
                )
            )

        fold = artifact[
            "outer_results"
        ][0]

        self.assertFalse(
            set(
                fold[
                    "outer_development_subjects"
                ]
            )
            & set(fold["test_subjects"])
        )

    def test_complete_evaluation_predicts_every_row_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, artifact, predictions = (
                self.run_fixture(
                    Path(directory)
                )
            )

        self.assertTrue(
            artifact[
                "complete_outer_evaluation"
            ]
        )

        self.assertEqual(
            len(predictions),
            fixture["bundle"].row_count,
        )

        self.assertEqual(
            predictions[
                "source_row_index"
            ].nunique(),
            fixture["bundle"].row_count,
        )

        np.testing.assert_array_equal(
            predictions[
                "source_row_index"
            ].to_numpy(dtype=int),
            np.arange(
                fixture["bundle"].row_count
            ),
        )

    def test_pooled_metrics_cover_every_prediction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, artifact, predictions = (
                self.run_fixture(
                    Path(directory)
                )
            )

        pooled = artifact[
            "aggregate"
        ]["pooled_test_metrics"]

        self.assertEqual(
            pooled["sample_count"],
            len(predictions),
        )

    def test_prediction_frame_is_test_partition_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, predictions = (
                self.run_fixture(
                    Path(directory),
                    outer_folds=[1],
                )
            )

        self.assertEqual(
            set(
                predictions[
                    "partition"
                ].tolist()
            ),
            {"test"},
        )

        self.assertEqual(
            set(
                predictions[
                    "subject_id"
                ].tolist()
            ),
            {0},
        )

        self.assertIn(
            (
                "predict_probability_"
                "argmax_agree"
            ),
            predictions.columns,
        )

    def test_repeated_outer_evaluation_is_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            fixture = self.fixture(root)

            first_artifact, first_predictions = (
                run_outer_evaluation(
                    bundle=fixture["bundle"],
                    split_manifest=(
                        fixture["manifest"]
                    ),
                    selection_artifact=(
                        fixture["selection"]
                    ),
                    registry=(
                        fixture["registry"]
                    ),
                    split_manifest_path=(
                        fixture["manifest_path"]
                    ),
                    selection_artifact_path=(
                        fixture["selection_path"]
                    ),
                    registry_path=(
                        fixture["registry_path"]
                    ),
                )
            )

            second_artifact, second_predictions = (
                run_outer_evaluation(
                    bundle=fixture["bundle"],
                    split_manifest=(
                        fixture["manifest"]
                    ),
                    selection_artifact=(
                        fixture["selection"]
                    ),
                    registry=(
                        fixture["registry"]
                    ),
                    split_manifest_path=(
                        fixture["manifest_path"]
                    ),
                    selection_artifact_path=(
                        fixture["selection_path"]
                    ),
                    registry_path=(
                        fixture["registry_path"]
                    ),
                )
            )

        self.assertEqual(
            json.dumps(
                first_artifact,
                sort_keys=True,
            ),
            json.dumps(
                second_artifact,
                sort_keys=True,
            ),
        )

        pd.testing.assert_frame_equal(
            first_predictions,
            second_predictions,
        )

    def test_artifact_writes_are_byte_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            _, artifact, predictions = (
                self.run_fixture(root)
            )

            json_path = (
                root / "outer.json"
            )

            summary_path = (
                root / "summary.csv"
            )

            predictions_path = (
                root / "predictions.csv"
            )

            write_outer_evaluation_artifacts(
                outer_artifact=artifact,
                predictions=predictions,
                json_output_path=json_path,
                summary_csv_output_path=(
                    summary_path
                ),
                predictions_csv_output_path=(
                    predictions_path
                ),
            )

            first_json = (
                json_path.read_bytes()
            )

            first_summary = (
                summary_path.read_bytes()
            )

            first_predictions = (
                predictions_path.read_bytes()
            )

            write_outer_evaluation_artifacts(
                outer_artifact=artifact,
                predictions=predictions,
                json_output_path=json_path,
                summary_csv_output_path=(
                    summary_path
                ),
                predictions_csv_output_path=(
                    predictions_path
                ),
            )

            self.assertEqual(
                first_json,
                json_path.read_bytes(),
            )

            self.assertEqual(
                first_summary,
                summary_path.read_bytes(),
            )

            self.assertEqual(
                first_predictions,
                predictions_path.read_bytes(),
            )

            summary = build_outer_summary_frame(
                artifact
            )

            self.assertEqual(
                len(summary),
                4,
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
            / "phase3_outer_evaluation.py"
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
