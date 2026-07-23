from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.phase3_dataset import (
    load_phase3_dataset,
)
from scripts.phase3_final_refit import (
    DEFAULT_REGISTRY_PATH,
    FINAL_MODEL_ARTIFACT_TYPE,
    build_final_manifest_frame,
    build_final_refit_logo_splits,
    build_selection_artifact,
    build_selection_summary_frame,
    evaluate_final_refit_candidate_space,
    load_final_refit_payload,
    load_registry,
    run_final_refit,
    validate_final_refit_payload,
    write_final_manifest,
    write_selection_artifacts,
)


class MinimalPipeline:
    def predict(
        self,
        X,
    ):
        return np.zeros(
            len(X),
            dtype=int,
        )

    def predict_proba(
        self,
        X,
    ):
        probabilities = np.zeros(
            (len(X), 5),
            dtype=float,
        )

        probabilities[:, 0] = 1.0

        return probabilities


class Phase3FinalRefitArtifactTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        try:
            cls.bundle = (
                load_phase3_dataset()
            )
        except FileNotFoundError as error:
            raise unittest.SkipTest(
                "Local Phase 3 model-input artifacts "
                "are not available in this environment: "
                f"{error}"
            ) from error

        cls.registry = (
            load_registry(
                DEFAULT_REGISTRY_PATH
            )
        )

        cls.splits = (
            build_final_refit_logo_splits(
                cls.bundle
            )
        )

        cls.partial_result = (
            evaluate_final_refit_candidate_space(
                bundle=cls.bundle,
                registry=cls.registry,
                logo_splits=cls.splits,
                model_names=[
                    "logistic_regression"
                ],
                max_candidates_per_model=1,
            )
        )

    def valid_payload(
        self,
        complete: bool = True,
    ) -> dict:
        return {
            "metadata": {
                "artifact_type": (
                    FINAL_MODEL_ARTIFACT_TYPE
                ),
                "inference_ready": True,
                "deployment_ready": False,
                "scientific_reporting_allowed": (
                    False
                ),
                "selection_candidate_space_complete": (
                    complete
                ),
                "training_subjects": [
                    0,
                    1,
                    2,
                    3,
                ],
                "excluded_test_subjects": [],
                "training_row_count": 3921,
                "test_feature_matrix_loaded": (
                    False
                ),
                "outer_test_metrics_loaded": (
                    False
                ),
                "outer_test_predictions_loaded": (
                    False
                ),
                "hyperparameter_search_performed": (
                    True
                ),
                "feature_names": list(
                    self.bundle.feature_names
                ),
                "feature_count": len(
                    self.bundle.feature_names
                ),
                "class_mapping": dict(
                    self.bundle.class_mapping
                ),
            },
            "pipeline": MinimalPipeline(),
        }

    def test_partial_candidate_space_is_marked(
        self,
    ) -> None:
        result = self.partial_result

        self.assertFalse(
            result[
                "candidate_space_complete"
            ]
        )

        self.assertEqual(
            result[
                "evaluated_candidate_count"
            ],
            1,
        )

        self.assertEqual(
            result["fold_count"],
            4,
        )

        self.assertEqual(
            result[
                "selected_candidate"
            ][
                "candidate_id"
            ],
            (
                "logistic_regression"
                "__candidate_001"
            ),
        )

    def test_selection_artifacts_are_byte_deterministic(
        self,
    ) -> None:
        artifact = (
            build_selection_artifact(
                bundle=self.bundle,
                registry=self.registry,
                registry_path=(
                    DEFAULT_REGISTRY_PATH
                ),
                logo_splits=self.splits,
                selection_result=(
                    self.partial_result
                ),
            )
        )

        frame = (
            build_selection_summary_frame(
                artifact
            )
        )

        self.assertEqual(
            frame.shape[0],
            1,
        )

        self.assertTrue(
            bool(
                frame.loc[
                    0,
                    "is_selected",
                ]
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first_json = (
                root / "first.json"
            )
            first_csv = (
                root / "first.csv"
            )
            second_json = (
                root / "second.json"
            )
            second_csv = (
                root / "second.csv"
            )

            write_selection_artifacts(
                artifact=artifact,
                json_output_path=(
                    first_json
                ),
                csv_output_path=(
                    first_csv
                ),
            )

            write_selection_artifacts(
                artifact=artifact,
                json_output_path=(
                    second_json
                ),
                csv_output_path=(
                    second_csv
                ),
            )

            self.assertEqual(
                first_json.read_bytes(),
                second_json.read_bytes(),
            )

            self.assertEqual(
                first_csv.read_bytes(),
                second_csv.read_bytes(),
            )

    def test_payload_contract_blocks_invalid_deployment(
        self,
    ) -> None:
        payload = self.valid_payload()

        validate_final_refit_payload(
            payload=payload,
            expected_feature_names=(
                self.bundle.feature_names
            ),
            expected_class_mapping=(
                self.bundle.class_mapping
            ),
            require_complete_selection=True,
        )

        invalid = copy.deepcopy(
            payload
        )

        invalid[
            "metadata"
        ][
            "deployment_ready"
        ] = True

        with self.assertRaisesRegex(
            ValueError,
            "cannot be deployment-ready",
        ):
            validate_final_refit_payload(
                payload=invalid
            )

        incomplete = (
            self.valid_payload(
                complete=False
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "complete candidate search",
        ):
            validate_final_refit_payload(
                payload=incomplete,
                require_complete_selection=(
                    True
                ),
            )

        validate_final_refit_payload(
            payload=incomplete,
            require_complete_selection=(
                False
            ),
        )

    def test_partial_refit_persists_roundtrippable_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                selection,
                manifest,
            ) = run_final_refit(
                bundle=self.bundle,
                registry=self.registry,
                registry_path=(
                    DEFAULT_REGISTRY_PATH
                ),
                model_output_path=(
                    root / "model.joblib"
                ),
                selection_json_path=(
                    root / "selection.json"
                ),
                selection_csv_path=(
                    root / "selection.csv"
                ),
                manifest_json_path=(
                    root / "manifest.json"
                ),
                manifest_csv_path=(
                    root / "manifest.csv"
                ),
                model_names=[
                    "logistic_regression"
                ],
                max_candidates_per_model=1,
                require_complete_selection=(
                    False
                ),
            )

            payload = (
                load_final_refit_payload(
                    path=(
                        root / "model.joblib"
                    ),
                    expected_feature_names=(
                        self.bundle.feature_names
                    ),
                    expected_class_mapping=(
                        self.bundle.class_mapping
                    ),
                    require_complete_selection=(
                        False
                    ),
                )
            )

            model = manifest[
                "models"
            ][0]

            self.assertEqual(
                selection[
                    "selection_result"
                ][
                    "selected_candidate"
                ][
                    "candidate_id"
                ],
                (
                    "logistic_regression"
                    "__candidate_001"
                ),
            )

            self.assertEqual(
                model[
                    "training_row_count"
                ],
                3921,
            )

            self.assertEqual(
                model[
                    "training_subjects"
                ],
                [0, 1, 2, 3],
            )

            self.assertTrue(
                model[
                    "reload_prediction_match"
                ]
            )

            self.assertTrue(
                model[
                    "reload_probability_match"
                ]
            )

            self.assertTrue(
                payload[
                    "metadata"
                ][
                    "inference_ready"
                ]
            )

            self.assertFalse(
                payload[
                    "metadata"
                ][
                    "deployment_ready"
                ]
            )

    def test_final_manifest_is_byte_deterministic(
        self,
    ) -> None:
        manifest = {
            "models": [
                {
                    "model_name": (
                        "logistic_regression"
                    ),
                    "candidate_id": (
                        "logistic_regression"
                        "__candidate_001"
                    ),
                    "candidate_parameters": {
                        "classifier__C": 0.1
                    },
                    "training_subjects": [
                        0,
                        1,
                        2,
                        3,
                    ],
                    "training_row_count": (
                        3921
                    ),
                    "feature_count": 28,
                    "validation_fold_count": (
                        4
                    ),
                    "selection_validation_summary": {
                        "mean_macro_f1": (
                            0.5
                        ),
                        "std_macro_f1": (
                            0.1
                        ),
                    },
                    "model_file_path": (
                        "model.joblib"
                    ),
                    "model_file_size_bytes": (
                        100
                    ),
                    "model_file_sha256": (
                        "a" * 64
                    ),
                    "reload_prediction_match": (
                        True
                    ),
                    "reload_probability_match": (
                        True
                    ),
                    "inference_ready": True,
                    "deployment_ready": False,
                    "scientific_reporting_allowed": (
                        False
                    ),
                }
            ]
        }

        frame = (
            build_final_manifest_frame(
                manifest
            )
        )

        self.assertEqual(
            frame.shape,
            (1, 17),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first_json = (
                root / "first.json"
            )
            first_csv = (
                root / "first.csv"
            )
            second_json = (
                root / "second.json"
            )
            second_csv = (
                root / "second.csv"
            )

            write_final_manifest(
                manifest=manifest,
                json_output_path=(
                    first_json
                ),
                csv_output_path=(
                    first_csv
                ),
            )

            write_final_manifest(
                manifest=manifest,
                json_output_path=(
                    second_json
                ),
                csv_output_path=(
                    second_csv
                ),
            )

            self.assertEqual(
                first_json.read_bytes(),
                second_json.read_bytes(),
            )

            self.assertEqual(
                first_csv.read_bytes(),
                second_csv.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
