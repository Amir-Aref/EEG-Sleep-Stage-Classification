from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.phase3_dataset import (
    select_subject_partition as real_select_subject_partition,
)
from scripts.phase3_model_artifacts import (
    ROUNDTRIP_PROBABILITY_ATOL,
    build_manifest_frame,
    load_trained_model_payload,
    resolve_artifact_path,
    sha256_file,
    train_and_save_outer_models,
    validate_model_payload,
    verify_roundtrip,
    write_model_manifest,
)
from tests.test_phase3_outer_evaluation import (
    CLASS_MAPPING,
    synthetic_bundle,
    synthetic_manifest,
    synthetic_selection_artifact,
    tiny_registry,
    write_json,
)


class Phase3ModelArtifactTests(
    unittest.TestCase
):
    def fixture(
        self,
        root: Path,
    ) -> dict:
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

        output_directory = (
            root / "models"
        )

        manifest = (
            train_and_save_outer_models(
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
                output_directory=(
                    output_directory
                ),
                outer_folds=outer_folds,
            )
        )

        return (
            fixture,
            output_directory,
            manifest,
        )

    def test_incomplete_selection_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)

            fixture["selection"][
                "candidate_space_complete"
            ] = False

            with self.assertRaisesRegex(
                ValueError,
                "complete",
            ):
                train_and_save_outer_models(
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
                    output_directory=(
                        root / "models"
                    ),
                    outer_folds=[1],
                )

    def test_source_hash_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)

            fixture["selection"][
                "source"
            ]["model_registry_sha256"] = (
                "incorrect"
            )

            with self.assertRaisesRegex(
                ValueError,
                "model_registry_sha256",
            ):
                train_and_save_outer_models(
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
                    output_directory=(
                        root / "models"
                    ),
                    outer_folds=[1],
                )

    def test_training_never_requests_test_partition(
        self,
    ) -> None:
        calls = []

        def tracking_partition(
            *args,
            **kwargs,
        ):
            calls.append(
                {
                    "subjects": tuple(
                        kwargs["subjects"]
                    ),
                    "name": kwargs["name"],
                }
            )

            return real_select_subject_partition(
                *args,
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with patch(
                (
                    "scripts.phase3_model_artifacts."
                    "select_subject_partition"
                ),
                side_effect=tracking_partition,
            ):
                self.run_fixture(
                    root,
                    outer_folds=[1],
                )

        self.assertEqual(
            calls,
            [
                {
                    "subjects": (1, 2, 3),
                    "name": (
                        "outer_development_"
                        "model_fit"
                    ),
                }
            ],
        )

    def test_saved_model_roundtrip_predicts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                fixture,
                output_directory,
                manifest,
            ) = self.run_fixture(
                root,
                outer_folds=[1],
            )

            model_files = list(
                output_directory.glob(
                    "*.joblib"
                )
            )

            self.assertEqual(
                len(model_files),
                1,
            )

            payload = (
                load_trained_model_payload(
                    model_files[0]
                )
            )

            development = (
                real_select_subject_partition(
                    bundle=fixture["bundle"],
                    subjects=(1, 2, 3),
                    name="development",
                    require_all_classes=True,
                )
            )

            predictions = payload[
                "pipeline"
            ].predict(
                development.X
            )

        self.assertEqual(
            len(predictions),
            development.row_count,
        )

        self.assertTrue(
            manifest["models"][0][
                "reload_prediction_match"
            ]
        )

        self.assertTrue(
            manifest["models"][0][
                "reload_probability_match"
            ]
        )

    def test_complete_manifest_covers_all_outer_folds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = (
                self.run_fixture(
                    Path(directory)
                )
            )

        self.assertTrue(
            manifest["complete_model_set"]
        )

        self.assertEqual(
            manifest["model_count"],
            4,
        )

        self.assertEqual(
            [
                model["outer_fold"]
                for model in manifest[
                    "models"
                ]
            ],
            [1, 2, 3, 4],
        )

        self.assertEqual(
            [
                model[
                    "excluded_test_subjects"
                ][0]
                for model in manifest[
                    "models"
                ]
            ],
            [0, 1, 2, 3],
        )

    def test_feature_order_contract_is_enforced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                fixture,
                output_directory,
                _,
            ) = self.run_fixture(
                Path(directory),
                outer_folds=[1],
            )

            model_path = next(
                output_directory.glob(
                    "*.joblib"
                )
            )

            payload = (
                load_trained_model_payload(
                    model_path
                )
            )

            tampered = {
                "metadata": copy.deepcopy(
                    payload["metadata"]
                ),
                "pipeline": payload[
                    "pipeline"
                ],
            }

            tampered["metadata"][
                "feature_names"
            ] = list(
                reversed(
                    tampered["metadata"][
                        "feature_names"
                    ]
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "feature order",
            ):
                validate_model_payload(
                    payload=tampered,
                    expected_feature_names=(
                        fixture["bundle"]
                        .feature_names
                    ),
                    expected_class_mapping=(
                        fixture["bundle"]
                        .class_mapping
                    ),
                )

    def test_manifest_writes_are_byte_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            _, _, manifest = (
                self.run_fixture(
                    root,
                    outer_folds=[1],
                )
            )

            json_path = (
                root / "models.json"
            )

            csv_path = (
                root / "models.csv"
            )

            write_model_manifest(
                manifest=manifest,
                json_output_path=json_path,
                csv_output_path=csv_path,
            )

            first_json = (
                json_path.read_bytes()
            )

            first_csv = (
                csv_path.read_bytes()
            )

            write_model_manifest(
                manifest=manifest,
                json_output_path=json_path,
                csv_output_path=csv_path,
            )

            self.assertEqual(
                first_json,
                json_path.read_bytes(),
            )

            self.assertEqual(
                first_csv,
                csv_path.read_bytes(),
            )

            frame = build_manifest_frame(
                manifest
            )

            self.assertEqual(
                len(frame),
                1,
            )

    def test_model_file_hash_and_paths_are_unique(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = (
                self.run_fixture(
                    Path(directory)
                )
            )

            paths = [
                model["model_file_path"]
                for model in manifest["models"]
            ]

            self.assertEqual(
                len(paths),
                len(set(paths)),
            )

            for model in manifest["models"]:
                model_path = (
                    resolve_artifact_path(
                        model[
                            "model_file_path"
                        ]
                    )
                )

                self.assertTrue(
                    model_path.exists()
                )

                self.assertEqual(
                    sha256_file(model_path),
                    model[
                        "model_file_sha256"
                    ],
                )

                self.assertGreater(
                    model[
                        "model_file_size_bytes"
                    ],
                    0,
                )

    def test_probability_roundtrip_uses_machine_precision_tolerance(
        self,
    ) -> None:
        class TinyClassifier:
            classes_ = np.array(
                [0, 1, 2, 3, 4],
                dtype=int,
            )

        class TinyPipeline:
            def __init__(
                self,
                probabilities,
            ):
                self._probabilities = np.asarray(
                    probabilities,
                    dtype=float,
                )

                self.named_steps = {
                    "classifier": TinyClassifier()
                }

            def predict(self, X):
                return np.argmax(
                    self._probabilities,
                    axis=1,
                )

            def predict_proba(self, X):
                return self._probabilities.copy()

        X = pd.DataFrame(
            {
                "feature": [0.0],
            }
        )

        original_probabilities = np.array(
            [
                [
                    0.6,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ]
            ],
            dtype=float,
        )

        machine_precision_probabilities = (
            original_probabilities.copy()
        )

        machine_precision_probabilities[
            0,
            0,
        ] += np.finfo(float).eps

        machine_precision_probabilities[
            0,
            1,
        ] -= np.finfo(float).eps

        (
            prediction_match,
            probability_match,
        ) = verify_roundtrip(
            original_pipeline=TinyPipeline(
                original_probabilities
            ),
            loaded_pipeline=TinyPipeline(
                machine_precision_probabilities
            ),
            X=X,
            class_mapping=CLASS_MAPPING,
        )

        self.assertTrue(
            prediction_match
        )

        self.assertTrue(
            probability_match
        )

        material_difference_probabilities = (
            original_probabilities.copy()
        )

        material_difference_probabilities[
            0,
            0,
        ] += 1e-10

        material_difference_probabilities[
            0,
            1,
        ] -= 1e-10

        (
            material_prediction_match,
            material_probability_match,
        ) = verify_roundtrip(
            original_pipeline=TinyPipeline(
                original_probabilities
            ),
            loaded_pipeline=TinyPipeline(
                material_difference_probabilities
            ),
            X=X,
            class_mapping=CLASS_MAPPING,
        )

        self.assertTrue(
            material_prediction_match
        )

        self.assertFalse(
            material_probability_match
        )

        self.assertEqual(
            ROUNDTRIP_PROBABILITY_ATOL,
            1e-15,
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
            / "phase3_model_artifacts.py"
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
