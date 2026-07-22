from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.phase3_inner_search as inner_search
from scripts.phase3_dataset import (
    Phase3DatasetBundle,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def synthetic_bundle() -> Phase3DatasetBundle:
    rows = []

    encoded_to_name = {
        encoded: name
        for name, encoded in (
            CLASS_MAPPING.items()
        )
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

    return Phase3DatasetBundle(
        X=frame[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].copy(),
        y=frame["encoded"].to_numpy(
            dtype=int
        ),
        groups=frame[
            "subject_id"
        ].to_numpy(),
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
        data_path=Path(
            "synthetic_model_input.csv"
        ),
        schema_path=Path(
            "synthetic_schema.json"
        ),
        protocol_path=Path(
            "synthetic_protocol.json"
        ),
        data_sha256="data-hash",
        schema_sha256="schema-hash",
        protocol_sha256="protocol-hash",
    )


def synthetic_manifest() -> dict:
    splits = []

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
        "row_count": 80,
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
                "data-hash"
            ),
            "protocol_sha256": (
                "protocol-hash"
            ),
        },
        "splits": splits,
    }


def tiny_registry() -> dict:
    return {
        "random_seed": 42,
        "primary_metric": "macro_f1",
        "models": {
            "dummy_prior": {
                "estimator": (
                    "sklearn.dummy."
                    "DummyClassifier"
                ),
                "eligible_for_selection": (
                    False
                ),
                "preprocessing": (
                    "passthrough"
                ),
                "complexity_rank": 0,
                "fixed_parameters": {
                    "strategy": "prior",
                    "random_state": 42,
                },
                "parameter_grid": [
                    {}
                ],
            },
            "logistic_regression": {
                "estimator": (
                    "sklearn.linear_model."
                    "LogisticRegression"
                ),
                "eligible_for_selection": (
                    True
                ),
                "preprocessing": (
                    "standard_scaler"
                ),
                "complexity_rank": 1,
                "fixed_parameters": {
                    "solver": "lbfgs",
                    "l1_ratio": 0.0,
                    "class_weight": (
                        "balanced"
                    ),
                    "max_iter": 1000,
                    "random_state": 42,
                },
                "parameter_grid": [
                    {
                        "classifier__C": [
                            0.1,
                            1.0,
                        ]
                    }
                ],
            },
        },
    }


def candidate_stub(
    model_name: str,
    mean_macro_f1: float,
    std_macro_f1: float,
    mean_balanced_accuracy: float,
    mean_log_loss: float,
    complexity_rank: int,
    candidate_index: int,
    eligible: bool = True,
) -> dict:
    return {
        "model_name": model_name,
        "candidate_index": (
            candidate_index
        ),
        "candidate_id": (
            f"{model_name}__"
            f"candidate_{candidate_index:03d}"
        ),
        "candidate_parameters": {},
        "eligible_for_selection": eligible,
        "complexity_rank": (
            complexity_rank
        ),
        "fold_count": 3,
        "fold_metrics": [],
        "aggregate": {
            "mean_macro_f1": (
                mean_macro_f1
            ),
            "std_macro_f1": (
                std_macro_f1
            ),
            "mean_balanced_accuracy": (
                mean_balanced_accuracy
            ),
            "mean_multiclass_log_loss": (
                mean_log_loss
            ),
        },
    }


class Phase3InnerSearchTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.bundle = synthetic_bundle()
        self.manifest = synthetic_manifest()
        self.registry = tiny_registry()

    def test_direct_script_entrypoint_imports(
        self,
    ) -> None:
        project_root = Path(
            __file__
        ).resolve().parents[1]

        script_path = (
            project_root
            / "scripts"
            / "phase3_inner_search.py"
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

    def test_manifest_contract_has_four_outer_folds(
        self,
    ) -> None:
        grouped = (
            inner_search
            .validate_local_split_manifest(
                manifest=self.manifest,
                bundle=self.bundle,
            )
        )

        self.assertEqual(
            list(grouped),
            [1, 2, 3, 4],
        )

        self.assertTrue(
            all(
                len(splits) == 3
                for splits in (
                    grouped.values()
                )
            )
        )

    def test_manifest_hash_mismatch_is_rejected(
        self,
    ) -> None:
        invalid = json.loads(
            json.dumps(self.manifest)
        )

        invalid["source"][
            "model_input_sha256"
        ] = "incorrect-hash"

        with self.assertRaisesRegex(
            ValueError,
            "model-input hash",
        ):
            inner_search.validate_local_split_manifest(
                manifest=invalid,
                bundle=self.bundle,
            )

    def test_inner_search_evaluates_every_requested_candidate(
        self,
    ) -> None:
        result = inner_search.run_inner_search(
            bundle=self.bundle,
            manifest=self.manifest,
            registry=self.registry,
            outer_folds=[1],
        )

        outer = result[
            "outer_results"
        ][0]

        self.assertEqual(
            outer[
                "evaluated_candidate_count"
            ],
            3,
        )

        self.assertTrue(
            all(
                candidate[
                    "fold_count"
                ] == 3
                for candidate in outer[
                    "all_candidate_results"
                ]
            )
        )

    def test_non_selectable_baseline_is_never_selected(
        self,
    ) -> None:
        result = inner_search.run_inner_search(
            bundle=self.bundle,
            manifest=self.manifest,
            registry=self.registry,
            outer_folds=[1],
        )

        selected = result[
            "outer_results"
        ][0]["selected_candidate"]

        self.assertEqual(
            selected["model_name"],
            "logistic_regression",
        )

    def test_ranking_prefers_lower_standard_deviation(
        self,
    ) -> None:
        candidates = [
            candidate_stub(
                model_name="model_a",
                mean_macro_f1=0.8,
                std_macro_f1=0.05,
                mean_balanced_accuracy=0.8,
                mean_log_loss=0.5,
                complexity_rank=1,
                candidate_index=1,
            ),
            candidate_stub(
                model_name="model_b",
                mean_macro_f1=0.8,
                std_macro_f1=0.02,
                mean_balanced_accuracy=0.8,
                mean_log_loss=0.5,
                complexity_rank=2,
                candidate_index=1,
            ),
        ]

        ranked = (
            inner_search
            .rank_selectable_candidates(
                candidates
            )
        )

        self.assertEqual(
            ranked[0]["model_name"],
            "model_b",
        )

    def test_ranking_prefers_simpler_model_after_metric_ties(
        self,
    ) -> None:
        candidates = [
            candidate_stub(
                model_name="complex_model",
                mean_macro_f1=0.8,
                std_macro_f1=0.02,
                mean_balanced_accuracy=0.8,
                mean_log_loss=0.5,
                complexity_rank=4,
                candidate_index=1,
            ),
            candidate_stub(
                model_name="simple_model",
                mean_macro_f1=0.8,
                std_macro_f1=0.02,
                mean_balanced_accuracy=0.8,
                mean_log_loss=0.5,
                complexity_rank=1,
                candidate_index=1,
            ),
        ]

        ranked = (
            inner_search
            .rank_selectable_candidates(
                candidates
            )
        )

        self.assertEqual(
            ranked[0]["model_name"],
            "simple_model",
        )

    def test_test_partition_is_never_requested(
        self,
    ) -> None:
        original_selector = (
            inner_search
            .select_subject_partition
        )

        calls = []

        def guarded_selector(*args, **kwargs):
            name = kwargs["name"]
            subjects = tuple(
                kwargs["subjects"]
            )

            calls.append(
                (name, subjects)
            )

            if name == "test":
                raise AssertionError(
                    "Test partition was accessed."
                )

            if 0 in subjects:
                raise AssertionError(
                    "Outer test subject entered "
                    "the dataset selector."
                )

            return original_selector(
                *args,
                **kwargs,
            )

        with patch.object(
            inner_search,
            "select_subject_partition",
            side_effect=guarded_selector,
        ):
            inner_search.run_inner_search(
                bundle=self.bundle,
                manifest=self.manifest,
                registry=self.registry,
                outer_folds=[1],
            )

        self.assertTrue(calls)

        self.assertEqual(
            {
                name
                for name, _ in calls
            },
            {
                "train",
                "validation",
            },
        )

    def test_repeated_search_is_deterministic(
        self,
    ) -> None:
        first = inner_search.run_inner_search(
            bundle=self.bundle,
            manifest=self.manifest,
            registry=self.registry,
            outer_folds=[1],
        )

        second = inner_search.run_inner_search(
            bundle=self.bundle,
            manifest=self.manifest,
            registry=self.registry,
            outer_folds=[1],
        )

        first_text = json.dumps(
            first,
            sort_keys=True,
            separators=(",", ":"),
        )

        second_text = json.dumps(
            second,
            sort_keys=True,
            separators=(",", ":"),
        )

        self.assertEqual(
            first_text,
            second_text,
        )

    def test_result_contains_no_test_metrics_or_predictions(
        self,
    ) -> None:
        result = inner_search.run_inner_search(
            bundle=self.bundle,
            manifest=self.manifest,
            registry=self.registry,
            outer_folds=[1],
        )

        self.assertFalse(
            result[
                "test_metrics_included"
            ]
        )

        self.assertFalse(
            result[
                "test_predictions_included"
            ]
        )

        self.assertFalse(
            result[
                "test_feature_matrix_loaded"
            ]
        )

        serialized = json.dumps(
            result,
            sort_keys=True,
        )

        self.assertNotIn(
            '"test_metrics"',
            serialized,
        )

        self.assertNotIn(
            '"test_predictions"',
            serialized,
        )


if __name__ == "__main__":
    unittest.main()
