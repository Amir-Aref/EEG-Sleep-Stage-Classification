from __future__ import annotations

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
from scripts.phase3_selection_artifacts import (
    build_selection_artifact,
    build_selection_summary_frame,
    validate_inner_search_result_safety,
    write_selection_artifacts,
)


def aggregate_metrics(
    macro_f1: float,
) -> dict[str, float]:
    return {
        "mean_macro_f1": macro_f1,
        "std_macro_f1": 0.01,
        "mean_balanced_accuracy": (
            macro_f1
        ),
        "std_balanced_accuracy": 0.01,
        "mean_weighted_f1": macro_f1,
        "std_weighted_f1": 0.01,
        "mean_accuracy": macro_f1,
        "std_accuracy": 0.01,
        "mean_cohen_kappa": (
            macro_f1 - 0.1
        ),
        "std_cohen_kappa": 0.01,
        "mean_multiclass_log_loss": (
            1.0 - macro_f1
        ),
        "std_multiclass_log_loss": 0.01,
    }


def candidate(
    model_name: str,
    candidate_index: int,
    macro_f1: float,
    selectable: bool,
    complexity_rank: int,
) -> dict:
    return {
        "model_name": model_name,
        "candidate_index": candidate_index,
        "candidate_id": (
            f"{model_name}__"
            f"candidate_{candidate_index:03d}"
        ),
        "candidate_parameters": {
            "classifier__value": (
                candidate_index
            )
        },
        "eligible_for_selection": (
            selectable
        ),
        "complexity_rank": (
            complexity_rank
        ),
        "fold_count": 3,
        "fold_metrics": [],
        "aggregate": aggregate_metrics(
            macro_f1
        ),
    }


def selection_result(
    complete: bool = True,
) -> dict:
    dummy = candidate(
        model_name="dummy_prior",
        candidate_index=1,
        macro_f1=0.20,
        selectable=False,
        complexity_rank=0,
    )

    logistic_one = candidate(
        model_name="logistic_regression",
        candidate_index=1,
        macro_f1=0.72,
        selectable=True,
        complexity_rank=1,
    )

    logistic_two = candidate(
        model_name="logistic_regression",
        candidate_index=2,
        macro_f1=0.68,
        selectable=True,
        complexity_rank=1,
    )

    ranked_one = dict(logistic_one)
    ranked_one["selection_rank"] = 1

    ranked_two = dict(logistic_two)
    ranked_two["selection_rank"] = 2

    return {
        "schema_version": "1.0.0",
        "primary_metric": "macro_f1",
        "selection_partition": (
            "validation"
        ),
        "test_metrics_included": False,
        "test_predictions_included": False,
        "test_feature_matrix_loaded": False,
        "candidate_space_complete": complete,
        "evaluated_outer_folds": [1],
        "evaluated_models": [
            "dummy_prior",
            "logistic_regression",
        ],
        "outer_results": [
            {
                "outer_fold": 1,
                "test_subjects": [0],
                "outer_development_subjects": [
                    1,
                    2,
                    3,
                ],
                "evaluated_candidate_count": 3,
                "selected_candidate": {
                    "selection_rank": 1,
                    "model_name": (
                        "logistic_regression"
                    ),
                    "candidate_index": 1,
                    "candidate_id": (
                        "logistic_regression__"
                        "candidate_001"
                    ),
                    "candidate_parameters": {
                        "classifier__value": 1
                    },
                    "complexity_rank": 1,
                    "aggregate": (
                        aggregate_metrics(
                            0.72
                        )
                    ),
                },
                "ranked_selectable_candidates": [
                    ranked_one,
                    ranked_two,
                ],
                "all_candidate_results": [
                    dummy,
                    logistic_one,
                    logistic_two,
                ],
            }
        ],
    }


def write_source_file(
    path: Path,
    content: str,
) -> None:
    path.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )


def synthetic_bundle(
    root: Path,
) -> Phase3DatasetBundle:
    data_path = root / "input.csv"
    schema_path = root / "schema.json"
    protocol_path = root / "protocol.json"

    write_source_file(
        data_path,
        "synthetic-data\n",
    )

    write_source_file(
        schema_path,
        '{"schema":1}\n',
    )

    write_source_file(
        protocol_path,
        '{"protocol":1}\n',
    )

    return Phase3DatasetBundle(
        X=pd.DataFrame(
            {
                "feature": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                ]
            }
        ),
        y=np.array(
            [0, 1, 2, 3],
            dtype=int,
        ),
        groups=np.array(
            [0, 1, 2, 3],
            dtype=int,
        ),
        identifiers=pd.DataFrame(
            {
                "subject_id": [
                    0,
                    1,
                    2,
                    3,
                ]
            }
        ),
        quality=pd.DataFrame(
            {
                "quality_issue_flag": [
                    False,
                    False,
                    False,
                    False,
                ]
            }
        ),
        target_names=np.array(
            [
                "Wake",
                "N1",
                "N2",
                "N3",
            ],
            dtype=object,
        ),
        row_indices=np.arange(
            4,
            dtype=int,
        ),
        feature_names=("feature",),
        identifier_columns=(
            "subject_id",
        ),
        quality_columns=(
            "quality_issue_flag",
        ),
        class_mapping={
            "Wake": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
        },
        group_column="subject_id",
        target_column=(
            "sleep_stage_encoded"
        ),
        target_name_column="sleep_stage",
        source_column_count=4,
        data_path=data_path,
        schema_path=schema_path,
        protocol_path=protocol_path,
        data_sha256=(
            "synthetic-data-hash"
        ),
        schema_sha256=(
            "synthetic-schema-hash"
        ),
        protocol_sha256=(
            "synthetic-protocol-hash"
        ),
    )


class Phase3SelectionArtifactTests(
    unittest.TestCase
):
    def build_artifact(
        self,
        root: Path,
        complete: bool = True,
        require_complete: bool = True,
    ) -> dict:
        manifest_path = (
            root / "manifest.json"
        )

        registry_path = (
            root / "registry.json"
        )

        write_source_file(
            manifest_path,
            '{"manifest":1}\n',
        )

        write_source_file(
            registry_path,
            '{"registry":1}\n',
        )

        return build_selection_artifact(
            result=selection_result(
                complete=complete
            ),
            bundle=synthetic_bundle(root),
            split_manifest_path=(
                manifest_path
            ),
            registry_path=registry_path,
            require_complete=require_complete,
        )

    def test_partial_result_is_rejected_when_complete_required(
        self,
    ) -> None:
        result = selection_result(
            complete=False
        )

        with self.assertRaisesRegex(
            ValueError,
            "complete candidate space",
        ):
            validate_inner_search_result_safety(
                result=result,
                require_complete=True,
            )

    def test_safe_partial_artifact_can_be_created_for_smoke_testing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.build_artifact(
                root=Path(directory),
                complete=False,
                require_complete=False,
            )

        self.assertFalse(
            artifact[
                "candidate_space_complete"
            ]
        )

    def test_forbidden_test_payload_is_rejected(
        self,
    ) -> None:
        result = selection_result()
        result["outer_results"][0][
            "test_metrics"
        ] = {
            "macro_f1": 0.99
        }

        with self.assertRaisesRegex(
            ValueError,
            "Forbidden test payload",
        ):
            validate_inner_search_result_safety(
                result=result,
                require_complete=True,
            )

    def test_artifact_contains_source_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.build_artifact(
                Path(directory)
            )

        source = artifact["source"]

        self.assertEqual(
            len(
                source[
                    "split_manifest_sha256"
                ]
            ),
            64,
        )

        self.assertEqual(
            len(
                source[
                    "model_registry_sha256"
                ]
            ),
            64,
        )

        self.assertFalse(
            artifact[
                "test_access_contract"
            ]["test_metrics_included"]
        )

    def test_summary_has_one_row_per_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.build_artifact(
                Path(directory)
            )

        summary = (
            build_selection_summary_frame(
                artifact
            )
        )

        self.assertEqual(
            len(summary),
            3,
        )

        self.assertEqual(
            summary[
                "candidate_id"
            ].nunique(),
            3,
        )

    def test_summary_marks_exactly_one_selected_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.build_artifact(
                Path(directory)
            )

        summary = (
            build_selection_summary_frame(
                artifact
            )
        )

        selected = summary.loc[
            summary["is_selected"]
        ]

        self.assertEqual(
            len(selected),
            1,
        )

        self.assertEqual(
            selected.iloc[0][
                "candidate_id"
            ],
            (
                "logistic_regression__"
                "candidate_001"
            ),
        )

    def test_artifact_writes_are_byte_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            artifact = self.build_artifact(
                root
            )

            json_path = (
                root / "output.json"
            )

            csv_path = (
                root / "output.csv"
            )

            write_selection_artifacts(
                artifact=artifact,
                json_output_path=json_path,
                csv_output_path=csv_path,
            )

            first_json = (
                json_path.read_bytes()
            )

            first_csv = (
                csv_path.read_bytes()
            )

            write_selection_artifacts(
                artifact=artifact,
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

            loaded = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded["primary_metric"],
                "macro_f1",
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
            / "phase3_selection_artifacts.py"
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
