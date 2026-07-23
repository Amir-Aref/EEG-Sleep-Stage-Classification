from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase3_dataset import (
    DEFAULT_DATA_PATH,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_SCHEMA_PATH,
    build_dataset_summary,
    load_phase3_dataset,
    select_subject_partition,
    write_dataset_summary,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def write_fixture(
    root: Path,
) -> tuple[Path, Path, Path]:
    rows = []

    for subject_id in range(4):
        for stage, encoded in (
            CLASS_MAPPING.items()
        ):
            for repetition in range(2):
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
                        "sleep_stage": stage,
                        "sleep_stage_encoded": (
                            encoded
                        ),
                        "quality_issue_flag": False,
                        "feature_mean": (
                            subject_id
                            + encoded * 0.1
                            + repetition * 0.01
                        ),
                        "feature_std": (
                            1.0
                            + encoded * 0.2
                        ),
                        "feature_entropy": (
                            0.5
                            + repetition * 0.1
                        ),
                    }
                )

    dataframe = pd.DataFrame(rows)

    data_path = root / "model_input.csv"
    schema_path = root / "schema.json"
    protocol_path = root / "protocol.json"

    dataframe.to_csv(
        data_path,
        index=False,
        lineterminator="\n",
    )

    schema_path.write_text(
        """{
  "selected_feature_count": 3,
  "feature_scaling_applied": false,
  "identifier_columns": [
    "subject_id",
    "recording_id",
    "night",
    "epoch_id"
  ],
  "target_columns": [
    "sleep_stage",
    "sleep_stage_encoded"
  ],
  "quality_columns": [
    "quality_issue_flag"
  ],
  "selected_features": [
    "feature_mean",
    "feature_std",
    "feature_entropy"
  ]
}
""",
        encoding="utf-8",
        newline="\n",
    )

    protocol_path.write_text(
        """{
  "group_column": "subject_id",
  "target_column": "sleep_stage_encoded",
  "target_name_column": "sleep_stage",
  "random_epoch_split_allowed": false,
  "class_mapping": {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4
  }
}
""",
        encoding="utf-8",
        newline="\n",
    )

    return (
        data_path,
        schema_path,
        protocol_path,
    )


class Phase3DatasetTests(unittest.TestCase):
    def load_fixture(
        self,
        root: Path,
        **kwargs,
    ):
        (
            data_path,
            schema_path,
            protocol_path,
        ) = write_fixture(root)

        return load_phase3_dataset(
            data_path=data_path,
            schema_path=schema_path,
            protocol_path=protocol_path,
            **kwargs,
        )

    def test_loads_exact_feature_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.load_fixture(
                Path(directory)
            )

        self.assertEqual(
            bundle.X.shape,
            (40, 3),
        )

        self.assertEqual(
            list(bundle.X.columns),
            [
                "feature_mean",
                "feature_std",
                "feature_entropy",
            ],
        )

        self.assertEqual(
            bundle.row_count,
            40,
        )

        self.assertEqual(
            bundle.feature_count,
            3,
        )

    def test_leakage_columns_never_enter_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.load_fixture(
                Path(directory)
            )

        forbidden = (
            set(bundle.identifier_columns)
            | {
                bundle.target_column,
                bundle.target_name_column,
            }
            | set(bundle.quality_columns)
        )

        self.assertFalse(
            forbidden
            & set(bundle.X.columns)
        )

    def test_non_finite_feature_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                data_path,
                schema_path,
                protocol_path,
            ) = write_fixture(root)

            dataframe = pd.read_csv(
                data_path
            )

            dataframe.loc[
                0,
                "feature_std",
            ] = np.inf

            dataframe.to_csv(
                data_path,
                index=False,
                lineterminator="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "non-finite",
            ):
                load_phase3_dataset(
                    data_path=data_path,
                    schema_path=schema_path,
                    protocol_path=protocol_path,
                )

    def test_unclassified_extra_column_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                data_path,
                schema_path,
                protocol_path,
            ) = write_fixture(root)

            dataframe = pd.read_csv(
                data_path
            )

            dataframe["leakage_column"] = (
                dataframe[
                    "sleep_stage_encoded"
                ]
            )

            dataframe.to_csv(
                data_path,
                index=False,
                lineterminator="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unexpected",
            ):
                load_phase3_dataset(
                    data_path=data_path,
                    schema_path=schema_path,
                    protocol_path=protocol_path,
                )

    def test_duplicate_identifiers_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                data_path,
                schema_path,
                protocol_path,
            ) = write_fixture(root)

            dataframe = pd.read_csv(
                data_path
            )

            dataframe = pd.concat(
                [
                    dataframe,
                    dataframe.iloc[[0]],
                ],
                ignore_index=True,
            )

            dataframe.to_csv(
                data_path,
                index=False,
                lineterminator="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate identifier",
            ):
                load_phase3_dataset(
                    data_path=data_path,
                    schema_path=schema_path,
                    protocol_path=protocol_path,
                )

    def test_target_mapping_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                data_path,
                schema_path,
                protocol_path,
            ) = write_fixture(root)

            protocol_path.write_text(
                """{
  "group_column": "subject_id",
  "target_column": "sleep_stage_encoded",
  "target_name_column": "sleep_stage",
  "random_epoch_split_allowed": false,
  "class_mapping": {
    "Wake": 0,
    "N1": 2,
    "N2": 1,
    "N3": 3,
    "REM": 4
  }
}
""",
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "target mapping",
            ):
                load_phase3_dataset(
                    data_path=data_path,
                    schema_path=schema_path,
                    protocol_path=protocol_path,
                )

    def test_quality_issue_is_rejected_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (
                data_path,
                schema_path,
                protocol_path,
            ) = write_fixture(root)

            dataframe = pd.read_csv(
                data_path
            )

            dataframe.loc[
                0,
                "quality_issue_flag",
            ] = True

            dataframe.to_csv(
                data_path,
                index=False,
                lineterminator="\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "quality-issue",
            ):
                load_phase3_dataset(
                    data_path=data_path,
                    schema_path=schema_path,
                    protocol_path=protocol_path,
                )

            bundle = load_phase3_dataset(
                data_path=data_path,
                schema_path=schema_path,
                protocol_path=protocol_path,
                reject_quality_issues=False,
            )

            self.assertEqual(
                int(
                    bundle.quality[
                        "quality_issue_flag"
                    ].sum()
                ),
                1,
            )

    def test_subject_partition_is_isolated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.load_fixture(
                Path(directory)
            )

        partition = select_subject_partition(
            bundle=bundle,
            subjects=[1, 0],
            name="train",
        )

        self.assertEqual(
            partition.subjects,
            (0, 1),
        )

        self.assertEqual(
            partition.row_count,
            20,
        )

        self.assertEqual(
            set(
                partition.groups.tolist()
            ),
            {0, 1},
        )

        self.assertEqual(
            partition.X.shape,
            (20, 3),
        )

        self.assertEqual(
            set(partition.y.tolist()),
            {0, 1, 2, 3, 4},
        )

        self.assertEqual(
            len(
                np.unique(
                    partition.row_indices
                )
            ),
            partition.row_count,
        )

    def test_unknown_subject_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.load_fixture(
                Path(directory)
            )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown subject",
        ):
            select_subject_partition(
                bundle=bundle,
                subjects=[99],
                name="test",
            )

    def test_summary_write_is_byte_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            bundle = self.load_fixture(
                root
            )

            summary = build_dataset_summary(
                bundle
            )

            output_path = (
                root / "summary.json"
            )

            write_dataset_summary(
                summary=summary,
                output_path=output_path,
            )

            first_content = (
                output_path.read_bytes()
            )

            write_dataset_summary(
                summary=summary,
                output_path=output_path,
            )

            self.assertEqual(
                first_content,
                output_path.read_bytes(),
            )

            self.assertEqual(
                summary["row_count"],
                40,
            )

            self.assertEqual(
                summary["feature_count"],
                3,
            )

    @unittest.skipUnless(
        DEFAULT_DATA_PATH.exists()
        and DEFAULT_SCHEMA_PATH.exists()
        and DEFAULT_PROTOCOL_PATH.exists(),
        "Local Phase 3 model input is unavailable.",
    )
    def test_local_model_input_contract(
        self,
    ) -> None:
        bundle = load_phase3_dataset()

        self.assertEqual(
            bundle.row_count,
            3921,
        )

        self.assertEqual(
            bundle.feature_count,
            28,
        )

        self.assertEqual(
            len(
                set(bundle.groups.tolist())
            ),
            4,
        )

        self.assertEqual(
            set(bundle.y.tolist()),
            {0, 1, 2, 3, 4},
        )

        self.assertEqual(
            int(
                bundle.quality.to_numpy(
                    dtype=bool
                ).sum()
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
