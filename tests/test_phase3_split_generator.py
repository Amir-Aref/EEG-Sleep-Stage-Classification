from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.generate_phase3_splits import (
    build_split_manifest,
    generate_nested_logo_splits,
    manifest_to_frame,
    write_manifest_outputs,
)


CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


def synthetic_protocol() -> dict:
    return {
        "group_column": "subject_id",
        "target_column": "sleep_stage_encoded",
        "target_name_column": "sleep_stage",
        "class_mapping": CLASS_MAPPING,
        "local_development_protocol": {
            "strategy": (
                "exhaustive_nested_leave_one_subject_out"
            ),
            "intended_use": (
                "engineering_smoke_test_only"
            ),
            "final_scientific_reporting_allowed": False,
        },
    }


def synthetic_dataframe() -> pd.DataFrame:
    rows = []

    for subject_id in range(4):
        for stage, encoded in CLASS_MAPPING.items():
            for repetition in range(2):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "sleep_stage": stage,
                        "sleep_stage_encoded": encoded,
                        "epoch_id": repetition,
                    }
                )

    return pd.DataFrame(rows)


class Phase3SplitGeneratorTests(unittest.TestCase):
    def test_generates_twelve_unique_splits(self) -> None:
        splits = generate_nested_logo_splits(
            [0, 1, 2, 3]
        )

        self.assertEqual(len(splits), 12)

        split_ids = {
            split["split_id"]
            for split in splits
        }

        self.assertEqual(len(split_ids), 12)

    def test_partitions_are_isolated_and_complete(
        self,
    ) -> None:
        expected_subjects = {0, 1, 2, 3}

        for split in generate_nested_logo_splits(
            expected_subjects
        ):
            train = set(split["train_subjects"])
            validation = set(
                split["validation_subjects"]
            )
            test = set(split["test_subjects"])

            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)

            self.assertEqual(
                train | validation | test,
                expected_subjects,
            )

            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 1)
            self.assertEqual(len(test), 1)

    def test_subject_role_counts_are_balanced(
        self,
    ) -> None:
        splits = generate_nested_logo_splits(
            [0, 1, 2, 3]
        )

        train_counter = Counter()
        validation_counter = Counter()
        test_counter = Counter()

        for split in splits:
            train_counter.update(
                split["train_subjects"]
            )
            validation_counter.update(
                split["validation_subjects"]
            )
            test_counter.update(
                split["test_subjects"]
            )

        for subject in range(4):
            self.assertEqual(
                train_counter[subject],
                6,
            )
            self.assertEqual(
                validation_counter[subject],
                3,
            )
            self.assertEqual(
                test_counter[subject],
                3,
            )

    def test_generation_is_input_order_independent(
        self,
    ) -> None:
        forward = generate_nested_logo_splits(
            [0, 1, 2, 3]
        )
        reverse = generate_nested_logo_splits(
            [3, 2, 1, 0]
        )

        self.assertEqual(forward, reverse)

    def test_manifest_has_complete_class_coverage(
        self,
    ) -> None:
        dataframe = synthetic_dataframe()

        manifest = build_split_manifest(
            dataframe=dataframe,
            protocol=synthetic_protocol(),
            data_sha256="data-hash",
            protocol_sha256="protocol-hash",
            data_path="synthetic.csv",
            protocol_path="protocol.json",
        )

        self.assertEqual(
            manifest["total_split_count"],
            12,
        )
        self.assertEqual(
            manifest["row_count"],
            len(dataframe),
        )

        for split in manifest["splits"]:
            for partition in (
                "train",
                "validation",
                "test",
            ):
                summary = split["partitions"][
                    partition
                ]

                self.assertEqual(
                    summary["missing_classes"],
                    [],
                )

                self.assertEqual(
                    set(summary["class_counts"]),
                    set(CLASS_MAPPING),
                )

    def test_csv_has_one_row_per_inner_split(
        self,
    ) -> None:
        manifest = build_split_manifest(
            dataframe=synthetic_dataframe(),
            protocol=synthetic_protocol(),
            data_sha256="data-hash",
            protocol_sha256="protocol-hash",
            data_path="synthetic.csv",
            protocol_path="protocol.json",
        )

        frame = manifest_to_frame(manifest)

        self.assertEqual(len(frame), 12)
        self.assertEqual(
            frame["outer_fold"].nunique(),
            4,
        )

        inner_counts = (
            frame.groupby("outer_fold")
            ["inner_fold"]
            .nunique()
        )

        self.assertTrue(
            (inner_counts == 3).all()
        )

    def test_output_files_are_byte_deterministic(
        self,
    ) -> None:
        manifest = build_split_manifest(
            dataframe=synthetic_dataframe(),
            protocol=synthetic_protocol(),
            data_sha256="data-hash",
            protocol_sha256="protocol-hash",
            data_path="synthetic.csv",
            protocol_path="protocol.json",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "splits.json"
            csv_path = root / "splits.csv"

            write_manifest_outputs(
                manifest=manifest,
                json_output=json_path,
                csv_output=csv_path,
            )

            first_json = json_path.read_bytes()
            first_csv = csv_path.read_bytes()

            write_manifest_outputs(
                manifest=manifest,
                json_output=json_path,
                csv_output=csv_path,
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
                loaded["total_split_count"],
                12,
            )


if __name__ == "__main__":
    unittest.main()
