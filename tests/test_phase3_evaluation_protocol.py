from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_evaluation_protocol.json"
)


class Phase3EvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with PROTOCOL_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            cls.protocol = json.load(file)

    def test_group_and_target_contract(self) -> None:
        self.assertEqual(
            self.protocol["group_column"],
            "subject_id",
        )
        self.assertEqual(
            self.protocol["target_column"],
            "sleep_stage_encoded",
        )
        self.assertFalse(
            self.protocol[
                "random_epoch_split_allowed"
            ]
        )

    def test_class_mapping_is_complete(self) -> None:
        expected_mapping = {
            "Wake": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "REM": 4,
        }

        self.assertEqual(
            self.protocol["class_mapping"],
            expected_mapping,
        )

    def test_primary_metric_handles_imbalance(self) -> None:
        self.assertEqual(
            self.protocol["primary_metric"],
            "macro_f1",
        )

        secondary_metrics = set(
            self.protocol["secondary_metrics"]
        )

        self.assertIn(
            "balanced_accuracy",
            secondary_metrics,
        )
        self.assertIn(
            "cohen_kappa",
            secondary_metrics,
        )
        self.assertIn(
            "per_class_recall",
            secondary_metrics,
        )

    def test_local_protocol_is_exhaustive_nested_logo(
        self,
    ) -> None:
        local = self.protocol[
            "local_development_protocol"
        ]

        self.assertEqual(
            local["strategy"],
            "exhaustive_nested_leave_one_subject_out",
        )
        self.assertEqual(
            local["outer_fold_count"],
            4,
        )
        self.assertEqual(
            local["inner_fold_count_per_outer"],
            3,
        )
        self.assertEqual(
            local[
                "total_train_validation_test_combinations"
            ],
            12,
        )
        self.assertFalse(
            local[
                "final_scientific_reporting_allowed"
            ]
        )

    def test_full_protocol_is_group_stratified(
        self,
    ) -> None:
        full = self.protocol[
            "full_dataset_protocol"
        ]

        self.assertEqual(
            full["strategy"],
            "nested_stratified_group_k_fold",
        )
        self.assertEqual(
            full["outer_fold_count"],
            5,
        )
        self.assertEqual(
            full["inner_fold_count"],
            3,
        )
        self.assertTrue(full["shuffle"])
        self.assertEqual(
            full["random_state"],
            42,
        )
        self.assertTrue(
            full[
                "final_scientific_reporting_allowed"
            ]
        )

    def test_preprocessing_is_training_only(
        self,
    ) -> None:
        leakage = self.protocol[
            "leakage_prevention"
        ]

        self.assertTrue(
            leakage["group_isolation_required"]
        )
        self.assertFalse(
            leakage[
                "test_partition_used_for_tuning"
            ]
        )

        for key in (
            "scaler_fit_scope",
            "feature_selection_fit_scope",
            "class_weight_fit_scope",
            "resampling_fit_scope",
        ):
            self.assertEqual(
                leakage[key],
                "training_partition_only",
            )

        self.assertEqual(
            leakage["default_resampling_policy"],
            "disabled",
        )


if __name__ == "__main__":
    unittest.main()
