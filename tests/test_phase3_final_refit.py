from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts.phase3_dataset import (
    load_phase3_dataset,
)
from scripts.phase3_final_refit import (
    DEFAULT_REGISTRY_PATH,
    build_final_refit_logo_splits,
    evaluate_candidate_on_logo_folds,
    load_registry,
)
from scripts.phase3_inner_search import (
    rank_selectable_candidates,
)
from scripts.phase3_model_registry import (
    enumerate_candidate_parameters,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "phase3_final_refit.py"
)


class Phase3FinalRefitTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.bundle = (
            load_phase3_dataset()
        )

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

        cls.parameters = (
            enumerate_candidate_parameters(
                model_name=(
                    "logistic_regression"
                ),
                registry=cls.registry,
            )[0]
        )

        cls.evaluation_result = (
            evaluate_candidate_on_logo_folds(
                bundle=cls.bundle,
                logo_splits=cls.splits,
                registry=cls.registry,
                model_name=(
                    "logistic_regression"
                ),
                candidate_index=1,
                candidate_parameters=(
                    cls.parameters
                ),
            )
        )

    def test_direct_script_entrypoint_imports(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=result.stderr,
        )

        self.assertIn(
            "--smoke-test",
            result.stdout,
        )

    def test_logo_splits_are_complete_and_isolated(
        self,
    ) -> None:
        self.assertEqual(
            len(self.splits),
            4,
        )

        validation_subjects = []

        for split in self.splits:
            train = set(
                split[
                    "train_subjects"
                ]
            )

            validation = set(
                split[
                    "validation_subjects"
                ]
            )

            self.assertFalse(
                train & validation
            )

            self.assertEqual(
                train | validation,
                {0, 1, 2, 3},
            )

            self.assertEqual(
                len(validation),
                1,
            )

            validation_subjects.extend(
                validation
            )

        self.assertEqual(
            sorted(
                validation_subjects
            ),
            [0, 1, 2, 3],
        )

    def test_candidate_evaluation_uses_every_subject_once(
        self,
    ) -> None:
        result = (
            self.evaluation_result
        )

        self.assertEqual(
            result["fold_count"],
            4,
        )

        self.assertEqual(
            result["candidate_id"],
            (
                "logistic_regression"
                "__candidate_001"
            ),
        )

        validation_subjects = []

        for fold in result[
            "fold_metrics"
        ]:
            self.assertTrue(
                set(
                    fold[
                        "train_subjects"
                    ]
                ).isdisjoint(
                    fold[
                        "validation_subjects"
                    ]
                )
            )

            validation_subjects.extend(
                fold[
                    "validation_subjects"
                ]
            )

            self.assertGreater(
                fold[
                    "train_row_count"
                ],
                0,
            )

            self.assertGreater(
                fold[
                    "validation_row_count"
                ],
                0,
            )

            self.assertTrue(
                np.isfinite(
                    fold["metrics"][
                        "macro_f1"
                    ]
                )
            )

        self.assertEqual(
            sorted(
                validation_subjects
            ),
            [0, 1, 2, 3],
        )

    def test_candidate_aggregate_contains_all_selection_metrics(
        self,
    ) -> None:
        aggregate = (
            self.evaluation_result[
                "aggregate"
            ]
        )

        for metric_name in (
            "macro_f1",
            "balanced_accuracy",
            "weighted_f1",
            "accuracy",
            "cohen_kappa",
            "multiclass_log_loss",
        ):
            self.assertIn(
                f"mean_{metric_name}",
                aggregate,
            )

            self.assertIn(
                f"std_{metric_name}",
                aggregate,
            )

            self.assertTrue(
                np.isfinite(
                    aggregate[
                        f"mean_"
                        f"{metric_name}"
                    ]
                )
            )

            self.assertGreaterEqual(
                aggregate[
                    f"std_"
                    f"{metric_name}"
                ],
                0.0,
            )

    def test_ranking_excludes_non_selectable_baseline(
        self,
    ) -> None:
        selectable = copy.deepcopy(
            self.evaluation_result
        )

        baseline = copy.deepcopy(
            self.evaluation_result
        )

        baseline[
            "model_name"
        ] = "dummy_prior"

        baseline[
            "candidate_id"
        ] = (
            "dummy_prior"
            "__candidate_001"
        )

        baseline[
            "eligible_for_selection"
        ] = False

        baseline[
            "complexity_rank"
        ] = 0

        baseline[
            "aggregate"
        ][
            "mean_macro_f1"
        ] = 1.0

        ranked = (
            rank_selectable_candidates(
                [
                    baseline,
                    selectable,
                ]
            )
        )

        self.assertEqual(
            len(ranked),
            1,
        )

        self.assertEqual(
            ranked[0][
                "candidate_id"
            ],
            selectable[
                "candidate_id"
            ],
        )

        self.assertEqual(
            ranked[0][
                "selection_rank"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
