from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from scripts.phase3_model_registry import (
    DEFAULT_CONFIG_PATH,
    build_model_pipeline,
    build_registry_summary,
    enumerate_candidate_parameters,
    load_registry_config,
    validate_registry,
    write_registry_summary,
)


EXPECTED_CANDIDATE_COUNTS = {
    "dummy_prior": 1,
    "logistic_regression": 3,
    "sgd_logistic": 9,
    "random_forest": 8,
    "extra_trees": 8,
}


class Phase3ModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry_config()
        cls.candidate_counts = (
            validate_registry(cls.registry)
        )

        cls.X, cls.y = make_classification(
            n_samples=250,
            n_features=28,
            n_informative=16,
            n_redundant=4,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=42,
        )

    def test_expected_model_families(self) -> None:
        self.assertEqual(
            set(self.registry["models"]),
            set(EXPECTED_CANDIDATE_COUNTS),
        )

    def test_candidate_counts_are_locked(
        self,
    ) -> None:
        self.assertEqual(
            self.candidate_counts,
            EXPECTED_CANDIDATE_COUNTS,
        )

        self.assertEqual(
            sum(self.candidate_counts.values()),
            29,
        )

    def test_selection_eligibility(self) -> None:
        models = self.registry["models"]

        self.assertFalse(
            models["dummy_prior"][
                "eligible_for_selection"
            ]
        )

        selectable = {
            name
            for name, spec in models.items()
            if spec["eligible_for_selection"]
        }

        self.assertEqual(
            selectable,
            {
                "logistic_regression",
                "sgd_logistic",
                "random_forest",
                "extra_trees",
            },
        )

    def test_logistic_regression_uses_current_l2_api(
        self,
    ) -> None:
        fixed_parameters = self.registry["models"][
            "logistic_regression"
        ]["fixed_parameters"]

        self.assertNotIn(
            "penalty",
            fixed_parameters,
        )
        self.assertEqual(
            fixed_parameters["l1_ratio"],
            0.0,
        )
        self.assertEqual(
            fixed_parameters["solver"],
            "lbfgs",
        )

    def test_scaling_policy_is_model_specific(
        self,
    ) -> None:
        scaled_models = {
            "logistic_regression",
            "sgd_logistic",
        }

        for model_name in self.registry[
            "models"
        ]:
            pipeline = build_model_pipeline(
                model_name=model_name,
                registry=self.registry,
            )

            preprocessor = pipeline.named_steps[
                "preprocessor"
            ]

            if model_name in scaled_models:
                self.assertIsInstance(
                    preprocessor,
                    StandardScaler,
                )
            else:
                self.assertEqual(
                    preprocessor,
                    "passthrough",
                )

    def test_every_grid_candidate_is_settable(
        self,
    ) -> None:
        for model_name in self.registry[
            "models"
        ]:
            pipeline = build_model_pipeline(
                model_name=model_name,
                registry=self.registry,
            )

            candidates = (
                enumerate_candidate_parameters(
                    model_name=model_name,
                    registry=self.registry,
                )
            )

            for candidate in candidates:
                configured = clone(
                    pipeline
                ).set_params(**candidate)

                self.assertIsNotNone(configured)

    def test_default_pipelines_fit_and_predict_probabilities(
        self,
    ) -> None:
        for model_name in self.registry[
            "models"
        ]:
            pipeline = build_model_pipeline(
                model_name=model_name,
                registry=self.registry,
            )

            fitted = clone(pipeline).fit(
                self.X,
                self.y,
            )

            predictions = fitted.predict(
                self.X[:20]
            )

            probabilities = (
                fitted.predict_proba(
                    self.X[:20]
                )
            )

            self.assertEqual(
                predictions.shape,
                (20,),
            )

            self.assertEqual(
                probabilities.shape,
                (20, 5),
            )

            self.assertTrue(
                np.isfinite(
                    probabilities
                ).all()
            )

            np.testing.assert_allclose(
                probabilities.sum(axis=1),
                np.ones(20),
                atol=1e-6,
            )

    def test_summary_output_is_byte_deterministic(
        self,
    ) -> None:
        summary = build_registry_summary(
            registry=self.registry,
            config_path=DEFAULT_CONFIG_PATH,
        )

        self.assertEqual(
            summary["model_count"],
            5,
        )
        self.assertEqual(
            summary["selection_model_count"],
            4,
        )
        self.assertEqual(
            summary["total_candidate_count"],
            29,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "registry-summary.json"
            )

            write_registry_summary(
                summary=summary,
                output_path=output_path,
            )

            first_content = (
                output_path.read_bytes()
            )

            write_registry_summary(
                summary=summary,
                output_path=output_path,
            )

            self.assertEqual(
                first_content,
                output_path.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
