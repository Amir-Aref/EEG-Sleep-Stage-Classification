from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from scripts import mlflow_tracking as tracking




class _FakeClient:
    def search_runs(self, **kwargs):
        return []

    def search_model_versions(self, filter_string):
        return []


class _FakeMlflow:
    def __init__(self) -> None:
        self.counter = 0
        self.logged_metrics = []
        self.logged_params = []
        self.logged_artifacts = []

    @contextmanager
    def start_run(self, **kwargs):
        self.counter += 1
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=f"fake-run-{self.counter}")
        )
        yield run

    def log_params(self, params):
        self.logged_params.append(dict(params))

    def log_metrics(self, metrics):
        self.logged_metrics.append(dict(metrics))

    def log_artifact(self, path, artifact_path=None):
        self.logged_artifacts.append((str(path), artifact_path))


class MlflowTrackingUnitTests(unittest.TestCase):
    def test_configuration_is_valid(self) -> None:
        config = tracking.load_tracking_config()
        self.assertEqual(config["schema_version"], "1.0.0")
        self.assertEqual(
            config["experiment_name"],
            "EEG Sleep Stage Classification",
        )

    def test_requirements_use_mlflow_compatible_pandas(self) -> None:
        requirements = (tracking.PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        pins = {
            name.strip().lower(): version.strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#") and "==" in line
            for name, version in [line.split("==", 1)]
        }
        self.assertEqual(pins.get("mlflow"), "3.14.0")
        self.assertEqual(pins.get("pandas"), "2.3.3")
        self.assertLess(int(pins["pandas"].split(".", 1)[0]), 3)

    def test_tracking_paths_are_project_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            config = tracking.load_tracking_config()
            paths = tracking.resolve_tracking_paths(
                config,
                project_root=project_root,
            )
            self.assertEqual(
                paths.root,
                (project_root / "mlflow").resolve(),
            )
            self.assertEqual(
                paths.backend_database,
                (project_root / "mlflow" / "mlflow.db").resolve(),
            )
            self.assertTrue(paths.tracking_uri.startswith("sqlite:///"))
            self.assertTrue(paths.artifact_uri.startswith("file:"))

    def test_tracking_root_override_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            config = tracking.load_tracking_config()
            paths = tracking.resolve_tracking_paths(
                config,
                project_root=project_root,
                tracking_root_override="custom-tracking",
            )
            self.assertEqual(
                paths.root,
                (project_root / "custom-tracking").resolve(),
            )

    def test_flatten_metrics_skips_confusion_matrix_and_nonfinite_values(self) -> None:
        metrics = tracking.flatten_metrics(
            {
                "macro_f1": 0.75,
                "invalid": float("nan"),
                "confusion_matrix": {"raw": [[1, 0], [0, 1]]},
                "per_class": [
                    {"class_name": "Wake", "f1": 0.8, "support": 10},
                    {"class_name": "N1", "f1": 0.5, "support": 4},
                ],
            },
            prefix="outer_test",
        )
        self.assertEqual(metrics["outer_test.macro_f1"], 0.75)
        self.assertEqual(metrics["outer_test.per_class.Wake.f1"], 0.8)
        self.assertNotIn("outer_test.invalid", metrics)
        self.assertFalse(any("confusion" in key for key in metrics))

    def test_flatten_parameters_is_deterministic(self) -> None:
        first = tracking.flatten_parameters(
            {"b": 2, "a": {"z": [3, 1]}},
            prefix="candidate",
        )
        second = tracking.flatten_parameters(
            {"a": {"z": [3, 1]}, "b": 2},
            prefix="candidate",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["candidate.a.z"], "[3,1]")

    def test_local_artifacts_validate_without_mlflow(self) -> None:
        model_input_path = (
            tracking.PROJECT_ROOT
            / "data"
            / "processed"
            / "sleep_edfx_model_input.csv"
        )
        if not model_input_path.is_file():
            self.skipTest(
                "Generated local Phase 3 model-input artifact is unavailable."
            )

        result = tracking.validate_local_artifacts()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["outer_model_count"], 4)
        self.assertEqual(result["final_model_count"], 1)
        self.assertFalse(result["scientific_reporting_allowed"])

    def test_local_scope_rejects_scientific_artifact(self) -> None:
        selection = {
            "candidate_space_complete": True,
            "scientific_reporting": {"allowed": True},
            "selection_result": {
                "test_metrics_included": False,
                "test_predictions_included": False,
                "test_feature_matrix_loaded": False,
            },
        }
        outer = {
            "complete_outer_evaluation": True,
            "scientific_reporting": {"allowed": False},
        }
        model_manifest = {"complete_model_set": True}
        final_manifest = {"scientific_reporting_allowed": False}
        with self.assertRaisesRegex(ValueError, "must not permit"):
            tracking.validate_scope_contracts(
                scope="local_validation",
                selection=selection,
                outer_evaluation=outer,
                model_manifest=model_manifest,
                final_refit_manifest=final_manifest,
            )

    def test_full_scope_requires_deployment_ready_final_model(self) -> None:
        selection = {
            "candidate_space_complete": True,
            "selection_result": {
                "test_metrics_included": False,
                "test_predictions_included": False,
                "test_feature_matrix_loaded": False,
            },
        }
        outer = {
            "complete_outer_evaluation": True,
            "scientific_reporting": {"allowed": True},
        }
        model_manifest = {"complete_model_set": True}
        final_manifest = {
            "scientific_reporting_allowed": True,
            "deployment": {"deployment_ready": False},
        }
        with self.assertRaisesRegex(ValueError, "deployment-ready"):
            tracking.validate_scope_contracts(
                scope="full_dataset",
                selection=selection,
                outer_evaluation=outer,
                model_manifest=model_manifest,
                final_refit_manifest=final_manifest,
            )

    def test_runtime_mismatch_is_rejected_by_default(self) -> None:
        metadata = {"runtime": {"scikit_learn": "0.0.0"}}
        with self.assertRaisesRegex(RuntimeError, "saved model runtime"):
            tracking.validate_runtime_compatibility(
                metadata,
                allow_version_mismatch=False,
            )
        tracking.validate_runtime_compatibility(
            metadata,
            allow_version_mismatch=True,
        )

    def test_pandas_only_runtime_mismatch_is_provenance_only(self) -> None:
        current = tracking.runtime_metadata()
        metadata = {
            "runtime": {
                "python": current["python"],
                "scikit_learn": current["scikit_learn"],
                "numpy": current["numpy"],
                "joblib": current["joblib"],
                "pandas": "0.0.0",
            }
        }
        tracking.validate_runtime_compatibility(
            metadata,
            allow_version_mismatch=False,
        )

    def test_source_hash_validation_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_file = project_root / "artifact.txt"
            source_file.write_text("original", encoding="utf-8")
            payload = {
                "source": {
                    "artifact_path": "artifact.txt",
                    "artifact_sha256": tracking.sha256_file(source_file),
                }
            }
            result = tracking.validate_source_hashes(
                payload,
                project_root=project_root,
            )
            self.assertEqual(len(result), 1)
            source_file.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                tracking.validate_source_hashes(
                    payload,
                    project_root=project_root,
                )

    def test_metrics_only_import_builds_parent_and_child_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            config = tracking.load_tracking_config()
            fake_mlflow = _FakeMlflow()
            fake_context = tracking.TrackingContext(
                mlflow=fake_mlflow,
                client=_FakeClient(),
                experiment_id="1",
                config=config,
                paths=tracking.TrackingPaths(
                    root=temporary_root / "mlflow",
                    backend_database=temporary_root / "mlflow" / "mlflow.db",
                    artifact_root=temporary_root / "mlflow" / "artifacts",
                    summary_path=temporary_root / "summary.json",
                    tracking_uri="sqlite:///fake",
                    artifact_uri=(temporary_root / "mlflow" / "artifacts").as_uri(),
                ),
            )
            with (
                patch.object(
                    tracking,
                    "initialize_tracking",
                    return_value=fake_context,
                ),
                patch.object(
                    tracking,
                    "validate_source_hashes",
                    return_value=[],
                ),
                patch.object(
                    tracking,
                    "validate_model_manifest",
                    return_value=[],
                ),
            ):
                summary = tracking.import_phase3_artifacts(
                    inputs=tracking.local_import_inputs(),
                    git_commit="11c9ff4",
                    log_models=False,
                )
            self.assertEqual(summary["status"], "imported")
            self.assertEqual(summary["scope"], "local_validation")
            self.assertEqual(len(summary["child_runs"]), 5)
            self.assertFalse(summary["models_logged"])
            self.assertTrue(fake_context.paths.summary_path.is_file())
            self.assertGreaterEqual(fake_mlflow.counter, 6)
            self.assertTrue(fake_mlflow.logged_metrics)
            self.assertTrue(fake_mlflow.logged_artifacts)

    def test_import_fingerprint_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("[]", encoding="utf-8")
            fingerprint_a = tracking.compute_import_fingerprint(
                [first, second],
                "local_validation",
            )
            fingerprint_b = tracking.compute_import_fingerprint(
                [second, first],
                "local_validation",
            )
            self.assertEqual(fingerprint_a, fingerprint_b)


class MlflowTrackingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mlflow = tracking.require_mlflow()
        except tracking.MlflowDependencyError as error:
            self.skipTest(str(error))

    def test_initialize_tracking_creates_sqlite_experiment(self) -> None:
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            project_root = Path(temporary_directory)
            config_path = project_root / "config.json"
            config = tracking.load_tracking_config()
            config["tracking"] = {
                "root_directory": "tracking",
                "backend_database": "mlflow.db",
                "artifact_directory": "artifacts",
                "summary_path": "summary.json",
            }
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            context = tracking.initialize_tracking(
                config_path=config_path,
                project_root=project_root,
            )
            self.assertTrue(context.paths.backend_database.exists())
            experiment = context.client.get_experiment(context.experiment_id)
            self.assertEqual(
                experiment.name,
                "EEG Sleep Stage Classification",
            )
            with context.mlflow.start_run(
                experiment_id=context.experiment_id,
                run_name="integration_test",
            ) as run:
                context.mlflow.log_metric("macro_f1", 0.5)
            fetched = context.client.get_run(run.info.run_id)
            self.assertEqual(fetched.data.metrics["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
