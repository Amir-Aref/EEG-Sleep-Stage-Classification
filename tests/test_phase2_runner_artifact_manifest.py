from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import (
    run_phase2_pipeline as runner,
)
from scripts.phase2_artifact_manifest import (
    ArtifactSpec,
)
from scripts.run_provenance import (
    sha256_file,
)


class PipelineRunnerArtifactManifestTests(
    unittest.TestCase
):
    def sample_arguments(
        self,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            refresh_manifest=False,
            download_subject_count=None,
            nights_per_subject=1,
            overwrite_features=False,
            skip_eda=False,
            skip_tests=False,
            plan=False,
        )

    def repository_snapshot(
        self,
    ) -> dict[str, object]:
        return {
            "commit": "a" * 40,
            "branch": "full-dataset-kaggle",
            "is_clean": True,
            "status_entries": [],
        }

    def path_contract(
        self,
        runtime_root: Path,
    ) -> dict[str, object]:
        return {
            "project_root": "/repo",
            "runtime_root": str(
                runtime_root
            ),
            "sleep_edfx_raw_dir": (
                "/input/sleep-cassette"
            ),
            "sleep_edfx_raw_dir_is_external": True,
        }

    def input_identity(
        self,
    ) -> dict[str, object]:
        return {
            "dataset_name": (
                "Sleep-EDF Expanded"
            ),
            "dataset_version": "1.0.0",
            "subset": "sleep-cassette",
            "expected_recording_count": 153,
            "manifest_path": (
                "/runtime/manifest.csv"
            ),
            "manifest_exists": True,
            "manifest_sha256": "b" * 64,
        }

    def test_successful_run_writes_linked_artifact_manifests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = (
                Path(directory)
                / "working"
            )

            reports_dir = (
                runtime_root
                / "reports"
            )

            artifact_path = (
                runtime_root
                / "data"
                / "dataset.csv"
            )

            artifact_spec = ArtifactSpec(
                logical_name="dataset",
                producer_step="Build dataset",
                artifact_type="dataset",
                path=artifact_path,
            )

            steps = [
                runner.PipelineStep(
                    name="Build dataset",
                    command=(
                        "python",
                        "-m",
                        "example.build",
                    ),
                )
            ]

            def execute_step(
                *,
                step: runner.PipelineStep,
                position: int,
                total: int,
            ) -> float:
                self.assertEqual(
                    position,
                    1,
                )

                self.assertEqual(
                    total,
                    1,
                )

                artifact_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                artifact_path.write_text(
                    "x,y\n1,2\n3,4\n",
                    encoding="utf-8",
                    newline="\n",
                )

                return 0.25

            with (
                mock.patch.object(
                    runner,
                    "REPORTS_DIR",
                    reports_dir,
                ),
                mock.patch.object(
                    runner,
                    "repository_snapshot",
                    return_value=(
                        self.repository_snapshot()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "runtime_path_contract",
                    return_value=(
                        self.path_contract(
                            runtime_root
                        )
                    ),
                ),
                mock.patch.object(
                    runner,
                    "build_input_identity",
                    return_value=(
                        self.input_identity()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "selected_artifact_specs",
                    return_value=(
                        artifact_spec,
                    ),
                ),
                mock.patch.object(
                    runner,
                    "run_step",
                    side_effect=execute_step,
                ),
            ):
                (
                    run_manifest_path,
                    run_manifest,
                    _,
                    _,
                ) = runner.execute_pipeline(
                    arguments=(
                        self.sample_arguments()
                    ),
                    steps=steps,
                )

            self.assertEqual(
                run_manifest["status"],
                "succeeded",
            )

            artifact_reference = (
                run_manifest[
                    "artifact_manifest"
                ]
            )

            self.assertEqual(
                artifact_reference["status"],
                "valid",
            )

            json_path = (
                runtime_root
                / artifact_reference[
                    "json_path"
                ]
            )

            csv_path = (
                runtime_root
                / artifact_reference[
                    "csv_path"
                ]
            )

            self.assertTrue(
                json_path.is_file()
            )

            self.assertTrue(
                csv_path.is_file()
            )

            self.assertEqual(
                artifact_reference[
                    "json_sha256"
                ],
                sha256_file(
                    json_path
                ),
            )

            self.assertEqual(
                artifact_reference[
                    "csv_sha256"
                ],
                sha256_file(
                    csv_path
                ),
            )

            artifact_manifest = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                artifact_manifest["status"],
                "valid",
            )

            self.assertEqual(
                artifact_manifest[
                    "artifact_count"
                ],
                1,
            )

            artifact = (
                artifact_manifest[
                    "artifacts"
                ][0]
            )

            self.assertEqual(
                artifact["change_type"],
                "created",
            )

            self.assertEqual(
                artifact["row_count"],
                2,
            )

            self.assertEqual(
                artifact["column_count"],
                2,
            )

            persisted_run_manifest = (
                json.loads(
                    run_manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

            self.assertEqual(
                persisted_run_manifest,
                run_manifest,
            )

    def test_missing_required_artifact_fails_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = (
                Path(directory)
                / "working"
            )

            reports_dir = (
                runtime_root
                / "reports"
            )

            missing_path = (
                runtime_root
                / "data"
                / "missing.csv"
            )

            artifact_spec = ArtifactSpec(
                logical_name="missing_dataset",
                producer_step="Build dataset",
                artifact_type="dataset",
                path=missing_path,
            )

            steps = [
                runner.PipelineStep(
                    name="Build dataset",
                    command=(
                        "python",
                        "-m",
                        "example.build",
                    ),
                )
            ]

            with (
                mock.patch.object(
                    runner,
                    "REPORTS_DIR",
                    reports_dir,
                ),
                mock.patch.object(
                    runner,
                    "repository_snapshot",
                    return_value=(
                        self.repository_snapshot()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "runtime_path_contract",
                    return_value=(
                        self.path_contract(
                            runtime_root
                        )
                    ),
                ),
                mock.patch.object(
                    runner,
                    "build_input_identity",
                    return_value=(
                        self.input_identity()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "selected_artifact_specs",
                    return_value=(
                        artifact_spec,
                    ),
                ),
                mock.patch.object(
                    runner,
                    "run_step",
                    return_value=0.1,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Artifact manifest validation failed",
                ):
                    runner.execute_pipeline(
                        arguments=(
                            self.sample_arguments()
                        ),
                        steps=steps,
                    )

            run_manifest_paths = list(
                (
                    reports_dir
                    / "runs"
                ).glob(
                    "*/run_manifest.json"
                )
            )

            self.assertEqual(
                len(run_manifest_paths),
                1,
            )

            run_manifest = json.loads(
                run_manifest_paths[
                    0
                ].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                run_manifest["status"],
                "failed",
            )

            self.assertEqual(
                run_manifest["error"]["type"],
                (
                    "ArtifactManifestValidationError"
                ),
            )

            reference = (
                run_manifest[
                    "artifact_manifest"
                ]
            )

            self.assertEqual(
                reference["status"],
                "invalid",
            )

            artifact_json_path = (
                runtime_root
                / reference["json_path"]
            )

            artifact_manifest = json.loads(
                artifact_json_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                artifact_manifest["status"],
                "invalid",
            )

            self.assertEqual(
                artifact_manifest[
                    "missing_required_specs"
                ][0]["logical_name"],
                "missing_dataset",
            )

    def test_failed_step_still_writes_artifact_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = (
                Path(directory)
                / "working"
            )

            reports_dir = (
                runtime_root
                / "reports"
            )

            partial_path = (
                runtime_root
                / "data"
                / "partial.csv"
            )

            artifact_spec = ArtifactSpec(
                logical_name="partial_dataset",
                producer_step="Build dataset",
                artifact_type="dataset",
                path=partial_path,
            )

            steps = [
                runner.PipelineStep(
                    name="Build dataset",
                    command=(
                        "python",
                        "-m",
                        "example.build",
                    ),
                )
            ]

            def fail_step(
                *,
                step: runner.PipelineStep,
                position: int,
                total: int,
            ) -> float:
                partial_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                partial_path.write_text(
                    "x\n1\n",
                    encoding="utf-8",
                    newline="\n",
                )

                raise subprocess.CalledProcessError(
                    returncode=5,
                    cmd=list(
                        step.command
                    ),
                )

            with (
                mock.patch.object(
                    runner,
                    "REPORTS_DIR",
                    reports_dir,
                ),
                mock.patch.object(
                    runner,
                    "repository_snapshot",
                    return_value=(
                        self.repository_snapshot()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "runtime_path_contract",
                    return_value=(
                        self.path_contract(
                            runtime_root
                        )
                    ),
                ),
                mock.patch.object(
                    runner,
                    "build_input_identity",
                    return_value=(
                        self.input_identity()
                    ),
                ),
                mock.patch.object(
                    runner,
                    "selected_artifact_specs",
                    return_value=(
                        artifact_spec,
                    ),
                ),
                mock.patch.object(
                    runner,
                    "run_step",
                    side_effect=fail_step,
                ),
            ):
                with self.assertRaises(
                    subprocess.CalledProcessError
                ):
                    runner.execute_pipeline(
                        arguments=(
                            self.sample_arguments()
                        ),
                        steps=steps,
                    )

            run_manifest_path = next(
                (
                    reports_dir
                    / "runs"
                ).glob(
                    "*/run_manifest.json"
                )
            )

            run_manifest = json.loads(
                run_manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                run_manifest["status"],
                "failed",
            )

            self.assertEqual(
                run_manifest[
                    "steps"
                ][0]["return_code"],
                5,
            )

            reference = (
                run_manifest[
                    "artifact_manifest"
                ]
            )

            self.assertEqual(
                reference["status"],
                "valid",
            )

            artifact_json_path = (
                runtime_root
                / reference["json_path"]
            )

            artifact_manifest = json.loads(
                artifact_json_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                artifact_manifest[
                    "counts_by_change_type"
                ]["created"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
