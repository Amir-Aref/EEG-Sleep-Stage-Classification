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
from scripts.run_provenance import (
    sha256_file,
)


class PipelineRunnerProvenanceTests(
    unittest.TestCase
):
    def sample_arguments(
        self,
        *,
        plan: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            refresh_manifest=False,
            download_subject_count=None,
            nights_per_subject=1,
            overwrite_features=False,
            skip_eda=False,
            skip_tests=False,
            plan=plan,
        )

    def sample_steps(
        self,
    ) -> list[runner.PipelineStep]:
        return [
            runner.PipelineStep(
                name="Step one",
                command=(
                    "python",
                    "-m",
                    "example.one",
                ),
            ),
            runner.PipelineStep(
                name="Step two",
                command=(
                    "python",
                    "-m",
                    "example.two",
                ),
            ),
        ]

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
                "/runtime/data/metadata/"
                "manifest.csv"
            ),
            "manifest_exists": True,
            "manifest_sha256": "b" * 64,
        }

    def test_input_identity_hashes_existing_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = (
                Path(directory)
                / "manifest.csv"
            )

            manifest_path.write_text(
                "recording_id\nSC4001\n",
                encoding="utf-8",
                newline="\n",
            )

            with mock.patch.object(
                runner,
                "SLEEP_EDFX_MANIFEST_PATH",
                manifest_path,
            ):
                identity = (
                    runner.build_input_identity()
                )

            self.assertTrue(
                identity["manifest_exists"]
            )

            self.assertEqual(
                identity["manifest_sha256"],
                sha256_file(
                    manifest_path
                ),
            )

    def test_successful_execution_persists_lifecycle(
        self,
    ) -> None:
        steps = self.sample_steps()
        arguments = self.sample_arguments()

        with tempfile.TemporaryDirectory() as directory:
            reports_dir = (
                Path(directory)
                / "reports"
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
                            reports_dir.parent
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
                    "run_step",
                    side_effect=[
                        0.25,
                        0.50,
                    ],
                ) as run_step_mock,
            ):
                (
                    manifest_path,
                    manifest,
                    elapsed_by_step,
                    total_elapsed,
                ) = runner.execute_pipeline(
                    arguments=arguments,
                    steps=steps,
                )

            self.assertTrue(
                manifest_path.is_file()
            )

            loaded = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded,
                manifest,
            )

            self.assertEqual(
                manifest["status"],
                "succeeded",
            )

            self.assertEqual(
                [
                    step["status"]
                    for step in manifest["steps"]
                ],
                [
                    "succeeded",
                    "succeeded",
                ],
            )

            self.assertEqual(
                [
                    step["return_code"]
                    for step in manifest["steps"]
                ],
                [
                    0,
                    0,
                ],
            )

            self.assertEqual(
                elapsed_by_step,
                [
                    ("Step one", 0.25),
                    ("Step two", 0.50),
                ],
            )

            self.assertGreaterEqual(
                total_elapsed,
                0.0,
            )

            self.assertEqual(
                run_step_mock.call_count,
                2,
            )

    def test_failed_execution_persists_failure(
        self,
    ) -> None:
        steps = self.sample_steps()
        arguments = self.sample_arguments()

        process_error = (
            subprocess.CalledProcessError(
                returncode=7,
                cmd=list(
                    steps[0].command
                ),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            reports_dir = (
                Path(directory)
                / "reports"
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
                            reports_dir.parent
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
                    "run_step",
                    side_effect=process_error,
                ),
            ):
                with self.assertRaises(
                    subprocess.CalledProcessError
                ):
                    runner.execute_pipeline(
                        arguments=arguments,
                        steps=steps,
                    )

            manifest_paths = list(
                (
                    reports_dir
                    / "runs"
                ).glob(
                    "*/run_manifest.json"
                )
            )

            self.assertEqual(
                len(manifest_paths),
                1,
            )

            manifest = json.loads(
                manifest_paths[0].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest["status"],
                "failed",
            )

            self.assertEqual(
                manifest["steps"][0]["status"],
                "failed",
            )

            self.assertEqual(
                manifest["steps"][0][
                    "return_code"
                ],
                7,
            )

            self.assertEqual(
                manifest["steps"][1]["status"],
                "pending",
            )

            self.assertEqual(
                manifest["error"][
                    "failed_step_position"
                ],
                1,
            )

    def test_plan_mode_does_not_execute_or_write(
        self,
    ) -> None:
        arguments = self.sample_arguments(
            plan=True
        )

        steps = self.sample_steps()

        with (
            mock.patch.object(
                runner,
                "parse_arguments",
                return_value=arguments,
            ),
            mock.patch.object(
                runner,
                "build_pipeline_steps",
                return_value=steps,
            ),
            mock.patch.object(
                runner,
                "execute_pipeline",
            ) as execute_mock,
        ):
            runner.main()

        execute_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
