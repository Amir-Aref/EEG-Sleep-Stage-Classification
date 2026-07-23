from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_provenance import (
    atomic_write_json,
    build_run_id,
    canonical_json,
    create_execution_id,
    create_run_manifest,
    repository_snapshot,
    sha256_file,
    validate_run_manifest,
    write_run_manifest,
    fail_run_manifest,
    fail_step_manifest,
    start_run_manifest,
    start_step_manifest,
    succeed_run_manifest,
    succeed_step_manifest,
)


FIXED_TIME = datetime(
    2026,
    7,
    24,
    1,
    2,
    3,
    456789,
    tzinfo=timezone.utc,
)


class RunProvenanceTests(unittest.TestCase):
    def sample_repository(
        self,
    ) -> dict[str, object]:
        return {
            "commit": "a" * 40,
            "branch": "full-dataset-kaggle",
            "is_clean": True,
            "status_entries": [],
        }

    def sample_arguments(
        self,
    ) -> dict[str, object]:
        return {
            "download_subject_count": 78,
            "nights_per_subject": 2,
            "skip_eda": False,
            "skip_tests": False,
        }

    def sample_path_contract(
        self,
    ) -> dict[str, object]:
        return {
            "project_root": "/repo",
            "runtime_root": "/runtime",
            "sleep_edfx_raw_dir": "/input/sleep-cassette",
            "sleep_edfx_raw_dir_is_external": True,
        }

    def sample_input_identity(
        self,
    ) -> dict[str, object]:
        return {
            "dataset_name": "Sleep-EDF Expanded",
            "dataset_version": "1.0.0",
            "subset": "sleep-cassette",
            "manifest_sha256": "b" * 64,
            "expected_recording_count": 153,
        }

    def sample_steps(
        self,
    ) -> list[dict[str, object]]:
        return [
            {
                "name": "Inspect verified EDF pairs",
                "command": [
                    sys.executable,
                    "-m",
                    "scripts.inspect_sleep_edfx",
                ],
            },
            {
                "name": "Build trimmed epoch metadata",
                "command": [
                    sys.executable,
                    "-m",
                    "scripts.build_epoch_metadata",
                ],
            },
        ]

    def create_sample_manifest(
        self,
        *,
        plan_mode: bool = True,
    ) -> dict[str, object]:
        return create_run_manifest(
            repository=self.sample_repository(),
            arguments=self.sample_arguments(),
            path_contract=self.sample_path_contract(),
            input_identity=self.sample_input_identity(),
            steps=self.sample_steps(),
            plan_mode=plan_mode,
            created_at=FIXED_TIME,
            execution_id=(
                "20260724T010203456789Z-012345abcdef"
            ),
            python_executable=sys.executable,
        )

    def test_canonical_json_is_order_independent(
        self,
    ) -> None:
        left = {
            "b": 2,
            "a": {
                "y": 2,
                "x": 1,
            },
        }

        right = {
            "a": {
                "x": 1,
                "y": 2,
            },
            "b": 2,
        }

        self.assertEqual(
            canonical_json(left),
            canonical_json(right),
        )

        self.assertEqual(
            canonical_json(left),
            '{"a":{"x":1,"y":2},"b":2}',
        )

    def test_canonical_json_rejects_nonfinite_values(
        self,
    ) -> None:
        for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
        ):
            with self.subTest(value=value):
                with self.assertRaises(
                    ValueError
                ):
                    canonical_json(
                        {"value": value}
                    )

    def test_sha256_file_matches_known_digest(
        self,
    ) -> None:
        payload = b"sleep-edfx-provenance\n"

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "payload.bin"
            )

            path.write_bytes(payload)

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(
                    payload
                ).hexdigest(),
            )

    def test_atomic_json_write_is_deterministic(
        self,
    ) -> None:
        payload = {
            "unicode": "خواب",
            "b": 2,
            "a": 1,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"

            atomic_write_json(
                path,
                payload,
            )

            first_bytes = path.read_bytes()

            atomic_write_json(
                path,
                payload,
            )

            second_bytes = path.read_bytes()

            self.assertEqual(
                first_bytes,
                second_bytes,
            )

            self.assertTrue(
                first_bytes.endswith(b"\n")
            )

            self.assertEqual(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                ),
                payload,
            )

            temporary_files = list(
                root.glob(
                    ".manifest.json.*.tmp"
                )
            )

            self.assertEqual(
                temporary_files,
                [],
            )

    def test_execution_id_is_canonical(
        self,
    ) -> None:
        execution_id = create_execution_id(
            created_at=FIXED_TIME,
            token="012345abcdef",
        )

        self.assertEqual(
            execution_id,
            (
                "20260724T010203456789Z-"
                "012345abcdef"
            ),
        )

    def test_run_id_is_stable_across_mapping_order(
        self,
    ) -> None:
        manifest = self.create_sample_manifest()

        run_id = build_run_id(
            repository_commit="a" * 40,
            arguments={
                "skip_tests": False,
                "skip_eda": False,
                "nights_per_subject": 2,
                "download_subject_count": 78,
            },
            path_contract={
                "sleep_edfx_raw_dir_is_external": True,
                "sleep_edfx_raw_dir": "/input/sleep-cassette",
                "runtime_root": "/runtime",
                "project_root": "/repo",
            },
            input_identity={
                "expected_recording_count": 153,
                "manifest_sha256": "b" * 64,
                "subset": "sleep-cassette",
                "dataset_version": "1.0.0",
                "dataset_name": "Sleep-EDF Expanded",
            },
            pipeline_plan=manifest[
                "pipeline_plan"
            ],
        )

        self.assertEqual(
            run_id,
            manifest["run_id"],
        )

    def test_run_id_changes_when_identity_changes(
        self,
    ) -> None:
        manifest = self.create_sample_manifest()
        original_run_id = manifest["run_id"]

        changed_arguments = (
            self.sample_arguments()
        )

        changed_arguments[
            "download_subject_count"
        ] = 77

        changed_run_id = build_run_id(
            repository_commit="a" * 40,
            arguments=changed_arguments,
            path_contract=self.sample_path_contract(),
            input_identity=self.sample_input_identity(),
            pipeline_plan=manifest[
                "pipeline_plan"
            ],
        )

        self.assertNotEqual(
            original_run_id,
            changed_run_id,
        )

    def test_manifest_contract_and_atomic_roundtrip(
        self,
    ) -> None:
        manifest = self.create_sample_manifest()

        validate_run_manifest(
            manifest
        )

        self.assertEqual(
            manifest["status"],
            "planned",
        )

        self.assertTrue(
            manifest["plan_mode"]
        )

        self.assertEqual(
            [
                step["status"]
                for step in manifest["steps"]
            ],
            [
                "planned",
                "planned",
            ],
        )

        self.assertEqual(
            manifest[
                "pipeline_plan"
            ][0]["command"][0],
            "<PYTHON>",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "run_manifest.json"
            )

            write_run_manifest(
                manifest,
                path,
            )

            loaded = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded,
                manifest,
            )

    def test_tampered_run_id_is_rejected(
        self,
    ) -> None:
        manifest = self.create_sample_manifest()

        tampered = copy.deepcopy(
            manifest
        )

        tampered["arguments"][
            "nights_per_subject"
        ] = 1

        with self.assertRaisesRegex(
            ValueError,
            "Run ID does not match",
        ):
            validate_run_manifest(
                tampered
            )

    @unittest.skipUnless(
        shutil.which("git"),
        "Git executable is unavailable.",
    )
    def test_repository_snapshot_detects_dirty_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run_git(
                *arguments: str,
            ) -> None:
                subprocess.run(
                    [
                        "git",
                        *arguments,
                    ],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            run_git("init")
            run_git(
                "config",
                "user.name",
                "Provenance Test",
            )
            run_git(
                "config",
                "user.email",
                "provenance@example.invalid",
            )

            tracked_path = (
                root / "tracked.txt"
            )

            tracked_path.write_text(
                "initial\n",
                encoding="utf-8",
                newline="\n",
            )

            run_git(
                "add",
                "tracked.txt",
            )

            run_git(
                "commit",
                "-m",
                "initial",
            )

            clean_snapshot = (
                repository_snapshot(root)
            )

            self.assertTrue(
                clean_snapshot["is_clean"]
            )

            self.assertEqual(
                clean_snapshot[
                    "status_entries"
                ],
                [],
            )

            self.assertEqual(
                len(
                    clean_snapshot["commit"]
                ),
                40,
            )

            tracked_path.write_text(
                "changed\n",
                encoding="utf-8",
                newline="\n",
            )

            dirty_snapshot = (
                repository_snapshot(root)
            )

            self.assertFalse(
                dirty_snapshot["is_clean"]
            )

            self.assertTrue(
                dirty_snapshot[
                    "status_entries"
                ]
            )



    def test_successful_run_lifecycle(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        self.assertEqual(
            manifest["status"],
            "running",
        )

        first_start = datetime(
            2026,
            7,
            24,
            1,
            2,
            4,
            tzinfo=timezone.utc,
        )

        first_finish = datetime(
            2026,
            7,
            24,
            1,
            2,
            5,
            tzinfo=timezone.utc,
        )

        manifest = start_step_manifest(
            manifest,
            position=1,
            started_at=first_start,
        )

        manifest = succeed_step_manifest(
            manifest,
            position=1,
            finished_at=first_finish,
            duration_seconds=1.25,
        )

        second_start = datetime(
            2026,
            7,
            24,
            1,
            2,
            6,
            tzinfo=timezone.utc,
        )

        second_finish = datetime(
            2026,
            7,
            24,
            1,
            2,
            7,
            tzinfo=timezone.utc,
        )

        manifest = start_step_manifest(
            manifest,
            position=2,
            started_at=second_start,
        )

        manifest = succeed_step_manifest(
            manifest,
            position=2,
            finished_at=second_finish,
            duration_seconds=0.75,
        )

        manifest = succeed_run_manifest(
            manifest,
            finished_at=second_finish,
            duration_seconds=3.0,
        )

        validate_run_manifest(
            manifest
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

    def test_steps_must_start_in_order(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        with self.assertRaisesRegex(
            ValueError,
            "must start in order",
        ):
            start_step_manifest(
                manifest,
                position=2,
                started_at=FIXED_TIME,
            )

    def test_run_cannot_succeed_with_pending_steps(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        with self.assertRaisesRegex(
            ValueError,
            "incomplete steps",
        ):
            succeed_run_manifest(
                manifest,
                finished_at=FIXED_TIME,
                duration_seconds=0.0,
            )

    def test_failed_step_and_run_capture_error(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        manifest = start_step_manifest(
            manifest,
            position=1,
            started_at=FIXED_TIME,
        )

        step_error = {
            "type": "CalledProcessError",
            "message": "Feature extraction failed.",
            "command": [
                "<PYTHON>",
                "-m",
                "scripts.extract_eeg_features",
            ],
        }

        manifest = fail_step_manifest(
            manifest,
            position=1,
            return_code=9,
            error=step_error,
            finished_at=FIXED_TIME,
            duration_seconds=2.5,
        )

        run_error = {
            "type": "PipelineStepError",
            "message": (
                "Pipeline stopped after step 1."
            ),
            "failed_step_position": 1,
        }

        manifest = fail_run_manifest(
            manifest,
            error=run_error,
            finished_at=FIXED_TIME,
            duration_seconds=2.5,
        )

        validate_run_manifest(
            manifest
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
            manifest["steps"][0]["return_code"],
            9,
        )

        self.assertEqual(
            manifest["steps"][1]["status"],
            "pending",
        )

        self.assertEqual(
            manifest["error"],
            run_error,
        )

    def test_failed_step_rejects_zero_return_code(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        manifest = start_step_manifest(
            manifest,
            position=1,
            started_at=FIXED_TIME,
        )

        with self.assertRaisesRegex(
            ValueError,
            "return code zero",
        ):
            fail_step_manifest(
                manifest,
                position=1,
                return_code=0,
                error={
                    "type": "Error",
                    "message": "failure",
                },
                finished_at=FIXED_TIME,
                duration_seconds=1.0,
            )

    def test_plan_manifest_cannot_start(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=True
        )

        with self.assertRaisesRegex(
            ValueError,
            "Plan-mode",
        ):
            start_run_manifest(
                manifest,
                started_at=FIXED_TIME,
            )

    def test_negative_duration_is_rejected(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        manifest = start_step_manifest(
            manifest,
            position=1,
            started_at=FIXED_TIME,
        )

        with self.assertRaisesRegex(
            ValueError,
            "must not be negative",
        ):
            succeed_step_manifest(
                manifest,
                position=1,
                finished_at=FIXED_TIME,
                duration_seconds=-0.1,
            )

    def test_lifecycle_manifest_roundtrip(
        self,
    ) -> None:
        manifest = self.create_sample_manifest(
            plan_mode=False
        )

        manifest = start_run_manifest(
            manifest,
            started_at=FIXED_TIME,
        )

        manifest = start_step_manifest(
            manifest,
            position=1,
            started_at=FIXED_TIME,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "running_manifest.json"
            )

            write_run_manifest(
                manifest,
                path,
            )

            loaded = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded,
                manifest,
            )

            self.assertEqual(
                loaded["status"],
                "running",
            )

            self.assertEqual(
                loaded["steps"][0]["status"],
                "running",
            )


if __name__ == "__main__":
    unittest.main()
