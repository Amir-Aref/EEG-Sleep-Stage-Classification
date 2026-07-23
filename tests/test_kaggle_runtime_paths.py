from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RUNTIME_ENV = "EEG_RUNTIME_ROOT"
RAW_ENV = "EEG_SLEEP_EDFX_RAW_DIR"

PROBE_SOURCE = r"""
import json
import os

from scripts import config


if os.environ.get("EEG_TEST_ENSURE_DIRECTORIES") == "1":
    config.ensure_runtime_directories()

print(
    json.dumps(
        config.runtime_path_contract(),
        sort_keys=True,
    )
)
"""


class KaggleRuntimePathTests(unittest.TestCase):
    def run_probe(
        self,
        *,
        runtime_root: str | None = None,
        raw_directory: str | None = None,
        ensure_directories: bool = False,
    ) -> dict[str, object]:
        environment = os.environ.copy()

        for variable_name in (
            RUNTIME_ENV,
            RAW_ENV,
            "EEG_TEST_ENSURE_DIRECTORIES",
        ):
            environment.pop(
                variable_name,
                None,
            )

        if runtime_root is not None:
            environment[RUNTIME_ENV] = runtime_root

        if raw_directory is not None:
            environment[RAW_ENV] = raw_directory

        if ensure_directories:
            environment[
                "EEG_TEST_ENSURE_DIRECTORIES"
            ] = "1"

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                PROBE_SOURCE,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        output_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.strip()
        ]

        self.assertTrue(
            output_lines,
            msg=completed.stderr,
        )

        return json.loads(
            output_lines[-1]
        )

    def test_default_paths_preserve_repository_layout(
        self,
    ) -> None:
        contract = self.run_probe()
        expected_root = PROJECT_ROOT.resolve()

        self.assertEqual(
            Path(contract["project_root"]),
            expected_root,
        )

        self.assertEqual(
            Path(contract["runtime_root"]),
            expected_root,
        )

        self.assertEqual(
            Path(contract["data_dir"]),
            expected_root / "data",
        )

        self.assertEqual(
            Path(contract["sleep_edfx_raw_dir"]),
            (
                expected_root
                / "data"
                / "raw"
                / "sleep-edfx"
                / "1.0.0"
                / "sleep-cassette"
            ),
        )

        self.assertFalse(
            contract[
                "sleep_edfx_raw_dir_is_external"
            ]
        )

    def test_runtime_override_relocates_writable_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = (
                Path(directory)
                / "runtime"
            ).resolve()

            contract = self.run_probe(
                runtime_root=str(runtime_root)
            )

            self.assertEqual(
                Path(contract["project_root"]),
                PROJECT_ROOT.resolve(),
            )

            self.assertEqual(
                Path(contract["runtime_root"]),
                runtime_root,
            )

            self.assertEqual(
                Path(contract["data_dir"]),
                runtime_root / "data",
            )

            self.assertEqual(
                Path(contract["database_path"]),
                (
                    runtime_root
                    / "database"
                    / "sleep_eeg.db"
                ),
            )

            self.assertEqual(
                Path(contract["reports_dir"]),
                runtime_root / "reports",
            )

            self.assertEqual(
                Path(contract["eda_output_dir"]),
                (
                    runtime_root
                    / "reports"
                    / "eda"
                ),
            )

    def test_external_raw_override_is_independent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            runtime_root = (
                root / "working"
            ).resolve()

            raw_directory = (
                root
                / "input"
                / "sleep-cassette"
            ).resolve()

            contract = self.run_probe(
                runtime_root=str(runtime_root),
                raw_directory=str(raw_directory),
            )

            self.assertEqual(
                Path(
                    contract[
                        "sleep_edfx_raw_dir"
                    ]
                ),
                raw_directory,
            )

            self.assertTrue(
                contract[
                    "sleep_edfx_raw_dir_is_external"
                ]
            )

            runtime_directories = {
                Path(value)
                for value in contract[
                    "runtime_directories"
                ]
            }

            self.assertNotIn(
                raw_directory,
                runtime_directories,
            )

            self.assertIn(
                (
                    runtime_root
                    / "data"
                    / "raw"
                ),
                runtime_directories,
            )

            self.assertIn(
                (
                    runtime_root
                    / "data"
                    / "metadata"
                ),
                runtime_directories,
            )

    def test_relative_runtime_override_is_project_relative(
        self,
    ) -> None:
        relative_value = (
            "runtime-relative-contract-test"
        )

        contract = self.run_probe(
            runtime_root=relative_value
        )

        self.assertEqual(
            Path(contract["runtime_root"]),
            (
                PROJECT_ROOT
                / relative_value
            ).resolve(),
        )

    def test_directory_creation_skips_external_raw_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            runtime_root = (
                root / "working"
            ).resolve()

            external_raw = (
                root
                / "read-only-input"
                / "sleep-cassette"
            ).resolve()

            self.assertFalse(
                external_raw.exists()
            )

            contract = self.run_probe(
                runtime_root=str(runtime_root),
                raw_directory=str(external_raw),
                ensure_directories=True,
            )

            self.assertTrue(
                (
                    runtime_root
                    / "data"
                    / "metadata"
                ).is_dir()
            )

            self.assertTrue(
                (
                    runtime_root
                    / "data"
                    / "interim"
                    / "features_by_recording"
                ).is_dir()
            )

            self.assertTrue(
                (
                    runtime_root
                    / "reports"
                ).is_dir()
            )

            self.assertFalse(
                external_raw.exists()
            )

            self.assertTrue(
                contract[
                    "sleep_edfx_raw_dir_is_external"
                ]
            )


if __name__ == "__main__":
    unittest.main()
