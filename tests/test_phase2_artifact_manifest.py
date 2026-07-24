from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts import config
from scripts.phase2_artifact_manifest import (
    ARTIFACT_CSV_COLUMNS,
    ArtifactSpec,
    build_artifact_manifest,
    capture_artifact_snapshot,
    phase2_artifact_specs,
    selected_artifact_specs,
    validate_artifact_specs,
    write_artifact_manifest,
)


FIXED_TIME = datetime(
    2026,
    7,
    24,
    2,
    3,
    4,
    567890,
    tzinfo=timezone.utc,
)


class Phase2ArtifactManifestTests(
    unittest.TestCase
):
    def test_registry_contains_only_current_pipeline_outputs(
        self,
    ) -> None:
        specs = phase2_artifact_specs()

        logical_names = {
            spec.logical_name
            for spec in specs
        }

        self.assertIn(
            "epoch_features",
            logical_names,
        )

        self.assertIn(
            "recording_feature_parts",
            logical_names,
        )

        self.assertIn(
            "model_input_dataset",
            logical_names,
        )

        self.assertIn(
            "eda_figures",
            logical_names,
        )

        registered_paths = {
            spec.path.resolve()
            for spec in specs
        }

        self.assertNotIn(
            config.DATABASE_PATH.resolve(),
            registered_paths,
        )

        self.assertNotIn(
            config.MODEL_READY_DATASET_PATH.resolve(),
            registered_paths,
        )

        self.assertNotIn(
            config.PREPROCESSED_FEATURES_PATH.resolve(),
            registered_paths,
        )

    def test_selected_specs_follow_selected_steps(
        self,
    ) -> None:
        specs = selected_artifact_specs(
            [
                "Inspect verified EDF pairs",
                "Build trimmed epoch metadata",
            ]
        )

        self.assertEqual(
            {
                spec.logical_name
                for spec in specs
            },
            {
                "edf_inspection_report",
                "epoch_metadata",
                "epoch_summary",
            },
        )

    def test_snapshot_fingerprints_csv_and_directory_members(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory)

            csv_path = (
                runtime_root
                / "data"
                / "dataset.csv"
            )

            csv_path.parent.mkdir(
                parents=True
            )

            csv_path.write_text(
                "a,b\n1,2\n3,4\n",
                encoding="utf-8",
                newline="\n",
            )

            figure_dir = (
                runtime_root
                / "reports"
                / "figures"
            )

            figure_dir.mkdir(
                parents=True
            )

            png_path = (
                figure_dir
                / "plot.png"
            )

            png_payload = b"\x89PNG\r\n\x1a\npayload"

            png_path.write_bytes(
                png_payload
            )

            specs = (
                ArtifactSpec(
                    logical_name="dataset",
                    producer_step="Build dataset",
                    artifact_type="dataset",
                    path=csv_path,
                ),
                ArtifactSpec(
                    logical_name="figures",
                    producer_step="Generate figures",
                    artifact_type="figure",
                    path=figure_dir,
                    recursive=True,
                ),
            )

            snapshot = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            csv_record = snapshot["files"][
                "data/dataset.csv"
            ]

            self.assertEqual(
                csv_record["row_count"],
                2,
            )

            self.assertEqual(
                csv_record["column_count"],
                2,
            )

            png_record = snapshot["files"][
                "reports/figures/plot.png"
            ]

            self.assertEqual(
                png_record["media_type"],
                "image/png",
            )

            self.assertEqual(
                png_record["sha256"],
                hashlib.sha256(
                    png_payload
                ).hexdigest(),
            )

    def test_manifest_classifies_created_modified_and_reused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory)

            modified_path = (
                runtime_root / "modified.csv"
            )

            reused_path = (
                runtime_root / "reused.json"
            )

            created_path = (
                runtime_root / "created.csv"
            )

            modified_path.write_text(
                "x\n1\n",
                encoding="utf-8",
                newline="\n",
            )

            reused_path.write_text(
                '{"ok":true}\n',
                encoding="utf-8",
                newline="\n",
            )

            specs = (
                ArtifactSpec(
                    logical_name="modified",
                    producer_step="Step",
                    artifact_type="dataset",
                    path=modified_path,
                ),
                ArtifactSpec(
                    logical_name="reused",
                    producer_step="Step",
                    artifact_type="report",
                    path=reused_path,
                ),
                ArtifactSpec(
                    logical_name="created",
                    producer_step="Step",
                    artifact_type="dataset",
                    path=created_path,
                ),
            )

            before = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            modified_path.write_text(
                "x\n1\n2\n",
                encoding="utf-8",
                newline="\n",
            )

            created_path.write_text(
                "y\n9\n",
                encoding="utf-8",
                newline="\n",
            )

            after = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            manifest = build_artifact_manifest(
                run_id="a" * 64,
                execution_id="execution-1",
                selected_steps=["Step"],
                before_snapshot=before,
                after_snapshot=after,
                created_at=FIXED_TIME,
            )

            changes = {
                artifact["logical_name"]: (
                    artifact["change_type"]
                )
                for artifact in manifest[
                    "artifacts"
                ]
            }

            self.assertEqual(
                changes,
                {
                    "created": "created",
                    "modified": "modified",
                    "reused": "reused",
                },
            )

            self.assertEqual(
                manifest["status"],
                "valid",
            )

    def test_missing_required_spec_invalidates_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory)

            specs = (
                ArtifactSpec(
                    logical_name="missing",
                    producer_step="Step",
                    artifact_type="dataset",
                    path=(
                        runtime_root
                        / "missing.csv"
                    ),
                ),
            )

            before = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            after = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            manifest = build_artifact_manifest(
                run_id="b" * 64,
                execution_id="execution-2",
                selected_steps=["Step"],
                before_snapshot=before,
                after_snapshot=after,
                created_at=FIXED_TIME,
            )

            self.assertEqual(
                manifest["status"],
                "invalid",
            )

            self.assertEqual(
                len(
                    manifest[
                        "missing_required_specs"
                    ]
                ),
                1,
            )

    def test_removed_required_artifact_invalidates_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory)

            path = (
                runtime_root
                / "dataset.csv"
            )

            path.write_text(
                "x\n1\n",
                encoding="utf-8",
                newline="\n",
            )

            specs = (
                ArtifactSpec(
                    logical_name="dataset",
                    producer_step="Step",
                    artifact_type="dataset",
                    path=path,
                ),
            )

            before = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            path.unlink()

            after = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            manifest = build_artifact_manifest(
                run_id="c" * 64,
                execution_id="execution-3",
                selected_steps=["Step"],
                before_snapshot=before,
                after_snapshot=after,
                created_at=FIXED_TIME,
            )

            self.assertEqual(
                manifest["status"],
                "invalid",
            )

            self.assertEqual(
                manifest[
                    "removed_required_paths"
                ],
                [
                    "dataset.csv"
                ],
            )

    def test_manifest_json_and_csv_writes_are_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory)

            dataset_path = (
                runtime_root
                / "dataset.csv"
            )

            dataset_path.write_text(
                "x,y\n1,2\n",
                encoding="utf-8",
                newline="\n",
            )

            specs = (
                ArtifactSpec(
                    logical_name="dataset",
                    producer_step="Step",
                    artifact_type="dataset",
                    path=dataset_path,
                ),
            )

            before = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            after = capture_artifact_snapshot(
                specs,
                runtime_root=runtime_root,
                captured_at=FIXED_TIME,
            )

            manifest = build_artifact_manifest(
                run_id="d" * 64,
                execution_id="execution-4",
                selected_steps=["Step"],
                before_snapshot=before,
                after_snapshot=after,
                created_at=FIXED_TIME,
            )

            json_path = (
                runtime_root
                / "artifact_manifest.json"
            )

            csv_path = (
                runtime_root
                / "artifact_manifest.csv"
            )

            write_artifact_manifest(
                manifest=manifest,
                json_output_path=json_path,
                csv_output_path=csv_path,
            )

            first_json = json_path.read_bytes()
            first_csv = csv_path.read_bytes()

            write_artifact_manifest(
                manifest=manifest,
                json_output_path=json_path,
                csv_output_path=csv_path,
            )

            self.assertEqual(
                first_json,
                json_path.read_bytes(),
            )

            self.assertEqual(
                first_csv,
                csv_path.read_bytes(),
            )

            loaded_json = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                loaded_json,
                manifest,
            )

            with csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file_handle:
                reader = csv.DictReader(
                    file_handle
                )

                self.assertEqual(
                    reader.fieldnames,
                    ARTIFACT_CSV_COLUMNS,
                )

                rows = list(reader)

            self.assertEqual(
                len(rows),
                1,
            )

    def test_duplicate_logical_names_and_external_paths_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as runtime_directory:
            with tempfile.TemporaryDirectory() as external_directory:
                runtime_root = Path(
                    runtime_directory
                )

                duplicate_specs = (
                    ArtifactSpec(
                        logical_name="same",
                        producer_step="Step",
                        artifact_type="dataset",
                        path=(
                            runtime_root / "a.csv"
                        ),
                    ),
                    ArtifactSpec(
                        logical_name="same",
                        producer_step="Step",
                        artifact_type="dataset",
                        path=(
                            runtime_root / "b.csv"
                        ),
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "Duplicate artifact logical name",
                ):
                    validate_artifact_specs(
                        duplicate_specs,
                        runtime_root=runtime_root,
                    )

                external_spec = (
                    ArtifactSpec(
                        logical_name="external",
                        producer_step="Step",
                        artifact_type="dataset",
                        path=(
                            Path(external_directory)
                            / "outside.csv"
                        ),
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "below runtime root",
                ):
                    validate_artifact_specs(
                        external_spec,
                        runtime_root=runtime_root,
                    )


if __name__ == "__main__":
    unittest.main()
