"""Artifact registry and deterministic manifests for Phase 2 outputs."""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

try:
    from .config import (
        EDA_OUTLIER_SUMMARY_PATH,
        EDA_OUTPUT_DIR,
        EDA_STAGE_SUMMARY_PATH,
        EDA_SUBJECT_SUMMARY_PATH,
        EDA_SUMMARY_PATH,
        EDF_INSPECTION_REPORT_PATH,
        EPOCH_FEATURES_PATH,
        EPOCH_METADATA_PATH,
        EPOCH_SUMMARY_PATH,
        FEATURE_AUDIT_REPORT_PATH,
        FEATURE_EXTRACTION_SUMMARY_PATH,
        FEATURE_PARTS_DIR,
        FEATURE_SCHEMA_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
        RUNTIME_ROOT,
        SLEEP_EDFX_CHECKSUMS_PATH,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_MANIFEST_PATH,
        STAGE_FEATURE_SUMMARY_PATH,
        SUBJECT_FEATURE_SUMMARY_PATH,
    )
    from .run_provenance import (
        atomic_write_json,
        atomic_write_text,
        isoformat_utc,
        sha256_file,
        utc_now,
    )
except ImportError:
    from config import (
        EDA_OUTLIER_SUMMARY_PATH,
        EDA_OUTPUT_DIR,
        EDA_STAGE_SUMMARY_PATH,
        EDA_SUBJECT_SUMMARY_PATH,
        EDA_SUMMARY_PATH,
        EDF_INSPECTION_REPORT_PATH,
        EPOCH_FEATURES_PATH,
        EPOCH_METADATA_PATH,
        EPOCH_SUMMARY_PATH,
        FEATURE_AUDIT_REPORT_PATH,
        FEATURE_EXTRACTION_SUMMARY_PATH,
        FEATURE_PARTS_DIR,
        FEATURE_SCHEMA_PATH,
        MODEL_FEATURE_SCHEMA_PATH,
        MODEL_INPUT_DATASET_PATH,
        RUNTIME_ROOT,
        SLEEP_EDFX_CHECKSUMS_PATH,
        SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        SLEEP_EDFX_MANIFEST_PATH,
        STAGE_FEATURE_SUMMARY_PATH,
        SUBJECT_FEATURE_SUMMARY_PATH,
    )
    from run_provenance import (
        atomic_write_json,
        atomic_write_text,
        isoformat_utc,
        sha256_file,
        utc_now,
    )


ARTIFACT_MANIFEST_SCHEMA_VERSION: Final[int] = 1

ARTIFACT_MANIFEST_TYPE: Final[str] = (
    "phase2_artifact_manifest"
)

CHANGE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "created",
        "modified",
        "reused",
        "removed",
    }
)

MEDIA_TYPES: Final[dict[str, str]] = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".txt": "text/plain",
}

ARTIFACT_CSV_COLUMNS: Final[list[str]] = [
    "logical_name",
    "producer_step",
    "artifact_type",
    "relative_path",
    "change_type",
    "required",
    "before_exists",
    "after_exists",
    "before_size_bytes",
    "after_size_bytes",
    "before_sha256",
    "after_sha256",
    "row_count",
    "column_count",
    "media_type",
]


@dataclass(frozen=True)
class ArtifactSpec:
    """One declared Phase 2 output contract."""

    logical_name: str
    producer_step: str
    artifact_type: str
    path: Path
    required: bool = True
    recursive: bool = False
    glob_pattern: str = "*"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            Path(self.path),
        )

        for field_name in (
            "logical_name",
            "producer_step",
            "artifact_type",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not isinstance(
                value,
                str,
            ) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a "
                    "non-empty string."
                )

        if not isinstance(
            self.required,
            bool,
        ):
            raise TypeError(
                "required must be boolean."
            )

        if not isinstance(
            self.recursive,
            bool,
        ):
            raise TypeError(
                "recursive must be boolean."
            )

        if not isinstance(
            self.glob_pattern,
            str,
        ) or not self.glob_pattern:
            raise ValueError(
                "glob_pattern must be non-empty."
            )


def phase2_artifact_specs() -> tuple[ArtifactSpec, ...]:
    """Return the authoritative Phase 2 output registry."""

    manifest_summary_path = (
        SLEEP_EDFX_MANIFEST_PATH.with_suffix(
            ".summary.json"
        )
    )

    return (
        ArtifactSpec(
            logical_name="official_dataset_manifest",
            producer_step=(
                "Refresh official dataset manifest"
            ),
            artifact_type="dataset_manifest",
            path=SLEEP_EDFX_MANIFEST_PATH,
        ),
        ArtifactSpec(
            logical_name="official_checksum_source",
            producer_step=(
                "Refresh official dataset manifest"
            ),
            artifact_type="checksum_source",
            path=SLEEP_EDFX_CHECKSUMS_PATH,
        ),
        ArtifactSpec(
            logical_name="official_manifest_summary",
            producer_step=(
                "Refresh official dataset manifest"
            ),
            artifact_type="metadata_summary",
            path=manifest_summary_path,
        ),
        ArtifactSpec(
            logical_name="download_inventory",
            producer_step=(
                "Download and verify raw EDF files"
            ),
            artifact_type="download_inventory",
            path=SLEEP_EDFX_DOWNLOAD_INVENTORY_PATH,
        ),
        ArtifactSpec(
            logical_name="edf_inspection_report",
            producer_step="Inspect verified EDF pairs",
            artifact_type="validation_report",
            path=EDF_INSPECTION_REPORT_PATH,
        ),
        ArtifactSpec(
            logical_name="epoch_metadata",
            producer_step=(
                "Build trimmed epoch metadata"
            ),
            artifact_type="epoch_dataset",
            path=EPOCH_METADATA_PATH,
        ),
        ArtifactSpec(
            logical_name="epoch_summary",
            producer_step=(
                "Build trimmed epoch metadata"
            ),
            artifact_type="metadata_summary",
            path=EPOCH_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="recording_feature_parts",
            producer_step="Extract EEG features",
            artifact_type="feature_partition",
            path=FEATURE_PARTS_DIR,
            recursive=True,
            glob_pattern="*.csv",
        ),
        ArtifactSpec(
            logical_name="epoch_features",
            producer_step="Extract EEG features",
            artifact_type="feature_dataset",
            path=EPOCH_FEATURES_PATH,
        ),
        ArtifactSpec(
            logical_name="feature_extraction_summary",
            producer_step="Extract EEG features",
            artifact_type="metadata_summary",
            path=FEATURE_EXTRACTION_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="feature_schema",
            producer_step="Extract EEG features",
            artifact_type="schema",
            path=FEATURE_SCHEMA_PATH,
        ),
        ArtifactSpec(
            logical_name="feature_audit_report",
            producer_step="Audit EEG features",
            artifact_type="validation_report",
            path=FEATURE_AUDIT_REPORT_PATH,
        ),
        ArtifactSpec(
            logical_name="stage_feature_summary",
            producer_step="Audit EEG features",
            artifact_type="analysis_summary",
            path=STAGE_FEATURE_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="subject_feature_summary",
            producer_step="Audit EEG features",
            artifact_type="analysis_summary",
            path=SUBJECT_FEATURE_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="model_input_dataset",
            producer_step=(
                "Build leakage-safe model input"
            ),
            artifact_type="model_input_dataset",
            path=MODEL_INPUT_DATASET_PATH,
        ),
        ArtifactSpec(
            logical_name="model_feature_schema",
            producer_step=(
                "Build leakage-safe model input"
            ),
            artifact_type="schema",
            path=MODEL_FEATURE_SCHEMA_PATH,
        ),
        ArtifactSpec(
            logical_name="eda_figures",
            producer_step=(
                "Generate EEG EDA reports"
            ),
            artifact_type="figure",
            path=EDA_OUTPUT_DIR,
            recursive=True,
            glob_pattern="*",
        ),
        ArtifactSpec(
            logical_name="eda_outlier_summary",
            producer_step=(
                "Generate EEG EDA reports"
            ),
            artifact_type="analysis_summary",
            path=EDA_OUTLIER_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="eda_stage_summary",
            producer_step=(
                "Generate EEG EDA reports"
            ),
            artifact_type="analysis_summary",
            path=EDA_STAGE_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="eda_subject_summary",
            producer_step=(
                "Generate EEG EDA reports"
            ),
            artifact_type="analysis_summary",
            path=EDA_SUBJECT_SUMMARY_PATH,
        ),
        ArtifactSpec(
            logical_name="eda_summary",
            producer_step=(
                "Generate EEG EDA reports"
            ),
            artifact_type="analysis_summary",
            path=EDA_SUMMARY_PATH,
        ),
    )


def selected_artifact_specs(
    selected_steps: Sequence[str],
    *,
    registry: Sequence[ArtifactSpec] | None = None,
) -> tuple[ArtifactSpec, ...]:
    """Select output contracts belonging to executable steps."""

    if registry is None:
        registry = phase2_artifact_specs()

    selected_step_names = {
        str(step_name)
        for step_name in selected_steps
    }

    return tuple(
        spec
        for spec in registry
        if spec.producer_step
        in selected_step_names
    )


def runtime_relative_path(
    path: Path,
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> str:
    """Create a portable POSIX-style path below runtime root."""

    resolved_root = runtime_root.resolve()
    resolved_path = path.resolve()

    try:
        relative_path = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise ValueError(
            "Artifact paths must be located below "
            f"runtime root: {resolved_path}"
        ) from error

    return relative_path.as_posix()


def validate_artifact_specs(
    specs: Sequence[ArtifactSpec],
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> None:
    """Validate uniqueness and runtime containment."""

    logical_names: set[str] = set()

    for spec in specs:
        if spec.logical_name in logical_names:
            raise ValueError(
                "Duplicate artifact logical name: "
                f"{spec.logical_name}"
            )

        logical_names.add(
            spec.logical_name
        )

        runtime_relative_path(
            spec.path,
            runtime_root=runtime_root,
        )


def artifact_spec_payload(
    spec: ArtifactSpec,
    *,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    """Serialize one artifact specification."""

    return {
        "logical_name": spec.logical_name,
        "producer_step": spec.producer_step,
        "artifact_type": spec.artifact_type,
        "relative_path": runtime_relative_path(
            spec.path,
            runtime_root=runtime_root,
        ),
        "required": spec.required,
        "recursive": spec.recursive,
        "glob_pattern": spec.glob_pattern,
    }


def csv_shape(
    path: Path,
) -> tuple[int, int]:
    """Return data-row and header-column counts for a CSV."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        reader = csv.reader(
            file_handle
        )

        try:
            header = next(reader)
        except StopIteration:
            return (
                0,
                0,
            )

        column_count = len(header)
        row_count = 0

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if len(row) != column_count:
                raise ValueError(
                    "Inconsistent CSV column count in "
                    f"{path} at row {row_number}: "
                    f"expected {column_count}, "
                    f"found {len(row)}"
                )

            row_count += 1

    return (
        row_count,
        column_count,
    )


def file_fingerprint(
    path: Path,
    *,
    spec: ArtifactSpec,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    """Fingerprint one concrete output file."""

    resolved_path = path.resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            resolved_path
        )

    suffix = resolved_path.suffix.lower()

    row_count: int | None = None
    column_count: int | None = None

    if suffix == ".csv":
        (
            row_count,
            column_count,
        ) = csv_shape(
            resolved_path
        )

    return {
        "logical_name": spec.logical_name,
        "producer_step": spec.producer_step,
        "artifact_type": spec.artifact_type,
        "relative_path": runtime_relative_path(
            resolved_path,
            runtime_root=runtime_root,
        ),
        "file_name": resolved_path.name,
        "suffix": suffix,
        "media_type": MEDIA_TYPES.get(
            suffix,
            "application/octet-stream",
        ),
        "size_bytes": (
            resolved_path.stat().st_size
        ),
        "sha256": sha256_file(
            resolved_path
        ),
        "row_count": row_count,
        "column_count": column_count,
        "required": spec.required,
    }


def _artifact_members(
    spec: ArtifactSpec,
) -> list[Path]:
    """Resolve concrete files represented by one spec."""

    path = spec.path.resolve()

    if spec.recursive:
        if not path.is_dir():
            return []

        return sorted(
            (
                member.resolve()
                for member in path.rglob(
                    spec.glob_pattern
                )
                if member.is_file()
            ),
            key=lambda value: value.as_posix(),
        )

    if path.is_file():
        return [
            path
        ]

    return []


def capture_artifact_snapshot(
    specs: Sequence[ArtifactSpec],
    *,
    runtime_root: Path = RUNTIME_ROOT,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Capture deterministic output fingerprints."""

    validate_artifact_specs(
        specs,
        runtime_root=runtime_root,
    )

    if captured_at is None:
        captured_at = utc_now()

    files: dict[
        str,
        dict[str, Any],
    ] = {}

    missing_specs: list[
        dict[str, Any]
    ] = []

    spec_payloads = [
        artifact_spec_payload(
            spec,
            runtime_root=runtime_root,
        )
        for spec in specs
    ]

    for spec in specs:
        members = _artifact_members(
            spec
        )

        if not members:
            missing_specs.append(
                artifact_spec_payload(
                    spec,
                    runtime_root=runtime_root,
                )
            )
            continue

        for member in members:
            fingerprint = file_fingerprint(
                member,
                spec=spec,
                runtime_root=runtime_root,
            )

            relative_path = str(
                fingerprint["relative_path"]
            )

            if relative_path in files:
                raise ValueError(
                    "Artifact file matched multiple "
                    f"specifications: {relative_path}"
                )

            files[relative_path] = (
                fingerprint
            )

    return {
        "captured_at_utc": isoformat_utc(
            captured_at
        ),
        "specs": spec_payloads,
        "files": {
            key: files[key]
            for key in sorted(files)
        },
        "missing_specs": sorted(
            missing_specs,
            key=lambda item: (
                item["logical_name"]
            ),
        ),
    }


def _artifact_change_record(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare one path before and after execution."""

    if before is None and after is None:
        raise ValueError(
            "At least one artifact state is required."
        )

    source = (
        after
        if after is not None
        else before
    )

    assert source is not None

    if before is None:
        change_type = "created"
    elif after is None:
        change_type = "removed"
    elif (
        before["sha256"]
        == after["sha256"]
        and before["size_bytes"]
        == after["size_bytes"]
    ):
        change_type = "reused"
    else:
        change_type = "modified"

    if change_type not in CHANGE_TYPES:
        raise AssertionError(
            change_type
        )

    return {
        "logical_name": source["logical_name"],
        "producer_step": source["producer_step"],
        "artifact_type": source["artifact_type"],
        "relative_path": source["relative_path"],
        "change_type": change_type,
        "required": bool(
            source["required"]
        ),
        "before_exists": before is not None,
        "after_exists": after is not None,
        "before_size_bytes": (
            before["size_bytes"]
            if before is not None
            else None
        ),
        "after_size_bytes": (
            after["size_bytes"]
            if after is not None
            else None
        ),
        "before_sha256": (
            before["sha256"]
            if before is not None
            else None
        ),
        "after_sha256": (
            after["sha256"]
            if after is not None
            else None
        ),
        "row_count": (
            after["row_count"]
            if after is not None
            else None
        ),
        "column_count": (
            after["column_count"]
            if after is not None
            else None
        ),
        "media_type": source["media_type"],
    }


def build_artifact_manifest(
    *,
    run_id: str,
    execution_id: str,
    selected_steps: Sequence[str],
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Compare snapshots and build the final artifact manifest."""

    if (
        not isinstance(run_id, str)
        or len(run_id) != 64
    ):
        raise ValueError(
            "run_id must contain 64 characters."
        )

    if (
        not isinstance(execution_id, str)
        or not execution_id
    ):
        raise ValueError(
            "execution_id must be non-empty."
        )

    if (
        before_snapshot.get("specs")
        != after_snapshot.get("specs")
    ):
        raise ValueError(
            "Before and after artifact "
            "specifications do not match."
        )

    if created_at is None:
        created_at = utc_now()

    before_files = before_snapshot.get(
        "files"
    )

    after_files = after_snapshot.get(
        "files"
    )

    if not isinstance(
        before_files,
        Mapping,
    ) or not isinstance(
        after_files,
        Mapping,
    ):
        raise ValueError(
            "Artifact snapshots must contain "
            "file mappings."
        )

    relative_paths = sorted(
        set(before_files)
        | set(after_files)
    )

    artifacts = [
        _artifact_change_record(
            before_files.get(
                relative_path
            ),
            after_files.get(
                relative_path
            ),
        )
        for relative_path in relative_paths
    ]

    missing_required_specs = [
        item
        for item in after_snapshot.get(
            "missing_specs",
            [],
        )
        if item.get("required") is True
    ]

    removed_required_paths = [
        artifact["relative_path"]
        for artifact in artifacts
        if (
            artifact["required"]
            and artifact["change_type"]
            == "removed"
        )
    ]

    status = (
        "valid"
        if (
            not missing_required_specs
            and not removed_required_paths
        )
        else "invalid"
    )

    change_counts = Counter(
        artifact["change_type"]
        for artifact in artifacts
    )

    return {
        "schema_version": (
            ARTIFACT_MANIFEST_SCHEMA_VERSION
        ),
        "manifest_type": (
            ARTIFACT_MANIFEST_TYPE
        ),
        "run_id": run_id,
        "execution_id": execution_id,
        "created_at_utc": isoformat_utc(
            created_at
        ),
        "status": status,
        "selected_steps": [
            str(step)
            for step in selected_steps
        ],
        "artifact_count": len(
            artifacts
        ),
        "counts_by_change_type": {
            change_type: int(
                change_counts.get(
                    change_type,
                    0,
                )
            )
            for change_type in sorted(
                CHANGE_TYPES
            )
        },
        "missing_required_specs": (
            missing_required_specs
        ),
        "removed_required_paths": (
            removed_required_paths
        ),
        "specs": after_snapshot["specs"],
        "artifacts": artifacts,
    }


def artifact_manifest_csv_text(
    manifest: Mapping[str, Any],
) -> str:
    """Serialize artifact records to deterministic CSV."""

    output = io.StringIO(
        newline=""
    )

    writer = csv.DictWriter(
        output,
        fieldnames=ARTIFACT_CSV_COLUMNS,
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()

    for artifact in manifest.get(
        "artifacts",
        [],
    ):
        row = {
            column: artifact.get(
                column
            )
            for column in ARTIFACT_CSV_COLUMNS
        }

        writer.writerow(
            {
                key: (
                    ""
                    if value is None
                    else value
                )
                for key, value in row.items()
            }
        )

    return output.getvalue()


def write_artifact_manifest(
    *,
    manifest: Mapping[str, Any],
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    """Write JSON and CSV artifact manifests atomically."""

    atomic_write_json(
        path=json_output_path,
        payload=manifest,
    )

    atomic_write_text(
        path=csv_output_path,
        text=artifact_manifest_csv_text(
            manifest
        ),
    )
