"""Deterministic and atomic provenance helpers for pipeline runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


RUN_MANIFEST_SCHEMA_VERSION: Final[int] = 1

RUN_MANIFEST_TYPE: Final[str] = (
    "eeg_pipeline_run"
)

RUN_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "planned",
        "pending",
        "running",
        "succeeded",
        "failed",
    }
)

STEP_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "planned",
        "pending",
        "running",
        "succeeded",
        "failed",
        "skipped",
    }
)

GIT_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{40}$"
)

EXECUTION_TOKEN_PATTERN: Final[
    re.Pattern[str]
] = re.compile(
    r"^[0-9a-f]{12}$"
)

FILE_HASH_CHUNK_SIZE_BYTES: Final[int] = (
    1024 * 1024
)


def isoformat_utc(
    value: datetime,
) -> str:
    """Serialize one timezone-aware datetime in canonical UTC form."""

    if value.tzinfo is None:
        raise ValueError(
            "Datetime values must be timezone-aware."
        )

    return (
        value.astimezone(timezone.utc)
        .isoformat(
            timespec="microseconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(
        timezone.utc
    )


def normalize_json_value(
    value: Any,
) -> Any:
    """Convert supported values into deterministic JSON-safe values."""

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite floating-point values "
                "are not allowed in provenance data."
            )

        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return isoformat_utc(value)

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Provenance mapping keys "
                    "must be strings."
                )

            normalized[key] = (
                normalize_json_value(item)
            )

        return normalized

    if isinstance(value, Sequence) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return [
            normalize_json_value(item)
            for item in value
        ]

    raise TypeError(
        "Unsupported provenance value type: "
        f"{type(value).__name__}"
    )


def canonical_json(
    value: Any,
) -> str:
    """Serialize a value to compact canonical JSON."""

    return json.dumps(
        normalize_json_value(value),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    )


def pretty_json(
    value: Any,
) -> str:
    """Serialize a value to deterministic human-readable JSON."""

    return (
        json.dumps(
            normalize_json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def sha256_bytes(
    payload: bytes,
) -> str:
    """Calculate the SHA-256 digest of an in-memory payload."""

    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    """Calculate the SHA-256 digest of one local file."""

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(
                FILE_HASH_CHUNK_SIZE_BYTES
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    """Atomically write UTF-8 text in the destination directory."""

    if not isinstance(text, str):
        raise TypeError(
            "Atomic text payload must be a string."
        )

    path = path.resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as file_handle:
            file_handle.write(text)
            file_handle.flush()
            os.fsync(
                file_handle.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    """Atomically write deterministic human-readable JSON."""

    atomic_write_text(
        path=path,
        text=pretty_json(payload),
    )


def normalize_command(
    command: Sequence[object],
    *,
    python_executable: str | Path | None = None,
) -> list[str]:
    """Normalize a command for portable run identity calculation."""

    if isinstance(
        command,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "Command must be a sequence of arguments, "
            "not one command string."
        )

    normalized = [
        str(part)
        for part in command
    ]

    if not normalized:
        raise ValueError(
            "Pipeline commands must not be empty."
        )

    if python_executable is not None:
        command_python = Path(
            normalized[0]
        )

        expected_python = Path(
            python_executable
        )

        try:
            same_python = (
                command_python.resolve()
                == expected_python.resolve()
            )
        except OSError:
            same_python = (
                str(command_python)
                == str(expected_python)
            )

        if same_python:
            normalized[0] = "<PYTHON>"

    return normalized


def normalize_pipeline_plan(
    steps: Sequence[Mapping[str, Any]],
    *,
    python_executable: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize ordered pipeline steps for storage and hashing."""

    normalized_steps: list[
        dict[str, Any]
    ] = []

    for position, step in enumerate(
        steps,
        start=1,
    ):
        if "name" not in step:
            raise ValueError(
                "Pipeline step is missing its name."
            )

        if "command" not in step:
            raise ValueError(
                "Pipeline step is missing its command."
            )

        name = str(
            step["name"]
        ).strip()

        if not name:
            raise ValueError(
                "Pipeline step names must not be empty."
            )

        normalized_steps.append(
            {
                "position": position,
                "name": name,
                "command": normalize_command(
                    step["command"],
                    python_executable=(
                        python_executable
                    ),
                ),
            }
        )

    if not normalized_steps:
        raise ValueError(
            "Pipeline plan must contain at least one step."
        )

    return normalized_steps


def repository_snapshot(
    project_root: Path,
) -> dict[str, Any]:
    """Capture the current Git commit, branch and dirty state."""

    project_root = (
        project_root.resolve()
    )

    def run_git(
        *arguments: str,
    ) -> str:
        completed = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return completed.stdout.strip()

    commit = run_git(
        "rev-parse",
        "HEAD",
    )

    branch = run_git(
        "branch",
        "--show-current",
    )

    status_text = run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    status_entries = [
        line
        for line in status_text.splitlines()
        if line
    ]

    return {
        "commit": commit,
        "branch": branch or None,
        "is_clean": not status_entries,
        "status_entries": status_entries,
    }


def build_run_identity_payload(
    *,
    repository_commit: str,
    arguments: Mapping[str, Any],
    path_contract: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    pipeline_plan: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Build the exact deterministic payload used to derive run_id."""

    repository_commit = (
        repository_commit.strip().lower()
    )

    if not GIT_COMMIT_PATTERN.fullmatch(
        repository_commit
    ):
        raise ValueError(
            "Repository commit must be a "
            "40-character Git SHA."
        )

    return {
        "schema_version": (
            RUN_MANIFEST_SCHEMA_VERSION
        ),
        "repository_commit": (
            repository_commit
        ),
        "arguments": normalize_json_value(
            arguments
        ),
        "path_contract": normalize_json_value(
            path_contract
        ),
        "input_identity": normalize_json_value(
            input_identity
        ),
        "pipeline_plan": normalize_json_value(
            pipeline_plan
        ),
    }


def build_run_id(
    *,
    repository_commit: str,
    arguments: Mapping[str, Any],
    path_contract: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    pipeline_plan: Sequence[
        Mapping[str, Any]
    ],
) -> str:
    """Calculate deterministic run_id from scientific run identity."""

    identity_payload = (
        build_run_identity_payload(
            repository_commit=(
                repository_commit
            ),
            arguments=arguments,
            path_contract=path_contract,
            input_identity=input_identity,
            pipeline_plan=pipeline_plan,
        )
    )

    return sha256_bytes(
        canonical_json(
            identity_payload
        ).encode("utf-8")
    )


def create_execution_id(
    *,
    created_at: datetime | None = None,
    token: str | None = None,
) -> str:
    """Create a unique timestamped execution identifier."""

    if created_at is None:
        created_at = utc_now()

    if created_at.tzinfo is None:
        raise ValueError(
            "Execution timestamp must be timezone-aware."
        )

    if token is None:
        token = uuid.uuid4().hex[:12]

    token = token.lower()

    if not EXECUTION_TOKEN_PATTERN.fullmatch(
        token
    ):
        raise ValueError(
            "Execution token must contain exactly "
            "12 lowercase hexadecimal characters."
        )

    timestamp = (
        created_at.astimezone(timezone.utc)
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    return f"{timestamp}-{token}"


def create_run_manifest(
    *,
    repository: Mapping[str, Any],
    arguments: Mapping[str, Any],
    path_contract: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    plan_mode: bool,
    created_at: datetime | None = None,
    execution_id: str | None = None,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Create a validated initial run-manifest document."""

    repository_data = (
        normalize_json_value(repository)
    )

    if not isinstance(
        repository_data,
        dict,
    ):
        raise TypeError(
            "Repository snapshot must be a mapping."
        )

    commit = repository_data.get(
        "commit"
    )

    if not isinstance(commit, str):
        raise ValueError(
            "Repository snapshot is missing commit."
        )

    if created_at is None:
        created_at = utc_now()

    created_at_text = isoformat_utc(
        created_at
    )

    actual_python_executable = str(
        python_executable
        if python_executable is not None
        else sys.executable
    )

    pipeline_plan = (
        normalize_pipeline_plan(
            steps,
            python_executable=(
                actual_python_executable
            ),
        )
    )

    arguments_data = (
        normalize_json_value(arguments)
    )

    path_contract_data = (
        normalize_json_value(
            path_contract
        )
    )

    input_identity_data = (
        normalize_json_value(
            input_identity
        )
    )

    run_id = build_run_id(
        repository_commit=commit,
        arguments=arguments_data,
        path_contract=path_contract_data,
        input_identity=input_identity_data,
        pipeline_plan=pipeline_plan,
    )

    if execution_id is None:
        execution_id = (
            create_execution_id(
                created_at=created_at
            )
        )

    initial_status = (
        "planned"
        if plan_mode
        else "pending"
    )

    initial_step_status = (
        "planned"
        if plan_mode
        else "pending"
    )

    execution_steps = [
        {
            "position": step["position"],
            "name": step["name"],
            "command": step["command"],
            "status": initial_step_status,
            "started_at_utc": None,
            "finished_at_utc": None,
            "duration_seconds": None,
            "return_code": None,
            "error": None,
        }
        for step in pipeline_plan
    ]

    manifest = {
        "schema_version": (
            RUN_MANIFEST_SCHEMA_VERSION
        ),
        "manifest_type": (
            RUN_MANIFEST_TYPE
        ),
        "run_id": run_id,
        "execution_id": execution_id,
        "status": initial_status,
        "plan_mode": bool(plan_mode),
        "created_at_utc": (
            created_at_text
        ),
        "started_at_utc": None,
        "finished_at_utc": None,
        "duration_seconds": None,
        "repository": repository_data,
        "python": {
            "executable": (
                actual_python_executable
            ),
            "version": (
                platform.python_version()
            ),
            "implementation": (
                platform.python_implementation()
            ),
            "platform": platform.platform(),
        },
        "arguments": arguments_data,
        "path_contract": (
            path_contract_data
        ),
        "input_identity": (
            input_identity_data
        ),
        "pipeline_plan": pipeline_plan,
        "steps": execution_steps,
        "error": None,
    }

    validate_run_manifest(
        manifest
    )

    return manifest


def validate_run_manifest(
    manifest: Mapping[str, Any],
) -> None:
    """Validate the structural and cryptographic run contract."""

    normalized = normalize_json_value(
        manifest
    )

    if not isinstance(normalized, dict):
        raise TypeError(
            "Run manifest must be a mapping."
        )

    if normalized.get(
        "schema_version"
    ) != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported run-manifest schema version."
        )

    if normalized.get(
        "manifest_type"
    ) != RUN_MANIFEST_TYPE:
        raise ValueError(
            "Unexpected run-manifest type."
        )

    status = normalized.get(
        "status"
    )

    if status not in RUN_STATUSES:
        raise ValueError(
            f"Invalid run status: {status!r}"
        )

    execution_id = normalized.get(
        "execution_id"
    )

    if not isinstance(
        execution_id,
        str,
    ) or not execution_id:
        raise ValueError(
            "Run manifest has no execution_id."
        )

    repository = normalized.get(
        "repository"
    )

    if not isinstance(repository, dict):
        raise ValueError(
            "Run manifest has no repository snapshot."
        )

    repository_commit = (
        repository.get("commit")
    )

    if not isinstance(
        repository_commit,
        str,
    ):
        raise ValueError(
            "Repository snapshot has no commit."
        )

    pipeline_plan = normalized.get(
        "pipeline_plan"
    )

    execution_steps = normalized.get(
        "steps"
    )

    if not isinstance(
        pipeline_plan,
        list,
    ) or not pipeline_plan:
        raise ValueError(
            "Run manifest has no pipeline plan."
        )

    if not isinstance(
        execution_steps,
        list,
    ):
        raise ValueError(
            "Run manifest has no execution steps."
        )

    if len(pipeline_plan) != len(
        execution_steps
    ):
        raise ValueError(
            "Pipeline-plan and execution-step "
            "counts do not match."
        )

    expected_run_id = build_run_id(
        repository_commit=(
            repository_commit
        ),
        arguments=normalized.get(
            "arguments",
            {},
        ),
        path_contract=normalized.get(
            "path_contract",
            {},
        ),
        input_identity=normalized.get(
            "input_identity",
            {},
        ),
        pipeline_plan=pipeline_plan,
    )

    actual_run_id = normalized.get(
        "run_id"
    )

    if actual_run_id != expected_run_id:
        raise ValueError(
            "Run ID does not match the "
            "deterministic identity payload."
        )

    for expected_position, (
        plan_step,
        execution_step,
    ) in enumerate(
        zip(
            pipeline_plan,
            execution_steps,
            strict=True,
        ),
        start=1,
    ):
        if plan_step.get(
            "position"
        ) != expected_position:
            raise ValueError(
                "Pipeline positions must be "
                "contiguous and one-based."
            )

        if execution_step.get(
            "position"
        ) != expected_position:
            raise ValueError(
                "Execution-step positions must be "
                "contiguous and one-based."
            )

        for field in (
            "name",
            "command",
        ):
            if execution_step.get(
                field
            ) != plan_step.get(field):
                raise ValueError(
                    "Execution step does not match "
                    f"pipeline plan field: {field}"
                )

        step_status = execution_step.get(
            "status"
        )

        if step_status not in STEP_STATUSES:
            raise ValueError(
                "Invalid execution-step status: "
                f"{step_status!r}"
            )

    if normalized.get("plan_mode"):
        if status != "planned":
            raise ValueError(
                "Plan-mode runs must have "
                "planned status."
            )

        if any(
            step.get("status") != "planned"
            for step in execution_steps
        ):
            raise ValueError(
                "Plan-mode steps must have "
                "planned status."
            )



def _mutable_manifest_copy(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a validated mutable copy of one run manifest."""

    validate_run_manifest(
        manifest
    )

    normalized = normalize_json_value(
        manifest
    )

    if not isinstance(
        normalized,
        dict,
    ):
        raise TypeError(
            "Run manifest must normalize to a dictionary."
        )

    return normalized


def _validate_duration_seconds(
    value: float,
) -> float:
    """Validate and normalize a measured duration."""

    if isinstance(value, bool) or not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            "Duration must be a numeric value."
        )

    duration = float(value)

    if not math.isfinite(duration):
        raise ValueError(
            "Duration must be finite."
        )

    if duration < 0:
        raise ValueError(
            "Duration must not be negative."
        )

    return duration


def _step_index(
    manifest: Mapping[str, Any],
    position: int,
) -> int:
    """Resolve a one-based pipeline position to its list index."""

    if isinstance(position, bool) or not isinstance(
        position,
        int,
    ):
        raise TypeError(
            "Step position must be an integer."
        )

    if position <= 0:
        raise ValueError(
            "Step position must be positive."
        )

    steps = manifest.get(
        "steps"
    )

    if not isinstance(
        steps,
        list,
    ):
        raise ValueError(
            "Run manifest has no execution-step list."
        )

    index = position - 1

    if index >= len(steps):
        raise IndexError(
            f"Unknown pipeline step position: {position}"
        )

    return index


def _normalize_error_payload(
    error: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a structured run or step error payload."""

    normalized = normalize_json_value(
        error
    )

    if not isinstance(
        normalized,
        dict,
    ) or not normalized:
        raise ValueError(
            "Error payload must be a non-empty mapping."
        )

    error_type = normalized.get(
        "type"
    )

    message = normalized.get(
        "message"
    )

    if not isinstance(
        error_type,
        str,
    ) or not error_type.strip():
        raise ValueError(
            "Error payload must contain a non-empty type."
        )

    if not isinstance(
        message,
        str,
    ) or not message.strip():
        raise ValueError(
            "Error payload must contain a non-empty message."
        )

    return normalized


def start_run_manifest(
    manifest: Mapping[str, Any],
    *,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Transition a non-plan run from pending to running."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["plan_mode"]:
        raise ValueError(
            "Plan-mode manifests cannot be started."
        )

    if updated["status"] != "pending":
        raise ValueError(
            "Only pending runs can be started."
        )

    if started_at is None:
        started_at = utc_now()

    updated["status"] = "running"
    updated["started_at_utc"] = (
        isoformat_utc(started_at)
    )
    updated["finished_at_utc"] = None
    updated["duration_seconds"] = None
    updated["error"] = None

    validate_run_manifest(
        updated
    )

    return updated


def start_step_manifest(
    manifest: Mapping[str, Any],
    *,
    position: int,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Transition the next pending step to running."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["status"] != "running":
        raise ValueError(
            "A step can start only while its run is running."
        )

    index = _step_index(
        updated,
        position,
    )

    steps = updated["steps"]
    step = steps[index]

    if step["status"] != "pending":
        raise ValueError(
            "Only pending steps can be started."
        )

    previous_statuses = [
        previous["status"]
        for previous in steps[:index]
    ]

    if any(
        status not in {
            "succeeded",
            "skipped",
        }
        for status in previous_statuses
    ):
        raise ValueError(
            "Pipeline steps must start in order."
        )

    if any(
        following["status"] != "pending"
        for following in steps[
            index + 1 :
        ]
    ):
        raise ValueError(
            "Later pipeline steps must remain pending."
        )

    if started_at is None:
        started_at = utc_now()

    step["status"] = "running"
    step["started_at_utc"] = (
        isoformat_utc(started_at)
    )
    step["finished_at_utc"] = None
    step["duration_seconds"] = None
    step["return_code"] = None
    step["error"] = None

    validate_run_manifest(
        updated
    )

    return updated


def succeed_step_manifest(
    manifest: Mapping[str, Any],
    *,
    position: int,
    finished_at: datetime | None = None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Mark one running step as successfully completed."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["status"] != "running":
        raise ValueError(
            "A step can succeed only while its run is running."
        )

    index = _step_index(
        updated,
        position,
    )

    step = updated["steps"][index]

    if step["status"] != "running":
        raise ValueError(
            "Only a running step can succeed."
        )

    if finished_at is None:
        finished_at = utc_now()

    step["status"] = "succeeded"
    step["finished_at_utc"] = (
        isoformat_utc(finished_at)
    )
    step["duration_seconds"] = (
        _validate_duration_seconds(
            duration_seconds
        )
    )
    step["return_code"] = 0
    step["error"] = None

    validate_run_manifest(
        updated
    )

    return updated


def fail_step_manifest(
    manifest: Mapping[str, Any],
    *,
    position: int,
    return_code: int,
    error: Mapping[str, Any],
    finished_at: datetime | None = None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Mark one running step as failed."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["status"] != "running":
        raise ValueError(
            "A step can fail only while its run is running."
        )

    if isinstance(return_code, bool) or not isinstance(
        return_code,
        int,
    ):
        raise TypeError(
            "Step return code must be an integer."
        )

    if return_code == 0:
        raise ValueError(
            "A failed step cannot have return code zero."
        )

    index = _step_index(
        updated,
        position,
    )

    step = updated["steps"][index]

    if step["status"] != "running":
        raise ValueError(
            "Only a running step can fail."
        )

    if finished_at is None:
        finished_at = utc_now()

    step["status"] = "failed"
    step["finished_at_utc"] = (
        isoformat_utc(finished_at)
    )
    step["duration_seconds"] = (
        _validate_duration_seconds(
            duration_seconds
        )
    )
    step["return_code"] = return_code
    step["error"] = (
        _normalize_error_payload(
            error
        )
    )

    validate_run_manifest(
        updated
    )

    return updated


def succeed_run_manifest(
    manifest: Mapping[str, Any],
    *,
    finished_at: datetime | None = None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Mark a running run as successfully completed."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["status"] != "running":
        raise ValueError(
            "Only a running run can succeed."
        )

    incomplete_steps = [
        step["position"]
        for step in updated["steps"]
        if step["status"] not in {
            "succeeded",
            "skipped",
        }
    ]

    if incomplete_steps:
        raise ValueError(
            "Run cannot succeed with incomplete steps: "
            f"{incomplete_steps}"
        )

    if finished_at is None:
        finished_at = utc_now()

    updated["status"] = "succeeded"
    updated["finished_at_utc"] = (
        isoformat_utc(finished_at)
    )
    updated["duration_seconds"] = (
        _validate_duration_seconds(
            duration_seconds
        )
    )
    updated["error"] = None

    validate_run_manifest(
        updated
    )

    return updated


def fail_run_manifest(
    manifest: Mapping[str, Any],
    *,
    error: Mapping[str, Any],
    finished_at: datetime | None = None,
    duration_seconds: float,
) -> dict[str, Any]:
    """Mark a pending or running run as failed."""

    updated = _mutable_manifest_copy(
        manifest
    )

    if updated["plan_mode"]:
        raise ValueError(
            "Plan-mode manifests cannot fail execution."
        )

    if updated["status"] not in {
        "pending",
        "running",
    }:
        raise ValueError(
            "Only pending or running runs can fail."
        )

    if finished_at is None:
        finished_at = utc_now()

    updated["status"] = "failed"
    updated["finished_at_utc"] = (
        isoformat_utc(finished_at)
    )
    updated["duration_seconds"] = (
        _validate_duration_seconds(
            duration_seconds
        )
    )
    updated["error"] = (
        _normalize_error_payload(
            error
        )
    )

    validate_run_manifest(
        updated
    )

    return updated


def write_run_manifest(
    manifest: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Validate and atomically persist one run manifest."""

    validate_run_manifest(
        manifest
    )

    atomic_write_json(
        path=output_path,
        payload=manifest,
    )
