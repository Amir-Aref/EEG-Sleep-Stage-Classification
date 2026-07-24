"""Run the rebuilt Phase 2 EEG pipeline with one portable command.

The runner uses the active Python interpreter, so the same command
works in Windows, Linux, CI, Colab, and Kaggle environments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .config import (
        EXPECTED_SLEEP_CASSETTE_RECORDINGS,
        PROJECT_ROOT,
        REPORTS_DIR,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_VERSION,
        runtime_path_contract,
    )
    from .phase2_artifact_manifest import (
        build_artifact_manifest,
        capture_artifact_snapshot,
        runtime_relative_path,
        selected_artifact_specs,
        write_artifact_manifest,
    )
    from .run_provenance import (
        create_run_manifest,
        fail_run_manifest,
        fail_step_manifest,
        repository_snapshot,
        sha256_file,
        start_run_manifest,
        start_step_manifest,
        succeed_run_manifest,
        succeed_step_manifest,
        utc_now,
        write_run_manifest,
    )
except ImportError:
    from config import (
        EXPECTED_SLEEP_CASSETTE_RECORDINGS,
        PROJECT_ROOT,
        REPORTS_DIR,
        SLEEP_EDFX_MANIFEST_PATH,
        SLEEP_EDFX_VERSION,
        runtime_path_contract,
    )
    from phase2_artifact_manifest import (
        build_artifact_manifest,
        capture_artifact_snapshot,
        runtime_relative_path,
        selected_artifact_specs,
        write_artifact_manifest,
    )
    from run_provenance import (
        create_run_manifest,
        fail_run_manifest,
        fail_step_manifest,
        repository_snapshot,
        sha256_file,
        start_run_manifest,
        start_step_manifest,
        succeed_run_manifest,
        succeed_step_manifest,
        utc_now,
        write_run_manifest,
    )


@dataclass(frozen=True)
class PipelineStep:
    """One executable pipeline stage."""

    name: str
    command: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    """Parse pipeline execution options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete rebuilt Phase 2 EEG pipeline."
        )
    )

    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help=(
            "Refresh the official Sleep-EDF manifest and "
            "checksum metadata. Requires internet access."
        ),
    )

    parser.add_argument(
        "--download-subject-count",
        type=int,
        default=None,
        help=(
            "Download the first N subjects before processing. "
            "Omit this argument when verified raw data already exists."
        ),
    )

    parser.add_argument(
        "--nights-per-subject",
        type=int,
        default=1,
        help=(
            "Maximum nights downloaded per selected subject. "
            "Used only with --download-subject-count."
        ),
    )

    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help=(
            "Recompute existing per-recording feature files."
        ),
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA report generation.",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the final automated contract tests.",
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print the execution plan without running commands."
        ),
    )

    arguments = parser.parse_args()

    if (
        arguments.download_subject_count is not None
        and arguments.download_subject_count <= 0
    ):
        parser.error(
            "--download-subject-count must be positive."
        )

    if arguments.nights_per_subject <= 0:
        parser.error(
            "--nights-per-subject must be positive."
        )

    return arguments


def python_module_command(
    module_name: str,
    *arguments: str,
) -> tuple[str, ...]:
    """Build a command using the active Python interpreter."""

    return (
        sys.executable,
        "-m",
        module_name,
        *arguments,
    )


def build_pipeline_steps(
    arguments: argparse.Namespace,
) -> list[PipelineStep]:
    """Build the deterministic pipeline execution plan."""

    steps: list[PipelineStep] = []

    if arguments.refresh_manifest:
        steps.append(
            PipelineStep(
                name="Refresh official dataset manifest",
                command=python_module_command(
                    "scripts.build_dataset_manifest"
                ),
            )
        )

    if arguments.download_subject_count is not None:
        steps.append(
            PipelineStep(
                name="Download and verify raw EDF files",
                command=python_module_command(
                    "scripts.download_sleep_edfx",
                    "--subject-count",
                    str(
                        arguments.download_subject_count
                    ),
                    "--nights-per-subject",
                    str(
                        arguments.nights_per_subject
                    ),
                ),
            )
        )

    steps.extend(
        [
            PipelineStep(
                name="Inspect verified EDF pairs",
                command=python_module_command(
                    "scripts.inspect_sleep_edfx"
                ),
            ),
            PipelineStep(
                name="Build trimmed epoch metadata",
                command=python_module_command(
                    "scripts.build_epoch_metadata"
                ),
            ),
        ]
    )

    feature_arguments = ["--all"]

    if arguments.overwrite_features:
        feature_arguments.append("--overwrite")

    steps.extend(
        [
            PipelineStep(
                name="Extract EEG features",
                command=python_module_command(
                    "scripts.extract_eeg_features",
                    *feature_arguments,
                ),
            ),
            PipelineStep(
                name="Audit EEG features",
                command=python_module_command(
                    "scripts.audit_eeg_features"
                ),
            ),
            PipelineStep(
                name="Build leakage-safe model input",
                command=python_module_command(
                    "scripts.build_model_input"
                ),
            ),
        ]
    )

    if not arguments.skip_eda:
        steps.append(
            PipelineStep(
                name="Generate EEG EDA reports",
                command=python_module_command(
                    "scripts.run_eeg_eda"
                ),
            )
        )

    if not arguments.skip_tests:
        steps.append(
            PipelineStep(
                name="Run automated contract tests",
                command=python_module_command(
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_*.py",
                    "-v",
                ),
            )
        )

    return steps


def format_command(
    command: Sequence[str],
) -> str:
    """Format one command for human-readable logs."""

    return " ".join(
        (
            f'"{part}"'
            if " " in part
            else part
        )
        for part in command
    )


def run_step(
    step: PipelineStep,
    position: int,
    total: int,
) -> float:
    """Run one pipeline step and return elapsed seconds."""

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"[{position}/{total}] {step.name}"
    )

    print(
        "Command:",
        format_command(step.command),
    )

    print("=" * 72)

    start = time.perf_counter()

    subprocess.run(
        step.command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    elapsed = time.perf_counter() - start

    print(
        f"\nCompleted: {step.name} "
        f"({elapsed:.2f} seconds)"
    )

    return elapsed


def pipeline_step_payloads(
    steps: Sequence[PipelineStep],
) -> list[dict[str, object]]:
    """Convert executable steps into provenance-safe payloads."""

    return [
        {
            "name": step.name,
            "command": list(step.command),
        }
        for step in steps
    ]


def build_input_identity() -> dict[str, object]:
    """Capture the authoritative Sleep-EDF input identity."""

    manifest_exists = (
        SLEEP_EDFX_MANIFEST_PATH.is_file()
    )

    return {
        "dataset_name": "Sleep-EDF Expanded",
        "dataset_version": SLEEP_EDFX_VERSION,
        "subset": "sleep-cassette",
        "expected_recording_count": (
            EXPECTED_SLEEP_CASSETTE_RECORDINGS
        ),
        "manifest_path": str(
            SLEEP_EDFX_MANIFEST_PATH
        ),
        "manifest_exists": manifest_exists,
        "manifest_sha256": (
            sha256_file(
                SLEEP_EDFX_MANIFEST_PATH
            )
            if manifest_exists
            else None
        ),
    }


def run_manifest_output_path(
    manifest: dict[str, Any],
) -> Path:
    """Resolve the execution-specific run-manifest path."""

    execution_id = str(
        manifest["execution_id"]
    )

    return (
        REPORTS_DIR
        / "runs"
        / execution_id
        / "run_manifest.json"
    ).resolve()


def exception_payload(
    error: BaseException,
    *,
    command: Sequence[str] | None = None,
    failed_step_position: int | None = None,
) -> dict[str, object]:
    """Create a non-empty structured error payload."""

    message = str(error).strip()

    if not message:
        message = repr(error)

    payload: dict[str, object] = {
        "type": type(error).__name__,
        "message": message,
    }

    if command is not None:
        payload["command"] = list(command)

    if failed_step_position is not None:
        payload[
            "failed_step_position"
        ] = failed_step_position

    return payload


def artifact_manifest_output_paths(
    run_manifest_path: Path,
) -> tuple[Path, Path]:
    """Resolve execution-specific artifact-manifest paths."""

    output_directory = (
        run_manifest_path.resolve().parent
    )

    return (
        output_directory
        / "artifact_manifest.json",
        output_directory
        / "artifact_manifest.csv",
    )


def persist_artifact_manifest(
    *,
    run_manifest: dict[str, Any],
    run_manifest_path: Path,
    selected_steps: Sequence[str],
    artifact_specs: Sequence[Any],
    before_snapshot: dict[str, Any],
    runtime_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Capture, compare and atomically persist output artifacts."""

    after_snapshot = (
        capture_artifact_snapshot(
            artifact_specs,
            runtime_root=runtime_root,
        )
    )

    artifact_manifest = (
        build_artifact_manifest(
            run_id=str(
                run_manifest["run_id"]
            ),
            execution_id=str(
                run_manifest[
                    "execution_id"
                ]
            ),
            selected_steps=selected_steps,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
    )

    (
        json_output_path,
        csv_output_path,
    ) = artifact_manifest_output_paths(
        run_manifest_path
    )

    write_artifact_manifest(
        manifest=artifact_manifest,
        json_output_path=json_output_path,
        csv_output_path=csv_output_path,
    )

    reference = {
        "status": artifact_manifest[
            "status"
        ],
        "artifact_count": (
            artifact_manifest[
                "artifact_count"
            ]
        ),
        "counts_by_change_type": (
            artifact_manifest[
                "counts_by_change_type"
            ]
        ),
        "json_path": runtime_relative_path(
            json_output_path,
            runtime_root=runtime_root,
        ),
        "json_size_bytes": (
            json_output_path.stat().st_size
        ),
        "json_sha256": sha256_file(
            json_output_path
        ),
        "csv_path": runtime_relative_path(
            csv_output_path,
            runtime_root=runtime_root,
        ),
        "csv_size_bytes": (
            csv_output_path.stat().st_size
        ),
        "csv_sha256": sha256_file(
            csv_output_path
        ),
    }

    return (
        artifact_manifest,
        reference,
    )


def execute_pipeline(
    arguments: argparse.Namespace,
    steps: Sequence[PipelineStep],
) -> tuple[
    Path,
    dict[str, Any],
    list[tuple[str, float]],
    float,
]:
    """Execute all steps while recording run and artifact provenance."""

    repository = repository_snapshot(
        PROJECT_ROOT
    )

    path_contract = (
        runtime_path_contract()
    )

    runtime_root = Path(
        str(
            path_contract[
                "runtime_root"
            ]
        )
    ).resolve()

    selected_steps = [
        step.name
        for step in steps
    ]

    artifact_specs = (
        selected_artifact_specs(
            selected_steps
        )
    )

    before_artifact_snapshot = (
        capture_artifact_snapshot(
            artifact_specs,
            runtime_root=runtime_root,
        )
    )

    manifest = create_run_manifest(
        repository=repository,
        arguments=vars(arguments),
        path_contract=path_contract,
        input_identity=build_input_identity(),
        steps=pipeline_step_payloads(
            steps
        ),
        plan_mode=False,
        python_executable=sys.executable,
    )

    manifest["artifact_tracking"] = {
        "status": "pending",
        "selected_spec_count": len(
            artifact_specs
        ),
        "selected_steps": (
            selected_steps
        ),
    }

    manifest_path = (
        run_manifest_output_path(
            manifest
        )
    )

    write_run_manifest(
        manifest,
        manifest_path,
    )

    total_start = time.perf_counter()

    manifest = start_run_manifest(
        manifest,
        started_at=utc_now(),
    )

    manifest[
        "artifact_tracking"
    ]["status"] = "running"

    write_run_manifest(
        manifest,
        manifest_path,
    )

    print(
        "\nRun manifest initialized:",
        manifest_path,
    )

    elapsed_by_step: list[
        tuple[str, float]
    ] = []

    for position, step in enumerate(
        steps,
        start=1,
    ):
        manifest = start_step_manifest(
            manifest,
            position=position,
            started_at=utc_now(),
        )

        write_run_manifest(
            manifest,
            manifest_path,
        )

        step_start = time.perf_counter()

        try:
            elapsed = run_step(
                step=step,
                position=position,
                total=len(steps),
            )
        except BaseException as error:
            elapsed = (
                time.perf_counter()
                - step_start
            )

            return_code = (
                error.returncode
                if isinstance(
                    error,
                    subprocess.CalledProcessError,
                )
                else -1
            )

            step_error = exception_payload(
                error,
                command=step.command,
                failed_step_position=(
                    position
                ),
            )

            manifest = fail_step_manifest(
                manifest,
                position=position,
                return_code=return_code,
                error=step_error,
                finished_at=utc_now(),
                duration_seconds=elapsed,
            )

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            manifest = fail_run_manifest(
                manifest,
                error={
                    "type": (
                        "PipelineExecutionError"
                    ),
                    "message": (
                        "Pipeline execution stopped "
                        f"at step {position}: "
                        f"{step.name}"
                    ),
                    "failed_step_position": (
                        position
                    ),
                    "cause": step_error,
                },
                finished_at=utc_now(),
                duration_seconds=(
                    total_elapsed
                ),
            )

            try:
                (
                    artifact_manifest,
                    artifact_reference,
                ) = persist_artifact_manifest(
                    run_manifest=manifest,
                    run_manifest_path=(
                        manifest_path
                    ),
                    selected_steps=(
                        selected_steps
                    ),
                    artifact_specs=(
                        artifact_specs
                    ),
                    before_snapshot=(
                        before_artifact_snapshot
                    ),
                    runtime_root=(
                        runtime_root
                    ),
                )
            except BaseException as artifact_error:
                manifest[
                    "artifact_manifest"
                ] = {
                    "status": "write_failed",
                    "error": exception_payload(
                        artifact_error
                    ),
                }

                manifest[
                    "artifact_tracking"
                ]["status"] = (
                    "write_failed"
                )
            else:
                manifest[
                    "artifact_manifest"
                ] = artifact_reference

                manifest[
                    "artifact_tracking"
                ]["status"] = (
                    artifact_manifest[
                        "status"
                    ]
                )

            write_run_manifest(
                manifest,
                manifest_path,
            )

            print(
                "\nFailed run manifest:",
                manifest_path,
            )

            raise
        else:
            manifest = (
                succeed_step_manifest(
                    manifest,
                    position=position,
                    finished_at=utc_now(),
                    duration_seconds=elapsed,
                )
            )

            write_run_manifest(
                manifest,
                manifest_path,
            )

            elapsed_by_step.append(
                (
                    step.name,
                    elapsed,
                )
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    try:
        (
            artifact_manifest,
            artifact_reference,
        ) = persist_artifact_manifest(
            run_manifest=manifest,
            run_manifest_path=manifest_path,
            selected_steps=selected_steps,
            artifact_specs=artifact_specs,
            before_snapshot=(
                before_artifact_snapshot
            ),
            runtime_root=runtime_root,
        )
    except BaseException as artifact_error:
        manifest[
            "artifact_manifest"
        ] = {
            "status": "write_failed",
            "error": exception_payload(
                artifact_error
            ),
        }

        manifest[
            "artifact_tracking"
        ]["status"] = "write_failed"

        manifest = fail_run_manifest(
            manifest,
            error={
                "type": (
                    "ArtifactManifestWriteError"
                ),
                "message": (
                    "Artifact manifest generation "
                    "or persistence failed."
                ),
                "cause": exception_payload(
                    artifact_error
                ),
            },
            finished_at=utc_now(),
            duration_seconds=(
                total_elapsed
            ),
        )

        write_run_manifest(
            manifest,
            manifest_path,
        )

        raise

    manifest[
        "artifact_manifest"
    ] = artifact_reference

    manifest[
        "artifact_tracking"
    ]["status"] = artifact_manifest[
        "status"
    ]

    if artifact_manifest["status"] != "valid":
        manifest = fail_run_manifest(
            manifest,
            error={
                "type": (
                    "ArtifactManifestValidationError"
                ),
                "message": (
                    "Required Phase 2 artifacts "
                    "are missing or removed."
                ),
                "missing_required_specs": (
                    artifact_manifest[
                        "missing_required_specs"
                    ]
                ),
                "removed_required_paths": (
                    artifact_manifest[
                        "removed_required_paths"
                    ]
                ),
            },
            finished_at=utc_now(),
            duration_seconds=(
                total_elapsed
            ),
        )

        write_run_manifest(
            manifest,
            manifest_path,
        )

        raise RuntimeError(
            "Artifact manifest validation failed."
        )

    manifest = succeed_run_manifest(
        manifest,
        finished_at=utc_now(),
        duration_seconds=total_elapsed,
    )

    write_run_manifest(
        manifest,
        manifest_path,
    )

    return (
        manifest_path,
        manifest,
        elapsed_by_step,
        total_elapsed,
    )


def main() -> None:
    """Execute or display the complete Phase 2 pipeline."""

    arguments = parse_arguments()
    steps = build_pipeline_steps(arguments)

    if not steps:
        raise RuntimeError(
            "No pipeline steps were selected."
        )

    print("\n=== PHASE 2 PIPELINE PLAN ===")
    print("Project root:", PROJECT_ROOT)
    print("Python:", sys.executable)
    print("Steps:", len(steps))

    for index, step in enumerate(
        steps,
        start=1,
    ):
        print(
            f"{index:02d}. {step.name}"
        )

        print(
            "    ",
            format_command(step.command),
        )

    if arguments.plan:
        print(
            "\nPlan mode complete. "
            "No commands were executed."
        )
        return

    (
        manifest_path,
        manifest,
        elapsed_by_step,
        total_elapsed,
    ) = execute_pipeline(
        arguments=arguments,
        steps=steps,
    )

    print(
        "\n=== PHASE 2 PIPELINE SUMMARY ==="
    )

    for step_name, elapsed in elapsed_by_step:
        print(
            f"{step_name}: "
            f"{elapsed:.2f} seconds"
        )

    print(
        f"Total elapsed: "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        "Run status:",
        manifest["status"],
    )

    print(
        "Run ID:",
        manifest["run_id"],
    )

    print(
        "Execution ID:",
        manifest["execution_id"],
    )

    print(
        "Run manifest:",
        manifest_path,
    )

    print(
        "Phase 2 pipeline validation: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            "\nPhase 2 pipeline validation: FAILED",
            file=sys.stderr,
        )

        print(
            "Failed command:",
            format_command(error.cmd),
            file=sys.stderr,
        )

        print(
            "Exit code:",
            error.returncode,
            file=sys.stderr,
        )

        sys.exit(error.returncode)
    except Exception as error:
        print(
            "\nPhase 2 pipeline validation: FAILED",
            file=sys.stderr,
        )

        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )

        sys.exit(1)
