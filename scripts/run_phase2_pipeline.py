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
from typing import Sequence

try:
    from .config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT


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

    total_start = time.perf_counter()
    elapsed_by_step: list[tuple[str, float]] = []

    for position, step in enumerate(
        steps,
        start=1,
    ):
        elapsed = run_step(
            step=step,
            position=position,
            total=len(steps),
        )

        elapsed_by_step.append(
            (step.name, elapsed)
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print("\n=== PHASE 2 PIPELINE SUMMARY ===")

    for step_name, elapsed in elapsed_by_step:
        print(
            f"{step_name}: {elapsed:.2f} seconds"
        )

    print(
        f"Total elapsed: {total_elapsed:.2f} seconds"
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
