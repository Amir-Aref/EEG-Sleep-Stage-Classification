from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.phase3_dataset import (
    DEFAULT_DATA_PATH,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_SCHEMA_PATH,
    Phase3DatasetBundle,
    load_phase3_dataset,
)
from scripts.phase3_inner_search import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    load_json_object,
    run_inner_search,
)
from scripts.phase3_model_registry import (
    DEFAULT_CONFIG_PATH as DEFAULT_REGISTRY_PATH,
    load_registry_config,
)


DEFAULT_OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_inner_search_results.json"
)

DEFAULT_OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_inner_search_results.csv"
)


FORBIDDEN_RESULT_KEYS = {
    "test_metrics",
    "test_predictions",
    "test_probabilities",
    "test_features",
    "test_feature_matrix",
}


SUMMARY_COLUMNS = [
    "outer_fold",
    "test_subjects",
    "outer_development_subjects",
    "model_name",
    "candidate_index",
    "candidate_id",
    "candidate_parameters_json",
    "eligible_for_selection",
    "selection_rank",
    "is_selected",
    "complexity_rank",
    "fold_count",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_balanced_accuracy",
    "std_balanced_accuracy",
    "mean_weighted_f1",
    "std_weighted_f1",
    "mean_accuracy",
    "std_accuracy",
    "mean_cohen_kappa",
    "std_cohen_kappa",
    "mean_multiclass_log_loss",
    "std_multiclass_log_loss",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def relative_display_path(path: Path) -> str:
    resolved = path.resolve()

    try:
        return resolved.relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved)


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def find_forbidden_keys(
    value: Any,
    path: str = "result",
) -> list[str]:
    violations: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key)

            if normalized_key in FORBIDDEN_RESULT_KEYS:
                violations.append(
                    f"{path}.{normalized_key}"
                )

            violations.extend(
                find_forbidden_keys(
                    item,
                    (
                        f"{path}."
                        f"{normalized_key}"
                    ),
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(
                find_forbidden_keys(
                    item,
                    f"{path}[{index}]",
                )
            )

    return violations


def validate_inner_search_result_safety(
    result: Mapping[str, Any],
    require_complete: bool,
) -> None:
    required_keys = {
        "candidate_space_complete",
        "evaluated_models",
        "evaluated_outer_folds",
        "outer_results",
        "primary_metric",
        "selection_partition",
        "test_feature_matrix_loaded",
        "test_metrics_included",
        "test_predictions_included",
    }

    missing = sorted(
        required_keys - set(result)
    )

    if missing:
        raise ValueError(
            "Inner-search result is missing keys: "
            + ", ".join(missing)
        )

    if result["primary_metric"] != "macro_f1":
        raise ValueError(
            "Selection artifact requires macro_f1 "
            "as its primary metric."
        )

    if result["selection_partition"] != (
        "validation"
    ):
        raise ValueError(
            "Model selection must use only the "
            "validation partition."
        )

    if bool(result["test_metrics_included"]):
        raise ValueError(
            "Inner-search result contains test metrics."
        )

    if bool(result["test_predictions_included"]):
        raise ValueError(
            "Inner-search result contains test "
            "predictions."
        )

    if bool(
        result["test_feature_matrix_loaded"]
    ):
        raise ValueError(
            "Inner-search result accessed the test "
            "feature matrix."
        )

    if (
        require_complete
        and not bool(
            result["candidate_space_complete"]
        )
    ):
        raise ValueError(
            "A complete selection artifact requires "
            "the complete candidate space."
        )

    forbidden_paths = find_forbidden_keys(
        result
    )

    if forbidden_paths:
        raise ValueError(
            "Forbidden test payload keys found: "
            + ", ".join(forbidden_paths)
        )

    outer_results = result["outer_results"]

    if not isinstance(
        outer_results,
        list,
    ) or not outer_results:
        raise ValueError(
            "Inner-search result has no outer folds."
        )

    observed_outer_folds: set[int] = set()

    for outer_result in outer_results:
        outer_fold = int(
            outer_result["outer_fold"]
        )

        if outer_fold in observed_outer_folds:
            raise ValueError(
                "Duplicate outer-fold result: "
                f"{outer_fold}."
            )

        observed_outer_folds.add(
            outer_fold
        )

        selected = outer_result[
            "selected_candidate"
        ]

        selected_id = str(
            selected["candidate_id"]
        )

        ranked = outer_result[
            "ranked_selectable_candidates"
        ]

        if not ranked:
            raise ValueError(
                f"Outer fold {outer_fold} has no "
                "ranked selectable candidates."
            )

        rank_one = [
            candidate
            for candidate in ranked
            if int(
                candidate["selection_rank"]
            ) == 1
        ]

        if len(rank_one) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} must have "
                "exactly one rank-one candidate."
            )

        if str(
            rank_one[0]["candidate_id"]
        ) != selected_id:
            raise ValueError(
                f"Outer fold {outer_fold} selected "
                "candidate does not match rank one."
            )

        all_candidates = outer_result[
            "all_candidate_results"
        ]

        candidate_ids = [
            str(candidate["candidate_id"])
            for candidate in all_candidates
        ]

        if len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError(
                f"Outer fold {outer_fold} contains "
                "duplicate candidate identifiers."
            )

        if selected_id not in candidate_ids:
            raise ValueError(
                f"Outer fold {outer_fold} selected "
                "candidate was not evaluated."
            )

        if int(
            outer_result[
                "evaluated_candidate_count"
            ]
        ) != len(all_candidates):
            raise ValueError(
                f"Outer fold {outer_fold} candidate "
                "count is inconsistent."
            )

    expected_outer_folds = {
        int(value)
        for value in result[
            "evaluated_outer_folds"
        ]
    }

    if observed_outer_folds != (
        expected_outer_folds
    ):
        raise ValueError(
            "Outer result identifiers do not match "
            "evaluated_outer_folds."
        )


def build_selection_artifact(
    result: Mapping[str, Any],
    bundle: Phase3DatasetBundle,
    split_manifest_path: Path,
    registry_path: Path,
    require_complete: bool,
) -> dict[str, Any]:
    validate_inner_search_result_safety(
        result=result,
        require_complete=require_complete,
    )

    split_manifest_path = (
        split_manifest_path.resolve()
    )

    registry_path = registry_path.resolve()

    for path, label in (
        (
            split_manifest_path,
            "split manifest",
        ),
        (
            registry_path,
            "model registry",
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    return {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_inner_model_selection"
        ),
        "primary_metric": "macro_f1",
        "selection_partition": "validation",
        "candidate_space_complete": bool(
            result["candidate_space_complete"]
        ),
        "test_access_contract": {
            "test_metrics_included": False,
            "test_predictions_included": False,
            "test_feature_matrix_loaded": False,
            "test_subject_ids_are_metadata_only": True,
        },
        "scientific_reporting": {
            "allowed": False,
            "reason": (
                "Local four-subject nested-LOGO "
                "engineering evaluation is not the "
                "final scientific experiment."
            ),
        },
        "environment": {
            "scikit_learn_version": version(
                "scikit-learn"
            ),
        },
        "source": {
            "model_input_path": relative_display_path(
                bundle.data_path
            ),
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_path": relative_display_path(
                bundle.schema_path
            ),
            "model_schema_sha256": (
                bundle.schema_sha256
            ),
            "evaluation_protocol_path": (
                relative_display_path(
                    bundle.protocol_path
                )
            ),
            "evaluation_protocol_sha256": (
                bundle.protocol_sha256
            ),
            "split_manifest_path": (
                relative_display_path(
                    split_manifest_path
                )
            ),
            "split_manifest_sha256": (
                sha256_file(
                    split_manifest_path
                )
            ),
            "model_registry_path": (
                relative_display_path(
                    registry_path
                )
            ),
            "model_registry_sha256": (
                sha256_file(
                    registry_path
                )
            ),
        },
        "dataset_contract": {
            "row_count": int(
                bundle.row_count
            ),
            "feature_count": int(
                bundle.feature_count
            ),
            "subject_count": int(
                len(
                    set(
                        bundle.groups.tolist()
                    )
                )
            ),
            "class_mapping": dict(
                bundle.class_mapping
            ),
            "group_column": (
                bundle.group_column
            ),
            "target_column": (
                bundle.target_column
            ),
        },
        "selection_result": dict(result),
    }


def build_selection_summary_frame(
    artifact: Mapping[str, Any],
) -> pd.DataFrame:
    result = artifact[
        "selection_result"
    ]

    rows: list[dict[str, Any]] = []

    for outer_result in result[
        "outer_results"
    ]:
        outer_fold = int(
            outer_result["outer_fold"]
        )

        selected_id = str(
            outer_result[
                "selected_candidate"
            ]["candidate_id"]
        )

        rank_by_candidate = {
            str(candidate["candidate_id"]): int(
                candidate["selection_rank"]
            )
            for candidate in outer_result[
                "ranked_selectable_candidates"
            ]
        }

        test_subjects = ",".join(
            str(value)
            for value in outer_result[
                "test_subjects"
            ]
        )

        development_subjects = ",".join(
            str(value)
            for value in outer_result[
                "outer_development_subjects"
            ]
        )

        for candidate in outer_result[
            "all_candidate_results"
        ]:
            candidate_id = str(
                candidate["candidate_id"]
            )

            aggregate = candidate[
                "aggregate"
            ]

            rows.append(
                {
                    "outer_fold": outer_fold,
                    "test_subjects": test_subjects,
                    "outer_development_subjects": (
                        development_subjects
                    ),
                    "model_name": str(
                        candidate["model_name"]
                    ),
                    "candidate_index": int(
                        candidate[
                            "candidate_index"
                        ]
                    ),
                    "candidate_id": candidate_id,
                    "candidate_parameters_json": (
                        canonical_json(
                            candidate[
                                "candidate_parameters"
                            ]
                        )
                    ),
                    "eligible_for_selection": bool(
                        candidate[
                            "eligible_for_selection"
                        ]
                    ),
                    "selection_rank": (
                        rank_by_candidate.get(
                            candidate_id,
                            "",
                        )
                    ),
                    "is_selected": (
                        candidate_id == selected_id
                    ),
                    "complexity_rank": int(
                        candidate[
                            "complexity_rank"
                        ]
                    ),
                    "fold_count": int(
                        candidate["fold_count"]
                    ),
                    "mean_macro_f1": float(
                        aggregate[
                            "mean_macro_f1"
                        ]
                    ),
                    "std_macro_f1": float(
                        aggregate[
                            "std_macro_f1"
                        ]
                    ),
                    "mean_balanced_accuracy": float(
                        aggregate[
                            "mean_balanced_accuracy"
                        ]
                    ),
                    "std_balanced_accuracy": float(
                        aggregate[
                            "std_balanced_accuracy"
                        ]
                    ),
                    "mean_weighted_f1": float(
                        aggregate[
                            "mean_weighted_f1"
                        ]
                    ),
                    "std_weighted_f1": float(
                        aggregate[
                            "std_weighted_f1"
                        ]
                    ),
                    "mean_accuracy": float(
                        aggregate[
                            "mean_accuracy"
                        ]
                    ),
                    "std_accuracy": float(
                        aggregate[
                            "std_accuracy"
                        ]
                    ),
                    "mean_cohen_kappa": float(
                        aggregate[
                            "mean_cohen_kappa"
                        ]
                    ),
                    "std_cohen_kappa": float(
                        aggregate[
                            "std_cohen_kappa"
                        ]
                    ),
                    "mean_multiclass_log_loss": float(
                        aggregate[
                            "mean_multiclass_log_loss"
                        ]
                    ),
                    "std_multiclass_log_loss": float(
                        aggregate[
                            "std_multiclass_log_loss"
                        ]
                    ),
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=SUMMARY_COLUMNS,
    )

    if frame.empty:
        raise ValueError(
            "Selection summary frame is empty."
        )

    frame = frame.sort_values(
        by=[
            "outer_fold",
            "complexity_rank",
            "model_name",
            "candidate_index",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    selected_counts = (
        frame.groupby(
            "outer_fold",
            sort=True,
        )["is_selected"]
        .sum()
    )

    if not (
        selected_counts == 1
    ).all():
        raise ValueError(
            "Every outer fold must contain exactly "
            "one selected candidate."
        )

    return frame


def write_selection_artifacts(
    artifact: Mapping[str, Any],
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_text = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    json_output_path.write_text(
        json_text,
        encoding="utf-8",
        newline="\n",
    )

    summary = build_selection_summary_frame(
        artifact
    )

    summary.to_csv(
        csv_output_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def run_selection(
    outer_folds: list[int] | None = None,
    model_names: list[str] | None = None,
    max_candidates_per_model: int | None = None,
) -> tuple[
    dict[str, Any],
    Phase3DatasetBundle,
]:
    bundle = load_phase3_dataset(
        data_path=DEFAULT_DATA_PATH,
        schema_path=DEFAULT_SCHEMA_PATH,
        protocol_path=DEFAULT_PROTOCOL_PATH,
    )

    manifest = load_json_object(
        DEFAULT_SPLIT_MANIFEST_PATH
    )

    registry = load_registry_config(
        DEFAULT_REGISTRY_PATH
    )

    result = run_inner_search(
        bundle=bundle,
        manifest=manifest,
        registry=registry,
        outer_folds=outer_folds,
        model_names=model_names,
        max_candidates_per_model=(
            max_candidates_per_model
        ),
    )

    return result, bundle


def smoke_test() -> None:
    result, bundle = run_selection(
        outer_folds=[1],
        model_names=[
            "dummy_prior",
            "logistic_regression",
        ],
        max_candidates_per_model=1,
    )

    artifact = build_selection_artifact(
        result=result,
        bundle=bundle,
        split_manifest_path=(
            DEFAULT_SPLIT_MANIFEST_PATH
        ),
        registry_path=DEFAULT_REGISTRY_PATH,
        require_complete=False,
    )

    summary = build_selection_summary_frame(
        artifact
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        json_path = (
            root / "selection.json"
        )

        csv_path = (
            root / "selection.csv"
        )

        write_selection_artifacts(
            artifact=artifact,
            json_output_path=json_path,
            csv_output_path=csv_path,
        )

        first_json = json_path.read_bytes()
        first_csv = csv_path.read_bytes()

        write_selection_artifacts(
            artifact=artifact,
            json_output_path=json_path,
            csv_output_path=csv_path,
        )

        if first_json != json_path.read_bytes():
            raise ValueError(
                "Selection JSON is not "
                "byte-deterministic."
            )

        if first_csv != csv_path.read_bytes():
            raise ValueError(
                "Selection CSV is not "
                "byte-deterministic."
            )

    selected = summary.loc[
        summary["is_selected"]
    ].iloc[0]

    print(
        "=== PHASE 3 SELECTION ARTIFACT "
        "SMOKE TEST ==="
    )
    print(
        "Candidate space complete:",
        artifact[
            "candidate_space_complete"
        ],
    )
    print(
        "Summary rows:",
        len(summary),
    )
    print(
        "Selected model:",
        selected["model_name"],
    )
    print(
        "Selected candidate:",
        selected["candidate_id"],
    )
    print(
        "Mean validation macro-F1:",
        round(
            float(
                selected["mean_macro_f1"]
            ),
            6,
        ),
    )
    print(
        "Test metrics included:",
        artifact[
            "test_access_contract"
        ]["test_metrics_included"],
    )
    print(
        "Byte-deterministic artifacts: PASS"
    )
    print(
        "Selection artifact smoke test: PASS"
    )


def run_full_local() -> None:
    result, bundle = run_selection()

    artifact = build_selection_artifact(
        result=result,
        bundle=bundle,
        split_manifest_path=(
            DEFAULT_SPLIT_MANIFEST_PATH
        ),
        registry_path=DEFAULT_REGISTRY_PATH,
        require_complete=True,
    )

    write_selection_artifacts(
        artifact=artifact,
        json_output_path=(
            DEFAULT_OUTPUT_JSON_PATH
        ),
        csv_output_path=(
            DEFAULT_OUTPUT_CSV_PATH
        ),
    )

    summary = build_selection_summary_frame(
        artifact
    )

    selected = summary.loc[
        summary["is_selected"]
    ].sort_values(
        "outer_fold"
    )

    print(
        "=== PHASE 3 FULL LOCAL "
        "INNER SEARCH ==="
    )
    print(
        "Outer folds:",
        sorted(
            summary[
                "outer_fold"
            ].unique().tolist()
        ),
    )
    print(
        "Candidate rows:",
        len(summary),
    )

    for _, row in selected.iterrows():
        print(
            "Outer fold",
            int(row["outer_fold"]),
            "| model=",
            row["model_name"],
            "| candidate=",
            row["candidate_id"],
            "| mean_macro_f1=",
            round(
                float(
                    row["mean_macro_f1"]
                ),
                6,
            ),
        )

    print(
        "JSON:",
        relative_display_path(
            DEFAULT_OUTPUT_JSON_PATH
        ),
    )
    print(
        "CSV:",
        relative_display_path(
            DEFAULT_OUTPUT_CSV_PATH
        ),
    )
    print(
        "Full local selection artifacts: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic Phase 3 "
            "inner-selection artifacts."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--run-full-local",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if (
        arguments.smoke_test
        and arguments.run_full_local
    ):
        raise SystemExit(
            "Choose only one execution mode."
        )

    if arguments.smoke_test:
        smoke_test()
        return

    if arguments.run_full_local:
        run_full_local()
        return

    print(
        "Use --smoke-test or "
        "--run-full-local."
    )


if __name__ == "__main__":
    main()
