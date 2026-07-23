from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    PROJECT_ROOT = Path(
        __file__
    ).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )
else:
    PROJECT_ROOT = Path(
        __file__
    ).resolve().parents[1]


from scripts.phase3_dataset import (
    Phase3DatasetBundle,
    load_phase3_dataset,
    select_subject_partition,
)
from scripts.phase3_inner_search import (
    SELECTION_METRICS,
    normalize_subject_list,
    rank_selectable_candidates,
    selected_metric_snapshot,
)
from scripts.phase3_metrics import (
    align_probability_columns,
    calculate_classification_metrics,
    write_metrics_json,
)
from scripts.phase3_model_registry import (
    build_model_pipeline,
    enumerate_candidate_parameters,
)
from scripts.phase3_model_artifacts import (
    atomic_joblib_dump,
    canonical_json,
    relative_display_path,
    runtime_metadata,
    sha256_file,
    verify_roundtrip,
)


SCHEMA_VERSION = "1.0.0"

FINAL_MODEL_ARTIFACT_TYPE = (
    "phase3_local_final_refit_pipeline"
)

FINAL_SELECTION_ARTIFACT_TYPE = (
    "phase3_local_final_refit_selection"
)

FINAL_MANIFEST_ARTIFACT_TYPE = (
    "phase3_local_final_refit_manifest"
)

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_model_registry.json"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_local_final_refit"
    / "phase3_local_final_refit.joblib"
)

DEFAULT_SELECTION_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_final_refit_selection.json"
)

DEFAULT_SELECTION_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_final_refit_selection.csv"
)

DEFAULT_MANIFEST_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_final_refit_manifest.json"
)

DEFAULT_MANIFEST_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_final_refit_manifest.csv"
)

SELECTION_SUMMARY_COLUMNS = [
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

FINAL_MANIFEST_COLUMNS = [
    "model_name",
    "candidate_id",
    "candidate_parameters_json",
    "training_subjects",
    "training_row_count",
    "feature_count",
    "validation_fold_count",
    "selection_mean_macro_f1",
    "selection_std_macro_f1",
    "model_file_path",
    "model_file_size_bytes",
    "model_file_sha256",
    "reload_prediction_match",
    "reload_probability_match",
    "inference_ready",
    "deployment_ready",
    "scientific_reporting_allowed",
]


def load_registry(
    path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    resolved = path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            resolved
        )

    registry = json.loads(
        resolved.read_text(
            encoding="utf-8-sig"
        )
    )

    required_keys = {
        "schema_version",
        "random_seed",
        "primary_metric",
        "probability_output_required",
        "expected_model_count",
        "expected_selection_model_count",
        "expected_total_candidate_count",
        "models",
    }

    missing = required_keys - set(
        registry
    )

    if missing:
        raise ValueError(
            "Model registry is missing "
            f"required keys: {sorted(missing)}."
        )

    if registry[
        "primary_metric"
    ] != "macro_f1":
        raise ValueError(
            "Final refit requires macro_f1 "
            "as its primary metric."
        )

    if not bool(
        registry[
            "probability_output_required"
        ]
    ):
        raise ValueError(
            "Final refit requires "
            "probability-capable models."
        )

    if not isinstance(
        registry["models"],
        dict,
    ):
        raise TypeError(
            "Registry models must be "
            "a mapping."
        )

    return registry


def build_final_refit_logo_splits(
    bundle: Phase3DatasetBundle,
) -> list[dict[str, Any]]:
    subjects = tuple(
        sorted(
            int(value)
            for value in np.unique(
                bundle.groups
            )
        )
    )

    if len(subjects) < 2:
        raise ValueError(
            "Grouped LOGO selection "
            "requires at least two subjects."
        )

    splits: list[
        dict[str, Any]
    ] = []

    for (
        fold_index,
        validation_subject,
    ) in enumerate(
        subjects,
        start=1,
    ):
        train_subjects = [
            subject
            for subject in subjects
            if subject
            != validation_subject
        ]

        splits.append(
            {
                "split_id": (
                    "final_refit_logo_"
                    "validation_subject_"
                    f"{validation_subject}"
                ),
                "fold_index": (
                    fold_index
                ),
                "train_subjects": (
                    train_subjects
                ),
                "validation_subjects": [
                    validation_subject
                ],
            }
        )

    validation_subjects = [
        int(
            split[
                "validation_subjects"
            ][0]
        )
        for split in splits
    ]

    if sorted(
        validation_subjects
    ) != list(subjects):
        raise ValueError(
            "Every subject must appear "
            "exactly once as validation."
        )

    for split in splits:
        train = set(
            split[
                "train_subjects"
            ]
        )

        validation = set(
            split[
                "validation_subjects"
            ]
        )

        if train & validation:
            raise ValueError(
                "Training and validation "
                "subjects overlap."
            )

        if (
            train | validation
        ) != set(subjects):
            raise ValueError(
                "A LOGO split does not "
                "cover every subject."
            )

    return splits


def evaluate_candidate_on_logo_folds(
    bundle: Phase3DatasetBundle,
    logo_splits: Sequence[
        Mapping[str, Any]
    ],
    registry: Mapping[str, Any],
    model_name: str,
    candidate_index: int,
    candidate_parameters: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    if model_name not in registry[
        "models"
    ]:
        raise KeyError(
            f"Unknown model: {model_name}."
        )

    if candidate_index < 1:
        raise ValueError(
            "candidate_index must "
            "start from one."
        )

    model_spec = registry[
        "models"
    ][model_name]

    fold_metrics: list[
        dict[str, Any]
    ] = []

    seen_validation_subjects: set[
        Any
    ] = set()

    for split in logo_splits:
        train_subjects = (
            normalize_subject_list(
                split[
                    "train_subjects"
                ],
                "train subjects",
            )
        )

        validation_subjects = (
            normalize_subject_list(
                split[
                    "validation_subjects"
                ],
                "validation subjects",
            )
        )

        if (
            set(train_subjects)
            & set(validation_subjects)
        ):
            raise ValueError(
                "Training and validation "
                "subjects overlap."
            )

        duplicate_validation = (
            seen_validation_subjects
            & set(validation_subjects)
        )

        if duplicate_validation:
            raise ValueError(
                "Validation subjects are "
                "repeated across LOGO folds: "
                f"{sorted(duplicate_validation)}."
            )

        seen_validation_subjects.update(
            validation_subjects
        )

        train_partition = (
            select_subject_partition(
                bundle=bundle,
                subjects=train_subjects,
                name=(
                    "final_refit_train"
                ),
                require_all_classes=True,
            )
        )

        validation_partition = (
            select_subject_partition(
                bundle=bundle,
                subjects=(
                    validation_subjects
                ),
                name=(
                    "final_refit_validation"
                ),
                require_all_classes=True,
            )
        )

        train_rows = set(
            train_partition
            .row_indices
            .tolist()
        )

        validation_rows = set(
            validation_partition
            .row_indices
            .tolist()
        )

        if train_rows & validation_rows:
            raise ValueError(
                "Training and validation "
                "rows overlap."
            )

        pipeline = (
            build_model_pipeline(
                model_name=model_name,
                registry=dict(
                    registry
                ),
            )
        )

        pipeline.set_params(
            **dict(
                candidate_parameters
            )
        )

        pipeline.fit(
            train_partition.X,
            train_partition.y,
        )

        predicted_labels = (
            pipeline.predict(
                validation_partition.X
            )
        )

        raw_probabilities = (
            pipeline.predict_proba(
                validation_partition.X
            )
        )

        classifier = (
            pipeline.named_steps[
                "classifier"
            ]
        )

        probabilities = (
            align_probability_columns(
                probabilities=(
                    raw_probabilities
                ),
                estimator_classes=(
                    classifier.classes_
                ),
                class_mapping=(
                    bundle.class_mapping
                ),
            )
        )

        metrics = (
            calculate_classification_metrics(
                y_true=(
                    validation_partition.y
                ),
                y_pred=predicted_labels,
                probabilities=probabilities,
                class_mapping=(
                    bundle.class_mapping
                ),
            )
        )

        fold_metrics.append(
            {
                "split_id": str(
                    split["split_id"]
                ),
                "fold_index": int(
                    split["fold_index"]
                ),
                "train_subjects": list(
                    train_subjects
                ),
                "validation_subjects": (
                    list(
                        validation_subjects
                    )
                ),
                "train_row_count": int(
                    train_partition
                    .row_count
                ),
                "validation_row_count": int(
                    validation_partition
                    .row_count
                ),
                "metrics": (
                    selected_metric_snapshot(
                        metrics
                    )
                ),
            }
        )

    expected_subjects = set(
        int(value)
        for value in np.unique(
            bundle.groups
        )
    )

    if (
        seen_validation_subjects
        != expected_subjects
    ):
        raise ValueError(
            "LOGO validation folds do not "
            "cover every subject exactly once."
        )

    aggregate: dict[
        str,
        float,
    ] = {}

    for metric_name in (
        SELECTION_METRICS
    ):
        values = np.asarray(
            [
                fold["metrics"][
                    metric_name
                ]
                for fold
                in fold_metrics
            ],
            dtype=float,
        )

        aggregate[
            f"mean_{metric_name}"
        ] = float(
            values.mean()
        )

        aggregate[
            f"std_{metric_name}"
        ] = float(
            values.std(
                ddof=0
            )
        )

    return {
        "model_name": (
            model_name
        ),
        "candidate_index": int(
            candidate_index
        ),
        "candidate_id": (
            f"{model_name}__"
            f"candidate_"
            f"{candidate_index:03d}"
        ),
        "candidate_parameters": dict(
            candidate_parameters
        ),
        "eligible_for_selection": (
            bool(
                model_spec[
                    "eligible_for_selection"
                ]
            )
        ),
        "complexity_rank": int(
            model_spec[
                "complexity_rank"
            ]
        ),
        "fold_count": len(
            fold_metrics
        ),
        "fold_metrics": (
            fold_metrics
        ),
        "aggregate": aggregate,
    }


def display_path(
    path: Path,
) -> str:
    resolved = path.resolve()

    try:
        return relative_display_path(
            resolved
        )
    except ValueError:
        return str(resolved)


def evaluate_final_refit_candidate_space(
    bundle: Phase3DatasetBundle,
    registry: Mapping[str, Any],
    logo_splits: Sequence[
        Mapping[str, Any]
    ],
    model_names: Sequence[str] | None = None,
    max_candidates_per_model: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    available_model_names = tuple(
        registry["models"]
    )

    if model_names is None:
        selected_model_names = (
            available_model_names
        )
    else:
        selected_model_names = tuple(
            str(value)
            for value in model_names
        )

        unknown_models = sorted(
            set(selected_model_names)
            - set(available_model_names)
        )

        if unknown_models:
            raise ValueError(
                "Unknown requested models: "
                f"{unknown_models}."
            )

        if len(
            selected_model_names
        ) != len(
            set(selected_model_names)
        ):
            raise ValueError(
                "Requested model names "
                "contain duplicates."
            )

    if (
        max_candidates_per_model
        is not None
        and max_candidates_per_model < 1
    ):
        raise ValueError(
            "max_candidates_per_model "
            "must be positive."
        )

    evaluation_plan = []

    for model_name in (
        selected_model_names
    ):
        candidate_parameters = (
            enumerate_candidate_parameters(
                model_name=model_name,
                registry=dict(registry),
            )
        )

        if (
            max_candidates_per_model
            is not None
        ):
            candidate_parameters = (
                candidate_parameters[
                    :max_candidates_per_model
                ]
            )

        for (
            candidate_index,
            parameters,
        ) in enumerate(
            candidate_parameters,
            start=1,
        ):
            evaluation_plan.append(
                (
                    model_name,
                    candidate_index,
                    parameters,
                )
            )

    if not evaluation_plan:
        raise ValueError(
            "Final-refit evaluation "
            "plan is empty."
        )

    candidate_results = []

    total = len(evaluation_plan)

    for current, (
        model_name,
        candidate_index,
        candidate_parameters,
    ) in enumerate(
        evaluation_plan,
        start=1,
    ):
        candidate_id = (
            f"{model_name}__"
            f"candidate_"
            f"{candidate_index:03d}"
        )

        if progress:
            print(
                f"[{current:02d}/{total:02d}] "
                f"Evaluating {candidate_id}"
            )

        result = (
            evaluate_candidate_on_logo_folds(
                bundle=bundle,
                logo_splits=logo_splits,
                registry=registry,
                model_name=model_name,
                candidate_index=(
                    candidate_index
                ),
                candidate_parameters=(
                    candidate_parameters
                ),
            )
        )

        candidate_results.append(
            result
        )

        if progress:
            print(
                "    mean macro-F1:",
                result["aggregate"][
                    "mean_macro_f1"
                ],
            )

    ranked_candidates = (
        rank_selectable_candidates(
            candidate_results
        )
    )

    selected_candidate = dict(
        ranked_candidates[0]
    )

    candidate_space_complete = (
        model_names is None
        and max_candidates_per_model
        is None
        and len(candidate_results)
        == int(
            registry[
                "expected_total_candidate_count"
            ]
        )
    )

    return {
        "candidate_space_complete": bool(
            candidate_space_complete
        ),
        "evaluated_model_names": list(
            selected_model_names
        ),
        "evaluated_candidate_count": len(
            candidate_results
        ),
        "fold_count": len(
            logo_splits
        ),
        "all_candidate_results": (
            candidate_results
        ),
        "ranked_selectable_candidates": (
            ranked_candidates
        ),
        "selected_candidate": (
            selected_candidate
        ),
    }


def build_selection_artifact(
    bundle: Phase3DatasetBundle,
    registry: Mapping[str, Any],
    registry_path: Path,
    logo_splits: Sequence[
        Mapping[str, Any]
    ],
    selection_result: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    subjects = sorted(
        int(value)
        for value in np.unique(
            bundle.groups
        )
    )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "artifact_type": (
            FINAL_SELECTION_ARTIFACT_TYPE
        ),
        "intended_use": (
            "local_engineering_"
            "model_selection"
        ),
        "scientific_reporting_allowed": (
            False
        ),
        "selection_partition": (
            "grouped_logo_"
            "validation_only"
        ),
        "test_partition_used": False,
        "outer_test_metrics_loaded": False,
        "outer_test_predictions_loaded": (
            False
        ),
        "candidate_space_complete": bool(
            selection_result[
                "candidate_space_complete"
            ]
        ),
        "primary_metric": "macro_f1",
        "ranking_policy": {
            "first": (
                "higher_mean_macro_f1"
            ),
            "second": (
                "lower_std_macro_f1"
            ),
            "third": (
                "higher_mean_"
                "balanced_accuracy"
            ),
            "fourth": (
                "lower_mean_"
                "multiclass_log_loss"
            ),
            "fifth": (
                "lower_model_complexity"
            ),
        },
        "subjects": subjects,
        "subject_count": len(subjects),
        "row_count": int(
            len(bundle.y)
        ),
        "feature_count": len(
            bundle.feature_names
        ),
        "fold_count": len(
            logo_splits
        ),
        "logo_splits": [
            dict(split)
            for split in logo_splits
        ],
        "selection_result": dict(
            selection_result
        ),
        "source": {
            "model_input_path": (
                display_path(
                    bundle.data_path
                )
            ),
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_path": (
                display_path(
                    bundle.schema_path
                )
            ),
            "model_schema_sha256": (
                bundle.schema_sha256
            ),
            "evaluation_protocol_path": (
                display_path(
                    bundle.protocol_path
                )
            ),
            "evaluation_protocol_sha256": (
                bundle.protocol_sha256
            ),
            "model_registry_path": (
                display_path(
                    registry_path
                )
            ),
            "model_registry_sha256": (
                sha256_file(
                    registry_path
                )
            ),
            "registry_schema_version": (
                str(
                    registry[
                        "schema_version"
                    ]
                )
            ),
        },
        "runtime": runtime_metadata(),
    }


def build_selection_summary_frame(
    artifact: Mapping[str, Any],
) -> pd.DataFrame:
    selection_result = artifact[
        "selection_result"
    ]

    selected_id = str(
        selection_result[
            "selected_candidate"
        ][
            "candidate_id"
        ]
    )

    rank_by_candidate = {
        str(
            result["candidate_id"]
        ): int(
            result["selection_rank"]
        )
        for result in selection_result[
            "ranked_selectable_candidates"
        ]
    }

    rows = []

    for result in selection_result[
        "all_candidate_results"
    ]:
        candidate_id = str(
            result["candidate_id"]
        )

        aggregate = result[
            "aggregate"
        ]

        row = {
            "model_name": str(
                result["model_name"]
            ),
            "candidate_index": int(
                result["candidate_index"]
            ),
            "candidate_id": (
                candidate_id
            ),
            "candidate_parameters_json": (
                canonical_json(
                    result[
                        "candidate_parameters"
                    ]
                )
            ),
            "eligible_for_selection": (
                bool(
                    result[
                        "eligible_for_selection"
                    ]
                )
            ),
            "selection_rank": (
                rank_by_candidate.get(
                    candidate_id
                )
            ),
            "is_selected": (
                candidate_id
                == selected_id
            ),
            "complexity_rank": int(
                result[
                    "complexity_rank"
                ]
            ),
            "fold_count": int(
                result["fold_count"]
            ),
        }

        for metric_name in (
            SELECTION_METRICS
        ):
            row[
                f"mean_{metric_name}"
            ] = float(
                aggregate[
                    f"mean_{metric_name}"
                ]
            )

            row[
                f"std_{metric_name}"
            ] = float(
                aggregate[
                    f"std_{metric_name}"
                ]
            )

        rows.append(row)

    frame = pd.DataFrame(
        rows,
        columns=(
            SELECTION_SUMMARY_COLUMNS
        ),
    )

    if frame.empty:
        raise ValueError(
            "Selection summary is empty."
        )

    if int(
        frame[
            "is_selected"
        ].sum()
    ) != 1:
        raise ValueError(
            "Selection summary must "
            "contain exactly one "
            "selected candidate."
        )

    return frame.sort_values(
        by=[
            "eligible_for_selection",
            "selection_rank",
            "model_name",
            "candidate_index",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
        na_position="last",
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def write_selection_artifacts(
    artifact: Mapping[str, Any],
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    write_metrics_json(
        metrics=artifact,
        output_path=(
            json_output_path
        ),
    )

    frame = (
        build_selection_summary_frame(
            artifact
        )
    )

    resolved_csv_path = (
        csv_output_path.resolve()
    )

    resolved_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        resolved_csv_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def build_final_refit_metadata(
    bundle: Phase3DatasetBundle,
    selected_candidate: Mapping[
        str,
        Any,
    ],
    selection_artifact: Mapping[
        str,
        Any,
    ],
    selection_json_path: Path,
) -> dict[str, Any]:
    subjects = sorted(
        int(value)
        for value in np.unique(
            bundle.groups
        )
    )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "artifact_type": (
            FINAL_MODEL_ARTIFACT_TYPE
        ),
        "intended_use": (
            "local_engineering_inference"
        ),
        "inference_ready": True,
        "deployment_ready": False,
        "deployment_block_reason": (
            "The local engineering "
            "dataset contains four "
            "subjects and does not "
            "satisfy the five-outer-fold "
            "full scientific protocol."
        ),
        "scientific_reporting_allowed": (
            False
        ),
        "training_scope": (
            "all_local_subjects_after_"
            "grouped_logo_selection"
        ),
        "selection_scope": (
            "four_fold_grouped_logo_"
            "validation"
        ),
        "selection_candidate_space_complete": (
            bool(
                selection_artifact[
                    "candidate_space_complete"
                ]
            )
        ),
        "training_subjects": subjects,
        "excluded_test_subjects": [],
        "training_row_count": int(
            len(bundle.y)
        ),
        "validation_fold_count": int(
            selection_artifact[
                "fold_count"
            ]
        ),
        "test_feature_matrix_loaded": (
            False
        ),
        "outer_test_metrics_loaded": (
            False
        ),
        "outer_test_predictions_loaded": (
            False
        ),
        "hyperparameter_search_performed": (
            True
        ),
        "model_name": str(
            selected_candidate[
                "model_name"
            ]
        ),
        "candidate_id": str(
            selected_candidate[
                "candidate_id"
            ]
        ),
        "candidate_parameters": dict(
            selected_candidate[
                "candidate_parameters"
            ]
        ),
        "selection_validation_summary": (
            dict(
                selected_candidate[
                    "aggregate"
                ]
            )
        ),
        "feature_names": list(
            bundle.feature_names
        ),
        "feature_count": len(
            bundle.feature_names
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
        "target_name_column": (
            bundle.target_name_column
        ),
        "source": {
            "model_input_path": (
                display_path(
                    bundle.data_path
                )
            ),
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_path": (
                display_path(
                    bundle.schema_path
                )
            ),
            "model_schema_sha256": (
                bundle.schema_sha256
            ),
            "evaluation_protocol_path": (
                display_path(
                    bundle.protocol_path
                )
            ),
            "evaluation_protocol_sha256": (
                bundle.protocol_sha256
            ),
            "selection_artifact_path": (
                display_path(
                    selection_json_path
                )
            ),
            "selection_artifact_sha256": (
                sha256_file(
                    selection_json_path
                )
            ),
        },
        "runtime": runtime_metadata(),
    }


def validate_final_refit_payload(
    payload: Mapping[str, Any],
    expected_feature_names: (
        Sequence[str] | None
    ) = None,
    expected_class_mapping: (
        Mapping[str, int] | None
    ) = None,
    require_complete_selection: (
        bool
    ) = True,
) -> None:
    if set(payload) != {
        "metadata",
        "pipeline",
    }:
        raise ValueError(
            "Final-refit payload must "
            "contain exactly metadata "
            "and pipeline."
        )

    metadata = payload[
        "metadata"
    ]

    pipeline = payload[
        "pipeline"
    ]

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise TypeError(
            "Final-refit metadata "
            "must be a mapping."
        )

    if metadata.get(
        "artifact_type"
    ) != FINAL_MODEL_ARTIFACT_TYPE:
        raise ValueError(
            "Unexpected final-refit "
            "model artifact type."
        )

    if not bool(
        metadata.get(
            "inference_ready"
        )
    ):
        raise ValueError(
            "Final-refit model must "
            "be inference-ready."
        )

    if bool(
        metadata.get(
            "deployment_ready"
        )
    ):
        raise ValueError(
            "Local final-refit model "
            "cannot be deployment-ready."
        )

    if bool(
        metadata.get(
            "scientific_reporting_allowed"
        )
    ):
        raise ValueError(
            "Local final-refit model "
            "cannot authorize "
            "scientific reporting."
        )

    if (
        require_complete_selection
        and not bool(
            metadata.get(
                "selection_candidate_"
                "space_complete"
            )
        )
    ):
        raise ValueError(
            "Final-refit model requires "
            "a complete candidate search."
        )

    if metadata.get(
        "excluded_test_subjects"
    ) != []:
        raise ValueError(
            "Final-refit model must "
            "not declare excluded "
            "test subjects."
        )

    for forbidden_flag in (
        "test_feature_matrix_loaded",
        "outer_test_metrics_loaded",
        "outer_test_predictions_loaded",
    ):
        if bool(
            metadata.get(
                forbidden_flag
            )
        ):
            raise ValueError(
                "Forbidden evaluation "
                "payload entered refit: "
                f"{forbidden_flag}."
            )

    if not bool(
        metadata.get(
            "hyperparameter_search_"
            "performed"
        )
    ):
        raise ValueError(
            "Final-refit metadata must "
            "record grouped model selection."
        )

    training_subjects = (
        normalize_subject_list(
            metadata[
                "training_subjects"
            ],
            "training subjects",
        )
    )

    if len(
        training_subjects
    ) < 2:
        raise ValueError(
            "Final-refit model requires "
            "at least two training subjects."
        )

    if int(
        metadata[
            "training_row_count"
        ]
    ) < 1:
        raise ValueError(
            "Final-refit training row "
            "count must be positive."
        )

    feature_names = tuple(
        str(value)
        for value in metadata[
            "feature_names"
        ]
    )

    if len(feature_names) != int(
        metadata["feature_count"]
    ):
        raise ValueError(
            "Feature count does not "
            "match the feature-name "
            "contract."
        )

    if len(
        set(feature_names)
    ) != len(feature_names):
        raise ValueError(
            "Duplicate feature names "
            "found in final-refit "
            "metadata."
        )

    if (
        expected_feature_names
        is not None
        and feature_names
        != tuple(
            expected_feature_names
        )
    ):
        raise ValueError(
            "Final-refit feature order "
            "does not match the "
            "expected contract."
        )

    class_mapping = {
        str(name): int(encoded)
        for name, encoded
        in metadata[
            "class_mapping"
        ].items()
    }

    if (
        expected_class_mapping
        is not None
    ):
        normalized_expected = {
            str(name): int(encoded)
            for name, encoded
            in expected_class_mapping
            .items()
        }

        if (
            class_mapping
            != normalized_expected
        ):
            raise ValueError(
                "Final-refit class mapping "
                "does not match the "
                "expected contract."
            )

    if not hasattr(
        pipeline,
        "predict",
    ):
        raise TypeError(
            "Final-refit pipeline "
            "has no predict method."
        )

    if not hasattr(
        pipeline,
        "predict_proba",
    ):
        raise TypeError(
            "Final-refit pipeline has "
            "no predict_proba method."
        )


def load_final_refit_payload(
    path: Path,
    expected_feature_names: (
        Sequence[str] | None
    ) = None,
    expected_class_mapping: (
        Mapping[str, int] | None
    ) = None,
    require_complete_selection: (
        bool
    ) = True,
) -> dict[str, Any]:
    resolved = path.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            resolved
        )

    payload = joblib.load(
        resolved
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Loaded final-refit payload "
            "must be a dictionary."
        )

    validate_final_refit_payload(
        payload=payload,
        expected_feature_names=(
            expected_feature_names
        ),
        expected_class_mapping=(
            expected_class_mapping
        ),
        require_complete_selection=(
            require_complete_selection
        ),
    )

    return payload


def build_final_manifest_frame(
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    models = manifest[
        "models"
    ]

    if len(models) != 1:
        raise ValueError(
            "Final-refit manifest must "
            "contain exactly one model."
        )

    model = models[0]

    aggregate = model[
        "selection_validation_summary"
    ]

    row = {
        "model_name": str(
            model["model_name"]
        ),
        "candidate_id": str(
            model["candidate_id"]
        ),
        "candidate_parameters_json": (
            canonical_json(
                model[
                    "candidate_parameters"
                ]
            )
        ),
        "training_subjects": ",".join(
            str(value)
            for value in model[
                "training_subjects"
            ]
        ),
        "training_row_count": int(
            model[
                "training_row_count"
            ]
        ),
        "feature_count": int(
            model["feature_count"]
        ),
        "validation_fold_count": int(
            model[
                "validation_fold_count"
            ]
        ),
        "selection_mean_macro_f1": (
            float(
                aggregate[
                    "mean_macro_f1"
                ]
            )
        ),
        "selection_std_macro_f1": (
            float(
                aggregate[
                    "std_macro_f1"
                ]
            )
        ),
        "model_file_path": str(
            model[
                "model_file_path"
            ]
        ),
        "model_file_size_bytes": int(
            model[
                "model_file_size_bytes"
            ]
        ),
        "model_file_sha256": str(
            model[
                "model_file_sha256"
            ]
        ),
        "reload_prediction_match": (
            bool(
                model[
                    "reload_prediction_match"
                ]
            )
        ),
        "reload_probability_match": (
            bool(
                model[
                    "reload_probability_match"
                ]
            )
        ),
        "inference_ready": bool(
            model["inference_ready"]
        ),
        "deployment_ready": bool(
            model["deployment_ready"]
        ),
        "scientific_reporting_allowed": (
            bool(
                model[
                    "scientific_reporting_allowed"
                ]
            )
        ),
    }

    return pd.DataFrame(
        [row],
        columns=(
            FINAL_MANIFEST_COLUMNS
        ),
    )


def write_final_manifest(
    manifest: Mapping[str, Any],
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    write_metrics_json(
        metrics=manifest,
        output_path=(
            json_output_path
        ),
    )

    frame = (
        build_final_manifest_frame(
            manifest
        )
    )

    resolved_csv_path = (
        csv_output_path.resolve()
    )

    resolved_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        resolved_csv_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def run_final_refit(
    bundle: Phase3DatasetBundle,
    registry: Mapping[str, Any],
    registry_path: Path,
    model_output_path: Path,
    selection_json_path: Path,
    selection_csv_path: Path,
    manifest_json_path: Path,
    manifest_csv_path: Path,
    model_names: (
        Sequence[str] | None
    ) = None,
    max_candidates_per_model: (
        int | None
    ) = None,
    require_complete_selection: (
        bool
    ) = True,
    progress: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    logo_splits = (
        build_final_refit_logo_splits(
            bundle
        )
    )

    selection_result = (
        evaluate_final_refit_candidate_space(
            bundle=bundle,
            registry=registry,
            logo_splits=logo_splits,
            model_names=model_names,
            max_candidates_per_model=(
                max_candidates_per_model
            ),
            progress=progress,
        )
    )

    if (
        require_complete_selection
        and not bool(
            selection_result[
                "candidate_space_complete"
            ]
        )
    ):
        raise ValueError(
            "A persistent final-refit "
            "artifact requires complete "
            "candidate evaluation."
        )

    selection_artifact = (
        build_selection_artifact(
            bundle=bundle,
            registry=registry,
            registry_path=(
                registry_path
            ),
            logo_splits=logo_splits,
            selection_result=(
                selection_result
            ),
        )
    )

    write_selection_artifacts(
        artifact=selection_artifact,
        json_output_path=(
            selection_json_path
        ),
        csv_output_path=(
            selection_csv_path
        ),
    )

    selected_candidate = (
        selection_result[
            "selected_candidate"
        ]
    )

    if progress:
        print(
            "Refitting selected candidate "
            "on all local rows:",
            selected_candidate[
                "candidate_id"
            ],
        )

    pipeline = build_model_pipeline(
        model_name=(
            selected_candidate[
                "model_name"
            ]
        ),
        registry=dict(registry),
    )

    pipeline.set_params(
        **dict(
            selected_candidate[
                "candidate_parameters"
            ]
        )
    )

    pipeline.fit(
        bundle.X,
        bundle.y,
    )

    metadata = (
        build_final_refit_metadata(
            bundle=bundle,
            selected_candidate=(
                selected_candidate
            ),
            selection_artifact=(
                selection_artifact
            ),
            selection_json_path=(
                selection_json_path
            ),
        )
    )

    payload = {
        "metadata": metadata,
        "pipeline": pipeline,
    }

    validate_final_refit_payload(
        payload=payload,
        expected_feature_names=(
            bundle.feature_names
        ),
        expected_class_mapping=(
            bundle.class_mapping
        ),
        require_complete_selection=(
            require_complete_selection
        ),
    )

    atomic_joblib_dump(
        payload=payload,
        output_path=model_output_path,
    )

    loaded_payload = (
        load_final_refit_payload(
            path=model_output_path,
            expected_feature_names=(
                bundle.feature_names
            ),
            expected_class_mapping=(
                bundle.class_mapping
            ),
            require_complete_selection=(
                require_complete_selection
            ),
        )
    )

    verification_row_count = min(
        256,
        len(bundle.X),
    )

    verification_X = bundle.X.iloc[
        :verification_row_count
    ].copy()

    (
        prediction_match,
        probability_match,
    ) = verify_roundtrip(
        original_pipeline=pipeline,
        loaded_pipeline=(
            loaded_payload[
                "pipeline"
            ]
        ),
        X=verification_X,
        class_mapping=(
            bundle.class_mapping
        ),
    )

    if not prediction_match:
        raise ValueError(
            "Final-refit prediction "
            "roundtrip failed."
        )

    if not probability_match:
        raise ValueError(
            "Final-refit probability "
            "roundtrip failed."
        )

    model_record = {
        "model_name": str(
            selected_candidate[
                "model_name"
            ]
        ),
        "candidate_id": str(
            selected_candidate[
                "candidate_id"
            ]
        ),
        "candidate_parameters": dict(
            selected_candidate[
                "candidate_parameters"
            ]
        ),
        "training_subjects": list(
            metadata[
                "training_subjects"
            ]
        ),
        "training_row_count": int(
            metadata[
                "training_row_count"
            ]
        ),
        "feature_count": int(
            metadata["feature_count"]
        ),
        "validation_fold_count": int(
            metadata[
                "validation_fold_count"
            ]
        ),
        "selection_validation_summary": (
            dict(
                metadata[
                    "selection_validation_summary"
                ]
            )
        ),
        "model_file_path": (
            display_path(
                model_output_path
            )
        ),
        "model_file_size_bytes": int(
            model_output_path
            .stat()
            .st_size
        ),
        "model_file_sha256": (
            sha256_file(
                model_output_path
            )
        ),
        "reload_prediction_match": (
            prediction_match
        ),
        "reload_probability_match": (
            probability_match
        ),
        "inference_ready": True,
        "deployment_ready": False,
        "scientific_reporting_allowed": (
            False
        ),
    }

    manifest = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "artifact_type": (
            FINAL_MANIFEST_ARTIFACT_TYPE
        ),
        "intended_use": (
            "local_engineering_inference"
        ),
        "model_count": 1,
        "candidate_space_complete": (
            bool(
                selection_result[
                    "candidate_space_complete"
                ]
            )
        ),
        "scientific_reporting_allowed": (
            False
        ),
        "deployment": {
            "inference_ready": True,
            "deployment_ready": False,
            "block_reason": (
                metadata[
                    "deployment_block_reason"
                ]
            ),
        },
        "models": [
            model_record
        ],
        "source": {
            "selection_artifact_path": (
                display_path(
                    selection_json_path
                )
            ),
            "selection_artifact_sha256": (
                sha256_file(
                    selection_json_path
                )
            ),
            "model_registry_path": (
                display_path(
                    registry_path
                )
            ),
            "model_registry_sha256": (
                sha256_file(
                    registry_path
                )
            ),
            "model_input_path": (
                display_path(
                    bundle.data_path
                )
            ),
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_path": (
                display_path(
                    bundle.schema_path
                )
            ),
            "model_schema_sha256": (
                bundle.schema_sha256
            ),
            "evaluation_protocol_path": (
                display_path(
                    bundle.protocol_path
                )
            ),
            "evaluation_protocol_sha256": (
                bundle.protocol_sha256
            ),
        },
        "runtime": runtime_metadata(),
    }

    write_final_manifest(
        manifest=manifest,
        json_output_path=(
            manifest_json_path
        ),
        csv_output_path=(
            manifest_csv_path
        ),
    )

    return (
        selection_artifact,
        manifest,
    )


def smoke_test() -> None:
    bundle = load_phase3_dataset()

    registry = load_registry(
        DEFAULT_REGISTRY_PATH
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        (
            selection,
            manifest,
        ) = run_final_refit(
            bundle=bundle,
            registry=registry,
            registry_path=(
                DEFAULT_REGISTRY_PATH
            ),
            model_output_path=(
                root / "model.joblib"
            ),
            selection_json_path=(
                root / "selection.json"
            ),
            selection_csv_path=(
                root / "selection.csv"
            ),
            manifest_json_path=(
                root / "manifest.json"
            ),
            manifest_csv_path=(
                root / "manifest.csv"
            ),
            model_names=[
                "logistic_regression"
            ],
            max_candidates_per_model=1,
            require_complete_selection=(
                False
            ),
        )

        selected = selection[
            "selection_result"
        ][
            "selected_candidate"
        ]

        model = manifest[
            "models"
        ][0]

        if selected[
            "candidate_id"
        ] != (
            "logistic_regression"
            "__candidate_001"
        ):
            raise ValueError(
                "Smoke test selected an "
                "unexpected candidate."
            )

        if model[
            "training_row_count"
        ] != len(bundle.y):
            raise ValueError(
                "Smoke refit did not "
                "use every local row."
            )

        if model[
            "training_subjects"
        ] != [0, 1, 2, 3]:
            raise ValueError(
                "Smoke refit did not "
                "use every local subject."
            )

        if not model[
            "reload_prediction_match"
        ]:
            raise ValueError(
                "Smoke prediction "
                "roundtrip failed."
            )

        if not model[
            "reload_probability_match"
        ]:
            raise ValueError(
                "Smoke probability "
                "roundtrip failed."
            )

        print(
            "=== PHASE 3 FINAL REFIT "
            "SMOKE TEST ==="
        )

        print(
            "Rows:",
            len(bundle.y),
        )

        print(
            "Features:",
            len(
                bundle.feature_names
            ),
        )

        print(
            "Subjects:",
            model[
                "training_subjects"
            ],
        )

        print(
            "LOGO folds:",
            selection[
                "fold_count"
            ],
        )

        print(
            "Candidate:",
            selected[
                "candidate_id"
            ],
        )

        print(
            "Candidate space complete:",
            selection[
                "candidate_space_complete"
            ],
        )

        print(
            "Inference ready:",
            model[
                "inference_ready"
            ],
        )

        print(
            "Deployment ready:",
            model[
                "deployment_ready"
            ],
        )

        print(
            "Scientific reporting allowed:",
            model[
                "scientific_reporting_allowed"
            ],
        )

        print(
            "Prediction roundtrip: PASS"
        )

        print(
            "Probability roundtrip: PASS"
        )

        print(
            "Final refit smoke test: PASS"
        )


def run_full_local() -> None:
    bundle = load_phase3_dataset()

    registry = load_registry(
        DEFAULT_REGISTRY_PATH
    )

    (
        selection,
        manifest,
    ) = run_final_refit(
        bundle=bundle,
        registry=registry,
        registry_path=(
            DEFAULT_REGISTRY_PATH
        ),
        model_output_path=(
            DEFAULT_MODEL_PATH
        ),
        selection_json_path=(
            DEFAULT_SELECTION_JSON_PATH
        ),
        selection_csv_path=(
            DEFAULT_SELECTION_CSV_PATH
        ),
        manifest_json_path=(
            DEFAULT_MANIFEST_JSON_PATH
        ),
        manifest_csv_path=(
            DEFAULT_MANIFEST_CSV_PATH
        ),
        require_complete_selection=True,
        progress=True,
    )

    selected = selection[
        "selection_result"
    ][
        "selected_candidate"
    ]

    aggregate = selected[
        "aggregate"
    ]

    model = manifest[
        "models"
    ][0]

    print(
        "=== PHASE 3 LOCAL "
        "FINAL REFIT ==="
    )

    print(
        "Evaluated candidates:",
        selection[
            "selection_result"
        ][
            "evaluated_candidate_count"
        ],
    )

    print(
        "LOGO folds:",
        selection[
            "fold_count"
        ],
    )

    print(
        "Selected model:",
        selected[
            "model_name"
        ],
    )

    print(
        "Selected candidate:",
        selected[
            "candidate_id"
        ],
    )

    print(
        "Selection mean macro-F1:",
        aggregate[
            "mean_macro_f1"
        ],
    )

    print(
        "Selection std macro-F1:",
        aggregate[
            "std_macro_f1"
        ],
    )

    print(
        "Training rows:",
        model[
            "training_row_count"
        ],
    )

    print(
        "Training subjects:",
        model[
            "training_subjects"
        ],
    )

    print(
        "Model path:",
        model[
            "model_file_path"
        ],
    )

    print(
        "Model SHA256:",
        model[
            "model_file_sha256"
        ],
    )

    print(
        "Inference ready:",
        model[
            "inference_ready"
        ],
    )

    print(
        "Deployment ready:",
        model[
            "deployment_ready"
        ],
    )

    print(
        "Scientific reporting allowed:",
        model[
            "scientific_reporting_allowed"
        ],
    )

    print(
        "Local final refit: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and refit a local "
            "Phase 3 engineering model "
            "using grouped LOGO validation."
        )
    )

    action = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    action.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run an isolated inexpensive "
            "final-refit smoke test."
        ),
    )

    action.add_argument(
        "--run-full-local",
        action="store_true",
        help=(
            "Evaluate every candidate and "
            "persist the local refit model."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.smoke_test:
        smoke_test()
        return

    if arguments.run_full_local:
        run_full_local()
        return

    raise RuntimeError(
        "No final-refit action "
        "was selected."
    )


if __name__ == "__main__":
    main()
