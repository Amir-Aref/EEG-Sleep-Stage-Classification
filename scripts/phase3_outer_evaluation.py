from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
    select_subject_partition,
)
from scripts.phase3_inner_search import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    load_json_object,
    normalize_subject_list,
    validate_local_split_manifest,
)
from scripts.phase3_metrics import (
    align_probability_columns,
    build_prediction_frame,
    calculate_classification_metrics,
    probability_column_name,
    write_metrics_json,
    write_prediction_csv,
)
from scripts.phase3_model_registry import (
    DEFAULT_CONFIG_PATH as DEFAULT_REGISTRY_PATH,
    build_model_pipeline,
    load_registry_config,
)
from scripts.phase3_selection_artifacts import (
    DEFAULT_OUTPUT_JSON_PATH as DEFAULT_SELECTION_PATH,
    validate_inner_search_result_safety,
)


DEFAULT_OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_outer_evaluation.json"
)

DEFAULT_OUTPUT_SUMMARY_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_outer_fold_metrics.csv"
)

DEFAULT_OUTPUT_PREDICTIONS_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_outer_test_predictions.csv"
)


SUMMARY_COLUMNS = [
    "outer_fold",
    "test_subjects",
    "outer_development_subjects",
    "model_name",
    "candidate_id",
    "candidate_parameters_json",
    "development_row_count",
    "test_row_count",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "multiclass_log_loss",
    "predict_probability_argmax_agreement",
    "predict_probability_argmax_mismatch_count",
    "mean_prediction_confidence",
    "mean_prediction_margin",
    "mean_normalized_entropy",
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def load_selection_artifact(
    path: Path = DEFAULT_SELECTION_PATH,
) -> dict[str, Any]:
    artifact = load_json_object(
        path.resolve()
    )

    if artifact.get("artifact_type") != (
        "phase3_inner_model_selection"
    ):
        raise ValueError(
            "Unexpected selection artifact type."
        )

    return artifact


def validate_outer_evaluation_inputs(
    bundle: Phase3DatasetBundle,
    split_manifest: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    split_manifest_path: Path,
    selection_artifact_path: Path,
    registry_path: Path,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, dict[str, Any]],
]:
    split_manifest_path = (
        split_manifest_path.resolve()
    )

    selection_artifact_path = (
        selection_artifact_path.resolve()
    )

    registry_path = registry_path.resolve()

    for path, label in (
        (split_manifest_path, "split manifest"),
        (
            selection_artifact_path,
            "selection artifact",
        ),
        (registry_path, "model registry"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    if selection_artifact.get(
        "artifact_type"
    ) != "phase3_inner_model_selection":
        raise ValueError(
            "Invalid selection artifact type."
        )

    if not bool(
        selection_artifact.get(
            "candidate_space_complete"
        )
    ):
        raise ValueError(
            "Outer evaluation requires a complete "
            "candidate-selection artifact."
        )

    selection_result = selection_artifact[
        "selection_result"
    ]

    validate_inner_search_result_safety(
        result=selection_result,
        require_complete=True,
    )

    source = selection_artifact["source"]

    expected_hashes = {
        "model_input_sha256": (
            bundle.data_sha256
        ),
        "model_schema_sha256": (
            bundle.schema_sha256
        ),
        "evaluation_protocol_sha256": (
            bundle.protocol_sha256
        ),
        "split_manifest_sha256": (
            sha256_file(split_manifest_path)
        ),
        "model_registry_sha256": (
            sha256_file(registry_path)
        ),
    }

    for key, expected_value in (
        expected_hashes.items()
    ):
        observed_value = source.get(key)

        if observed_value != expected_value:
            raise ValueError(
                f"Selection artifact {key} "
                "does not match the current input."
            )

    grouped_splits = (
        validate_local_split_manifest(
            manifest=split_manifest,
            bundle=bundle,
        )
    )

    selection_by_outer: dict[
        int,
        dict[str, Any],
    ] = {}

    for outer_result in selection_result[
        "outer_results"
    ]:
        outer_fold = int(
            outer_result["outer_fold"]
        )

        if outer_fold in selection_by_outer:
            raise ValueError(
                f"Duplicate selected outer fold: "
                f"{outer_fold}."
            )

        if outer_fold not in grouped_splits:
            raise ValueError(
                f"Unknown selected outer fold: "
                f"{outer_fold}."
            )

        manifest_splits = grouped_splits[
            outer_fold
        ]

        expected_test_subjects = (
            normalize_subject_list(
                manifest_splits[0][
                    "test_subjects"
                ],
                "test subjects",
            )
        )

        expected_development_subjects = (
            normalize_subject_list(
                manifest_splits[0][
                    "outer_development_subjects"
                ],
                "development subjects",
            )
        )

        selected_test_subjects = (
            normalize_subject_list(
                outer_result["test_subjects"],
                "selected test subjects",
            )
        )

        selected_development_subjects = (
            normalize_subject_list(
                outer_result[
                    "outer_development_subjects"
                ],
                "selected development subjects",
            )
        )

        if selected_test_subjects != (
            expected_test_subjects
        ):
            raise ValueError(
                f"Outer fold {outer_fold} test "
                "subjects do not match the manifest."
            )

        if selected_development_subjects != (
            expected_development_subjects
        ):
            raise ValueError(
                f"Outer fold {outer_fold} development "
                "subjects do not match the manifest."
            )

        if set(
            selected_test_subjects
        ) & set(
            selected_development_subjects
        ):
            raise ValueError(
                f"Outer fold {outer_fold} development "
                "and test subjects overlap."
            )

        selected_candidate = outer_result[
            "selected_candidate"
        ]

        model_name = str(
            selected_candidate["model_name"]
        )

        if model_name not in registry["models"]:
            raise ValueError(
                f"Selected model is not registered: "
                f"{model_name}."
            )

        model_spec = registry["models"][
            model_name
        ]

        if not bool(
            model_spec[
                "eligible_for_selection"
            ]
        ):
            raise ValueError(
                "A non-selectable model entered "
                "outer evaluation."
            )

        selected_candidate_id = str(
            selected_candidate["candidate_id"]
        )

        evaluated_matches = [
            candidate
            for candidate in outer_result[
                "all_candidate_results"
            ]
            if str(
                candidate["candidate_id"]
            ) == selected_candidate_id
        ]

        if len(evaluated_matches) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} selected "
                "candidate is not uniquely present "
                "in evaluated candidates."
            )

        evaluated_candidate = (
            evaluated_matches[0]
        )

        if str(
            evaluated_candidate["model_name"]
        ) != model_name:
            raise ValueError(
                "Selected candidate model name "
                "is inconsistent."
            )

        selected_parameters = dict(
            selected_candidate[
                "candidate_parameters"
            ]
        )

        evaluated_parameters = dict(
            evaluated_candidate[
                "candidate_parameters"
            ]
        )

        if selected_parameters != (
            evaluated_parameters
        ):
            raise ValueError(
                "Selected candidate parameters "
                "do not match the evaluated candidate."
            )

        pipeline = build_model_pipeline(
            model_name=model_name,
            registry=dict(registry),
        )

        try:
            pipeline.set_params(
                **selected_parameters
            )
        except ValueError as error:
            raise ValueError(
                "Selected candidate contains invalid "
                "pipeline parameters."
            ) from error

        selection_by_outer[
            outer_fold
        ] = dict(outer_result)

    expected_outer_folds = set(
        grouped_splits
    )

    if set(selection_by_outer) != (
        expected_outer_folds
    ):
        raise ValueError(
            "Selection artifact does not cover "
            "every outer fold."
        )

    return (
        grouped_splits,
        selection_by_outer,
    )


def selected_outer_folds(
    available_outer_folds: Sequence[int],
    requested_outer_folds: (
        Sequence[int] | None
    ),
) -> tuple[int, ...]:
    available = tuple(
        sorted(
            int(value)
            for value in available_outer_folds
        )
    )

    if requested_outer_folds is None:
        return available

    selected = tuple(
        sorted(
            {
                int(value)
                for value in requested_outer_folds
            }
        )
    )

    unknown = sorted(
        set(selected) - set(available)
    )

    if unknown:
        raise ValueError(
            f"Unknown outer folds: {unknown}."
        )

    if not selected:
        raise ValueError(
            "At least one outer fold is required."
        )

    return selected


def evaluate_selected_outer_fold(
    bundle: Phase3DatasetBundle,
    registry: Mapping[str, Any],
    outer_result: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    outer_fold = int(
        outer_result["outer_fold"]
    )

    test_subjects = normalize_subject_list(
        outer_result["test_subjects"],
        "test subjects",
    )

    development_subjects = (
        normalize_subject_list(
            outer_result[
                "outer_development_subjects"
            ],
            "development subjects",
        )
    )

    if set(test_subjects) & set(
        development_subjects
    ):
        raise ValueError(
            "Development and test subjects overlap."
        )

    development = select_subject_partition(
        bundle=bundle,
        subjects=development_subjects,
        name="outer_development",
        require_all_classes=True,
    )

    test = select_subject_partition(
        bundle=bundle,
        subjects=test_subjects,
        name="outer_test",
        require_all_classes=True,
    )

    if set(
        development.row_indices.tolist()
    ) & set(
        test.row_indices.tolist()
    ):
        raise ValueError(
            "Development and test rows overlap."
        )

    selected_candidate = outer_result[
        "selected_candidate"
    ]

    model_name = str(
        selected_candidate["model_name"]
    )

    candidate_id = str(
        selected_candidate["candidate_id"]
    )

    candidate_parameters = dict(
        selected_candidate[
            "candidate_parameters"
        ]
    )

    pipeline = build_model_pipeline(
        model_name=model_name,
        registry=dict(registry),
    )

    pipeline.set_params(
        **candidate_parameters
    )

    pipeline.fit(
        development.X,
        development.y,
    )

    predicted_labels = pipeline.predict(
        test.X
    )

    raw_probabilities = (
        pipeline.predict_proba(
            test.X
        )
    )

    estimator_classes = (
        pipeline.named_steps[
            "classifier"
        ].classes_
    )

    probabilities = align_probability_columns(
        probabilities=raw_probabilities,
        estimator_classes=estimator_classes,
        class_mapping=bundle.class_mapping,
    )

    metrics = calculate_classification_metrics(
        y_true=test.y,
        y_pred=predicted_labels,
        probabilities=probabilities,
        class_mapping=bundle.class_mapping,
    )

    identifiers = test.identifiers.copy()

    identifiers["source_row_index"] = (
        test.row_indices
    )

    predictions = build_prediction_frame(
        identifiers=identifiers,
        y_true=test.y,
        y_pred=predicted_labels,
        probabilities=probabilities,
        class_mapping=bundle.class_mapping,
        run_metadata={
            "candidate_id": candidate_id,
            "evaluation_scope": (
                "local_engineering_outer_test"
            ),
            "model_name": model_name,
            "outer_fold": outer_fold,
            "partition": "test",
        },
    )

    fold_result = {
        "outer_fold": outer_fold,
        "test_subjects": list(
            test_subjects
        ),
        "outer_development_subjects": list(
            development_subjects
        ),
        "model_name": model_name,
        "candidate_id": candidate_id,
        "candidate_parameters": (
            candidate_parameters
        ),
        "selection_validation_summary": dict(
            selected_candidate["aggregate"]
        ),
        "development_row_count": int(
            development.row_count
        ),
        "test_row_count": int(
            test.row_count
        ),
        "metrics": metrics,
    }

    return fold_result, predictions


def build_outer_summary_frame(
    outer_artifact: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []

    for fold in outer_artifact[
        "outer_results"
    ]:
        metrics = fold["metrics"]

        rows.append(
            {
                "outer_fold": int(
                    fold["outer_fold"]
                ),
                "test_subjects": ",".join(
                    str(value)
                    for value in fold[
                        "test_subjects"
                    ]
                ),
                "outer_development_subjects": (
                    ",".join(
                        str(value)
                        for value in fold[
                            "outer_development_subjects"
                        ]
                    )
                ),
                "model_name": str(
                    fold["model_name"]
                ),
                "candidate_id": str(
                    fold["candidate_id"]
                ),
                "candidate_parameters_json": (
                    canonical_json(
                        fold[
                            "candidate_parameters"
                        ]
                    )
                ),
                "development_row_count": int(
                    fold[
                        "development_row_count"
                    ]
                ),
                "test_row_count": int(
                    fold["test_row_count"]
                ),
                "accuracy": float(
                    metrics["accuracy"]
                ),
                "balanced_accuracy": float(
                    metrics[
                        "balanced_accuracy"
                    ]
                ),
                "macro_f1": float(
                    metrics["macro_f1"]
                ),
                "weighted_f1": float(
                    metrics["weighted_f1"]
                ),
                "cohen_kappa": float(
                    metrics["cohen_kappa"]
                ),
                "multiclass_log_loss": float(
                    metrics[
                        "multiclass_log_loss"
                    ]
                ),
                (
                    "predict_probability_"
                    "argmax_agreement"
                ): float(
                    metrics[
                        (
                            "predict_probability_"
                            "argmax_agreement"
                        )
                    ]
                ),
                (
                    "predict_probability_"
                    "argmax_mismatch_count"
                ): int(
                    metrics[
                        (
                            "predict_probability_"
                            "argmax_mismatch_count"
                        )
                    ]
                ),
                "mean_prediction_confidence": float(
                    metrics[
                        "mean_prediction_confidence"
                    ]
                ),
                "mean_prediction_margin": float(
                    metrics[
                        "mean_prediction_margin"
                    ]
                ),
                "mean_normalized_entropy": float(
                    metrics[
                        "mean_normalized_entropy"
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
            "Outer summary frame is empty."
        )

    return frame.sort_values(
        "outer_fold",
        kind="mergesort",
    ).reset_index(drop=True)


def calculate_pooled_metrics(
    predictions: pd.DataFrame,
    class_mapping: Mapping[str, int],
) -> dict[str, Any]:
    ordered_classes = sorted(
        class_mapping.items(),
        key=lambda item: item[1],
    )

    probability_columns = [
        probability_column_name(name)
        for name, _ in ordered_classes
    ]

    return calculate_classification_metrics(
        y_true=predictions[
            "true_label_encoded"
        ].to_numpy(dtype=int),
        y_pred=predictions[
            "predicted_label_encoded"
        ].to_numpy(dtype=int),
        probabilities=predictions[
            probability_columns
        ].to_numpy(dtype=float),
        class_mapping=class_mapping,
    )


def run_outer_evaluation(
    bundle: Phase3DatasetBundle,
    split_manifest: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    split_manifest_path: Path,
    selection_artifact_path: Path,
    registry_path: Path,
    outer_folds: Sequence[int] | None = None,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    (
        grouped_splits,
        selection_by_outer,
    ) = validate_outer_evaluation_inputs(
        bundle=bundle,
        split_manifest=split_manifest,
        selection_artifact=selection_artifact,
        registry=registry,
        split_manifest_path=(
            split_manifest_path
        ),
        selection_artifact_path=(
            selection_artifact_path
        ),
        registry_path=registry_path,
    )

    selected_folds = selected_outer_folds(
        available_outer_folds=tuple(
            grouped_splits
        ),
        requested_outer_folds=outer_folds,
    )

    outer_results = []
    prediction_frames = []

    for outer_fold in selected_folds:
        fold_result, predictions = (
            evaluate_selected_outer_fold(
                bundle=bundle,
                registry=registry,
                outer_result=(
                    selection_by_outer[
                        outer_fold
                    ]
                ),
            )
        )

        outer_results.append(
            fold_result
        )

        prediction_frames.append(
            predictions
        )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions = (
        all_predictions.sort_values(
            "source_row_index",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if all_predictions[
        "source_row_index"
    ].duplicated().any():
        raise ValueError(
            "An epoch received more than one "
            "outer-test prediction."
        )

    complete_outer_evaluation = (
        set(selected_folds)
        == set(grouped_splits)
    )

    if complete_outer_evaluation:
        observed_indices = (
            all_predictions[
                "source_row_index"
            ].to_numpy(dtype=int)
        )

        expected_indices = np.arange(
            bundle.row_count,
            dtype=int,
        )

        if not np.array_equal(
            observed_indices,
            expected_indices,
        ):
            raise ValueError(
                "Complete outer evaluation did not "
                "predict every dataset row exactly once."
            )

    pooled_metrics = calculate_pooled_metrics(
        predictions=all_predictions,
        class_mapping=bundle.class_mapping,
    )

    macro_f1_values = np.asarray(
        [
            fold["metrics"]["macro_f1"]
            for fold in outer_results
        ],
        dtype=float,
    )

    balanced_accuracy_values = np.asarray(
        [
            fold["metrics"][
                "balanced_accuracy"
            ]
            for fold in outer_results
        ],
        dtype=float,
    )

    outer_artifact = {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_local_outer_evaluation"
        ),
        "intended_use": (
            "local_engineering_evaluation"
        ),
        "scientific_reporting": {
            "allowed": False,
            "reason": (
                "The local dataset contains only "
                "four subjects and is reserved for "
                "engineering validation."
            ),
        },
        "evaluation_contract": {
            "selection_artifact_frozen": True,
            "hyperparameter_search_performed": False,
            "test_access_stage": (
                "after_inner_model_selection"
            ),
            "each_epoch_tested_once_when_complete": (
                True
            ),
        },
        "source": {
            "model_input_path": (
                relative_display_path(
                    bundle.data_path
                )
            ),
            "model_input_sha256": (
                bundle.data_sha256
            ),
            "model_schema_path": (
                relative_display_path(
                    bundle.schema_path
                )
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
                sha256_file(registry_path)
            ),
            "selection_artifact_path": (
                relative_display_path(
                    selection_artifact_path
                )
            ),
            "selection_artifact_sha256": (
                sha256_file(
                    selection_artifact_path
                )
            ),
        },
        "evaluated_outer_folds": list(
            selected_folds
        ),
        "complete_outer_evaluation": (
            complete_outer_evaluation
        ),
        "outer_results": outer_results,
        "aggregate": {
            "outer_fold_count": len(
                outer_results
            ),
            "prediction_row_count": int(
                len(all_predictions)
            ),
            "mean_outer_macro_f1": float(
                macro_f1_values.mean()
            ),
            "std_outer_macro_f1": float(
                macro_f1_values.std(ddof=0)
            ),
            (
                "mean_outer_balanced_accuracy"
            ): float(
                balanced_accuracy_values.mean()
            ),
            (
                "std_outer_balanced_accuracy"
            ): float(
                balanced_accuracy_values.std(
                    ddof=0
                )
            ),
            "pooled_test_metrics": (
                pooled_metrics
            ),
        },
    }

    return outer_artifact, all_predictions


def write_outer_evaluation_artifacts(
    outer_artifact: Mapping[str, Any],
    predictions: pd.DataFrame,
    json_output_path: Path,
    summary_csv_output_path: Path,
    predictions_csv_output_path: Path,
) -> None:
    write_metrics_json(
        metrics=outer_artifact,
        output_path=json_output_path,
    )

    summary = build_outer_summary_frame(
        outer_artifact
    )

    summary_csv_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_csv_output_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )

    write_prediction_csv(
        predictions=predictions,
        output_path=predictions_csv_output_path,
    )


def load_default_inputs() -> tuple[
    Phase3DatasetBundle,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    bundle = load_phase3_dataset(
        data_path=DEFAULT_DATA_PATH,
        schema_path=DEFAULT_SCHEMA_PATH,
        protocol_path=DEFAULT_PROTOCOL_PATH,
    )

    split_manifest = load_json_object(
        DEFAULT_SPLIT_MANIFEST_PATH
    )

    selection_artifact = (
        load_selection_artifact(
            DEFAULT_SELECTION_PATH
        )
    )

    registry = load_registry_config(
        DEFAULT_REGISTRY_PATH
    )

    return (
        bundle,
        split_manifest,
        selection_artifact,
        registry,
    )


def smoke_test() -> None:
    (
        bundle,
        split_manifest,
        selection_artifact,
        registry,
    ) = load_default_inputs()

    artifact, predictions = (
        run_outer_evaluation(
            bundle=bundle,
            split_manifest=split_manifest,
            selection_artifact=(
                selection_artifact
            ),
            registry=registry,
            split_manifest_path=(
                DEFAULT_SPLIT_MANIFEST_PATH
            ),
            selection_artifact_path=(
                DEFAULT_SELECTION_PATH
            ),
            registry_path=(
                DEFAULT_REGISTRY_PATH
            ),
            outer_folds=[1],
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        json_path = root / "outer.json"
        summary_path = root / "summary.csv"
        predictions_path = (
            root / "predictions.csv"
        )

        write_outer_evaluation_artifacts(
            outer_artifact=artifact,
            predictions=predictions,
            json_output_path=json_path,
            summary_csv_output_path=(
                summary_path
            ),
            predictions_csv_output_path=(
                predictions_path
            ),
        )

        first_json = json_path.read_bytes()
        first_summary = (
            summary_path.read_bytes()
        )
        first_predictions = (
            predictions_path.read_bytes()
        )

        write_outer_evaluation_artifacts(
            outer_artifact=artifact,
            predictions=predictions,
            json_output_path=json_path,
            summary_csv_output_path=(
                summary_path
            ),
            predictions_csv_output_path=(
                predictions_path
            ),
        )

        if first_json != json_path.read_bytes():
            raise ValueError(
                "Outer JSON is not deterministic."
            )

        if first_summary != (
            summary_path.read_bytes()
        ):
            raise ValueError(
                "Outer summary CSV is not deterministic."
            )

        if first_predictions != (
            predictions_path.read_bytes()
        ):
            raise ValueError(
                "Outer predictions CSV is not "
                "deterministic."
            )

    fold = artifact["outer_results"][0]

    print(
        "=== PHASE 3 OUTER EVALUATION "
        "SMOKE TEST ==="
    )
    print(
        "Outer fold:",
        fold["outer_fold"],
    )
    print(
        "Test subjects:",
        fold["test_subjects"],
    )
    print(
        "Selected model:",
        fold["model_name"],
    )
    print(
        "Selected candidate:",
        fold["candidate_id"],
    )
    print(
        "Development rows:",
        fold["development_row_count"],
    )
    print(
        "Test rows:",
        fold["test_row_count"],
    )
    print(
        "Test macro-F1:",
        round(
            fold["metrics"]["macro_f1"],
            6,
        ),
    )
    print(
        "Hyperparameter search performed:",
        artifact[
            "evaluation_contract"
        ][
            "hyperparameter_search_performed"
        ],
    )
    print(
        "Byte-deterministic artifacts: PASS"
    )
    print(
        "Outer evaluation smoke test: PASS"
    )


def run_full_local() -> None:
    (
        bundle,
        split_manifest,
        selection_artifact,
        registry,
    ) = load_default_inputs()

    artifact, predictions = (
        run_outer_evaluation(
            bundle=bundle,
            split_manifest=split_manifest,
            selection_artifact=(
                selection_artifact
            ),
            registry=registry,
            split_manifest_path=(
                DEFAULT_SPLIT_MANIFEST_PATH
            ),
            selection_artifact_path=(
                DEFAULT_SELECTION_PATH
            ),
            registry_path=(
                DEFAULT_REGISTRY_PATH
            ),
        )
    )

    write_outer_evaluation_artifacts(
        outer_artifact=artifact,
        predictions=predictions,
        json_output_path=(
            DEFAULT_OUTPUT_JSON_PATH
        ),
        summary_csv_output_path=(
            DEFAULT_OUTPUT_SUMMARY_CSV_PATH
        ),
        predictions_csv_output_path=(
            DEFAULT_OUTPUT_PREDICTIONS_CSV_PATH
        ),
    )

    summary = build_outer_summary_frame(
        artifact
    )

    print(
        "=== PHASE 3 FULL LOCAL "
        "OUTER EVALUATION ==="
    )

    for _, row in summary.iterrows():
        print(
            "Outer fold",
            int(row["outer_fold"]),
            "| test_subject=",
            row["test_subjects"],
            "| model=",
            row["model_name"],
            "| macro_f1=",
            round(
                float(row["macro_f1"]),
                6,
            ),
        )

    pooled = artifact[
        "aggregate"
    ]["pooled_test_metrics"]

    print(
        "Prediction rows:",
        len(predictions),
    )
    print(
        "Pooled macro-F1:",
        round(
            pooled["macro_f1"],
            6,
        ),
    )
    print(
        "Pooled balanced accuracy:",
        round(
            pooled[
                "balanced_accuracy"
            ],
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
        "Summary CSV:",
        relative_display_path(
            DEFAULT_OUTPUT_SUMMARY_CSV_PATH
        ),
    )
    print(
        "Predictions CSV:",
        relative_display_path(
            DEFAULT_OUTPUT_PREDICTIONS_CSV_PATH
        ),
    )
    print(
        "Full local outer evaluation: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Phase 3 model "
            "selections on outer test subjects."
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
