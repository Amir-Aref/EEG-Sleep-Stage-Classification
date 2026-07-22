from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import numpy as np

from scripts.phase3_dataset import (
    DEFAULT_DATA_PATH,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_SCHEMA_PATH,
    Phase3DatasetBundle,
    load_phase3_dataset,
    select_subject_partition,
)
from scripts.phase3_metrics import (
    align_probability_columns,
    calculate_classification_metrics,
)
from scripts.phase3_model_registry import (
    DEFAULT_CONFIG_PATH as DEFAULT_REGISTRY_PATH,
    build_model_pipeline,
    enumerate_candidate_parameters,
    load_registry_config,
)


DEFAULT_SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_split_manifest.json"
)

SELECTION_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "weighted_f1",
    "accuracy",
    "cohen_kappa",
    "multiclass_log_loss",
)


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return value


def normalize_subject_list(
    values: Sequence[Any],
    name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence."
        )

    normalized = []

    for value in values:
        item_method = getattr(
            value,
            "item",
            None,
        )

        if callable(item_method):
            value = item_method()

        normalized.append(value)

    if not normalized:
        raise ValueError(
            f"{name} must not be empty."
        )

    if len(normalized) != len(
        set(normalized)
    ):
        raise ValueError(
            f"{name} contains duplicates."
        )

    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                type(item).__name__,
                str(item),
            ),
        )
    )


def normalize_class_mapping(
    mapping: Mapping[str, Any],
) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise ValueError(
            "class_mapping must be an object."
        )

    return {
        str(name): int(encoded)
        for name, encoded in mapping.items()
    }


def validate_local_split_manifest(
    manifest: Mapping[str, Any],
    bundle: Phase3DatasetBundle,
) -> dict[int, list[dict[str, Any]]]:
    required_keys = {
        "class_mapping",
        "group_column",
        "inner_fold_count_per_outer",
        "outer_fold_count",
        "row_count",
        "source",
        "splits",
        "subject_count",
        "subjects",
        "target_column",
        "target_name_column",
        "total_split_count",
    }

    missing_keys = sorted(
        required_keys - set(manifest)
    )

    if missing_keys:
        raise ValueError(
            "Split manifest is missing keys: "
            + ", ".join(missing_keys)
        )

    if int(manifest["row_count"]) != (
        bundle.row_count
    ):
        raise ValueError(
            "Manifest row count does not match "
            "the dataset."
        )

    if str(manifest["group_column"]) != (
        bundle.group_column
    ):
        raise ValueError(
            "Manifest group column does not match "
            "the dataset contract."
        )

    if str(manifest["target_column"]) != (
        bundle.target_column
    ):
        raise ValueError(
            "Manifest target column does not match "
            "the dataset contract."
        )

    if str(
        manifest["target_name_column"]
    ) != bundle.target_name_column:
        raise ValueError(
            "Manifest target-name column does not "
            "match the dataset contract."
        )

    if normalize_class_mapping(
        manifest["class_mapping"]
    ) != bundle.class_mapping:
        raise ValueError(
            "Manifest class mapping does not match "
            "the dataset contract."
        )

    source = manifest["source"]

    if not isinstance(source, dict):
        raise ValueError(
            "Manifest source must be an object."
        )

    if source.get(
        "model_input_sha256"
    ) != bundle.data_sha256:
        raise ValueError(
            "Manifest model-input hash does not "
            "match the loaded dataset."
        )

    if source.get(
        "protocol_sha256"
    ) != bundle.protocol_sha256:
        raise ValueError(
            "Manifest protocol hash does not match "
            "the loaded protocol."
        )

    manifest_subjects = set(
        normalize_subject_list(
            manifest["subjects"],
            "manifest subjects",
        )
    )

    dataset_subjects = set(
        bundle.groups.tolist()
    )

    if manifest_subjects != dataset_subjects:
        raise ValueError(
            "Manifest subjects do not match "
            "dataset subjects."
        )

    if int(manifest["subject_count"]) != len(
        dataset_subjects
    ):
        raise ValueError(
            "Manifest subject count is invalid."
        )

    splits = manifest["splits"]

    if not isinstance(splits, list):
        raise ValueError(
            "Manifest splits must be a list."
        )

    if len(splits) != int(
        manifest["total_split_count"]
    ):
        raise ValueError(
            "Manifest split count is inconsistent."
        )

    split_ids: set[str] = set()
    grouped: dict[int, list[dict[str, Any]]] = {}

    for split in splits:
        if not isinstance(split, dict):
            raise ValueError(
                "Every split must be an object."
            )

        required_split_keys = {
            "inner_fold",
            "outer_development_subjects",
            "outer_fold",
            "split_id",
            "test_subjects",
            "train_subjects",
            "validation_subjects",
        }

        missing_split_keys = sorted(
            required_split_keys - set(split)
        )

        if missing_split_keys:
            raise ValueError(
                "Split is missing keys: "
                + ", ".join(missing_split_keys)
            )

        split_id = str(split["split_id"])

        if split_id in split_ids:
            raise ValueError(
                f"Duplicate split id: {split_id}"
            )

        split_ids.add(split_id)

        outer_fold = int(
            split["outer_fold"]
        )

        inner_fold = int(
            split["inner_fold"]
        )

        development_subjects = set(
            normalize_subject_list(
                split[
                    "outer_development_subjects"
                ],
                "outer development subjects",
            )
        )

        train_subjects = set(
            normalize_subject_list(
                split["train_subjects"],
                "train subjects",
            )
        )

        validation_subjects = set(
            normalize_subject_list(
                split["validation_subjects"],
                "validation subjects",
            )
        )

        test_subjects = set(
            normalize_subject_list(
                split["test_subjects"],
                "test subjects",
            )
        )

        if len(validation_subjects) != 1:
            raise ValueError(
                f"{split_id} must contain exactly "
                "one validation subject."
            )

        if len(test_subjects) != 1:
            raise ValueError(
                f"{split_id} must contain exactly "
                "one test subject."
            )

        if (
            train_subjects
            & validation_subjects
        ):
            raise ValueError(
                f"{split_id} train and validation "
                "subjects overlap."
            )

        if (
            train_subjects
            & test_subjects
            or validation_subjects
            & test_subjects
        ):
            raise ValueError(
                f"{split_id} exposes the test subject "
                "to inner model selection."
            )

        if (
            train_subjects
            | validation_subjects
        ) != development_subjects:
            raise ValueError(
                f"{split_id} development subjects "
                "are inconsistent."
            )

        if (
            development_subjects
            | test_subjects
        ) != dataset_subjects:
            raise ValueError(
                f"{split_id} does not cover all "
                "dataset subjects."
            )

        normalized_split = dict(split)
        normalized_split["outer_fold"] = outer_fold
        normalized_split["inner_fold"] = inner_fold

        grouped.setdefault(
            outer_fold,
            []
        ).append(normalized_split)

    expected_outer_count = int(
        manifest["outer_fold_count"]
    )

    expected_inner_count = int(
        manifest[
            "inner_fold_count_per_outer"
        ]
    )

    if len(grouped) != expected_outer_count:
        raise ValueError(
            "Unexpected number of outer folds."
        )

    for outer_fold, outer_splits in grouped.items():
        outer_splits.sort(
            key=lambda split: split[
                "inner_fold"
            ]
        )

        if len(outer_splits) != (
            expected_inner_count
        ):
            raise ValueError(
                f"Outer fold {outer_fold} has an "
                "unexpected inner-fold count."
            )

        inner_fold_ids = [
            split["inner_fold"]
            for split in outer_splits
        ]

        if inner_fold_ids != list(
            range(
                1,
                expected_inner_count + 1,
            )
        ):
            raise ValueError(
                f"Outer fold {outer_fold} has invalid "
                "inner-fold identifiers."
            )

        test_subject_sets = {
            tuple(
                normalize_subject_list(
                    split["test_subjects"],
                    "test subjects",
                )
            )
            for split in outer_splits
        }

        if len(test_subject_sets) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} changes "
                "its test subject."
            )

        development_sets = {
            tuple(
                normalize_subject_list(
                    split[
                        "outer_development_subjects"
                    ],
                    "development subjects",
                )
            )
            for split in outer_splits
        }

        if len(development_sets) != 1:
            raise ValueError(
                f"Outer fold {outer_fold} changes "
                "its development subjects."
            )

        development_subjects = set(
            next(iter(development_sets))
        )

        validation_subjects = [
            normalize_subject_list(
                split["validation_subjects"],
                "validation subjects",
            )[0]
            for split in outer_splits
        ]

        if set(validation_subjects) != (
            development_subjects
        ):
            raise ValueError(
                f"Outer fold {outer_fold} does not "
                "rotate every development subject "
                "through validation."
            )

    return dict(
        sorted(grouped.items())
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def selected_metric_snapshot(
    metrics: Mapping[str, Any],
) -> dict[str, float]:
    return {
        metric_name: float(
            metrics[metric_name]
        )
        for metric_name in SELECTION_METRICS
    }


def evaluate_candidate_on_inner_folds(
    bundle: Phase3DatasetBundle,
    outer_splits: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    model_name: str,
    candidate_index: int,
    candidate_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    model_spec = registry["models"][
        model_name
    ]

    fold_metrics = []

    for split in outer_splits:
        train_subjects = (
            normalize_subject_list(
                split["train_subjects"],
                "train subjects",
            )
        )

        validation_subjects = (
            normalize_subject_list(
                split["validation_subjects"],
                "validation subjects",
            )
        )

        test_subjects = set(
            normalize_subject_list(
                split["test_subjects"],
                "test subjects",
            )
        )

        if (
            set(train_subjects)
            & test_subjects
            or set(validation_subjects)
            & test_subjects
        ):
            raise ValueError(
                "Test subjects entered an inner "
                "training partition."
            )

        train_partition = (
            select_subject_partition(
                bundle=bundle,
                subjects=train_subjects,
                name="train",
                require_all_classes=True,
            )
        )

        validation_partition = (
            select_subject_partition(
                bundle=bundle,
                subjects=validation_subjects,
                name="validation",
                require_all_classes=True,
            )
        )

        if set(
            train_partition.row_indices.tolist()
        ) & set(
            validation_partition.row_indices.tolist()
        ):
            raise ValueError(
                "Train and validation rows overlap."
            )

        pipeline = build_model_pipeline(
            model_name=model_name,
            registry=dict(registry),
        )

        pipeline.set_params(
            **dict(candidate_parameters)
        )

        pipeline.fit(
            train_partition.X,
            train_partition.y,
        )

        predicted_labels = pipeline.predict(
            validation_partition.X
        )

        raw_probabilities = (
            pipeline.predict_proba(
                validation_partition.X
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
            y_true=validation_partition.y,
            y_pred=predicted_labels,
            probabilities=probabilities,
            class_mapping=bundle.class_mapping,
        )

        fold_metrics.append(
            {
                "split_id": str(
                    split["split_id"]
                ),
                "outer_fold": int(
                    split["outer_fold"]
                ),
                "inner_fold": int(
                    split["inner_fold"]
                ),
                "train_subjects": list(
                    train_subjects
                ),
                "validation_subjects": list(
                    validation_subjects
                ),
                "train_row_count": int(
                    train_partition.row_count
                ),
                "validation_row_count": int(
                    validation_partition.row_count
                ),
                "metrics": (
                    selected_metric_snapshot(
                        metrics
                    )
                ),
            }
        )

    aggregate: dict[str, float] = {}

    for metric_name in SELECTION_METRICS:
        values = np.asarray(
            [
                fold["metrics"][metric_name]
                for fold in fold_metrics
            ],
            dtype=float,
        )

        aggregate[
            f"mean_{metric_name}"
        ] = float(values.mean())

        aggregate[
            f"std_{metric_name}"
        ] = float(
            values.std(ddof=0)
        )

    return {
        "model_name": model_name,
        "candidate_index": int(
            candidate_index
        ),
        "candidate_id": (
            f"{model_name}__"
            f"candidate_{candidate_index:03d}"
        ),
        "candidate_parameters": json_safe(
            dict(candidate_parameters)
        ),
        "eligible_for_selection": bool(
            model_spec[
                "eligible_for_selection"
            ]
        ),
        "complexity_rank": int(
            model_spec["complexity_rank"]
        ),
        "fold_count": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
    }


def candidate_ranking_key(
    result: Mapping[str, Any],
) -> tuple[Any, ...]:
    aggregate = result["aggregate"]

    return (
        -float(
            aggregate["mean_macro_f1"]
        ),
        float(
            aggregate["std_macro_f1"]
        ),
        -float(
            aggregate[
                "mean_balanced_accuracy"
            ]
        ),
        float(
            aggregate[
                "mean_multiclass_log_loss"
            ]
        ),
        int(result["complexity_rank"]),
        str(result["model_name"]),
        int(result["candidate_index"]),
    )


def rank_selectable_candidates(
    candidate_results: Sequence[
        Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    selectable = [
        dict(result)
        for result in candidate_results
        if bool(
            result["eligible_for_selection"]
        )
    ]

    if not selectable:
        raise ValueError(
            "No selectable candidate was evaluated."
        )

    selectable.sort(
        key=candidate_ranking_key
    )

    for rank, result in enumerate(
        selectable,
        start=1,
    ):
        result["selection_rank"] = rank

    return selectable


def run_inner_search(
    bundle: Phase3DatasetBundle,
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    outer_folds: Sequence[int] | None = None,
    model_names: Sequence[str] | None = None,
    max_candidates_per_model: int | None = None,
) -> dict[str, Any]:
    if registry.get(
        "primary_metric"
    ) != "macro_f1":
        raise ValueError(
            "Inner search requires macro_f1 as "
            "its primary metric."
        )

    grouped_splits = (
        validate_local_split_manifest(
            manifest=manifest,
            bundle=bundle,
        )
    )

    available_outer_folds = tuple(
        grouped_splits
    )

    if outer_folds is None:
        selected_outer_folds = (
            available_outer_folds
        )
    else:
        selected_outer_folds = tuple(
            sorted(
                {
                    int(value)
                    for value in outer_folds
                }
            )
        )

        unknown_outer_folds = sorted(
            set(selected_outer_folds)
            - set(available_outer_folds)
        )

        if unknown_outer_folds:
            raise ValueError(
                "Unknown outer folds: "
                f"{unknown_outer_folds}."
            )

    available_models = registry[
        "models"
    ]

    if model_names is None:
        selected_models = list(
            available_models
        )
    else:
        selected_models = [
            str(value)
            for value in model_names
        ]

        unknown_models = sorted(
            set(selected_models)
            - set(available_models)
        )

        if unknown_models:
            raise ValueError(
                "Unknown models: "
                f"{unknown_models}."
            )

    selected_models.sort(
        key=lambda model_name: (
            int(
                available_models[
                    model_name
                ]["complexity_rank"]
            ),
            model_name,
        )
    )

    if (
        max_candidates_per_model is not None
        and max_candidates_per_model < 1
    ):
        raise ValueError(
            "max_candidates_per_model must be "
            "positive."
        )

    outer_results = []

    for outer_fold in selected_outer_folds:
        outer_splits = grouped_splits[
            outer_fold
        ]

        candidate_results = []

        for model_name in selected_models:
            candidates = (
                enumerate_candidate_parameters(
                    model_name=model_name,
                    registry=dict(registry),
                )
            )

            if (
                max_candidates_per_model
                is not None
            ):
                candidates = candidates[
                    :max_candidates_per_model
                ]

            for candidate_index, parameters in enumerate(
                candidates,
                start=1,
            ):
                candidate_results.append(
                    evaluate_candidate_on_inner_folds(
                        bundle=bundle,
                        outer_splits=outer_splits,
                        registry=registry,
                        model_name=model_name,
                        candidate_index=(
                            candidate_index
                        ),
                        candidate_parameters=(
                            parameters
                        ),
                    )
                )

        ranked_candidates = (
            rank_selectable_candidates(
                candidate_results
            )
        )

        best_candidate = (
            ranked_candidates[0]
        )

        test_subjects = (
            normalize_subject_list(
                outer_splits[0][
                    "test_subjects"
                ],
                "test subjects",
            )
        )

        development_subjects = (
            normalize_subject_list(
                outer_splits[0][
                    "outer_development_subjects"
                ],
                "development subjects",
            )
        )

        outer_results.append(
            {
                "outer_fold": int(
                    outer_fold
                ),
                "test_subjects": list(
                    test_subjects
                ),
                "outer_development_subjects": (
                    list(
                        development_subjects
                    )
                ),
                "evaluated_candidate_count": (
                    len(candidate_results)
                ),
                "selected_candidate": {
                    "selection_rank": 1,
                    "model_name": (
                        best_candidate[
                            "model_name"
                        ]
                    ),
                    "candidate_index": (
                        best_candidate[
                            "candidate_index"
                        ]
                    ),
                    "candidate_id": (
                        best_candidate[
                            "candidate_id"
                        ]
                    ),
                    "candidate_parameters": (
                        best_candidate[
                            "candidate_parameters"
                        ]
                    ),
                    "complexity_rank": (
                        best_candidate[
                            "complexity_rank"
                        ]
                    ),
                    "aggregate": (
                        best_candidate[
                            "aggregate"
                        ]
                    ),
                },
                "ranked_selectable_candidates": (
                    ranked_candidates
                ),
                "all_candidate_results": (
                    candidate_results
                ),
            }
        )

    complete_candidate_space = (
        outer_folds is None
        and model_names is None
        and max_candidates_per_model is None
    )

    return {
        "schema_version": "1.0.0",
        "primary_metric": "macro_f1",
        "selection_partition": "validation",
        "test_metrics_included": False,
        "test_predictions_included": False,
        "test_feature_matrix_loaded": False,
        "candidate_space_complete": (
            complete_candidate_space
        ),
        "ranking_contract": [
            "mean_macro_f1_desc",
            "std_macro_f1_asc",
            "mean_balanced_accuracy_desc",
            "mean_multiclass_log_loss_asc",
            "complexity_rank_asc",
            "model_name_asc",
            "candidate_index_asc",
        ],
        "evaluated_outer_folds": list(
            selected_outer_folds
        ),
        "evaluated_models": (
            selected_models
        ),
        "outer_results": outer_results,
    }


def smoke_test() -> None:
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
        outer_folds=[1],
        model_names=[
            "dummy_prior",
            "logistic_regression",
        ],
        max_candidates_per_model=1,
    )

    outer_result = result[
        "outer_results"
    ][0]

    if result["test_metrics_included"]:
        raise ValueError(
            "Smoke test exposed test metrics."
        )

    if result[
        "test_feature_matrix_loaded"
    ]:
        raise ValueError(
            "Smoke test loaded test features."
        )

    if outer_result[
        "selected_candidate"
    ]["model_name"] != (
        "logistic_regression"
    ):
        raise ValueError(
            "Non-selectable baseline was selected."
        )

    print(
        "=== PHASE 3 INNER SEARCH SMOKE TEST ==="
    )
    print(
        "Outer fold:",
        outer_result["outer_fold"],
    )
    print(
        "Test subjects recorded only:",
        outer_result["test_subjects"],
    )
    print(
        "Evaluated candidates:",
        outer_result[
            "evaluated_candidate_count"
        ],
    )
    print(
        "Inner folds per candidate:",
        outer_result[
            "all_candidate_results"
        ][0]["fold_count"],
    )
    print(
        "Selected model:",
        outer_result[
            "selected_candidate"
        ]["model_name"],
    )
    print(
        "Selected candidate:",
        outer_result[
            "selected_candidate"
        ]["candidate_id"],
    )
    print(
        "Mean validation macro-F1:",
        round(
            outer_result[
                "selected_candidate"
            ]["aggregate"][
                "mean_macro_f1"
            ],
            6,
        ),
    )
    print(
        "Test feature access: BLOCKED"
    )
    print("Inner search smoke test: PASS")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Phase 3 inner-fold "
            "hyperparameter search."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.smoke_test:
        smoke_test()
        return

    print(
        "Use --smoke-test to run the guarded "
        "inner-search engineering check."
    )


if __name__ == "__main__":
    main()
