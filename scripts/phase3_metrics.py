from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ordered_class_items(
    class_mapping: Mapping[str, int],
) -> list[tuple[str, int]]:
    if not class_mapping:
        raise ValueError(
            "Class mapping must not be empty."
        )

    normalized = {
        str(name): int(encoded)
        for name, encoded in class_mapping.items()
    }

    if len(normalized.values()) != len(
        set(normalized.values())
    ):
        raise ValueError(
            "Class encodings must be unique."
        )

    items = sorted(
        normalized.items(),
        key=lambda item: item[1],
    )

    encoded_values = [
        encoded
        for _, encoded in items
    ]

    expected_values = list(
        range(len(items))
    )

    if encoded_values != expected_values:
        raise ValueError(
            "Class encodings must be contiguous and "
            f"start at zero; found {encoded_values}."
        )

    return items


def ordered_class_labels(
    class_mapping: Mapping[str, int],
) -> list[int]:
    return [
        encoded
        for _, encoded in ordered_class_items(
            class_mapping
        )
    ]


def ordered_class_names(
    class_mapping: Mapping[str, int],
) -> list[str]:
    return [
        name
        for name, _ in ordered_class_items(
            class_mapping
        )
    ]


def normalize_label_array(
    values: Sequence[Any] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"found shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        try:
            array = array.astype(int)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must contain integer labels."
            ) from error

    numeric = array.astype(float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    integer = numeric.astype(int)

    if not np.array_equal(
        numeric,
        integer.astype(float),
    ):
        raise ValueError(
            f"{name} contains non-integer labels."
        )

    return integer


def validate_and_normalize_probabilities(
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    expected_row_count: int,
    expected_class_count: int,
    tolerance: float = 1e-8,
) -> np.ndarray:
    array = np.asarray(
        probabilities,
        dtype=float,
    )

    expected_shape = (
        expected_row_count,
        expected_class_count,
    )

    if array.shape != expected_shape:
        raise ValueError(
            "Probability matrix has an invalid shape: "
            f"expected {expected_shape}, found "
            f"{array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Probability matrix contains non-finite "
            "values."
        )

    if np.any(array < -tolerance):
        raise ValueError(
            "Probability matrix contains negative "
            "values."
        )

    if np.any(array > 1.0 + tolerance):
        raise ValueError(
            "Probability matrix contains values "
            "greater than one."
        )

    row_sums = array.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=tolerance,
        rtol=0.0,
    ):
        maximum_error = float(
            np.max(
                np.abs(row_sums - 1.0)
            )
        )

        raise ValueError(
            "Probability rows must sum to one; "
            f"maximum error is {maximum_error}."
        )

    normalized = np.clip(
        array,
        0.0,
        1.0,
    )

    normalized /= normalized.sum(
        axis=1,
        keepdims=True,
    )

    return normalized


def align_probability_columns(
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    estimator_classes: Sequence[Any]
    | np.ndarray,
    class_mapping: Mapping[str, int],
) -> np.ndarray:
    source_classes = normalize_label_array(
        estimator_classes,
        "estimator_classes",
    )

    expected_labels = ordered_class_labels(
        class_mapping
    )

    if len(source_classes) != len(
        set(source_classes.tolist())
    ):
        raise ValueError(
            "Estimator classes contain duplicates."
        )

    if set(source_classes.tolist()) != set(
        expected_labels
    ):
        raise ValueError(
            "Estimator classes do not match the "
            "evaluation class mapping."
        )

    source_probabilities = (
        validate_and_normalize_probabilities(
            probabilities=probabilities,
            expected_row_count=len(probabilities),
            expected_class_count=len(
                source_classes
            ),
        )
    )

    source_index = {
        int(label): index
        for index, label in enumerate(
            source_classes
        )
    }

    aligned = np.column_stack(
        [
            source_probabilities[
                :,
                source_index[label],
            ]
            for label in expected_labels
        ]
    )

    return validate_and_normalize_probabilities(
        probabilities=aligned,
        expected_row_count=len(aligned),
        expected_class_count=len(
            expected_labels
        ),
    )


def validate_prediction_contract(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    class_mapping: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_array = normalize_label_array(
        y_true,
        "y_true",
    )

    predicted_array = normalize_label_array(
        y_pred,
        "y_pred",
    )

    if len(true_array) != len(
        predicted_array
    ):
        raise ValueError(
            "y_true and y_pred have different "
            "lengths."
        )

    class_labels = ordered_class_labels(
        class_mapping
    )

    allowed_labels = set(class_labels)

    unknown_true = sorted(
        set(true_array.tolist())
        - allowed_labels
    )

    unknown_predicted = sorted(
        set(predicted_array.tolist())
        - allowed_labels
    )

    if unknown_true:
        raise ValueError(
            "y_true contains unknown classes: "
            f"{unknown_true}."
        )

    if unknown_predicted:
        raise ValueError(
            "y_pred contains unknown classes: "
            f"{unknown_predicted}."
        )

    probability_array = (
        validate_and_normalize_probabilities(
            probabilities=probabilities,
            expected_row_count=len(
                true_array
            ),
            expected_class_count=len(
                class_labels
            ),
        )
    )

    probability_predictions = np.asarray(
        class_labels,
        dtype=int,
    )[
        np.argmax(
            probability_array,
            axis=1,
        )
    ]

    if not np.array_equal(
        predicted_array,
        probability_predictions,
    ):
        mismatch_count = int(
            np.sum(
                predicted_array
                != probability_predictions
            )
        )

        raise ValueError(
            "Predicted labels do not match the "
            "maximum-probability classes; "
            f"{mismatch_count} mismatches found."
        )

    return (
        true_array,
        predicted_array,
        probability_array,
    )


def ensure_finite_metric(
    name: str,
    value: float,
) -> float:
    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"Metric {name} is non-finite: "
            f"{numeric}."
        )

    return numeric


def calculate_classification_metrics(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    class_mapping: Mapping[str, int],
) -> dict[str, Any]:
    (
        true_array,
        predicted_array,
        probability_array,
    ) = validate_prediction_contract(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_mapping=class_mapping,
    )

    class_items = ordered_class_items(
        class_mapping
    )

    class_names = [
        name
        for name, _ in class_items
    ]

    class_labels = [
        encoded
        for _, encoded in class_items
    ]

    (
        precision,
        recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        true_array,
        predicted_array,
        labels=class_labels,
        zero_division=0,
    )

    raw_confusion = confusion_matrix(
        true_array,
        predicted_array,
        labels=class_labels,
    )

    normalized_confusion = confusion_matrix(
        true_array,
        predicted_array,
        labels=class_labels,
        normalize="true",
    )

    per_class_metrics = []

    for index, (
        class_name,
        encoded_label,
    ) in enumerate(class_items):
        per_class_metrics.append(
            {
                "class_name": class_name,
                "class_encoded": int(
                    encoded_label
                ),
                "support": int(
                    support[index]
                ),
                "precision": ensure_finite_metric(
                    (
                        f"{class_name}.precision"
                    ),
                    precision[index],
                ),
                "recall": ensure_finite_metric(
                    f"{class_name}.recall",
                    recall[index],
                ),
                "f1": ensure_finite_metric(
                    f"{class_name}.f1",
                    class_f1[index],
                ),
            }
        )

    confidence = probability_array.max(
        axis=1
    )

    sorted_probabilities = np.sort(
        probability_array,
        axis=1,
    )

    prediction_margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    epsilon = np.finfo(float).eps

    entropy = -np.sum(
        probability_array
        * np.log(
            np.clip(
                probability_array,
                epsilon,
                1.0,
            )
        ),
        axis=1,
    )

    normalized_entropy = entropy / math.log(
        len(class_labels)
    )

    correct_mask = (
        true_array == predicted_array
    )

    metrics = {
        "schema_version": "1.0.0",
        "sample_count": int(
            len(true_array)
        ),
        "class_count": int(
            len(class_labels)
        ),
        "class_order": class_names,
        "class_mapping": {
            name: int(encoded)
            for name, encoded in class_items
        },
        "primary_metric_name": "macro_f1",
        "accuracy": ensure_finite_metric(
            "accuracy",
            accuracy_score(
                true_array,
                predicted_array,
            ),
        ),
        "balanced_accuracy": ensure_finite_metric(
            "balanced_accuracy",
            balanced_accuracy_score(
                true_array,
                predicted_array,
            ),
        ),
        "macro_f1": ensure_finite_metric(
            "macro_f1",
            f1_score(
                true_array,
                predicted_array,
                labels=class_labels,
                average="macro",
                zero_division=0,
            ),
        ),
        "weighted_f1": ensure_finite_metric(
            "weighted_f1",
            f1_score(
                true_array,
                predicted_array,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            ),
        ),
        "cohen_kappa": ensure_finite_metric(
            "cohen_kappa",
            cohen_kappa_score(
                true_array,
                predicted_array,
            ),
        ),
        "multiclass_log_loss": (
            ensure_finite_metric(
                "multiclass_log_loss",
                log_loss(
                    true_array,
                    probability_array,
                    labels=class_labels,
                ),
            )
        ),
        "mean_prediction_confidence": (
            ensure_finite_metric(
                "mean_prediction_confidence",
                confidence.mean(),
            )
        ),
        "mean_prediction_margin": (
            ensure_finite_metric(
                "mean_prediction_margin",
                prediction_margin.mean(),
            )
        ),
        "mean_normalized_entropy": (
            ensure_finite_metric(
                "mean_normalized_entropy",
                normalized_entropy.mean(),
            )
        ),
        "mean_confidence_when_correct": (
            ensure_finite_metric(
                "mean_confidence_when_correct",
                (
                    confidence[correct_mask].mean()
                    if correct_mask.any()
                    else 0.0
                ),
            )
        ),
        "mean_confidence_when_incorrect": (
            ensure_finite_metric(
                "mean_confidence_when_incorrect",
                (
                    confidence[
                        ~correct_mask
                    ].mean()
                    if (~correct_mask).any()
                    else 0.0
                ),
            )
        ),
        "per_class": per_class_metrics,
        "confusion_matrix": {
            "labels": class_names,
            "encoded_labels": class_labels,
            "raw": raw_confusion.astype(
                int
            ).tolist(),
            "normalized_by_true_class": (
                normalized_confusion.astype(
                    float
                ).tolist()
            ),
        },
    }

    return metrics


def probability_column_name(
    class_name: str,
) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        class_name.lower(),
    ).strip("_")

    if not normalized:
        raise ValueError(
            f"Invalid class name: {class_name!r}"
        )

    return f"probability_{normalized}"


def build_prediction_frame(
    identifiers: pd.DataFrame,
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]]
    | np.ndarray,
    class_mapping: Mapping[str, int],
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    if not isinstance(
        identifiers,
        pd.DataFrame,
    ):
        raise TypeError(
            "identifiers must be a pandas DataFrame."
        )

    if identifiers.empty:
        raise ValueError(
            "Identifier frame must not be empty."
        )

    if identifiers.columns.duplicated().any():
        raise ValueError(
            "Identifier frame contains duplicate "
            "column names."
        )

    if identifiers.isna().any().any():
        raise ValueError(
            "Identifier frame contains missing "
            "values."
        )

    if identifiers.duplicated().any():
        raise ValueError(
            "Identifier frame contains duplicate "
            "rows."
        )

    (
        true_array,
        predicted_array,
        probability_array,
    ) = validate_prediction_contract(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_mapping=class_mapping,
    )

    if len(identifiers) != len(
        true_array
    ):
        raise ValueError(
            "Identifier row count does not match "
            "prediction row count."
        )

    class_items = ordered_class_items(
        class_mapping
    )

    encoded_to_name = {
        encoded: name
        for name, encoded in class_items
    }

    confidence = probability_array.max(
        axis=1
    )

    sorted_probabilities = np.sort(
        probability_array,
        axis=1,
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    epsilon = np.finfo(float).eps

    entropy = -np.sum(
        probability_array
        * np.log(
            np.clip(
                probability_array,
                epsilon,
                1.0,
            )
        ),
        axis=1,
    )

    normalized_entropy = entropy / math.log(
        len(class_items)
    )

    output = pd.DataFrame(
        index=range(len(true_array))
    )

    for key in sorted(run_metadata):
        value = run_metadata[key]

        if isinstance(
            value,
            (
                dict,
                list,
                tuple,
                set,
            ),
        ):
            raise TypeError(
                f"Run metadata '{key}' must be "
                "a scalar value."
            )

        output[key] = value

    identifier_frame = identifiers.reset_index(
        drop=True
    )

    for column in identifier_frame.columns:
        if column in output.columns:
            raise ValueError(
                "Run metadata and identifier columns "
                f"overlap: {column}."
            )

        output[column] = identifier_frame[
            column
        ]

    output["true_label_encoded"] = (
        true_array
    )

    output["true_label"] = [
        encoded_to_name[int(value)]
        for value in true_array
    ]

    output["predicted_label_encoded"] = (
        predicted_array
    )

    output["predicted_label"] = [
        encoded_to_name[int(value)]
        for value in predicted_array
    ]

    output["is_correct"] = (
        true_array == predicted_array
    )

    output["prediction_confidence"] = (
        confidence
    )

    output["prediction_margin"] = margin

    output["prediction_entropy"] = entropy

    output[
        "prediction_normalized_entropy"
    ] = normalized_entropy

    for column_index, (
        class_name,
        _,
    ) in enumerate(class_items):
        output[
            probability_column_name(
                class_name
            )
        ] = probability_array[
            :,
            column_index,
        ]

    return output


def write_metrics_json(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        metrics,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    output_path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def write_prediction_csv(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )


def smoke_test() -> None:
    class_mapping = {
        "Wake": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4,
    }

    y_true = np.tile(
        np.arange(5),
        4,
    )

    y_pred = y_true.copy()

    y_pred[[1, 7, 13, 19]] = np.array(
        [2, 1, 4, 0]
    )

    probabilities = np.full(
        (len(y_pred), 5),
        0.025,
        dtype=float,
    )

    probabilities[
        np.arange(len(y_pred)),
        y_pred,
    ] = 0.9

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_mapping=class_mapping,
    )

    identifiers = pd.DataFrame(
        {
            "subject_id": np.repeat(
                np.arange(4),
                5,
            ),
            "recording_id": [
                f"recording_{index // 5}"
                for index in range(20)
            ],
            "night": 1,
            "epoch_id": np.arange(20),
        }
    )

    predictions = build_prediction_frame(
        identifiers=identifiers,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_mapping=class_mapping,
        run_metadata={
            "model_name": "synthetic_model",
            "outer_fold": 1,
            "partition": "test",
        },
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        metrics_path = (
            root / "metrics.json"
        )

        predictions_path = (
            root / "predictions.csv"
        )

        write_metrics_json(
            metrics=metrics,
            output_path=metrics_path,
        )

        write_prediction_csv(
            predictions=predictions,
            output_path=predictions_path,
        )

        if not metrics_path.exists():
            raise ValueError(
                "Metrics artifact was not written."
            )

        if not predictions_path.exists():
            raise ValueError(
                "Prediction artifact was not written."
            )

    print("=== PHASE 3 METRICS SMOKE TEST ===")
    print(
        "Samples:",
        metrics["sample_count"],
    )
    print(
        "Accuracy:",
        round(metrics["accuracy"], 6),
    )
    print(
        "Balanced accuracy:",
        round(
            metrics["balanced_accuracy"],
            6,
        ),
    )
    print(
        "Macro F1:",
        round(metrics["macro_f1"], 6),
    )
    print(
        "Weighted F1:",
        round(
            metrics["weighted_f1"],
            6,
        ),
    )
    print(
        "Cohen kappa:",
        round(
            metrics["cohen_kappa"],
            6,
        ),
    )
    print(
        "Log loss:",
        round(
            metrics[
                "multiclass_log_loss"
            ],
            6,
        ),
    )
    print(
        "Prediction rows:",
        len(predictions),
    )
    print("Metrics smoke test: PASS")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 classification metrics and "
            "prediction-artifact utilities."
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
        "Use --smoke-test to validate the "
        "metrics engine."
    )


if __name__ == "__main__":
    main()
