from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.phase3_dataset import (
    load_phase3_dataset,
    select_subject_partition,
)
from scripts.phase3_metrics import (
    align_probability_columns,
    probability_column_name,
    validate_and_normalize_probabilities,
    write_prediction_csv,
)
from scripts.phase3_model_artifacts import (
    DEFAULT_MANIFEST_JSON_PATH,
    load_trained_model_payload,
    resolve_artifact_path,
    sha256_file,
    validate_model_payload,
)
from scripts.phase3_prediction_store import (
    DEFAULT_PREDICTION_DB_PATH,
    PredictionStoreResult,
    audit_prediction_database,
    persist_prediction_run,
    prediction_frame_sha256,
    read_prediction_run,
    validate_prediction_frame,
)


IDENTIFIER_COLUMNS = (
    "subject_id",
    "recording_id",
    "night",
    "epoch_id",
)

QUALITY_COLUMN = "quality_issue_flag"

SOURCE_ROW_COLUMN = "source_row_index"

LOCKED_CLASS_MAPPING = {
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}


@dataclass(frozen=True)
class LoadedPredictionModel:
    manifest_path: Path
    manifest_sha256: str
    model_path: Path
    model_record: dict[str, Any]
    payload: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.payload["metadata"]

    @property
    def pipeline(self) -> Any:
        return self.payload["pipeline"]


@dataclass(frozen=True)
class PreparedPredictionInput:
    identifiers: pd.DataFrame
    X: pd.DataFrame
    source_row_indices: np.ndarray
    y_true: np.ndarray | None
    target_names: np.ndarray | None
    input_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class PredictionExecution:
    predictions: pd.DataFrame
    run_metadata: dict[str, Any]
    store_result: PredictionStoreResult | None


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


def runtime_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def dataframe_sha256(
    frame: pd.DataFrame,
) -> str:
    serialized = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")

    return hashlib.sha256(
        serialized
    ).hexdigest()


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def is_git_lfs_pointer(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    if path.stat().st_size > 1024:
        return False

    prefix = path.read_bytes()[:200]

    return prefix.startswith(
        b"version https://git-lfs.github.com/spec/v1"
    )


def normalize_class_mapping(
    class_mapping: Mapping[str, int],
) -> dict[str, int]:
    normalized = {
        str(name): int(encoded)
        for name, encoded
        in class_mapping.items()
    }

    if normalized != LOCKED_CLASS_MAPPING:
        raise ValueError(
            "Prediction pipeline requires the "
            "locked five-class mapping."
        )

    return normalized


def load_prediction_model(
    manifest_path: Path,
    outer_fold: int,
) -> LoadedPredictionModel:
    manifest_path = Path(
        manifest_path
    ).resolve()

    manifest = load_json_mapping(
        manifest_path
    )

    if manifest.get(
        "artifact_type"
    ) != "phase3_local_trained_model_manifest":
        raise ValueError(
            "Unexpected trained-model "
            "manifest artifact type."
        )

    models = manifest.get("models")

    if not isinstance(models, list):
        raise TypeError(
            "Trained-model manifest models "
            "must be a list."
        )

    matches = [
        model
        for model in models
        if int(model["outer_fold"])
        == int(outer_fold)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one model for "
            f"outer fold {outer_fold}; "
            f"found {len(matches)}."
        )

    record = dict(matches[0])

    model_path = resolve_artifact_path(
        str(record["model_file_path"])
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if is_git_lfs_pointer(model_path):
        raise ValueError(
            "The model path contains only a "
            "Git LFS pointer. Run `git lfs pull` "
            "before prediction."
        )

    observed_size = int(
        model_path.stat().st_size
    )

    expected_size = int(
        record["model_file_size_bytes"]
    )

    if observed_size != expected_size:
        raise ValueError(
            "Model file size does not match "
            "the trained-model manifest."
        )

    observed_hash = sha256_file(
        model_path
    )

    expected_hash = str(
        record["model_file_sha256"]
    ).lower()

    if observed_hash.lower() != expected_hash:
        raise ValueError(
            "Model file SHA256 does not match "
            "the trained-model manifest."
        )

    payload = load_trained_model_payload(
        model_path
    )

    validate_model_payload(payload)

    metadata = payload["metadata"]

    scalar_contracts = (
        "outer_fold",
        "model_name",
        "candidate_id",
        "feature_count",
        "deployment_ready",
    )

    for key in scalar_contracts:
        if metadata[key] != record[key]:
            raise ValueError(
                f"Manifest/payload mismatch: {key}."
            )

    sequence_contracts = (
        "training_subjects",
        "excluded_test_subjects",
    )

    for key in sequence_contracts:
        if list(metadata[key]) != list(
            record[key]
        ):
            raise ValueError(
                f"Manifest/payload mismatch: {key}."
            )

    normalize_class_mapping(
        metadata["class_mapping"]
    )

    return LoadedPredictionModel(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(
            manifest_path
        ),
        model_path=model_path,
        model_record=record,
        payload=payload,
    )


def normalize_integer_series(
    series: pd.Series,
    name: str,
    minimum: int | None = None,
) -> pd.Series:
    if series.isna().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    numeric = pd.to_numeric(
        series,
        errors="raise",
    )

    numeric_array = numeric.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    rounded = np.round(
        numeric_array
    )

    if not np.array_equal(
        numeric_array,
        rounded,
    ):
        raise ValueError(
            f"{name} must contain integers."
        )

    integers = rounded.astype(
        np.int64
    )

    if (
        minimum is not None
        and np.any(integers < minimum)
    ):
        raise ValueError(
            f"{name} contains values below "
            f"{minimum}."
        )

    return pd.Series(
        integers,
        index=series.index,
        name=series.name,
    )


def normalize_boolean_series(
    series: pd.Series,
    name: str,
) -> pd.Series:
    if series.isna().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.astype(bool)

    normalized_strings = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    invalid = ~normalized_strings.isin(
        mapping
    )

    if invalid.any():
        raise ValueError(
            f"{name} must contain only "
            "boolean or 0/1 values."
        )

    return normalized_strings.map(
        mapping
    ).astype(bool)


def validate_identifier_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    identifiers = frame.loc[
        :,
        list(IDENTIFIER_COLUMNS),
    ].copy()

    identifiers["subject_id"] = (
        normalize_integer_series(
            identifiers["subject_id"],
            "subject_id",
            minimum=0,
        )
    )

    identifiers["night"] = (
        normalize_integer_series(
            identifiers["night"],
            "night",
            minimum=0,
        )
    )

    identifiers["epoch_id"] = (
        normalize_integer_series(
            identifiers["epoch_id"],
            "epoch_id",
            minimum=0,
        )
    )

    if identifiers[
        "recording_id"
    ].isna().any():
        raise ValueError(
            "recording_id contains "
            "missing values."
        )

    identifiers["recording_id"] = (
        identifiers["recording_id"]
        .astype(str)
        .str.strip()
    )

    if (
        identifiers["recording_id"]
        == ""
    ).any():
        raise ValueError(
            "recording_id contains "
            "empty values."
        )

    if identifiers.duplicated().any():
        raise ValueError(
            "Prediction input identifiers "
            "are not unique."
        )

    return identifiers.reset_index(
        drop=True
    )


def prepare_prediction_input(
    frame: pd.DataFrame,
    model_metadata: Mapping[str, Any],
) -> PreparedPredictionInput:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "Prediction input must be "
            "a DataFrame."
        )

    if frame.empty:
        raise ValueError(
            "Prediction input is empty."
        )

    feature_names = tuple(
        str(value)
        for value in model_metadata[
            "feature_names"
        ]
    )

    if len(feature_names) != int(
        model_metadata["feature_count"]
    ):
        raise ValueError(
            "Model feature-count metadata "
            "is inconsistent."
        )

    target_column = str(
        model_metadata["target_column"]
    )

    target_name_column = str(
        model_metadata[
            "target_name_column"
        ]
    )

    required_columns = {
        *IDENTIFIER_COLUMNS,
        *feature_names,
    }

    missing = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing prediction input columns: "
            + ", ".join(missing)
        )

    allowed_columns = {
        *required_columns,
        target_column,
        target_name_column,
        QUALITY_COLUMN,
        SOURCE_ROW_COLUMN,
    }

    unexpected = sorted(
        set(frame.columns)
        - allowed_columns
    )

    if unexpected:
        raise ValueError(
            "Unexpected prediction input "
            "columns: "
            + ", ".join(unexpected)
        )

    has_target = target_column in (
        frame.columns
    )

    has_target_name = target_name_column in (
        frame.columns
    )

    if has_target != has_target_name:
        raise ValueError(
            "Ground-truth encoded and name "
            "columns must be provided together."
        )

    if QUALITY_COLUMN in frame.columns:
        quality = normalize_boolean_series(
            frame[QUALITY_COLUMN],
            QUALITY_COLUMN,
        )

        if quality.any():
            raise ValueError(
                "Prediction input contains "
                "quality-issue rows."
            )

    identifiers = validate_identifier_frame(
        frame
    )

    if SOURCE_ROW_COLUMN in frame.columns:
        source_row_indices = (
            normalize_integer_series(
                frame[SOURCE_ROW_COLUMN],
                SOURCE_ROW_COLUMN,
                minimum=0,
            ).to_numpy(
                dtype=np.int64
            )
        )

        if len(
            np.unique(source_row_indices)
        ) != len(source_row_indices):
            raise ValueError(
                "source_row_index values "
                "are not unique."
            )
    else:
        source_row_indices = np.arange(
            len(frame),
            dtype=np.int64,
        )

    X = frame.loc[
        :,
        list(feature_names),
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    feature_values = X.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Prediction features contain "
            "missing or non-finite values."
        )

    X = pd.DataFrame(
        feature_values,
        columns=list(feature_names),
    )

    mapping = normalize_class_mapping(
        model_metadata["class_mapping"]
    )

    inverse_mapping = {
        encoded: name
        for name, encoded
        in mapping.items()
    }

    y_true = None
    target_names = None

    if has_target:
        y_series = normalize_integer_series(
            frame[target_column],
            target_column,
            minimum=0,
        )

        y_true = y_series.to_numpy(
            dtype=np.int64
        )

        if not set(
            y_true.tolist()
        ).issubset(
            set(inverse_mapping)
        ):
            raise ValueError(
                "Ground truth contains an "
                "unknown encoded class."
            )

        target_names = (
            frame[target_name_column]
            .astype(str)
            .to_numpy(
                dtype=object
            )
        )

        expected_names = np.array(
            [
                inverse_mapping[
                    int(value)
                ]
                for value in y_true
            ],
            dtype=object,
        )

        if not np.array_equal(
            target_names,
            expected_names,
        ):
            raise ValueError(
                "Ground-truth label names "
                "do not match encoded labels."
            )

    hash_frame = identifiers.copy()

    hash_frame[
        SOURCE_ROW_COLUMN
    ] = source_row_indices

    for feature_name in feature_names:
        hash_frame[
            feature_name
        ] = X[feature_name]

    if has_target:
        hash_frame[
            target_column
        ] = y_true

        hash_frame[
            target_name_column
        ] = target_names

    return PreparedPredictionInput(
        identifiers=identifiers,
        X=X,
        source_row_indices=(
            source_row_indices
        ),
        y_true=y_true,
        target_names=target_names,
        input_sha256=dataframe_sha256(
            hash_frame
        ),
    )


def classifier_classes(
    pipeline: Any,
) -> np.ndarray:
    if (
        hasattr(pipeline, "named_steps")
        and "classifier"
        in pipeline.named_steps
    ):
        classifier = pipeline.named_steps[
            "classifier"
        ]
    else:
        classifier = pipeline

    if not hasattr(
        classifier,
        "classes_",
    ):
        raise TypeError(
            "Prediction classifier has no "
            "classes_ attribute."
        )

    return np.asarray(
        classifier.classes_
    )


def build_prediction_output(
    model: LoadedPredictionModel,
    prepared: PreparedPredictionInput,
    allow_non_deployment_model: bool,
) -> pd.DataFrame:
    metadata = model.metadata

    deployment_ready = bool(
        metadata["deployment_ready"]
    )

    if (
        not deployment_ready
        and not allow_non_deployment_model
    ):
        raise ValueError(
            "This model is not deployment-ready. "
            "Use an explicit non-deployment "
            "override for engineering inference."
        )

    expected_features = tuple(
        metadata["feature_names"]
    )

    if tuple(
        prepared.X.columns
    ) != expected_features:
        raise ValueError(
            "Prepared feature order does not "
            "match model metadata."
        )

    pipeline = model.pipeline

    predictions = np.asarray(
        pipeline.predict(
            prepared.X
        )
    )

    if predictions.ndim != 1:
        raise ValueError(
            "Model predictions must be "
            "one-dimensional."
        )

    if len(predictions) != (
        prepared.row_count
    ):
        raise ValueError(
            "Prediction count does not match "
            "input row count."
        )

    try:
        predictions = predictions.astype(
            np.int64
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Model predictions are not "
            "integer-encoded labels."
        ) from error

    raw_probabilities = (
        pipeline.predict_proba(
            prepared.X
        )
    )

    class_mapping = (
        normalize_class_mapping(
            metadata["class_mapping"]
        )
    )

    probabilities = (
        align_probability_columns(
            probabilities=raw_probabilities,
            estimator_classes=(
                classifier_classes(
                    pipeline
                )
            ),
            class_mapping=class_mapping,
        )
    )

    probabilities = (
        validate_and_normalize_probabilities(
            probabilities=probabilities,
            expected_row_count=(
                prepared.row_count
            ),
            expected_class_count=len(
                class_mapping
            ),
        )
    )

    inverse_mapping = {
        encoded: name
        for name, encoded
        in class_mapping.items()
    }

    valid_encoded = set(
        inverse_mapping
    )

    if not set(
        predictions.tolist()
    ).issubset(valid_encoded):
        raise ValueError(
            "Model produced an unknown "
            "encoded class."
        )

    predicted_names = np.array(
        [
            inverse_mapping[int(value)]
            for value in predictions
        ],
        dtype=object,
    )

    probability_argmax = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int64
    )

    probability_argmax_names = np.array(
        [
            inverse_mapping[int(value)]
            for value
            in probability_argmax
        ],
        dtype=object,
    )

    agreement = (
        predictions
        == probability_argmax
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    confidence = (
        sorted_probabilities[:, -1]
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    safe_probabilities = np.clip(
        probabilities,
        np.finfo(float).tiny,
        1.0,
    )

    entropy = -np.sum(
        probabilities
        * np.log(safe_probabilities),
        axis=1,
    )

    normalized_entropy = (
        entropy
        / math.log(
            len(class_mapping)
        )
    )

    output = prepared.identifiers.copy()

    output[
        SOURCE_ROW_COLUMN
    ] = prepared.source_row_indices

    output[
        "predicted_label_encoded"
    ] = predictions

    output[
        "predicted_label"
    ] = predicted_names

    output[
        "probability_argmax_label_encoded"
    ] = probability_argmax

    output[
        "probability_argmax_label"
    ] = probability_argmax_names

    output[
        "predict_probability_argmax_agree"
    ] = agreement

    output[
        "prediction_confidence"
    ] = confidence

    output[
        "prediction_margin"
    ] = margin

    output[
        "prediction_entropy"
    ] = entropy

    output[
        "prediction_normalized_entropy"
    ] = normalized_entropy

    ordered_class_names = [
        name
        for name, _
        in sorted(
            class_mapping.items(),
            key=lambda item: item[1],
        )
    ]

    for encoded, class_name in enumerate(
        ordered_class_names
    ):
        output[
            probability_column_name(
                class_name
            )
        ] = probabilities[
            :,
            encoded,
        ]

    if prepared.y_true is not None:
        output[
            "true_label_encoded"
        ] = prepared.y_true

        output[
            "true_label"
        ] = prepared.target_names

        output[
            "is_correct"
        ] = (
            prepared.y_true
            == predictions
        )

    return validate_prediction_frame(
        predictions=output,
        class_mapping=class_mapping,
    )


def build_run_metadata(
    model: LoadedPredictionModel,
    prepared: PreparedPredictionInput,
    input_scope: str,
    allow_non_deployment_model: bool,
    input_file_path: Path | None = None,
) -> dict[str, Any]:
    input_scope = str(
        input_scope
    ).strip()

    if not input_scope:
        raise ValueError(
            "input_scope cannot be empty."
        )

    metadata = model.metadata

    run_metadata = {
        "artifact_type": (
            "phase3_prediction_run"
        ),
        "model_artifact_type": (
            metadata["artifact_type"]
        ),
        "model_file_path": str(
            model.model_record[
                "model_file_path"
            ]
        ),
        "model_file_sha256": str(
            model.model_record[
                "model_file_sha256"
            ]
        ).lower(),
        "model_file_size_bytes": int(
            model.model_record[
                "model_file_size_bytes"
            ]
        ),
        "model_name": str(
            metadata["model_name"]
        ),
        "candidate_id": str(
            metadata["candidate_id"]
        ),
        "outer_fold": int(
            metadata["outer_fold"]
        ),
        "deployment_ready": bool(
            metadata["deployment_ready"]
        ),
        "non_deployment_override": bool(
            allow_non_deployment_model
        ),
        "input_scope": input_scope,
        "input_row_count": int(
            prepared.row_count
        ),
        "input_dataframe_sha256": (
            prepared.input_sha256
        ),
        "feature_names": list(
            metadata["feature_names"]
        ),
        "class_mapping": dict(
            metadata["class_mapping"]
        ),
        "trained_model_manifest_path": str(
            model.manifest_path
        ),
        "trained_model_manifest_sha256": (
            model.manifest_sha256
        ),
        "runtime": runtime_metadata(),
    }

    if input_file_path is not None:
        input_file_path = Path(
            input_file_path
        ).resolve()

        if not input_file_path.exists():
            raise FileNotFoundError(
                input_file_path
            )

        run_metadata[
            "input_file_path"
        ] = str(input_file_path)

        run_metadata[
            "input_file_sha256"
        ] = sha256_file(
            input_file_path
        )

    return run_metadata


def execute_prediction_pipeline(
    model: LoadedPredictionModel,
    input_frame: pd.DataFrame,
    input_scope: str,
    allow_non_deployment_model: bool,
    database_path: Path | None = None,
    output_csv_path: Path | None = None,
    input_file_path: Path | None = None,
) -> PredictionExecution:
    prepared = prepare_prediction_input(
        frame=input_frame,
        model_metadata=model.metadata,
    )

    predictions = build_prediction_output(
        model=model,
        prepared=prepared,
        allow_non_deployment_model=(
            allow_non_deployment_model
        ),
    )

    run_metadata = build_run_metadata(
        model=model,
        prepared=prepared,
        input_scope=input_scope,
        allow_non_deployment_model=(
            allow_non_deployment_model
        ),
        input_file_path=input_file_path,
    )

    run_metadata[
        "prediction_sha256"
    ] = prediction_frame_sha256(
        predictions
    )

    store_result = None

    if database_path is not None:
        store_result = persist_prediction_run(
            database_path=Path(
                database_path
            ),
            run_metadata=run_metadata,
            predictions=predictions,
        )

    if output_csv_path is not None:
        write_prediction_csv(
            predictions=predictions,
            output_path=Path(
                output_csv_path
            ),
        )

    return PredictionExecution(
        predictions=predictions,
        run_metadata=run_metadata,
        store_result=store_result,
    )


def smoke_test() -> None:
    model = load_prediction_model(
        manifest_path=(
            DEFAULT_MANIFEST_JSON_PATH
        ),
        outer_fold=1,
    )

    training_subject = int(
        model.metadata[
            "training_subjects"
        ][0]
    )

    bundle = load_phase3_dataset()

    partition = select_subject_partition(
        bundle=bundle,
        subjects=(
            training_subject,
        ),
        name=(
            "prediction_pipeline_smoke"
        ),
        require_all_classes=True,
    )

    row_count = min(
        64,
        partition.row_count,
    )

    input_frame = pd.concat(
        [
            partition.identifiers.iloc[
                :row_count
            ].reset_index(drop=True),
            partition.X.iloc[
                :row_count
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    input_frame[
        SOURCE_ROW_COLUMN
    ] = partition.row_indices[
        :row_count
    ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        database_path = (
            root
            / "predictions.sqlite3"
        )

        output_path = (
            root
            / "predictions.csv"
        )

        first = execute_prediction_pipeline(
            model=model,
            input_frame=input_frame,
            input_scope=(
                "smoke_outer_fold_01_"
                f"training_subject_{training_subject}"
            ),
            allow_non_deployment_model=True,
            database_path=database_path,
            output_csv_path=output_path,
        )

        first_csv = output_path.read_bytes()

        second = execute_prediction_pipeline(
            model=model,
            input_frame=input_frame,
            input_scope=(
                "smoke_outer_fold_01_"
                f"training_subject_{training_subject}"
            ),
            allow_non_deployment_model=True,
            database_path=database_path,
            output_csv_path=output_path,
        )

        second_csv = output_path.read_bytes()

        if first.store_result is None:
            raise ValueError(
                "Smoke prediction was not "
                "persisted."
            )

        if second.store_result is None:
            raise ValueError(
                "Repeated smoke prediction was "
                "not checked."
            )

        if not first.store_result.inserted:
            raise ValueError(
                "Initial smoke prediction was "
                "not inserted."
            )

        if second.store_result.inserted:
            raise ValueError(
                "Repeated smoke prediction "
                "created a duplicate run."
            )

        if (
            first.store_result.run_id
            != second.store_result.run_id
        ):
            raise ValueError(
                "Repeated prediction run IDs "
                "do not match."
            )

        if first_csv != second_csv:
            raise ValueError(
                "Prediction CSV is not "
                "byte-deterministic."
            )

        forbidden_ground_truth = {
            "true_label_encoded",
            "true_label",
            "is_correct",
        }

        if forbidden_ground_truth & set(
            first.predictions.columns
        ):
            raise ValueError(
                "Targetless smoke prediction "
                "contains ground-truth columns."
            )

        audit = audit_prediction_database(
            database_path
        )

        stored_run, stored_rows = (
            read_prediction_run(
                database_path=database_path,
                run_id=(
                    first.store_result.run_id
                ),
            )
        )

        if len(stored_rows) != row_count:
            raise ValueError(
                "Stored smoke prediction row "
                "count is incorrect."
            )

        print(
            "=== PHASE 3 PREDICTION "
            "PIPELINE SMOKE TEST ==="
        )
        print(
            "Outer fold:",
            model.metadata[
                "outer_fold"
            ],
        )
        print(
            "Model:",
            model.metadata[
                "model_name"
            ],
        )
        print(
            "Candidate:",
            model.metadata[
                "candidate_id"
            ],
        )
        print(
            "Deployment ready:",
            model.metadata[
                "deployment_ready"
            ],
        )
        print(
            "Explicit override:",
            True,
        )
        print(
            "Input subject:",
            training_subject,
        )
        print(
            "Input rows:",
            row_count,
        )
        print(
            "Ground truth supplied:",
            False,
        )
        print(
            "Prediction rows:",
            len(
                first.predictions
            ),
        )
        print(
            "First DB insert:",
            first.store_result.inserted,
        )
        print(
            "Second DB insert:",
            second.store_result.inserted,
        )
        print(
            "Stored run ID:",
            stored_run["run_id"],
        )
        print(
            "SQLite quick check:",
            audit["quick_check"],
        )
        print(
            "Foreign-key issues:",
            audit[
                "foreign_key_issue_count"
            ],
        )
        print(
            "Byte-deterministic CSV: PASS"
        )
        print(
            "Targetless prediction: PASS"
        )
        print(
            "Transactional persistence: PASS"
        )
        print(
            "Prediction pipeline smoke test: PASS"
        )


def run_cli_prediction(
    arguments: argparse.Namespace,
) -> None:
    if arguments.outer_fold is None:
        raise SystemExit(
            "--outer-fold is required."
        )

    if arguments.input_csv is None:
        raise SystemExit(
            "--input-csv is required."
        )

    input_path = Path(
        arguments.input_csv
    ).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    input_frame = pd.read_csv(
        input_path
    )

    model = load_prediction_model(
        manifest_path=Path(
            arguments.model_manifest
        ),
        outer_fold=int(
            arguments.outer_fold
        ),
    )

    database_path = Path(
        arguments.database_path
    )

    output_path = (
        Path(arguments.output_csv)
        if arguments.output_csv
        is not None
        else None
    )

    execution = execute_prediction_pipeline(
        model=model,
        input_frame=input_frame,
        input_scope=(
            arguments.input_scope
        ),
        allow_non_deployment_model=(
            arguments.allow_non_deployment_model
        ),
        database_path=database_path,
        output_csv_path=output_path,
        input_file_path=input_path,
    )

    print(
        "=== PHASE 3 PREDICTION RUN ==="
    )
    print(
        "Outer fold:",
        model.metadata["outer_fold"],
    )
    print(
        "Model:",
        model.metadata["model_name"],
    )
    print(
        "Candidate:",
        model.metadata[
            "candidate_id"
        ],
    )
    print(
        "Deployment ready:",
        model.metadata[
            "deployment_ready"
        ],
    )
    print(
        "Input rows:",
        len(execution.predictions),
    )
    print(
        "Ground truth supplied:",
        (
            "true_label_encoded"
            in execution.predictions.columns
        ),
    )

    if execution.store_result is not None:
        print(
            "Database:",
            execution.store_result.database_path,
        )
        print(
            "Run ID:",
            execution.store_result.run_id,
        )
        print(
            "Inserted:",
            execution.store_result.inserted,
        )

    if output_path is not None:
        print(
            "Prediction CSV:",
            output_path.resolve(),
        )

    print(
        "Prediction pipeline run: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validated Phase 3 model "
            "against a feature CSV and persist "
            "predictions transactionally."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--model-manifest",
        default=str(
            DEFAULT_MANIFEST_JSON_PATH
        ),
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
    )

    parser.add_argument(
        "--input-csv",
    )

    parser.add_argument(
        "--input-scope",
        default="phase3_prediction_input",
    )

    parser.add_argument(
        "--database-path",
        default=str(
            DEFAULT_PREDICTION_DB_PATH
        ),
    )

    parser.add_argument(
        "--output-csv",
    )

    parser.add_argument(
        "--allow-non-deployment-model",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.smoke_test:
        smoke_test()
        return

    if (
        arguments.outer_fold is not None
        or arguments.input_csv is not None
    ):
        run_cli_prediction(
            arguments
        )
        return

    print(
        "Use --smoke-test or provide "
        "--outer-fold and --input-csv."
    )


if __name__ == "__main__":
    main()
