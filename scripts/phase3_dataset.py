from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_edfx_model_input.csv"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "sleep_edfx_model_feature_schema.json"
)

DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_evaluation_protocol.json"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_dataset_contract_summary.json"
)


@dataclass
class DatasetPartition:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    identifiers: pd.DataFrame
    quality: pd.DataFrame
    target_names: np.ndarray
    row_indices: np.ndarray
    subjects: tuple[Any, ...]

    @property
    def row_count(self) -> int:
        return len(self.y)


@dataclass
class Phase3DatasetBundle:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    identifiers: pd.DataFrame
    quality: pd.DataFrame
    target_names: np.ndarray
    row_indices: np.ndarray
    feature_names: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    quality_columns: tuple[str, ...]
    class_mapping: dict[str, int]
    group_column: str
    target_column: str
    target_name_column: str
    source_column_count: int
    data_path: Path
    schema_path: Path
    protocol_path: Path
    data_sha256: str
    schema_sha256: str
    protocol_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.y)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
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


def ensure_unique_strings(
    values: Sequence[Any],
    name: str,
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(
            f"{name} must be a JSON list."
        )

    normalized = [
        str(value)
        for value in values
    ]

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

    return normalized


def normalize_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    if is_bool_dtype(series.dtype):
        return series.astype(bool)

    normalized = (
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

    unknown = sorted(
        set(normalized.unique())
        - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"Quality column {column_name} contains "
            f"invalid boolean values: {unknown}."
        )

    return normalized.map(mapping).astype(bool)


def normalize_class_mapping(
    mapping: Mapping[str, Any],
) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise ValueError(
            "class_mapping must be a JSON object."
        )

    normalized = {
        str(name): int(encoded)
        for name, encoded in mapping.items()
    }

    if not normalized:
        raise ValueError(
            "class_mapping must not be empty."
        )

    encoded_values = sorted(
        normalized.values()
    )

    expected = list(
        range(len(normalized))
    )

    if encoded_values != expected:
        raise ValueError(
            "Class encodings must be contiguous and "
            f"start at zero; found {encoded_values}."
        )

    if len(normalized.values()) != len(
        set(normalized.values())
    ):
        raise ValueError(
            "Class encodings must be unique."
        )

    return normalized


def relative_display_path(path: Path) -> str:
    resolved = path.resolve()

    try:
        return resolved.relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return str(resolved)


def load_phase3_dataset(
    data_path: Path = DEFAULT_DATA_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    reject_quality_issues: bool = True,
) -> Phase3DatasetBundle:
    data_path = data_path.resolve()
    schema_path = schema_path.resolve()
    protocol_path = protocol_path.resolve()

    for path, label in (
        (data_path, "model input"),
        (schema_path, "model feature schema"),
        (protocol_path, "evaluation protocol"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    dataframe = pd.read_csv(data_path)
    schema = load_json(schema_path)
    protocol = load_json(protocol_path)

    identifier_columns = ensure_unique_strings(
        schema["identifier_columns"],
        "identifier_columns",
    )

    target_columns = ensure_unique_strings(
        schema["target_columns"],
        "target_columns",
    )

    quality_columns = ensure_unique_strings(
        schema["quality_columns"],
        "quality_columns",
    )

    selected_features = ensure_unique_strings(
        schema["selected_features"],
        "selected_features",
    )

    declared_groups = [
        set(identifier_columns),
        set(target_columns),
        set(quality_columns),
        set(selected_features),
    ]

    for left_index in range(
        len(declared_groups)
    ):
        for right_index in range(
            left_index + 1,
            len(declared_groups),
        ):
            overlap = sorted(
                declared_groups[left_index]
                & declared_groups[right_index]
            )

            if overlap:
                raise ValueError(
                    "Schema column categories overlap: "
                    f"{overlap}."
                )

    expected_columns = (
        identifier_columns
        + target_columns
        + quality_columns
        + selected_features
    )

    if list(dataframe.columns) != expected_columns:
        missing = sorted(
            set(expected_columns)
            - set(dataframe.columns)
        )

        unexpected = sorted(
            set(dataframe.columns)
            - set(expected_columns)
        )

        raise ValueError(
            "Dataset columns do not exactly match "
            "the model schema. "
            f"Missing={missing}; "
            f"Unexpected={unexpected}."
        )

    selected_feature_count = int(
        schema["selected_feature_count"]
    )

    if selected_feature_count != len(
        selected_features
    ):
        raise ValueError(
            "selected_feature_count does not match "
            "selected_features."
        )

    if bool(
        schema["feature_scaling_applied"]
    ):
        raise ValueError(
            "Model input must remain unscaled before "
            "subject-wise splitting."
        )

    if dataframe.empty:
        raise ValueError(
            "Model input dataset is empty."
        )

    if dataframe.isna().any().any():
        missing_count = int(
            dataframe.isna().sum().sum()
        )

        raise ValueError(
            "Model input contains missing values: "
            f"{missing_count} cells."
        )

    if dataframe.duplicated(
        subset=identifier_columns
    ).any():
        duplicate_count = int(
            dataframe.duplicated(
                subset=identifier_columns
            ).sum()
        )

        raise ValueError(
            "Model input contains duplicate identifier "
            f"rows: {duplicate_count}."
        )

    group_column = str(
        protocol["group_column"]
    )

    target_column = str(
        protocol["target_column"]
    )

    target_name_column = str(
        protocol["target_name_column"]
    )

    if group_column not in identifier_columns:
        raise ValueError(
            "Protocol group column is not an "
            "identifier column."
        )

    if target_column not in target_columns:
        raise ValueError(
            "Protocol encoded target is not a "
            "target column."
        )

    if target_name_column not in target_columns:
        raise ValueError(
            "Protocol target name is not a "
            "target column."
        )

    if bool(
        protocol["random_epoch_split_allowed"]
    ):
        raise ValueError(
            "Random epoch splitting must remain disabled."
        )

    class_mapping = normalize_class_mapping(
        protocol["class_mapping"]
    )

    observed_mapping_frame = (
        dataframe[
            [target_name_column, target_column]
        ]
        .drop_duplicates()
        .sort_values(target_column)
    )

    if (
        observed_mapping_frame[
            target_name_column
        ].duplicated().any()
        or observed_mapping_frame[
            target_column
        ].duplicated().any()
    ):
        raise ValueError(
            "Dataset target mapping is not one-to-one."
        )

    observed_mapping = {
        str(row[target_name_column]): int(
            row[target_column]
        )
        for _, row in observed_mapping_frame.iterrows()
    }

    if observed_mapping != class_mapping:
        raise ValueError(
            "Dataset target mapping does not match "
            "the evaluation protocol."
        )

    y_numeric = pd.to_numeric(
        dataframe[target_column],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(y_numeric).all():
        raise ValueError(
            "Encoded target contains non-finite values."
        )

    y = y_numeric.astype(int)

    if not np.array_equal(
        y_numeric,
        y.astype(float),
    ):
        raise ValueError(
            "Encoded target contains non-integer values."
        )

    allowed_labels = set(
        class_mapping.values()
    )

    unknown_labels = sorted(
        set(y.tolist())
        - allowed_labels
    )

    if unknown_labels:
        raise ValueError(
            "Encoded target contains unknown labels: "
            f"{unknown_labels}."
        )

    target_names = (
        dataframe[target_name_column]
        .astype(str)
        .to_numpy(copy=True)
    )

    encoded_to_name = {
        encoded: name
        for name, encoded in class_mapping.items()
    }

    expected_target_names = np.asarray(
        [
            encoded_to_name[int(encoded)]
            for encoded in y
        ],
        dtype=object,
    )

    if not np.array_equal(
        target_names,
        expected_target_names,
    ):
        raise ValueError(
            "Target names do not match encoded targets."
        )

    non_numeric_features = [
        feature
        for feature in selected_features
        if not is_numeric_dtype(
            dataframe[feature]
        )
    ]

    if non_numeric_features:
        raise ValueError(
            "Selected model features must be numeric: "
            f"{non_numeric_features}."
        )

    X = dataframe[
        selected_features
    ].astype(float).copy()

    feature_array = X.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_array
    ).all():
        non_finite_count = int(
            (~np.isfinite(feature_array)).sum()
        )

        raise ValueError(
            "Selected features contain non-finite "
            f"values: {non_finite_count} cells."
        )

    quality = dataframe[
        quality_columns
    ].copy()

    for column in quality_columns:
        quality[column] = normalize_boolean_series(
            quality[column],
            column,
        )

    quality_issue_count = int(
        quality.to_numpy(dtype=bool).sum()
    )

    if (
        reject_quality_issues
        and quality_issue_count > 0
    ):
        raise ValueError(
            "Model input contains quality-issue flags: "
            f"{quality_issue_count}."
        )

    identifiers = dataframe[
        identifier_columns
    ].copy()

    groups = dataframe[
        group_column
    ].to_numpy(copy=True)

    if pd.isna(groups).any():
        raise ValueError(
            "Group column contains missing values."
        )

    row_indices = np.arange(
        len(dataframe),
        dtype=int,
    )

    return Phase3DatasetBundle(
        X=X.reset_index(drop=True),
        y=y.copy(),
        groups=groups.copy(),
        identifiers=identifiers.reset_index(
            drop=True
        ),
        quality=quality.reset_index(drop=True),
        target_names=target_names.copy(),
        row_indices=row_indices,
        feature_names=tuple(selected_features),
        identifier_columns=tuple(
            identifier_columns
        ),
        quality_columns=tuple(
            quality_columns
        ),
        class_mapping=class_mapping,
        group_column=group_column,
        target_column=target_column,
        target_name_column=target_name_column,
        source_column_count=len(
            dataframe.columns
        ),
        data_path=data_path,
        schema_path=schema_path,
        protocol_path=protocol_path,
        data_sha256=sha256_file(data_path),
        schema_sha256=sha256_file(
            schema_path
        ),
        protocol_sha256=sha256_file(
            protocol_path
        ),
    )


def normalize_subjects(
    subjects: Sequence[Any],
) -> tuple[Any, ...]:
    if isinstance(
        subjects,
        (str, bytes),
    ):
        raise TypeError(
            "subjects must be a sequence, not a string."
        )

    normalized = []

    for subject in subjects:
        item_method = getattr(
            subject,
            "item",
            None,
        )

        if callable(item_method):
            subject = item_method()

        normalized.append(subject)

    if not normalized:
        raise ValueError(
            "At least one subject is required."
        )

    if len(normalized) != len(
        set(normalized)
    ):
        raise ValueError(
            "Subject selection contains duplicates."
        )

    return tuple(
        sorted(
            normalized,
            key=lambda value: (
                type(value).__name__,
                str(value),
            ),
        )
    )


def select_subject_partition(
    bundle: Phase3DatasetBundle,
    subjects: Sequence[Any],
    name: str,
    require_all_classes: bool = True,
) -> DatasetPartition:
    selected_subjects = normalize_subjects(
        subjects
    )

    available_subjects = set(
        bundle.groups.tolist()
    )

    unknown_subjects = sorted(
        set(selected_subjects)
        - available_subjects,
        key=str,
    )

    if unknown_subjects:
        raise ValueError(
            "Unknown subject identifiers: "
            f"{unknown_subjects}."
        )

    mask = np.isin(
        bundle.groups,
        selected_subjects,
    )

    positions = np.flatnonzero(mask)

    if len(positions) == 0:
        raise ValueError(
            f"Partition {name} is empty."
        )

    partition_y = bundle.y[
        positions
    ]

    if require_all_classes:
        missing_classes = sorted(
            set(
                bundle.class_mapping.values()
            )
            - set(partition_y.tolist())
        )

        if missing_classes:
            raise ValueError(
                f"Partition {name} is missing encoded "
                f"classes: {missing_classes}."
            )

    partition_groups = bundle.groups[
        positions
    ]

    if not set(
        partition_groups.tolist()
    ).issubset(
        set(selected_subjects)
    ):
        raise ValueError(
            f"Partition {name} contains unexpected "
            "subjects."
        )

    return DatasetPartition(
        name=str(name),
        X=bundle.X.iloc[
            positions
        ].reset_index(drop=True),
        y=partition_y.copy(),
        groups=partition_groups.copy(),
        identifiers=bundle.identifiers.iloc[
            positions
        ].reset_index(drop=True),
        quality=bundle.quality.iloc[
            positions
        ].reset_index(drop=True),
        target_names=bundle.target_names[
            positions
        ].copy(),
        row_indices=bundle.row_indices[
            positions
        ].copy(),
        subjects=selected_subjects,
    )


def build_dataset_summary(
    bundle: Phase3DatasetBundle,
) -> dict[str, Any]:
    ordered_classes = sorted(
        bundle.class_mapping.items(),
        key=lambda item: item[1],
    )

    class_counts = {
        name: int(
            np.sum(
                bundle.y == encoded
            )
        )
        for name, encoded in ordered_classes
    }

    unique_subjects = sorted(
        {
            value.item()
            if callable(
                getattr(value, "item", None)
            )
            else value
            for value in bundle.groups
        },
        key=lambda value: (
            type(value).__name__,
            str(value),
        ),
    )

    rows_per_subject = {
        str(subject): int(
            np.sum(
                bundle.groups == subject
            )
        )
        for subject in unique_subjects
    }

    quality_issue_counts = {
        column: int(
            bundle.quality[column].sum()
        )
        for column in bundle.quality_columns
    }

    return {
        "schema_version": "1.0.0",
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
        },
        "row_count": bundle.row_count,
        "source_column_count": (
            bundle.source_column_count
        ),
        "feature_count": bundle.feature_count,
        "feature_names": list(
            bundle.feature_names
        ),
        "identifier_columns": list(
            bundle.identifier_columns
        ),
        "quality_columns": list(
            bundle.quality_columns
        ),
        "group_column": bundle.group_column,
        "target_column": bundle.target_column,
        "target_name_column": (
            bundle.target_name_column
        ),
        "class_mapping": (
            bundle.class_mapping
        ),
        "class_counts": class_counts,
        "subject_count": len(
            unique_subjects
        ),
        "subjects": unique_subjects,
        "rows_per_subject": (
            rows_per_subject
        ),
        "duplicate_identifier_rows": 0,
        "missing_cell_count": 0,
        "non_finite_feature_cell_count": 0,
        "quality_issue_counts": (
            quality_issue_counts
        ),
        "feature_scaling_applied": False,
        "random_epoch_split_allowed": False,
    }


def write_dataset_summary(
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        summary,
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and load the leakage-safe "
            "Phase 3 model input."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )

    parser.add_argument(
        "--write-summary",
        action="store_true",
    )

    parser.add_argument(
        "--allow-quality-issues",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    bundle = load_phase3_dataset(
        data_path=arguments.data,
        schema_path=arguments.schema,
        protocol_path=arguments.protocol,
        reject_quality_issues=(
            not arguments.allow_quality_issues
        ),
    )

    print("=== PHASE 3 DATASET CONTRACT ===")
    print("Rows:", bundle.row_count)
    print(
        "Source columns:",
        bundle.source_column_count,
    )
    print(
        "Model features:",
        bundle.feature_count,
    )
    print(
        "Subjects:",
        sorted(
            set(bundle.groups.tolist())
        ),
    )
    print(
        "Classes:",
        bundle.class_mapping,
    )
    print(
        "Identifier columns:",
        list(bundle.identifier_columns),
    )
    print(
        "Quality columns:",
        list(bundle.quality_columns),
    )
    print(
        "Feature matrix shape:",
        bundle.X.shape,
    )
    print(
        "Dataset contract validation: PASS"
    )

    if arguments.write_summary:
        summary = build_dataset_summary(
            bundle
        )

        write_dataset_summary(
            summary=summary,
            output_path=(
                arguments.summary_output.resolve()
            ),
        )

        print(
            "Summary:",
            relative_display_path(
                arguments.summary_output
            ),
        )
        print(
            "Dataset summary write: PASS"
        )


if __name__ == "__main__":
    main()
