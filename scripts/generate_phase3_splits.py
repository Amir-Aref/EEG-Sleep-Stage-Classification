from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_edfx_model_input.csv"
)

DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_evaluation_protocol.json"
)

DEFAULT_JSON_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_split_manifest.json"
)

DEFAULT_CSV_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_split_manifest.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def to_python_scalar(value: Any) -> Any:
    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    return value


def sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted(
        {
            to_python_scalar(value)
            for value in values
        }
    )


def generate_nested_logo_splits(
    subjects: Iterable[Any],
) -> list[dict[str, Any]]:
    ordered_subjects = sorted_unique(subjects)

    if len(ordered_subjects) != 4:
        raise ValueError(
            "The local development protocol requires "
            f"exactly four subjects; found "
            f"{len(ordered_subjects)}."
        )

    splits: list[dict[str, Any]] = []

    for outer_index, test_subject in enumerate(
        ordered_subjects,
        start=1,
    ):
        outer_development_subjects = [
            subject
            for subject in ordered_subjects
            if subject != test_subject
        ]

        for inner_index, validation_subject in enumerate(
            outer_development_subjects,
            start=1,
        ):
            train_subjects = [
                subject
                for subject in outer_development_subjects
                if subject != validation_subject
            ]

            splits.append(
                {
                    "split_id": (
                        f"outer_{outer_index:02d}"
                        f"_inner_{inner_index:02d}"
                    ),
                    "outer_fold": outer_index,
                    "inner_fold": inner_index,
                    "outer_development_subjects": (
                        outer_development_subjects
                    ),
                    "train_subjects": train_subjects,
                    "validation_subjects": [
                        validation_subject
                    ],
                    "test_subjects": [
                        test_subject
                    ],
                }
            )

    validate_split_assignments(
        splits=splits,
        all_subjects=ordered_subjects,
    )

    return splits


def validate_split_assignments(
    splits: list[dict[str, Any]],
    all_subjects: list[Any],
) -> None:
    if len(splits) != 12:
        raise ValueError(
            f"Expected 12 local splits; found {len(splits)}."
        )

    split_ids = [
        split["split_id"]
        for split in splits
    ]

    if len(split_ids) != len(set(split_ids)):
        raise ValueError("Split identifiers are not unique.")

    expected_subjects = set(all_subjects)

    test_counter: Counter[Any] = Counter()
    validation_counter: Counter[Any] = Counter()
    train_counter: Counter[Any] = Counter()

    for split in splits:
        train = set(split["train_subjects"])
        validation = set(
            split["validation_subjects"]
        )
        test = set(split["test_subjects"])
        outer_development = set(
            split["outer_development_subjects"]
        )

        if len(train) != 2:
            raise ValueError(
                f"{split['split_id']} does not have "
                "exactly two training subjects."
            )

        if len(validation) != 1:
            raise ValueError(
                f"{split['split_id']} does not have "
                "exactly one validation subject."
            )

        if len(test) != 1:
            raise ValueError(
                f"{split['split_id']} does not have "
                "exactly one test subject."
            )

        if train & validation:
            raise ValueError(
                f"Train/validation overlap in "
                f"{split['split_id']}."
            )

        if train & test:
            raise ValueError(
                f"Train/test overlap in "
                f"{split['split_id']}."
            )

        if validation & test:
            raise ValueError(
                f"Validation/test overlap in "
                f"{split['split_id']}."
            )

        if train | validation | test != expected_subjects:
            raise ValueError(
                f"Incomplete subject coverage in "
                f"{split['split_id']}."
            )

        if outer_development != train | validation:
            raise ValueError(
                f"Outer development subjects are invalid in "
                f"{split['split_id']}."
            )

        test_counter.update(test)
        validation_counter.update(validation)
        train_counter.update(train)

    for subject in all_subjects:
        if test_counter[subject] != 3:
            raise ValueError(
                f"Subject {subject} appears "
                f"{test_counter[subject]} times as test; "
                "expected 3."
            )

        if validation_counter[subject] != 3:
            raise ValueError(
                f"Subject {subject} appears "
                f"{validation_counter[subject]} times as "
                "validation; expected 3."
            )

        if train_counter[subject] != 6:
            raise ValueError(
                f"Subject {subject} appears "
                f"{train_counter[subject]} times as train; "
                "expected 6."
            )


def partition_summary(
    dataframe: pd.DataFrame,
    group_column: str,
    target_name_column: str,
    class_order: list[str],
    subjects: list[Any],
) -> dict[str, Any]:
    partition = dataframe[
        dataframe[group_column].isin(subjects)
    ]

    counts = (
        partition[target_name_column]
        .value_counts()
        .reindex(
            class_order,
            fill_value=0,
        )
    )

    class_counts = {
        stage: int(counts.loc[stage])
        for stage in class_order
    }

    missing_classes = [
        stage
        for stage, count in class_counts.items()
        if count == 0
    ]

    return {
        "subjects": subjects,
        "row_count": int(len(partition)),
        "class_counts": class_counts,
        "missing_classes": missing_classes,
    }


def build_split_manifest(
    dataframe: pd.DataFrame,
    protocol: dict[str, Any],
    data_sha256: str,
    protocol_sha256: str,
    data_path: str,
    protocol_path: str,
) -> dict[str, Any]:
    group_column = protocol["group_column"]
    target_column = protocol["target_column"]
    target_name_column = protocol[
        "target_name_column"
    ]
    class_mapping = protocol["class_mapping"]

    required_columns = {
        group_column,
        target_column,
        target_name_column,
    }

    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if dataframe[
        [group_column, target_column, target_name_column]
    ].isna().any().any():
        raise ValueError(
            "Group or target columns contain missing values."
        )

    observed_mapping_frame = (
        dataframe[
            [target_name_column, target_column]
        ]
        .drop_duplicates()
        .sort_values(target_column)
    )

    observed_mapping = {
        str(row[target_name_column]): int(
            row[target_column]
        )
        for _, row in observed_mapping_frame.iterrows()
    }

    if observed_mapping != class_mapping:
        raise ValueError(
            "Dataset target mapping does not match the "
            "evaluation protocol."
        )

    subjects = sorted_unique(
        dataframe[group_column].tolist()
    )

    class_order = [
        stage
        for stage, _ in sorted(
            class_mapping.items(),
            key=lambda item: item[1],
        )
    ]

    assignments = generate_nested_logo_splits(
        subjects
    )

    manifest_splits: list[dict[str, Any]] = []

    for assignment in assignments:
        train_summary = partition_summary(
            dataframe=dataframe,
            group_column=group_column,
            target_name_column=target_name_column,
            class_order=class_order,
            subjects=assignment["train_subjects"],
        )

        validation_summary = partition_summary(
            dataframe=dataframe,
            group_column=group_column,
            target_name_column=target_name_column,
            class_order=class_order,
            subjects=assignment[
                "validation_subjects"
            ],
        )

        test_summary = partition_summary(
            dataframe=dataframe,
            group_column=group_column,
            target_name_column=target_name_column,
            class_order=class_order,
            subjects=assignment["test_subjects"],
        )

        total_rows = (
            train_summary["row_count"]
            + validation_summary["row_count"]
            + test_summary["row_count"]
        )

        if total_rows != len(dataframe):
            raise ValueError(
                f"Row coverage failure in "
                f"{assignment['split_id']}."
            )

        for partition_name, summary in (
            ("train", train_summary),
            ("validation", validation_summary),
            ("test", test_summary),
        ):
            if summary["missing_classes"]:
                raise ValueError(
                    f"{assignment['split_id']} "
                    f"{partition_name} partition is missing: "
                    f"{summary['missing_classes']}."
                )

        manifest_splits.append(
            {
                **assignment,
                "partitions": {
                    "train": train_summary,
                    "validation": validation_summary,
                    "test": test_summary,
                },
            }
        )

    return {
        "schema_version": "1.0.0",
        "strategy": protocol[
            "local_development_protocol"
        ]["strategy"],
        "intended_use": protocol[
            "local_development_protocol"
        ]["intended_use"],
        "final_scientific_reporting_allowed": (
            protocol[
                "local_development_protocol"
            ]["final_scientific_reporting_allowed"]
        ),
        "source": {
            "model_input_path": data_path,
            "model_input_sha256": data_sha256,
            "protocol_path": protocol_path,
            "protocol_sha256": protocol_sha256,
        },
        "group_column": group_column,
        "target_column": target_column,
        "target_name_column": target_name_column,
        "class_mapping": class_mapping,
        "class_order": class_order,
        "subject_count": len(subjects),
        "subjects": subjects,
        "row_count": int(len(dataframe)),
        "outer_fold_count": 4,
        "inner_fold_count_per_outer": 3,
        "total_split_count": len(manifest_splits),
        "splits": manifest_splits,
    }


def manifest_to_frame(
    manifest: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    class_order = manifest["class_order"]

    for split in manifest["splits"]:
        row: dict[str, Any] = {
            "split_id": split["split_id"],
            "outer_fold": split["outer_fold"],
            "inner_fold": split["inner_fold"],
            "outer_development_subjects": ",".join(
                map(
                    str,
                    split[
                        "outer_development_subjects"
                    ],
                )
            ),
            "train_subjects": ",".join(
                map(str, split["train_subjects"])
            ),
            "validation_subject": split[
                "validation_subjects"
            ][0],
            "test_subject": split[
                "test_subjects"
            ][0],
        }

        for partition_name in (
            "train",
            "validation",
            "test",
        ):
            partition = split["partitions"][
                partition_name
            ]

            row[
                f"{partition_name}_row_count"
            ] = partition["row_count"]

            row[
                f"{partition_name}_missing_classes"
            ] = ",".join(
                partition["missing_classes"]
            )

            for stage in class_order:
                row[
                    f"{partition_name}_{stage}_count"
                ] = partition["class_counts"][stage]

        rows.append(row)

    return pd.DataFrame(rows)


def write_manifest_outputs(
    manifest: dict[str, Any],
    json_output: Path,
    csv_output: Path,
) -> None:
    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_text = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"

    json_output.write_text(
        json_text,
        encoding="utf-8",
        newline="\n",
    )

    manifest_to_frame(
        manifest
    ).to_csv(
        csv_output,
        index=False,
        lineterminator="\n",
    )


def relative_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic nested subject-wise "
            "splits for local Phase 3 development."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write deterministic JSON and CSV manifests.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    data_path = arguments.data.resolve()
    protocol_path = arguments.protocol.resolve()

    if not data_path.exists():
        raise SystemExit(
            f"Model input does not exist: {data_path}"
        )

    if not protocol_path.exists():
        raise SystemExit(
            f"Evaluation protocol does not exist: "
            f"{protocol_path}"
        )

    dataframe = pd.read_csv(data_path)

    with protocol_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        protocol = json.load(file)

    manifest = build_split_manifest(
        dataframe=dataframe,
        protocol=protocol,
        data_sha256=sha256_file(data_path),
        protocol_sha256=sha256_file(
            protocol_path
        ),
        data_path=relative_display_path(
            data_path
        ),
        protocol_path=relative_display_path(
            protocol_path
        ),
    )

    print("=== PHASE 3 LOCAL SPLIT MANIFEST ===")
    print("Rows:", manifest["row_count"])
    print("Subjects:", manifest["subjects"])
    print(
        "Outer folds:",
        manifest["outer_fold_count"],
    )
    print(
        "Inner folds per outer:",
        manifest[
            "inner_fold_count_per_outer"
        ],
    )
    print(
        "Total splits:",
        manifest["total_split_count"],
    )

    for split in manifest["splits"]:
        print(
            split["split_id"],
            "| train=",
            split["train_subjects"],
            "| validation=",
            split["validation_subjects"],
            "| test=",
            split["test_subjects"],
        )

    if arguments.write:
        write_manifest_outputs(
            manifest=manifest,
            json_output=arguments.json_output.resolve(),
            csv_output=arguments.csv_output.resolve(),
        )

        print(
            "JSON:",
            relative_display_path(
                arguments.json_output.resolve()
            ),
        )
        print(
            "CSV:",
            relative_display_path(
                arguments.csv_output.resolve()
            ),
        )
        print("Split manifest write: PASS")
    else:
        print(
            "Validation only. Use --write to save outputs."
        )


if __name__ == "__main__":
    main()
