from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import tempfile
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
    Phase3DatasetBundle,
    select_subject_partition,
)
from scripts.phase3_inner_search import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    normalize_subject_list,
)
from scripts.phase3_metrics import (
    align_probability_columns,
    write_metrics_json,
)
from scripts.phase3_model_registry import (
    DEFAULT_CONFIG_PATH as DEFAULT_REGISTRY_PATH,
    build_model_pipeline,
)
from scripts.phase3_outer_evaluation import (
    load_default_inputs,
    relative_display_path,
    selected_outer_folds,
    sha256_file,
    validate_outer_evaluation_inputs,
)
from scripts.phase3_selection_artifacts import (
    DEFAULT_OUTPUT_JSON_PATH as DEFAULT_SELECTION_PATH,
)


DEFAULT_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "phase3_local_outer"
)

DEFAULT_MANIFEST_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_trained_model_manifest.json"
)

DEFAULT_MANIFEST_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_trained_model_manifest.csv"
)


ROUNDTRIP_PROBABILITY_ATOL = 1e-15


MANIFEST_COLUMNS = [
    "outer_fold",
    "training_subjects",
    "excluded_test_subjects",
    "model_name",
    "candidate_id",
    "candidate_parameters_json",
    "training_row_count",
    "feature_count",
    "model_file_path",
    "model_file_size_bytes",
    "model_file_sha256",
    "reload_prediction_match",
    "reload_probability_match",
    "deployment_ready",
]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def safe_filename(value: str) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value.strip(),
    )

    normalized = normalized.strip("._")

    if not normalized:
        raise ValueError(
            "Model artifact filename is empty."
        )

    return normalized


def runtime_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def resolve_artifact_path(
    display_path: str,
) -> Path:
    path = Path(display_path)

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def build_source_metadata(
    bundle: Phase3DatasetBundle,
    split_manifest_path: Path,
    selection_artifact_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    return {
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
                split_manifest_path.resolve()
            )
        ),
        "model_registry_path": (
            relative_display_path(
                registry_path
            )
        ),
        "model_registry_sha256": (
            sha256_file(
                registry_path.resolve()
            )
        ),
        "selection_artifact_path": (
            relative_display_path(
                selection_artifact_path
            )
        ),
        "selection_artifact_sha256": (
            sha256_file(
                selection_artifact_path.resolve()
            )
        ),
    }


def build_model_metadata(
    bundle: Phase3DatasetBundle,
    outer_result: Mapping[str, Any],
    training_row_count: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    outer_fold = int(
        outer_result["outer_fold"]
    )

    training_subjects = (
        normalize_subject_list(
            outer_result[
                "outer_development_subjects"
            ],
            "training subjects",
        )
    )

    excluded_test_subjects = (
        normalize_subject_list(
            outer_result["test_subjects"],
            "excluded test subjects",
        )
    )

    if set(training_subjects) & set(
        excluded_test_subjects
    ):
        raise ValueError(
            "Training and excluded test "
            "subjects overlap."
        )

    selected_candidate = outer_result[
        "selected_candidate"
    ]

    return {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_trained_outer_pipeline"
        ),
        "intended_use": (
            "local_engineering_reproducibility"
        ),
        "deployment_ready": False,
        "deployment_block_reason": (
            "This outer-fold model excludes its "
            "held-out test subject and is retained "
            "for evaluation reproducibility only."
        ),
        "outer_fold": outer_fold,
        "training_scope": (
            "outer_development_subjects_only"
        ),
        "training_subjects": list(
            training_subjects
        ),
        "excluded_test_subjects": list(
            excluded_test_subjects
        ),
        "training_row_count": int(
            training_row_count
        ),
        "test_feature_matrix_loaded": False,
        "test_predictions_loaded": False,
        "outer_test_metrics_loaded": False,
        "hyperparameter_search_performed": False,
        "model_name": str(
            selected_candidate["model_name"]
        ),
        "candidate_id": str(
            selected_candidate["candidate_id"]
        ),
        "candidate_parameters": dict(
            selected_candidate[
                "candidate_parameters"
            ]
        ),
        "selection_validation_summary": dict(
            selected_candidate["aggregate"]
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
        "source": dict(source),
        "runtime": runtime_metadata(),
    }


def validate_model_payload(
    payload: Mapping[str, Any],
    expected_feature_names: (
        Sequence[str] | None
    ) = None,
    expected_class_mapping: (
        Mapping[str, int] | None
    ) = None,
) -> None:
    if set(payload) != {
        "metadata",
        "pipeline",
    }:
        raise ValueError(
            "Model payload must contain exactly "
            "metadata and pipeline."
        )

    metadata = payload["metadata"]
    pipeline = payload["pipeline"]

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise TypeError(
            "Model metadata must be a mapping."
        )

    if metadata.get(
        "artifact_type"
    ) != "phase3_trained_outer_pipeline":
        raise ValueError(
            "Unexpected trained-model "
            "artifact type."
        )

    if bool(
        metadata.get("deployment_ready")
    ):
        raise ValueError(
            "Outer-fold evaluation model "
            "cannot be deployment-ready."
        )

    if bool(
        metadata.get(
            "test_feature_matrix_loaded"
        )
    ):
        raise ValueError(
            "Test feature matrix entered "
            "model persistence."
        )

    if bool(
        metadata.get(
            "hyperparameter_search_performed"
        )
    ):
        raise ValueError(
            "Hyperparameter search entered "
            "model persistence."
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
            "Feature count does not match "
            "the feature-name contract."
        )

    if len(set(feature_names)) != len(
        feature_names
    ):
        raise ValueError(
            "Duplicate feature names found "
            "in model metadata."
        )

    if expected_feature_names is not None:
        if feature_names != tuple(
            expected_feature_names
        ):
            raise ValueError(
                "Model feature order does not "
                "match the expected contract."
            )

    class_mapping = {
        str(name): int(encoded)
        for name, encoded
        in metadata["class_mapping"].items()
    }

    if expected_class_mapping is not None:
        normalized_expected = {
            str(name): int(encoded)
            for name, encoded
            in expected_class_mapping.items()
        }

        if class_mapping != (
            normalized_expected
        ):
            raise ValueError(
                "Model class mapping does not "
                "match the expected contract."
            )

    if not hasattr(
        pipeline,
        "predict",
    ):
        raise TypeError(
            "Loaded pipeline has no predict method."
        )

    if not hasattr(
        pipeline,
        "predict_proba",
    ):
        raise TypeError(
            "Loaded pipeline has no "
            "predict_proba method."
        )


def atomic_joblib_dump(
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        joblib.dump(
            dict(payload),
            temporary_path,
            compress=3,
        )

        temporary_path.replace(
            output_path
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_trained_model_payload(
    path: Path,
) -> dict[str, Any]:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    payload = joblib.load(path)

    if not isinstance(payload, dict):
        raise TypeError(
            "Loaded joblib payload must "
            "be a dictionary."
        )

    validate_model_payload(payload)

    return payload


def verify_roundtrip(
    original_pipeline: Any,
    loaded_pipeline: Any,
    X: pd.DataFrame,
    class_mapping: Mapping[str, int],
) -> tuple[bool, bool]:
    if X.empty:
        raise ValueError(
            "Roundtrip verification input "
            "is empty."
        )

    original_predictions = (
        original_pipeline.predict(X)
    )

    loaded_predictions = (
        loaded_pipeline.predict(X)
    )

    prediction_match = bool(
        np.array_equal(
            original_predictions,
            loaded_predictions,
        )
    )

    original_classifier = (
        original_pipeline.named_steps[
            "classifier"
        ]
    )

    loaded_classifier = (
        loaded_pipeline.named_steps[
            "classifier"
        ]
    )

    original_probabilities = (
        align_probability_columns(
            probabilities=(
                original_pipeline.predict_proba(
                    X
                )
            ),
            estimator_classes=(
                original_classifier.classes_
            ),
            class_mapping=class_mapping,
        )
    )

    loaded_probabilities = (
        align_probability_columns(
            probabilities=(
                loaded_pipeline.predict_proba(
                    X
                )
            ),
            estimator_classes=(
                loaded_classifier.classes_
            ),
            class_mapping=class_mapping,
        )
    )

    probability_match = bool(
        np.allclose(
            original_probabilities,
            loaded_probabilities,
            rtol=0.0,
            atol=ROUNDTRIP_PROBABILITY_ATOL,
        )
    )

    return (
        prediction_match,
        probability_match,
    )


def train_and_save_outer_models(
    bundle: Phase3DatasetBundle,
    split_manifest: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    split_manifest_path: Path,
    selection_artifact_path: Path,
    registry_path: Path,
    output_directory: Path,
    outer_folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    (
        grouped_splits,
        selection_by_outer,
    ) = validate_outer_evaluation_inputs(
        bundle=bundle,
        split_manifest=split_manifest,
        selection_artifact=(
            selection_artifact
        ),
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

    source = build_source_metadata(
        bundle=bundle,
        split_manifest_path=(
            split_manifest_path
        ),
        selection_artifact_path=(
            selection_artifact_path
        ),
        registry_path=registry_path,
    )

    output_directory = (
        output_directory.resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_records = []

    for outer_fold in selected_folds:
        outer_result = selection_by_outer[
            outer_fold
        ]

        training_subjects = (
            normalize_subject_list(
                outer_result[
                    "outer_development_subjects"
                ],
                "training subjects",
            )
        )

        excluded_test_subjects = (
            normalize_subject_list(
                outer_result[
                    "test_subjects"
                ],
                "excluded test subjects",
            )
        )

        if set(training_subjects) & set(
            excluded_test_subjects
        ):
            raise ValueError(
                f"Outer fold {outer_fold} "
                "contains subject leakage."
            )

        development = (
            select_subject_partition(
                bundle=bundle,
                subjects=training_subjects,
                name=(
                    "outer_development_"
                    "model_fit"
                ),
                require_all_classes=True,
            )
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

        metadata = build_model_metadata(
            bundle=bundle,
            outer_result=outer_result,
            training_row_count=(
                development.row_count
            ),
            source=source,
        )

        payload = {
            "metadata": metadata,
            "pipeline": pipeline,
        }

        validate_model_payload(
            payload=payload,
            expected_feature_names=(
                bundle.feature_names
            ),
            expected_class_mapping=(
                bundle.class_mapping
            ),
        )

        filename = (
            f"outer_fold_{outer_fold:02d}"
            f"__{safe_filename(model_name)}"
            f"__{safe_filename(candidate_id)}"
            ".joblib"
        )

        model_path = (
            output_directory
            / filename
        )

        atomic_joblib_dump(
            payload=payload,
            output_path=model_path,
        )

        loaded_payload = (
            load_trained_model_payload(
                model_path
            )
        )

        validate_model_payload(
            payload=loaded_payload,
            expected_feature_names=(
                bundle.feature_names
            ),
            expected_class_mapping=(
                bundle.class_mapping
            ),
        )

        verification_count = min(
            64,
            development.row_count,
        )

        verification_X = (
            development.X.iloc[
                :verification_count
            ].copy()
        )

        (
            prediction_match,
            probability_match,
        ) = verify_roundtrip(
            original_pipeline=pipeline,
            loaded_pipeline=(
                loaded_payload["pipeline"]
            ),
            X=verification_X,
            class_mapping=(
                bundle.class_mapping
            ),
        )

        if not prediction_match:
            raise ValueError(
                f"Outer fold {outer_fold} "
                "prediction roundtrip failed."
            )

        if not probability_match:
            raise ValueError(
                f"Outer fold {outer_fold} "
                "probability roundtrip failed."
            )

        model_records.append(
            {
                "outer_fold": outer_fold,
                "training_subjects": list(
                    training_subjects
                ),
                "excluded_test_subjects": list(
                    excluded_test_subjects
                ),
                "model_name": model_name,
                "candidate_id": candidate_id,
                "candidate_parameters": (
                    candidate_parameters
                ),
                "training_row_count": int(
                    development.row_count
                ),
                "feature_count": len(
                    bundle.feature_names
                ),
                "model_file_path": (
                    relative_display_path(
                        model_path
                    )
                ),
                "model_file_size_bytes": int(
                    model_path.stat().st_size
                ),
                "model_file_sha256": (
                    sha256_file(model_path)
                ),
                "verification_row_count": (
                    verification_count
                ),
                "reload_prediction_match": (
                    prediction_match
                ),
                "reload_probability_match": (
                    probability_match
                ),
                "deployment_ready": False,
            }
        )

    complete_model_set = (
        set(selected_folds)
        == set(grouped_splits)
    )

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": (
            "phase3_local_trained_model_manifest"
        ),
        "intended_use": (
            "local_engineering_reproducibility"
        ),
        "scientific_reporting": {
            "allowed": False,
            "reason": (
                "The local dataset contains only "
                "four subjects and remains an "
                "engineering validation dataset."
            ),
        },
        "deployment": {
            "ready": False,
            "reason": (
                "These are outer-fold evaluation "
                "models. Each model excludes one "
                "held-out subject."
            ),
        },
        "persistence_contract": {
            "selection_artifact_frozen": True,
            "hyperparameter_search_performed": (
                False
            ),
            "test_feature_matrix_loaded": False,
            "test_predictions_loaded": False,
            "outer_test_metrics_loaded": False,
            "roundtrip_verification_partition": (
                "outer_development"
            ),
            "roundtrip_probability_atol": (
                ROUNDTRIP_PROBABILITY_ATOL
            ),
            "joblib_files_must_be_trusted": True,
        },
        "source": source,
        "runtime": runtime_metadata(),
        "evaluated_outer_folds": list(
            selected_folds
        ),
        "complete_model_set": (
            complete_model_set
        ),
        "model_count": len(
            model_records
        ),
        "models": sorted(
            model_records,
            key=lambda item: int(
                item["outer_fold"]
            ),
        ),
    }

    return manifest


def build_manifest_frame(
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []

    for model in manifest["models"]:
        rows.append(
            {
                "outer_fold": int(
                    model["outer_fold"]
                ),
                "training_subjects": ",".join(
                    str(value)
                    for value in model[
                        "training_subjects"
                    ]
                ),
                "excluded_test_subjects": (
                    ",".join(
                        str(value)
                        for value in model[
                            "excluded_test_subjects"
                        ]
                    )
                ),
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
                "training_row_count": int(
                    model[
                        "training_row_count"
                    ]
                ),
                "feature_count": int(
                    model["feature_count"]
                ),
                "model_file_path": str(
                    model["model_file_path"]
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
                "reload_prediction_match": bool(
                    model[
                        "reload_prediction_match"
                    ]
                ),
                "reload_probability_match": bool(
                    model[
                        "reload_probability_match"
                    ]
                ),
                "deployment_ready": bool(
                    model["deployment_ready"]
                ),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=MANIFEST_COLUMNS,
    )

    if frame.empty:
        raise ValueError(
            "Model manifest frame is empty."
        )

    if frame[
        "outer_fold"
    ].duplicated().any():
        raise ValueError(
            "Duplicate outer fold found "
            "in model manifest."
        )

    if frame[
        "model_file_path"
    ].duplicated().any():
        raise ValueError(
            "Duplicate model path found "
            "in model manifest."
        )

    return frame.sort_values(
        "outer_fold",
        kind="mergesort",
    ).reset_index(drop=True)


def write_model_manifest(
    manifest: Mapping[str, Any],
    json_output_path: Path,
    csv_output_path: Path,
) -> None:
    write_metrics_json(
        metrics=manifest,
        output_path=json_output_path,
    )

    frame = build_manifest_frame(
        manifest
    )

    csv_output_path = (
        csv_output_path.resolve()
    )

    csv_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        csv_output_path,
        index=False,
        lineterminator="\n",
    )


def smoke_test() -> None:
    (
        bundle,
        split_manifest,
        selection_artifact,
        registry,
    ) = load_default_inputs()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        model_directory = (
            root / "models"
        )

        manifest = (
            train_and_save_outer_models(
                bundle=bundle,
                split_manifest=(
                    split_manifest
                ),
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
                output_directory=(
                    model_directory
                ),
                outer_folds=[1],
            )
        )

        json_path = (
            root / "manifest.json"
        )

        csv_path = (
            root / "manifest.csv"
        )

        write_model_manifest(
            manifest=manifest,
            json_output_path=json_path,
            csv_output_path=csv_path,
        )

        first_json = json_path.read_bytes()
        first_csv = csv_path.read_bytes()

        write_model_manifest(
            manifest=manifest,
            json_output_path=json_path,
            csv_output_path=csv_path,
        )

        if first_json != (
            json_path.read_bytes()
        ):
            raise ValueError(
                "Model manifest JSON is "
                "not deterministic."
            )

        if first_csv != (
            csv_path.read_bytes()
        ):
            raise ValueError(
                "Model manifest CSV is "
                "not deterministic."
            )

        model = manifest["models"][0]

        model_path = resolve_artifact_path(
            model["model_file_path"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        print(
            "=== PHASE 3 TRAINED MODEL "
            "ARTIFACT SMOKE TEST ==="
        )
        print(
            "Outer fold:",
            model["outer_fold"],
        )
        print(
            "Training subjects:",
            model["training_subjects"],
        )
        print(
            "Excluded test subjects:",
            model[
                "excluded_test_subjects"
            ],
        )
        print(
            "Selected model:",
            model["model_name"],
        )
        print(
            "Selected candidate:",
            model["candidate_id"],
        )
        print(
            "Training rows:",
            model[
                "training_row_count"
            ],
        )
        print(
            "Model size bytes:",
            model[
                "model_file_size_bytes"
            ],
        )
        print(
            "Test feature matrix loaded:",
            manifest[
                "persistence_contract"
            ][
                "test_feature_matrix_loaded"
            ],
        )
        print(
            "Reload prediction match:",
            model[
                "reload_prediction_match"
            ],
        )
        print(
            "Reload probability match:",
            model[
                "reload_probability_match"
            ],
        )
        print(
            "Deployment ready:",
            model[
                "deployment_ready"
            ],
        )
        print(
            "Byte-deterministic manifest: PASS"
        )
        print(
            "Trained model artifact "
            "smoke test: PASS"
        )


def run_full_local() -> None:
    (
        bundle,
        split_manifest,
        selection_artifact,
        registry,
    ) = load_default_inputs()

    manifest = train_and_save_outer_models(
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
        output_directory=(
            DEFAULT_MODEL_DIRECTORY
        ),
    )

    write_model_manifest(
        manifest=manifest,
        json_output_path=(
            DEFAULT_MANIFEST_JSON_PATH
        ),
        csv_output_path=(
            DEFAULT_MANIFEST_CSV_PATH
        ),
    )

    frame = build_manifest_frame(
        manifest
    )

    print(
        "=== PHASE 3 FULL LOCAL "
        "TRAINED MODEL ARTIFACTS ==="
    )

    for _, row in frame.iterrows():
        print(
            "Outer fold",
            int(row["outer_fold"]),
            "| model=",
            row["model_name"],
            "| candidate=",
            row["candidate_id"],
            "| size_bytes=",
            int(
                row[
                    "model_file_size_bytes"
                ]
            ),
        )

    print(
        "Model count:",
        manifest["model_count"],
    )
    print(
        "Complete model set:",
        manifest[
            "complete_model_set"
        ],
    )
    print(
        "Deployment ready:",
        manifest[
            "deployment"
        ]["ready"],
    )
    print(
        "Manifest JSON:",
        relative_display_path(
            DEFAULT_MANIFEST_JSON_PATH
        ),
    )
    print(
        "Manifest CSV:",
        relative_display_path(
            DEFAULT_MANIFEST_CSV_PATH
        ),
    )
    print(
        "Full local trained model "
        "artifacts: PASS"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist frozen-selection Phase 3 "
            "outer-fold pipelines."
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
