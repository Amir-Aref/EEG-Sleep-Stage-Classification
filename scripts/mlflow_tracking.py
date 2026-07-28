"""MLflow tracking and model-registry integration for Phase 3 artifacts.

The module is intentionally separate from training. It imports already
validated Phase 3 artifacts into an MLflow experiment without rerunning
hyperparameter search or touching held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "mlflow_tracking.json"

DEFAULT_LOCAL_SELECTION_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "phase3_local_inner_search_results.json"
)
DEFAULT_LOCAL_OUTER_EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "phase3_local_outer_evaluation.json"
)
DEFAULT_LOCAL_MODEL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_trained_model_manifest.json"
)
DEFAULT_LOCAL_FINAL_REFIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_local_final_refit_manifest.json"
)

SUPPORTED_SCOPE_NAMES = {"local_validation", "full_dataset"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
METRIC_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.\-/]+")
TAG_MAX_LENGTH = 5000
PARAM_MAX_LENGTH = 6000


@dataclass(frozen=True)
class TrackingPaths:
    root: Path
    backend_database: Path
    artifact_root: Path
    summary_path: Path
    tracking_uri: str
    artifact_uri: str


@dataclass(frozen=True)
class TrackingContext:
    mlflow: Any
    client: Any
    experiment_id: str
    config: Mapping[str, Any]
    paths: TrackingPaths


@dataclass(frozen=True)
class ImportInputs:
    scope: str
    selection_path: Path
    outer_evaluation_path: Path
    model_manifest_path: Path
    final_refit_manifest_path: Path


class MlflowDependencyError(RuntimeError):
    """Raised when MLflow is required but not installed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(serialized)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return payload


def require_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        raise MlflowDependencyError(
            "MLflow is not installed. Run: python -m pip install -r requirements.txt"
        ) from error


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "mlflow": package_version("mlflow"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "joblib": package_version("joblib"),
    }


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be an object.")
    return value


def load_tracking_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_json(path.resolve())
    if config.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported MLflow tracking configuration schema.")
    experiment_name = config.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must be a non-empty string.")

    tracking = _require_mapping(config, "tracking")
    for key in (
        "root_directory",
        "backend_database",
        "artifact_directory",
        "summary_path",
    ):
        value = tracking.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tracking.{key} must be a non-empty string.")

    registered_models = _require_mapping(config, "registered_models")
    for scope in SUPPORTED_SCOPE_NAMES:
        value = registered_models.get(scope)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"registered_models.{scope} must be defined.")

    safety = _require_mapping(config, "safety")
    required_safety_keys = {
        "require_exact_model_runtime",
        "allow_local_scientific_reporting",
        "register_local_validation_model",
        "promote_full_model_alias",
    }
    missing_safety_keys = sorted(required_safety_keys - set(safety))
    if missing_safety_keys:
        raise ValueError(f"Missing safety configuration keys: {missing_safety_keys}")
    return config


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def sqlite_tracking_uri(database_path: Path) -> str:
    normalized = database_path.resolve().as_posix()
    return f"sqlite:///{normalized}"


def resolve_tracking_paths(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    tracking_root_override: str | Path | None = None,
) -> TrackingPaths:
    tracking = _require_mapping(config, "tracking")
    configured_override = tracking_root_override or os.environ.get("EEG_MLFLOW_ROOT")
    root_value = configured_override or str(tracking["root_directory"])
    root = resolve_project_path(root_value, project_root)
    backend_database = root / str(tracking["backend_database"])
    artifact_root = root / str(tracking["artifact_directory"])
    summary_path = resolve_project_path(str(tracking["summary_path"]), project_root)
    return TrackingPaths(
        root=root,
        backend_database=backend_database.resolve(),
        artifact_root=artifact_root.resolve(),
        summary_path=summary_path.resolve(),
        tracking_uri=sqlite_tracking_uri(backend_database),
        artifact_uri=artifact_root.resolve().as_uri(),
    )


def initialize_tracking(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
    tracking_root_override: str | Path | None = None,
) -> TrackingContext:
    config = load_tracking_config(config_path)
    paths = resolve_tracking_paths(
        config,
        project_root=project_root,
        tracking_root_override=tracking_root_override,
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.artifact_root.mkdir(parents=True, exist_ok=True)
    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow = require_mlflow()
    mlflow.set_tracking_uri(paths.tracking_uri)
    mlflow.set_registry_uri(paths.tracking_uri)

    client_module = importlib.import_module("mlflow.tracking")
    client = client_module.MlflowClient()
    experiment_name = str(config["experiment_name"])
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=paths.artifact_uri,
        )
    else:
        experiment_id = str(experiment.experiment_id)
        existing_location = str(experiment.artifact_location).rstrip("/")
        expected_location = paths.artifact_uri.rstrip("/")
        if existing_location != expected_location:
            raise ValueError(
                "Existing MLflow experiment uses a different artifact location: "
                f"{existing_location} != {expected_location}"
            )
    mlflow.set_experiment(experiment_name)
    return TrackingContext(
        mlflow=mlflow,
        client=client,
        experiment_id=str(experiment_id),
        config=config,
        paths=paths,
    )


def detect_git_commit(project_root: Path = PROJECT_ROOT) -> str | None:
    for variable_name in ("GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        value = os.environ.get(variable_name)
        if value and value.strip():
            return value.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def sanitize_metric_name(name: str) -> str:
    sanitized = METRIC_NAME_PATTERN.sub("_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_.")
    if not sanitized:
        raise ValueError("Metric name became empty after sanitization.")
    return sanitized


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    flattened: dict[str, float] = {}

    def visit(value: Any, key_path: list[str]) -> None:
        if _is_numeric_scalar(value):
            numeric = float(value)
            if math.isfinite(numeric):
                key = sanitize_metric_name(".".join(key_path))
                flattened[key] = numeric
            return
        if isinstance(value, Mapping):
            for key in sorted(value):
                if key in {"confusion_matrix", "class_mapping"}:
                    continue
                visit(value[key], [*key_path, str(key)])
            return
        if isinstance(value, list) and value and all(
            isinstance(item, Mapping) for item in value
        ):
            for index, item in enumerate(value):
                label = item.get("class_name", item.get("name", index))
                for key in sorted(item):
                    if key in {"class_name", "class_encoded", "name"}:
                        continue
                    visit(item[key], [*key_path, str(label), str(key)])

    starting_prefix = [prefix] if prefix else []
    visit(payload, starting_prefix)
    return flattened


def flatten_parameters(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str]:
    flattened: dict[str, str] = {}

    def visit(value: Any, key_path: list[str]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], [*key_path, str(key)])
            return
        key = ".".join(key_path)
        if isinstance(value, (list, tuple, set, dict)):
            rendered = canonical_json(value)
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        flattened[key] = rendered[:PARAM_MAX_LENGTH]

    starting_prefix = [prefix] if prefix else []
    visit(payload, starting_prefix)
    return flattened


def normalize_tags(tags: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)[:TAG_MAX_LENGTH]
        for key, value in tags.items()
        if value is not None
    }


def scientific_reporting_allowed(payload: Mapping[str, Any]) -> bool:
    direct = payload.get("scientific_reporting_allowed")
    if isinstance(direct, bool):
        return direct
    nested = payload.get("scientific_reporting")
    if isinstance(nested, Mapping) and isinstance(nested.get("allowed"), bool):
        return bool(nested["allowed"])
    return False


def validate_source_hashes(
    payload: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return []
    validations: list[dict[str, Any]] = []
    for path_key in sorted(key for key in source if key.endswith("_path")):
        hash_key = path_key[:-5] + "_sha256"
        if hash_key not in source:
            continue
        display_path = str(source[path_key])
        expected_hash = str(source[hash_key]).lower()
        if not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"Invalid SHA-256 value in source metadata: {hash_key}")
        resolved_path = resolve_project_path(display_path, project_root)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Source artifact referenced by {path_key} does not exist: {resolved_path}"
            )
        actual_hash = sha256_file(resolved_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source artifact hash mismatch for {display_path}: "
                f"{actual_hash} != {expected_hash}"
            )
        validations.append(
            {
                "path_key": path_key,
                "path": display_path,
                "sha256": actual_hash,
                "valid": True,
            }
        )
    return validations


def validate_model_manifest(
    manifest: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Model manifest must contain a non-empty models list.")
    if int(manifest.get("model_count", -1)) != len(models):
        raise ValueError("model_count does not match the models list length.")

    validations: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise TypeError(f"Model record {index} must be an object.")
        model_path = resolve_project_path(str(model["model_file_path"]), project_root)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        expected_size = int(model["model_file_size_bytes"])
        actual_size = model_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Model size mismatch for {model_path}: {actual_size} != {expected_size}"
            )
        expected_hash = str(model["model_file_sha256"]).lower()
        actual_hash = sha256_file(model_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Model hash mismatch for {model_path}: {actual_hash} != {expected_hash}"
            )
        if model.get("reload_prediction_match") is not True:
            raise ValueError(f"Prediction roundtrip was not validated for {model_path}.")
        if model.get("reload_probability_match") is not True:
            raise ValueError(f"Probability roundtrip was not validated for {model_path}.")
        validations.append(
            {
                "model_file_path": str(model["model_file_path"]),
                "model_file_sha256": actual_hash,
                "model_file_size_bytes": actual_size,
                "valid": True,
            }
        )
    return validations


def validate_scope_contracts(
    *,
    scope: str,
    selection: Mapping[str, Any],
    outer_evaluation: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    final_refit_manifest: Mapping[str, Any],
) -> None:
    if scope not in SUPPORTED_SCOPE_NAMES:
        raise ValueError(f"Unsupported MLflow import scope: {scope}")
    if selection.get("candidate_space_complete") is not True:
        raise ValueError("Candidate space must be complete before MLflow import.")

    selection_result = selection.get("selection_result")
    if isinstance(selection_result, Mapping):
        for forbidden_key in (
            "test_metrics_included",
            "test_predictions_included",
            "test_feature_matrix_loaded",
        ):
            if selection_result.get(forbidden_key) is True:
                raise ValueError(
                    f"Selection artifact violates test isolation: {forbidden_key}=true"
                )
    if selection.get("test_partition_used") is True:
        raise ValueError("Selection artifact reports test_partition_used=true.")

    if outer_evaluation.get("complete_outer_evaluation") is not True:
        raise ValueError("Outer evaluation must be complete before MLflow import.")
    if model_manifest.get("complete_model_set") is not True:
        raise ValueError("Outer model manifest must contain a complete model set.")

    selection_allowed = scientific_reporting_allowed(selection)
    outer_allowed = scientific_reporting_allowed(outer_evaluation)
    final_allowed = scientific_reporting_allowed(final_refit_manifest)

    if scope == "local_validation":
        if selection_allowed or outer_allowed or final_allowed:
            raise ValueError(
                "Local validation artifacts must not permit final scientific reporting."
            )
    else:
        if not outer_allowed or not final_allowed:
            raise ValueError(
                "Full-dataset import requires scientifically reportable outer and final artifacts."
            )
        deployment = final_refit_manifest.get("deployment")
        deployment_ready = (
            isinstance(deployment, Mapping)
            and deployment.get("deployment_ready") is True
        )
        if not deployment_ready:
            raise ValueError(
                "Full-dataset final refit must be deployment-ready before registration."
            )


def validate_runtime_compatibility(
    payload_metadata: Mapping[str, Any],
    *,
    allow_version_mismatch: bool,
) -> None:
    expected_runtime = payload_metadata.get("runtime")
    if not isinstance(expected_runtime, Mapping):
        raise ValueError("Model payload does not contain runtime metadata.")

    current_runtime = runtime_metadata()
    mismatches: list[str] = []

    expected_python = str(expected_runtime.get("python", ""))
    current_python = str(current_runtime.get("python", ""))
    if expected_python and current_python:
        expected_major_minor = ".".join(expected_python.split(".")[:2])
        current_major_minor = ".".join(current_python.split(".")[:2])
        if expected_major_minor != current_major_minor:
            mismatches.append(
                f"python {expected_python} != {current_python}"
            )

    # The serialized estimator is a scikit-learn pipeline. Python,
    # scikit-learn, NumPy, and joblib are therefore treated as strict
    # persistence dependencies. pandas is recorded for provenance but is not
    # a strict unpickling dependency for these fitted pipelines. This also
    # permits the MLflow-compatible delivery environment (pandas 2.x) to load
    # artifacts that were originally created with pandas 3.x.
    runtime_key_pairs = (
        ("scikit_learn", "scikit_learn"),
        ("numpy", "numpy"),
        ("joblib", "joblib"),
    )
    for expected_key, current_key in runtime_key_pairs:
        expected_value = str(expected_runtime.get(expected_key, ""))
        current_value = str(current_runtime.get(current_key, "") or "")
        if expected_value and expected_value != current_value:
            mismatches.append(
                f"{expected_key} {expected_value} != {current_value}"
            )

    if mismatches and not allow_version_mismatch:
        raise RuntimeError(
            "The saved model runtime does not match the current environment: "
            + "; ".join(mismatches)
        )


def load_model_payload(
    path: Path,
    *,
    allow_version_mismatch: bool = False,
) -> tuple[Mapping[str, Any], Any]:
    payload = joblib.load(path)
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "pipeline"}:
        raise ValueError(f"Unexpected model payload structure: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Model metadata is invalid: {path}")
    validate_runtime_compatibility(
        metadata,
        allow_version_mismatch=allow_version_mismatch,
    )
    pipeline = payload["pipeline"]
    if not hasattr(pipeline, "predict"):
        raise TypeError(f"Saved pipeline does not implement predict(): {path}")
    return metadata, pipeline


def resolve_model_input_path(
    model_metadata: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    source = model_metadata.get("source")
    if not isinstance(source, Mapping) or "model_input_path" not in source:
        raise ValueError("Model metadata does not reference model_input_path.")
    return resolve_project_path(str(source["model_input_path"]), project_root)


def build_input_example(
    model_metadata: Mapping[str, Any],
    *,
    project_root: Path,
    row_count: int = 8,
) -> pd.DataFrame:
    feature_names = model_metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Model metadata does not contain feature_names.")
    input_path = resolve_model_input_path(model_metadata, project_root=project_root)
    frame = pd.read_csv(input_path, usecols=[str(name) for name in feature_names], nrows=row_count)
    if frame.empty:
        raise ValueError("Model input example is empty.")
    return frame


def compute_import_fingerprint(paths: Sequence[Path], scope: str) -> str:
    components = [f"scope={scope}"]
    for path in sorted(path.resolve() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        components.append(f"{path.name}={sha256_file(path)}")
    return sha256_bytes("\n".join(components).encode("utf-8"))


def _run_tags(
    context: TrackingContext,
    *,
    scope: str,
    role: str,
    scientific_allowed: bool,
    git_commit: str | None,
    additional: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    tags: dict[str, Any] = dict(_require_mapping(context.config, "default_tags"))
    tags.update(
        {
            "eeg.scope": scope,
            "eeg.role": role,
            "eeg.scientific_reporting_allowed": str(scientific_allowed).lower(),
            "eeg.git_commit": git_commit or "unavailable",
        }
    )
    if additional:
        tags.update(additional)
    return normalize_tags(tags)


def _log_artifact_set(mlflow: Any, paths: Iterable[Path], artifact_path: str) -> None:
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_model(
    context: TrackingContext,
    *,
    pipeline: Any,
    input_example: pd.DataFrame,
    registered_model_name: str | None,
    model_metadata: Mapping[str, Any],
) -> Any:
    infer_signature = importlib.import_module("mlflow.models").infer_signature
    predictions = pipeline.predict(input_example)
    signature = infer_signature(input_example, predictions)
    model_tags = normalize_tags(
        {
            "eeg.scope": model_metadata.get("scope"),
            "eeg.model_name": model_metadata.get("model_name"),
            "eeg.candidate_id": model_metadata.get("candidate_id"),
            "eeg.scientific_reporting_allowed": model_metadata.get(
                "scientific_reporting_allowed"
            ),
            "eeg.deployment_ready": model_metadata.get("deployment_ready"),
        }
    )
    sklearn_flavor = importlib.import_module("mlflow.sklearn")
    return sklearn_flavor.log_model(
        sk_model=pipeline,
        name="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        serialization_format="cloudpickle",
        await_registration_for=300,
        metadata=dict(model_metadata),
        tags=model_tags,
    )


def _registered_versions_for_run(client: Any, run_id: str) -> list[Any]:
    return list(client.search_model_versions(f"run_id = '{run_id}'"))


def _tag_model_versions(
    context: TrackingContext,
    *,
    run_id: str,
    tags: Mapping[str, Any],
    alias: str | None = None,
) -> list[dict[str, str]]:
    versions = _registered_versions_for_run(context.client, run_id)
    results: list[dict[str, str]] = []
    normalized_tags = normalize_tags(tags)
    for version in versions:
        for key, value in normalized_tags.items():
            context.client.set_model_version_tag(
                name=version.name,
                version=version.version,
                key=key,
                value=value,
            )
        if alias:
            context.client.set_registered_model_alias(
                name=version.name,
                alias=alias,
                version=version.version,
            )
        results.append(
            {
                "name": str(version.name),
                "version": str(version.version),
                "run_id": str(version.run_id),
                "alias": alias or "",
            }
        )
    return results


def _find_existing_import_run(
    context: TrackingContext,
    *,
    fingerprint: str,
) -> Any | None:
    escaped = fingerprint.replace("'", "\\'")
    runs = context.client.search_runs(
        experiment_ids=[context.experiment_id],
        filter_string=f"tags.eeg.import_fingerprint = '{escaped}'",
        max_results=10,
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        if str(run.info.status) == "FINISHED":
            return run
    return None


def _outer_result_index(outer_evaluation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    results = outer_evaluation.get("outer_results")
    if not isinstance(results, list):
        raise ValueError("Outer evaluation does not contain outer_results.")
    index: dict[int, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("Outer evaluation result must be an object.")
        fold = int(result["outer_fold"])
        if fold in index:
            raise ValueError(f"Duplicate outer fold in evaluation artifact: {fold}")
        index[fold] = result
    return index


def _log_common_metadata(
    context: TrackingContext,
    *,
    inputs: ImportInputs,
    config_path: Path,
    project_root: Path,
) -> None:
    _log_artifact_set(
        context.mlflow,
        [
            config_path,
            inputs.selection_path,
            inputs.outer_evaluation_path,
            inputs.model_manifest_path,
            inputs.final_refit_manifest_path,
            project_root / "config" / "phase3_evaluation_protocol.json",
            project_root / "config" / "phase3_model_registry.json",
            project_root / "data" / "metadata" / "sleep_edfx_model_feature_schema.json",
        ],
        "metadata",
    )


def import_phase3_artifacts(
    *,
    inputs: ImportInputs,
    config_path: Path = DEFAULT_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
    tracking_root_override: str | Path | None = None,
    git_commit: str | None = None,
    force: bool = False,
    log_models: bool = True,
    allow_version_mismatch: bool = False,
) -> dict[str, Any]:
    selection = load_json(inputs.selection_path)
    outer_evaluation = load_json(inputs.outer_evaluation_path)
    model_manifest = load_json(inputs.model_manifest_path)
    final_refit_manifest = load_json(inputs.final_refit_manifest_path)

    validate_scope_contracts(
        scope=inputs.scope,
        selection=selection,
        outer_evaluation=outer_evaluation,
        model_manifest=model_manifest,
        final_refit_manifest=final_refit_manifest,
    )

    source_validation = {
        "selection": validate_source_hashes(selection, project_root=project_root),
        "outer_evaluation": validate_source_hashes(
            outer_evaluation, project_root=project_root
        ),
        "model_manifest": validate_source_hashes(
            model_manifest, project_root=project_root
        ),
        "final_refit_manifest": validate_source_hashes(
            final_refit_manifest, project_root=project_root
        ),
    }
    model_validation = {
        "outer_models": validate_model_manifest(
            model_manifest, project_root=project_root
        ),
        "final_refit": validate_model_manifest(
            final_refit_manifest, project_root=project_root
        ),
    }

    context = initialize_tracking(
        config_path=config_path,
        project_root=project_root,
        tracking_root_override=tracking_root_override,
    )
    fingerprint_scope = (
        f"{inputs.scope}:models={str(bool(log_models)).lower()}"
    )
    fingerprint = compute_import_fingerprint(
        [
            inputs.selection_path,
            inputs.outer_evaluation_path,
            inputs.model_manifest_path,
            inputs.final_refit_manifest_path,
        ],
        fingerprint_scope,
    )
    if not force:
        existing_run = _find_existing_import_run(context, fingerprint=fingerprint)
        if existing_run is not None:
            return {
                "schema_version": "1.0.0",
                "artifact_type": "mlflow_phase3_import_summary",
                "status": "already_imported",
                "scope": inputs.scope,
                "experiment_id": context.experiment_id,
                "parent_run_id": str(existing_run.info.run_id),
                "tracking_uri": context.paths.tracking_uri,
                "artifact_root": context.paths.artifact_uri,
                "import_fingerprint": fingerprint,
                "scientific_reporting_allowed": scientific_reporting_allowed(
                    outer_evaluation
                ),
            }

    resolved_commit = git_commit or detect_git_commit(project_root)
    scientific_allowed = scientific_reporting_allowed(outer_evaluation)
    outer_index = _outer_result_index(outer_evaluation)
    registered_models = _require_mapping(context.config, "registered_models")
    registered_model_name = str(registered_models[inputs.scope])
    safety = _require_mapping(context.config, "safety")
    register_local = bool(safety["register_local_validation_model"])
    effective_allow_version_mismatch = (
        allow_version_mismatch
        or not bool(safety["require_exact_model_runtime"])
    )

    parent_tags = _run_tags(
        context,
        scope=inputs.scope,
        role="phase3_artifact_import",
        scientific_allowed=scientific_allowed,
        git_commit=resolved_commit,
        additional={
            "eeg.import_fingerprint": fingerprint,
            "eeg.candidate_space_complete": selection.get(
                "candidate_space_complete"
            ),
        },
    )

    child_runs: list[dict[str, Any]] = []
    registered_versions: list[dict[str, str]] = []
    with context.mlflow.start_run(
        experiment_id=context.experiment_id,
        run_name=f"{inputs.scope}__phase3_artifact_import",
        tags=parent_tags,
    ) as parent_run:
        context.mlflow.log_params(
            {
                "scope": inputs.scope,
                "selection_artifact_sha256": sha256_file(inputs.selection_path),
                "outer_evaluation_sha256": sha256_file(
                    inputs.outer_evaluation_path
                ),
                "model_manifest_sha256": sha256_file(inputs.model_manifest_path),
                "final_refit_manifest_sha256": sha256_file(
                    inputs.final_refit_manifest_path
                ),
                "git_commit": resolved_commit or "unavailable",
            }
        )
        aggregate = outer_evaluation.get("aggregate")
        if isinstance(aggregate, Mapping):
            context.mlflow.log_metrics(flatten_metrics(aggregate, prefix="outer"))
        _log_common_metadata(
            context,
            inputs=inputs,
            config_path=config_path,
            project_root=project_root,
        )

        for model_record in model_manifest["models"]:
            outer_fold = int(model_record["outer_fold"])
            outer_result = outer_index.get(outer_fold)
            if outer_result is None:
                raise ValueError(
                    f"Outer model manifest references unevaluated fold {outer_fold}."
                )
            model_path = resolve_project_path(
                str(model_record["model_file_path"]), project_root
            )
            run_tags = _run_tags(
                context,
                scope=inputs.scope,
                role="outer_fold_model",
                scientific_allowed=scientific_allowed,
                git_commit=resolved_commit,
                additional={
                    "eeg.outer_fold": outer_fold,
                    "eeg.model_name": model_record["model_name"],
                    "eeg.candidate_id": model_record["candidate_id"],
                    "eeg.deployment_ready": model_record.get(
                        "deployment_ready", False
                    ),
                },
            )
            with context.mlflow.start_run(
                experiment_id=context.experiment_id,
                run_name=(
                    f"{inputs.scope}__outer_fold_{outer_fold:02d}__"
                    f"{model_record['model_name']}"
                ),
                nested=True,
                tags=run_tags,
            ) as child_run:
                parameters = flatten_parameters(
                    dict(model_record.get("candidate_parameters", {})),
                    prefix="candidate",
                )
                parameters.update(
                    {
                        "outer_fold": str(outer_fold),
                        "candidate_id": str(model_record["candidate_id"]),
                        "model_name": str(model_record["model_name"]),
                        "feature_count": str(model_record["feature_count"]),
                        "training_row_count": str(model_record["training_row_count"]),
                        "model_file_sha256": str(
                            model_record["model_file_sha256"]
                        ),
                    }
                )
                context.mlflow.log_params(parameters)
                metrics = outer_result.get("metrics")
                if isinstance(metrics, Mapping):
                    context.mlflow.log_metrics(
                        flatten_metrics(metrics, prefix="outer_test")
                    )
                selection_summary = outer_result.get("selection_validation_summary")
                if isinstance(selection_summary, Mapping):
                    context.mlflow.log_metrics(
                        flatten_metrics(selection_summary, prefix="inner_validation")
                    )
                context.mlflow.log_artifact(
                    str(model_path), artifact_path="source_model"
                )

                if log_models:
                    model_metadata, pipeline = load_model_payload(
                        model_path,
                        allow_version_mismatch=effective_allow_version_mismatch,
                    )
                    input_example = build_input_example(
                        model_metadata,
                        project_root=project_root,
                    )
                    should_register = (
                        inputs.scope == "full_dataset" or register_local
                    )
                    _log_model(
                        context,
                        pipeline=pipeline,
                        input_example=input_example,
                        registered_model_name=(
                            registered_model_name if should_register else None
                        ),
                        model_metadata={
                            "scope": inputs.scope,
                            "role": "outer_fold_model",
                            "outer_fold": outer_fold,
                            "model_name": model_record["model_name"],
                            "candidate_id": model_record["candidate_id"],
                            "scientific_reporting_allowed": scientific_allowed,
                            "deployment_ready": False,
                        },
                    )
                    if should_register:
                        registered_versions.extend(
                            _tag_model_versions(
                                context,
                                run_id=child_run.info.run_id,
                                tags={
                                    "eeg.scope": inputs.scope,
                                    "eeg.role": "outer_fold_model",
                                    "eeg.outer_fold": outer_fold,
                                    "eeg.scientific_reporting_allowed": scientific_allowed,
                                    "eeg.deployment_ready": False,
                                },
                            )
                        )
                child_runs.append(
                    {
                        "role": "outer_fold_model",
                        "outer_fold": outer_fold,
                        "run_id": child_run.info.run_id,
                        "model_name": str(model_record["model_name"]),
                        "candidate_id": str(model_record["candidate_id"]),
                    }
                )

        final_model_record = final_refit_manifest["models"][0]
        final_model_path = resolve_project_path(
            str(final_model_record["model_file_path"]), project_root
        )
        final_deployment_ready = bool(final_model_record.get("deployment_ready", False))
        final_tags = _run_tags(
            context,
            scope=inputs.scope,
            role="final_refit_model",
            scientific_allowed=scientific_allowed,
            git_commit=resolved_commit,
            additional={
                "eeg.model_name": final_model_record["model_name"],
                "eeg.candidate_id": final_model_record["candidate_id"],
                "eeg.deployment_ready": final_deployment_ready,
            },
        )
        with context.mlflow.start_run(
            experiment_id=context.experiment_id,
            run_name=f"{inputs.scope}__final_refit__{final_model_record['model_name']}",
            nested=True,
            tags=final_tags,
        ) as final_run:
            parameters = flatten_parameters(
                dict(final_model_record.get("candidate_parameters", {})),
                prefix="candidate",
            )
            parameters.update(
                {
                    "candidate_id": str(final_model_record["candidate_id"]),
                    "model_name": str(final_model_record["model_name"]),
                    "feature_count": str(final_model_record["feature_count"]),
                    "training_row_count": str(final_model_record["training_row_count"]),
                    "model_file_sha256": str(
                        final_model_record["model_file_sha256"]
                    ),
                    "deployment_ready": str(final_deployment_ready).lower(),
                }
            )
            context.mlflow.log_params(parameters)
            selection_validation_summary = final_model_record.get(
                "selection_validation_summary"
            )
            if isinstance(selection_validation_summary, Mapping):
                context.mlflow.log_metrics(
                    flatten_metrics(
                        selection_validation_summary,
                        prefix="selection_validation",
                    )
                )
            context.mlflow.log_artifact(
                str(final_model_path), artifact_path="source_model"
            )
            if log_models:
                model_metadata, pipeline = load_model_payload(
                    final_model_path,
                    allow_version_mismatch=effective_allow_version_mismatch,
                )
                input_example = build_input_example(
                    model_metadata,
                    project_root=project_root,
                )
                _log_model(
                    context,
                    pipeline=pipeline,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    model_metadata={
                        "scope": inputs.scope,
                        "role": "final_refit_model",
                        "model_name": final_model_record["model_name"],
                        "candidate_id": final_model_record["candidate_id"],
                        "scientific_reporting_allowed": scientific_allowed,
                        "deployment_ready": final_deployment_ready,
                    },
                )
                alias = None
                if (
                    inputs.scope == "full_dataset"
                    and scientific_allowed
                    and final_deployment_ready
                ):
                    alias = str(safety["promote_full_model_alias"])
                registered_versions.extend(
                    _tag_model_versions(
                        context,
                        run_id=final_run.info.run_id,
                        tags={
                            "eeg.scope": inputs.scope,
                            "eeg.role": "final_refit_model",
                            "eeg.scientific_reporting_allowed": scientific_allowed,
                            "eeg.deployment_ready": final_deployment_ready,
                        },
                        alias=alias,
                    )
                )
            child_runs.append(
                {
                    "role": "final_refit_model",
                    "run_id": final_run.info.run_id,
                    "model_name": str(final_model_record["model_name"]),
                    "candidate_id": str(final_model_record["candidate_id"]),
                }
            )

        summary = {
            "schema_version": "1.0.0",
            "artifact_type": "mlflow_phase3_import_summary",
            "status": "imported",
            "scope": inputs.scope,
            "experiment_name": str(context.config["experiment_name"]),
            "experiment_id": context.experiment_id,
            "parent_run_id": parent_run.info.run_id,
            "child_runs": child_runs,
            "registered_model_versions": registered_versions,
            "tracking_uri": context.paths.tracking_uri,
            "artifact_root": context.paths.artifact_uri,
            "tracking_root": str(context.paths.root),
            "summary_path": str(context.paths.summary_path),
            "import_fingerprint": fingerprint,
            "git_commit": resolved_commit,
            "scientific_reporting_allowed": scientific_allowed,
            "models_logged": bool(log_models),
            "source_validation": source_validation,
            "model_validation": model_validation,
            "runtime": runtime_metadata(),
        }
        atomic_write_json(context.paths.summary_path, summary)
        context.mlflow.log_artifact(
            str(context.paths.summary_path), artifact_path="metadata"
        )
    return summary


def local_import_inputs() -> ImportInputs:
    return ImportInputs(
        scope="local_validation",
        selection_path=DEFAULT_LOCAL_SELECTION_PATH,
        outer_evaluation_path=DEFAULT_LOCAL_OUTER_EVALUATION_PATH,
        model_manifest_path=DEFAULT_LOCAL_MODEL_MANIFEST_PATH,
        final_refit_manifest_path=DEFAULT_LOCAL_FINAL_REFIT_MANIFEST_PATH,
    )


def validate_local_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    inputs = local_import_inputs()
    selection = load_json(inputs.selection_path)
    outer_evaluation = load_json(inputs.outer_evaluation_path)
    model_manifest = load_json(inputs.model_manifest_path)
    final_refit_manifest = load_json(inputs.final_refit_manifest_path)
    validate_scope_contracts(
        scope=inputs.scope,
        selection=selection,
        outer_evaluation=outer_evaluation,
        model_manifest=model_manifest,
        final_refit_manifest=final_refit_manifest,
    )
    source_validations = sum(
        (
            validate_source_hashes(payload, project_root=project_root)
            for payload in (
                selection,
                outer_evaluation,
                model_manifest,
                final_refit_manifest,
            )
        ),
        [],
    )
    model_validations = [
        *validate_model_manifest(model_manifest, project_root=project_root),
        *validate_model_manifest(final_refit_manifest, project_root=project_root),
    ]
    return {
        "scope": inputs.scope,
        "candidate_space_complete": True,
        "outer_fold_count": len(outer_evaluation["outer_results"]),
        "outer_model_count": int(model_manifest["model_count"]),
        "final_model_count": int(final_refit_manifest["model_count"]),
        "source_hash_count": len(source_validations),
        "model_hash_count": len(model_validations),
        "scientific_reporting_allowed": False,
        "status": "PASS",
    }


def ui_command(paths: TrackingPaths) -> str:
    return (
        "python -m mlflow server "
        f'--backend-store-uri "{paths.tracking_uri}" '
        f'--default-artifact-root "{paths.artifact_uri}" '
        "--host 127.0.0.1 --port 5000"
    )


def _add_common_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config/mlflow_tracking.json.",
    )
    parser.add_argument(
        "--tracking-root",
        type=Path,
        default=None,
        help="Optional MLflow root override. Relative paths use the project root.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config_parser = subparsers.add_parser(
        "validate-config", help="Validate the MLflow configuration without importing MLflow."
    )
    _add_common_tracking_arguments(validate_config_parser)

    init_parser = subparsers.add_parser(
        "init", help="Initialize the local SQLite tracking store and experiment."
    )
    _add_common_tracking_arguments(init_parser)

    validate_local_parser = subparsers.add_parser(
        "validate-local", help="Validate local Phase 3 artifacts without creating runs."
    )
    _add_common_tracking_arguments(validate_local_parser)

    import_local_parser = subparsers.add_parser(
        "import-local", help="Import validated local Phase 3 artifacts into MLflow."
    )
    _add_common_tracking_arguments(import_local_parser)
    import_local_parser.add_argument("--git-commit", default=None)
    import_local_parser.add_argument("--force", action="store_true")
    import_local_parser.add_argument("--skip-models", action="store_true")
    import_local_parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Unsafe override for loading trusted model files with another sklearn version.",
    )

    import_phase3_parser = subparsers.add_parser(
        "import-phase3",
        help="Import explicitly supplied local or full-dataset Phase 3 artifacts.",
    )
    _add_common_tracking_arguments(import_phase3_parser)
    import_phase3_parser.add_argument("--scope", choices=sorted(SUPPORTED_SCOPE_NAMES), required=True)
    import_phase3_parser.add_argument("--selection", type=Path, required=True)
    import_phase3_parser.add_argument("--outer-evaluation", type=Path, required=True)
    import_phase3_parser.add_argument("--model-manifest", type=Path, required=True)
    import_phase3_parser.add_argument("--final-refit-manifest", type=Path, required=True)
    import_phase3_parser.add_argument("--git-commit", default=None)
    import_phase3_parser.add_argument("--force", action="store_true")
    import_phase3_parser.add_argument("--skip-models", action="store_true")
    import_phase3_parser.add_argument("--allow-version-mismatch", action="store_true")

    ui_parser = subparsers.add_parser(
        "ui-command", help="Print the exact local MLflow server command."
    )
    _add_common_tracking_arguments(ui_parser)
    return parser


def _resolved_input_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        config = load_tracking_config(args.config)
        paths = resolve_tracking_paths(
            config,
            tracking_root_override=args.tracking_root,
        )
        print("MLFLOW CONFIGURATION: PASS")
        print(f"Experiment: {config['experiment_name']}")
        print(f"Tracking URI: {paths.tracking_uri}")
        print(f"Artifact root: {paths.artifact_uri}")
        return 0

    if args.command == "validate-local":
        result = validate_local_artifacts()
        print(json.dumps(result, indent=2, sort_keys=True))
        print("LOCAL PHASE 3 ARTIFACT VALIDATION: PASS")
        return 0

    if args.command == "init":
        context = initialize_tracking(
            config_path=args.config,
            tracking_root_override=args.tracking_root,
        )
        print("MLFLOW INITIALIZATION: PASS")
        print(f"Experiment ID: {context.experiment_id}")
        print(f"Tracking URI: {context.paths.tracking_uri}")
        print(f"Artifact root: {context.paths.artifact_uri}")
        return 0

    if args.command == "ui-command":
        config = load_tracking_config(args.config)
        paths = resolve_tracking_paths(
            config,
            tracking_root_override=args.tracking_root,
        )
        print(ui_command(paths))
        return 0

    if args.command == "import-local":
        inputs = local_import_inputs()
    else:
        inputs = ImportInputs(
            scope=args.scope,
            selection_path=_resolved_input_path(args.selection),
            outer_evaluation_path=_resolved_input_path(args.outer_evaluation),
            model_manifest_path=_resolved_input_path(args.model_manifest),
            final_refit_manifest_path=_resolved_input_path(args.final_refit_manifest),
        )

    summary = import_phase3_artifacts(
        inputs=inputs,
        config_path=args.config,
        tracking_root_override=args.tracking_root,
        git_commit=args.git_commit,
        force=bool(args.force),
        log_models=not bool(args.skip_models),
        allow_version_mismatch=bool(args.allow_version_mismatch),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("MLFLOW PHASE 3 IMPORT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
