from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_model_registry.json"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "phase3_model_registry_summary.json"
)


ESTIMATOR_CLASSES = {
    "sklearn.dummy.DummyClassifier": DummyClassifier,
    "sklearn.linear_model.LogisticRegression": (
        LogisticRegression
    ),
    "sklearn.linear_model.SGDClassifier": (
        SGDClassifier
    ),
    "sklearn.ensemble.RandomForestClassifier": (
        RandomForestClassifier
    ),
    "sklearn.ensemble.ExtraTreesClassifier": (
        ExtraTreesClassifier
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_registry_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        registry = json.load(file)

    return registry


def get_model_spec(
    model_name: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    models = registry.get("models", {})

    if model_name not in models:
        available = ", ".join(sorted(models))

        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    return models[model_name]


def build_preprocessor(
    preprocessing_name: str,
) -> Any:
    if preprocessing_name == "standard_scaler":
        return StandardScaler()

    if preprocessing_name == "passthrough":
        return "passthrough"

    raise ValueError(
        "Unsupported preprocessing strategy: "
        f"{preprocessing_name}"
    )


def build_model_pipeline(
    model_name: str,
    registry: dict[str, Any] | None = None,
) -> Pipeline:
    if registry is None:
        registry = load_registry_config()

    spec = get_model_spec(
        model_name=model_name,
        registry=registry,
    )

    estimator_path = spec["estimator"]

    if estimator_path not in ESTIMATOR_CLASSES:
        raise ValueError(
            "Estimator is not allowed by the registry: "
            f"{estimator_path}"
        )

    estimator_class = ESTIMATOR_CLASSES[
        estimator_path
    ]

    estimator = estimator_class(
        **spec["fixed_parameters"]
    )

    preprocessor = build_preprocessor(
        spec["preprocessing"]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def get_parameter_grid(
    model_name: str,
    registry: dict[str, Any] | None = None,
) -> list[dict[str, list[Any]]]:
    if registry is None:
        registry = load_registry_config()

    spec = get_model_spec(
        model_name=model_name,
        registry=registry,
    )

    grids = spec["parameter_grid"]

    if not isinstance(grids, list) or not grids:
        raise ValueError(
            f"{model_name} must define a non-empty "
            "parameter-grid list."
        )

    return grids


def enumerate_candidate_parameters(
    model_name: str,
    registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    grids = get_parameter_grid(
        model_name=model_name,
        registry=registry,
    )

    return [
        dict(candidate)
        for candidate in ParameterGrid(grids)
    ]


def validate_registry(
    registry: dict[str, Any],
) -> dict[str, int]:
    required_top_level_keys = {
        "schema_version",
        "random_seed",
        "primary_metric",
        "probability_output_required",
        "expected_model_count",
        "expected_selection_model_count",
        "expected_total_candidate_count",
        "models",
    }

    missing_top_level = sorted(
        required_top_level_keys
        - set(registry)
    )

    if missing_top_level:
        raise ValueError(
            "Missing registry keys: "
            + ", ".join(missing_top_level)
        )

    models = registry["models"]

    if len(models) != registry[
        "expected_model_count"
    ]:
        raise ValueError(
            "Unexpected number of registered models."
        )

    selection_model_count = sum(
        bool(spec["eligible_for_selection"])
        for spec in models.values()
    )

    if selection_model_count != registry[
        "expected_selection_model_count"
    ]:
        raise ValueError(
            "Unexpected number of selectable models."
        )

    complexity_ranks = [
        int(spec["complexity_rank"])
        for spec in models.values()
    ]

    if len(complexity_ranks) != len(
        set(complexity_ranks)
    ):
        raise ValueError(
            "Model complexity ranks must be unique."
        )

    candidate_counts: dict[str, int] = {}

    for model_name in sorted(models):
        spec = models[model_name]

        pipeline = build_model_pipeline(
            model_name=model_name,
            registry=registry,
        )

        available_parameters = pipeline.get_params(
            deep=True
        )

        grids = get_parameter_grid(
            model_name=model_name,
            registry=registry,
        )

        for grid in grids:
            if not isinstance(grid, dict):
                raise ValueError(
                    f"{model_name} contains a "
                    "non-dictionary parameter grid."
                )

            for parameter_name, values in grid.items():
                if parameter_name not in (
                    available_parameters
                ):
                    raise ValueError(
                        f"{model_name} contains unknown "
                        f"parameter: {parameter_name}"
                    )

                if not isinstance(values, list):
                    raise ValueError(
                        f"{model_name} parameter "
                        f"{parameter_name} must use a list."
                    )

                if not values:
                    raise ValueError(
                        f"{model_name} parameter "
                        f"{parameter_name} has no values."
                    )

        candidates = enumerate_candidate_parameters(
            model_name=model_name,
            registry=registry,
        )

        if not candidates:
            raise ValueError(
                f"{model_name} has no candidates."
            )

        if (
            registry["probability_output_required"]
            and not hasattr(
                pipeline,
                "predict_proba",
            )
        ):
            raise ValueError(
                f"{model_name} does not expose "
                "predict_proba."
            )

        candidate_counts[model_name] = len(
            candidates
        )

    total_candidate_count = sum(
        candidate_counts.values()
    )

    if total_candidate_count != registry[
        "expected_total_candidate_count"
    ]:
        raise ValueError(
            "Unexpected total candidate count: "
            f"{total_candidate_count}"
        )

    return candidate_counts


def build_registry_summary(
    registry: dict[str, Any],
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    candidate_counts = validate_registry(
        registry
    )

    models = []

    for model_name in sorted(
        registry["models"]
    ):
        spec = registry["models"][model_name]

        models.append(
            {
                "model_name": model_name,
                "estimator": spec["estimator"],
                "role": spec["role"],
                "eligible_for_selection": bool(
                    spec[
                        "eligible_for_selection"
                    ]
                ),
                "preprocessing": spec[
                    "preprocessing"
                ],
                "complexity_rank": int(
                    spec["complexity_rank"]
                ),
                "candidate_count": int(
                    candidate_counts[model_name]
                ),
                "fixed_parameters": spec[
                    "fixed_parameters"
                ],
                "parameter_grid": spec[
                    "parameter_grid"
                ],
            }
        )

    return {
        "schema_version": "1.0.0",
        "registry_schema_version": registry[
            "schema_version"
        ],
        "registry_path": (
            config_path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix()
        ),
        "registry_sha256": sha256_file(
            config_path
        ),
        "scikit_learn_version": version(
            "scikit-learn"
        ),
        "random_seed": registry[
            "random_seed"
        ],
        "primary_metric": registry[
            "primary_metric"
        ],
        "probability_output_required": registry[
            "probability_output_required"
        ],
        "model_count": len(models),
        "selection_model_count": sum(
            model["eligible_for_selection"]
            for model in models
        ),
        "total_candidate_count": sum(
            model["candidate_count"]
            for model in models
        ),
        "models": models,
    }


def write_registry_summary(
    summary: dict[str, Any],
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
    ) + "\n"

    output_path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def smoke_fit_registry(
    registry: dict[str, Any],
) -> None:
    X, y = make_classification(
        n_samples=250,
        n_features=28,
        n_informative=16,
        n_redundant=4,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=registry["random_seed"],
    )

    for model_name in sorted(
        registry["models"]
    ):
        pipeline = build_model_pipeline(
            model_name=model_name,
            registry=registry,
        )

        fitted = clone(pipeline).fit(
            X,
            y,
        )

        predictions = fitted.predict(
            X[:20]
        )

        probabilities = fitted.predict_proba(
            X[:20]
        )

        if predictions.shape != (20,):
            raise ValueError(
                f"{model_name} returned invalid "
                "prediction shape."
            )

        if probabilities.shape != (20, 5):
            raise ValueError(
                f"{model_name} returned invalid "
                "probability shape."
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                f"{model_name} returned non-finite "
                "probabilities."
            )

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{model_name} probabilities do "
                "not sum to one."
            )

        print(
            f"{model_name}: "
            "fit/predict/predict_proba PASS"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Phase 3 model registry "
            "and its hyperparameter spaces."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
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
        "--smoke-fit",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()

    registry = load_registry_config(
        config_path
    )

    candidate_counts = validate_registry(
        registry
    )

    print("=== PHASE 3 MODEL REGISTRY ===")
    print(
        "Models:",
        len(registry["models"]),
    )

    for model_name in sorted(
        candidate_counts
    ):
        spec = registry["models"][
            model_name
        ]

        print(
            model_name,
            "| candidates=",
            candidate_counts[model_name],
            "| selectable=",
            spec["eligible_for_selection"],
            "| preprocessing=",
            spec["preprocessing"],
        )

    print(
        "Total candidates:",
        sum(candidate_counts.values()),
    )
    print("Registry validation: PASS")

    if arguments.smoke_fit:
        smoke_fit_registry(registry)
        print("Registry smoke fit: PASS")

    if arguments.write_summary:
        summary = build_registry_summary(
            registry=registry,
            config_path=config_path,
        )

        write_registry_summary(
            summary=summary,
            output_path=(
                arguments.summary_output.resolve()
            ),
        )

        print(
            "Summary:",
            arguments.summary_output,
        )
        print("Registry summary write: PASS")


if __name__ == "__main__":
    main()
