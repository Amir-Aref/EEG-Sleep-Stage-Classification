from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
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


from scripts.database_connection import (
    get_connection,
)
from scripts.phase3_metrics import (
    probability_column_name,
    validate_and_normalize_probabilities,
    write_prediction_csv,
)


DEFAULT_PREDICTION_DB_PATH = (
    PROJECT_ROOT
    / "sqlite-db"
    / "phase3_predictions.sqlite3"
)

CLASS_NAMES = (
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
)

IDENTIFIER_COLUMNS = (
    "subject_id",
    "recording_id",
    "night",
    "epoch_id",
)

PROBABILITY_COLUMNS = tuple(
    probability_column_name(class_name)
    for class_name in CLASS_NAMES
)

CANONICAL_FLOAT_COLUMNS = (
    "prediction_confidence",
    "prediction_margin",
    "prediction_entropy",
    "prediction_normalized_entropy",
    *PROBABILITY_COLUMNS,
)

REQUIRED_PREDICTION_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    "predicted_label_encoded",
    "predicted_label",
    "probability_argmax_label_encoded",
    "probability_argmax_label",
    "predict_probability_argmax_agree",
    "prediction_confidence",
    "prediction_margin",
    "prediction_entropy",
    "prediction_normalized_entropy",
    *PROBABILITY_COLUMNS,
)

OPTIONAL_PREDICTION_COLUMNS = (
    "source_row_index",
    "true_label_encoded",
    "true_label",
    "is_correct",
)


MAX_PREDICTION_ENTROPY = math.log(
    len(CLASS_NAMES)
)


CREATE_PREDICTION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id TEXT PRIMARY KEY,
    run_fingerprint TEXT NOT NULL UNIQUE,
    created_at_utc TEXT NOT NULL,

    artifact_type TEXT NOT NULL,
    model_file_path TEXT NOT NULL,
    model_file_sha256 TEXT NOT NULL,
    model_name TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    outer_fold INTEGER,

    deployment_ready INTEGER NOT NULL
        CHECK (deployment_ready IN (0, 1)),

    non_deployment_override INTEGER NOT NULL
        CHECK (non_deployment_override IN (0, 1)),

    input_scope TEXT NOT NULL,
    input_row_count INTEGER NOT NULL
        CHECK (input_row_count > 0),

    feature_count INTEGER NOT NULL
        CHECK (feature_count > 0),

    feature_names_json TEXT NOT NULL,
    class_mapping_json TEXT NOT NULL,

    prediction_sha256 TEXT NOT NULL,
    run_metadata_json TEXT NOT NULL
);
"""


CREATE_PREDICTION_ROWS_TABLE = f"""
CREATE TABLE IF NOT EXISTS prediction_rows (
    run_id TEXT NOT NULL,
    row_position INTEGER NOT NULL
        CHECK (row_position >= 0),

    subject_id INTEGER NOT NULL,
    recording_id TEXT NOT NULL,
    night INTEGER NOT NULL,
    epoch_id INTEGER NOT NULL,
    source_row_index INTEGER,

    predicted_label_encoded INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,

    probability_argmax_label_encoded INTEGER NOT NULL,
    probability_argmax_label TEXT NOT NULL,

    predict_probability_argmax_agree INTEGER NOT NULL
        CHECK (
            predict_probability_argmax_agree
            IN (0, 1)
        ),

    prediction_confidence REAL NOT NULL
        CHECK (
            prediction_confidence >= 0.0
            AND prediction_confidence <= 1.0
        ),

    prediction_margin REAL NOT NULL
        CHECK (
            prediction_margin >= 0.0
            AND prediction_margin <= 1.0
        ),

    prediction_entropy REAL NOT NULL
        CHECK (
            prediction_entropy >= 0.0
            AND prediction_entropy <= {MAX_PREDICTION_ENTROPY:.17g}
        ),

    prediction_normalized_entropy REAL NOT NULL
        CHECK (
            prediction_normalized_entropy >= 0.0
            AND prediction_normalized_entropy <= 1.0
        ),

    probability_wake REAL NOT NULL
        CHECK (
            probability_wake >= 0.0
            AND probability_wake <= 1.0
        ),

    probability_n1 REAL NOT NULL
        CHECK (
            probability_n1 >= 0.0
            AND probability_n1 <= 1.0
        ),

    probability_n2 REAL NOT NULL
        CHECK (
            probability_n2 >= 0.0
            AND probability_n2 <= 1.0
        ),

    probability_n3 REAL NOT NULL
        CHECK (
            probability_n3 >= 0.0
            AND probability_n3 <= 1.0
        ),

    probability_rem REAL NOT NULL
        CHECK (
            probability_rem >= 0.0
            AND probability_rem <= 1.0
        ),

    true_label_encoded INTEGER,
    true_label TEXT,

    is_correct INTEGER
        CHECK (
            is_correct IS NULL
            OR is_correct IN (0, 1)
        ),

    PRIMARY KEY (
        run_id,
        row_position
    ),

    UNIQUE (
        run_id,
        subject_id,
        recording_id,
        night,
        epoch_id
    ),

    FOREIGN KEY (run_id)
        REFERENCES prediction_runs(run_id)
        ON DELETE CASCADE
);
"""


CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS
    idx_prediction_rows_subject
ON prediction_rows (
    subject_id,
    recording_id,
    night,
    epoch_id
);

CREATE INDEX IF NOT EXISTS
    idx_prediction_rows_predicted_label
ON prediction_rows (
    predicted_label_encoded
);

CREATE INDEX IF NOT EXISTS
    idx_prediction_runs_model_hash
ON prediction_runs (
    model_file_sha256
);
"""


@dataclass(frozen=True)
class PredictionStoreResult:
    database_path: Path
    run_id: str
    inserted: bool
    prediction_row_count: int
    prediction_sha256: str


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


def sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(
            timespec="microseconds"
        )
        .replace("+00:00", "Z")
    )


def normalize_scalar(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if pd.isna(value):
        return None

    return value


def clamp_unit_interval(
    value: Any,
) -> float:
    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            "Unit-interval value must be finite."
        )

    return float(
        np.clip(
            numeric,
            0.0,
            1.0,
        )
    )


def clamp_prediction_entropy(
    value: Any,
) -> float:
    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            "Prediction entropy must be finite."
        )

    return float(
        np.clip(
            numeric,
            0.0,
            MAX_PREDICTION_ENTROPY,
        )
    )


def validate_class_mapping(
    class_mapping: Mapping[str, int],
) -> dict[str, int]:
    normalized = {
        str(name): int(encoded)
        for name, encoded
        in class_mapping.items()
    }

    expected = {
        "Wake": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4,
    }

    if normalized != expected:
        raise ValueError(
            "Prediction store requires the "
            "locked five-class mapping."
        )

    return normalized


def validate_run_metadata(
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "artifact_type",
        "model_file_path",
        "model_file_sha256",
        "model_name",
        "candidate_id",
        "deployment_ready",
        "non_deployment_override",
        "input_scope",
        "feature_names",
        "class_mapping",
    }

    missing = sorted(
        required - set(run_metadata)
    )

    if missing:
        raise ValueError(
            "Missing prediction run metadata: "
            + ", ".join(missing)
        )

    normalized = dict(
        run_metadata
    )

    model_hash = str(
        normalized[
            "model_file_sha256"
        ]
    ).lower()

    if (
        len(model_hash) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in model_hash
        )
    ):
        raise ValueError(
            "model_file_sha256 must be a "
            "64-character hexadecimal SHA256."
        )

    normalized[
        "model_file_sha256"
    ] = model_hash

    deployment_ready = bool(
        normalized["deployment_ready"]
    )

    override = bool(
        normalized[
            "non_deployment_override"
        ]
    )

    if (
        not deployment_ready
        and not override
    ):
        raise ValueError(
            "A non-deployment model requires "
            "an explicit override."
        )

    feature_names = tuple(
        str(value)
        for value in normalized[
            "feature_names"
        ]
    )

    if not feature_names:
        raise ValueError(
            "Feature contract is empty."
        )

    if len(set(feature_names)) != len(
        feature_names
    ):
        raise ValueError(
            "Feature contract contains "
            "duplicate names."
        )

    normalized[
        "feature_names"
    ] = list(feature_names)

    normalized[
        "class_mapping"
    ] = validate_class_mapping(
        normalized["class_mapping"]
    )

    outer_fold = normalized.get(
        "outer_fold"
    )

    if outer_fold is not None:
        normalized[
            "outer_fold"
        ] = int(outer_fold)

    normalized[
        "deployment_ready"
    ] = deployment_ready

    normalized[
        "non_deployment_override"
    ] = override

    normalized[
        "input_scope"
    ] = str(
        normalized["input_scope"]
    )

    if not normalized[
        "input_scope"
    ].strip():
        raise ValueError(
            "input_scope cannot be empty."
        )

    return normalized


def canonicalize_signed_zero(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    canonical = frame.copy()

    for column in CANONICAL_FLOAT_COLUMNS:
        if column not in canonical.columns:
            continue

        values = canonical[
            column
        ].to_numpy(
            dtype=float,
            copy=True,
        )

        zero_mask = values == 0.0

        values[zero_mask] = 0.0

        canonical[column] = values

    return canonical


def validate_prediction_frame(
    predictions: pd.DataFrame,
    class_mapping: Mapping[str, int],
) -> pd.DataFrame:
    if not isinstance(
        predictions,
        pd.DataFrame,
    ):
        raise TypeError(
            "Predictions must be a DataFrame."
        )

    if predictions.empty:
        raise ValueError(
            "Prediction frame is empty."
        )

    missing = [
        column
        for column
        in REQUIRED_PREDICTION_COLUMNS
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(missing)
        )

    mapping = validate_class_mapping(
        class_mapping
    )

    inverse_mapping = {
        encoded: name
        for name, encoded
        in mapping.items()
    }

    frame = predictions.copy()

    if frame[
        list(IDENTIFIER_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Prediction identifiers "
            "contain missing values."
        )

    if frame[
        list(IDENTIFIER_COLUMNS)
    ].duplicated().any():
        raise ValueError(
            "Prediction identifiers "
            "are not unique."
        )

    probabilities = (
        validate_and_normalize_probabilities(
            probabilities=frame[
                list(PROBABILITY_COLUMNS)
            ].to_numpy(
                dtype=float
            ),
            expected_row_count=len(frame),
            expected_class_count=len(mapping),
        )
    )

    frame.loc[
        :,
        list(PROBABILITY_COLUMNS),
    ] = probabilities

    predicted_encoded = frame[
        "predicted_label_encoded"
    ].to_numpy(
        dtype=int
    )

    valid_encoded = set(
        inverse_mapping
    )

    if not set(
        predicted_encoded.tolist()
    ).issubset(valid_encoded):
        raise ValueError(
            "Prediction frame contains "
            "an unknown encoded label."
        )

    expected_predicted_names = np.array(
        [
            inverse_mapping[int(value)]
            for value in predicted_encoded
        ],
        dtype=object,
    )

    observed_predicted_names = frame[
        "predicted_label"
    ].astype(str).to_numpy()

    if not np.array_equal(
        expected_predicted_names,
        observed_predicted_names,
    ):
        raise ValueError(
            "Predicted label names do not "
            "match encoded labels."
        )

    probability_argmax = np.argmax(
        probabilities,
        axis=1,
    ).astype(int)

    stored_probability_argmax = frame[
        "probability_argmax_label_encoded"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        probability_argmax,
        stored_probability_argmax,
    ):
        raise ValueError(
            "Stored probability argmax does "
            "not match probability columns."
        )

    expected_argmax_names = np.array(
        [
            inverse_mapping[int(value)]
            for value in probability_argmax
        ],
        dtype=object,
    )

    observed_argmax_names = frame[
        "probability_argmax_label"
    ].astype(str).to_numpy()

    if not np.array_equal(
        expected_argmax_names,
        observed_argmax_names,
    ):
        raise ValueError(
            "Probability argmax names do not "
            "match encoded labels."
        )

    expected_agreement = (
        predicted_encoded
        == probability_argmax
    )

    observed_agreement = frame[
        "predict_probability_argmax_agree"
    ].astype(bool).to_numpy()

    if not np.array_equal(
        expected_agreement,
        observed_agreement,
    ):
        raise ValueError(
            "Predict/probability agreement "
            "column is inconsistent."
        )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    expected_confidence = (
        sorted_probabilities[:, -1]
    )

    expected_margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    safe_probabilities = np.clip(
        probabilities,
        np.finfo(float).tiny,
        1.0,
    )

    expected_entropy = -np.sum(
        probabilities
        * np.log(safe_probabilities),
        axis=1,
    )

    expected_normalized_entropy = (
        expected_entropy
        / math.log(len(mapping))
    )

    numeric_contracts = (
        (
            "prediction_confidence",
            expected_confidence,
        ),
        (
            "prediction_margin",
            expected_margin,
        ),
        (
            "prediction_entropy",
            expected_entropy,
        ),
        (
            "prediction_normalized_entropy",
            expected_normalized_entropy,
        ),
    )

    for column, expected in numeric_contracts:
        observed = frame[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            observed
        ).all():
            raise ValueError(
                f"{column} contains "
                "non-finite values."
            )

        if not np.allclose(
            observed,
            expected,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"{column} is inconsistent "
                "with probability columns."
            )

    optional_ground_truth = {
        "true_label_encoded",
        "true_label",
        "is_correct",
    }

    present_ground_truth = (
        optional_ground_truth
        & set(frame.columns)
    )

    if present_ground_truth and (
        present_ground_truth
        != optional_ground_truth
    ):
        raise ValueError(
            "Ground-truth columns must be "
            "provided together."
        )

    if present_ground_truth:
        truth_missing = frame[
            [
                "true_label_encoded",
                "true_label",
                "is_correct",
            ]
        ].isna().any().any()

        if truth_missing:
            raise ValueError(
                "Ground-truth columns contain "
                "missing values."
            )

        true_encoded = frame[
            "true_label_encoded"
        ].to_numpy(
            dtype=int
        )

        if not set(
            true_encoded.tolist()
        ).issubset(valid_encoded):
            raise ValueError(
                "Ground truth contains an "
                "unknown encoded label."
            )

        expected_true_names = np.array(
            [
                inverse_mapping[int(value)]
                for value in true_encoded
            ],
            dtype=object,
        )

        observed_true_names = frame[
            "true_label"
        ].astype(str).to_numpy()

        if not np.array_equal(
            expected_true_names,
            observed_true_names,
        ):
            raise ValueError(
                "True label names do not match "
                "encoded labels."
            )

        expected_correct = (
            true_encoded
            == predicted_encoded
        )

        observed_correct = frame[
            "is_correct"
        ].astype(bool).to_numpy()

        if not np.array_equal(
            expected_correct,
            observed_correct,
        ):
            raise ValueError(
                "is_correct is inconsistent."
            )

    frame = canonicalize_signed_zero(
        frame
    )

    return frame.reset_index(
        drop=True
    )


def prediction_frame_sha256(
    predictions: pd.DataFrame,
) -> str:
    with tempfile.TemporaryDirectory() as directory:
        canonical_path = (
            Path(directory)
            / "predictions.csv"
        )

        write_prediction_csv(
            predictions=predictions,
            output_path=canonical_path,
        )

        return sha256_bytes(
            canonical_path.read_bytes()
        )


def build_run_fingerprint(
    run_metadata: Mapping[str, Any],
    prediction_sha256: str,
    prediction_row_count: int,
) -> str:
    fingerprint_payload = {
        "artifact_type": (
            run_metadata["artifact_type"]
        ),
        "model_file_path": (
            run_metadata["model_file_path"]
        ),
        "model_file_sha256": (
            run_metadata[
                "model_file_sha256"
            ]
        ),
        "model_name": (
            run_metadata["model_name"]
        ),
        "candidate_id": (
            run_metadata["candidate_id"]
        ),
        "outer_fold": (
            run_metadata.get("outer_fold")
        ),
        "deployment_ready": (
            run_metadata["deployment_ready"]
        ),
        "non_deployment_override": (
            run_metadata[
                "non_deployment_override"
            ]
        ),
        "input_scope": (
            run_metadata["input_scope"]
        ),
        "feature_names": (
            run_metadata["feature_names"]
        ),
        "class_mapping": (
            run_metadata["class_mapping"]
        ),
        "prediction_sha256": (
            prediction_sha256
        ),
        "prediction_row_count": int(
            prediction_row_count
        ),
    }

    return sha256_bytes(
        canonical_json(
            fingerprint_payload
        ).encode("utf-8")
    )


def setup_prediction_database(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        "PRAGMA foreign_keys = ON;"
    )

    connection.execute(
        "PRAGMA busy_timeout = 10000;"
    )

    connection.execute(
        CREATE_PREDICTION_RUNS_TABLE
    )

    connection.execute(
        CREATE_PREDICTION_ROWS_TABLE
    )

    connection.executescript(
        CREATE_INDEXES
    )


def prediction_rows_for_insert(
    run_id: str,
    predictions: pd.DataFrame,
) -> list[tuple[Any, ...]]:
    has_source_index = (
        "source_row_index"
        in predictions.columns
    )

    has_ground_truth = all(
        column in predictions.columns
        for column in (
            "true_label_encoded",
            "true_label",
            "is_correct",
        )
    )

    rows = []

    for row_position, row in (
        predictions.iterrows()
    ):
        rows.append(
            (
                run_id,
                int(row_position),
                int(row["subject_id"]),
                str(row["recording_id"]),
                int(row["night"]),
                int(row["epoch_id"]),
                (
                    int(row["source_row_index"])
                    if has_source_index
                    and not pd.isna(
                        row["source_row_index"]
                    )
                    else None
                ),
                int(
                    row[
                        "predicted_label_encoded"
                    ]
                ),
                str(
                    row["predicted_label"]
                ),
                int(
                    row[
                        "probability_argmax_label_encoded"
                    ]
                ),
                str(
                    row[
                        "probability_argmax_label"
                    ]
                ),
                int(
                    bool(
                        row[
                            "predict_probability_argmax_agree"
                        ]
                    )
                ),
                clamp_unit_interval(
                    row[
                        "prediction_confidence"
                    ]
                ),
                clamp_unit_interval(
                    row[
                        "prediction_margin"
                    ]
                ),
                clamp_prediction_entropy(
                    row[
                        "prediction_entropy"
                    ]
                ),
                clamp_unit_interval(
                    row[
                        "prediction_normalized_entropy"
                    ]
                ),
                clamp_unit_interval(
                    row["probability_wake"]
                ),
                clamp_unit_interval(
                    row["probability_n1"]
                ),
                clamp_unit_interval(
                    row["probability_n2"]
                ),
                clamp_unit_interval(
                    row["probability_n3"]
                ),
                clamp_unit_interval(
                    row["probability_rem"]
                ),
                (
                    int(
                        row[
                            "true_label_encoded"
                        ]
                    )
                    if has_ground_truth
                    else None
                ),
                (
                    str(row["true_label"])
                    if has_ground_truth
                    else None
                ),
                (
                    int(
                        bool(row["is_correct"])
                    )
                    if has_ground_truth
                    else None
                ),
            )
        )

    return rows


INSERT_PREDICTION_ROW = """
INSERT INTO prediction_rows (
    run_id,
    row_position,
    subject_id,
    recording_id,
    night,
    epoch_id,
    source_row_index,
    predicted_label_encoded,
    predicted_label,
    probability_argmax_label_encoded,
    probability_argmax_label,
    predict_probability_argmax_agree,
    prediction_confidence,
    prediction_margin,
    prediction_entropy,
    prediction_normalized_entropy,
    probability_wake,
    probability_n1,
    probability_n2,
    probability_n3,
    probability_rem,
    true_label_encoded,
    true_label,
    is_correct
)
VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?
);
"""


def persist_prediction_run(
    database_path: Path,
    run_metadata: Mapping[str, Any],
    predictions: pd.DataFrame,
) -> PredictionStoreResult:
    metadata = validate_run_metadata(
        run_metadata
    )

    frame = validate_prediction_frame(
        predictions=predictions,
        class_mapping=metadata[
            "class_mapping"
        ],
    )

    prediction_hash = (
        prediction_frame_sha256(
            frame
        )
    )

    fingerprint = build_run_fingerprint(
        run_metadata=metadata,
        prediction_sha256=(
            prediction_hash
        ),
        prediction_row_count=len(frame),
    )

    run_id = fingerprint

    database_path = (
        Path(database_path).resolve()
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = get_connection(
        database_path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        setup_prediction_database(
            connection
        )

        existing = connection.execute(
            """
            SELECT
                run_id,
                prediction_sha256,
                input_row_count
            FROM prediction_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if existing is not None:
            if (
                existing[
                    "prediction_sha256"
                ]
                != prediction_hash
            ):
                raise ValueError(
                    "Existing prediction run "
                    "contains a different hash."
                )

            if int(
                existing[
                    "input_row_count"
                ]
            ) != len(frame):
                raise ValueError(
                    "Existing prediction run "
                    "contains a different "
                    "row count."
                )

            stored_rows = connection.execute(
                """
                SELECT COUNT(*)
                FROM prediction_rows
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()[0]

            if int(stored_rows) != len(
                frame
            ):
                raise ValueError(
                    "Existing prediction run is "
                    "incomplete."
                )

            return PredictionStoreResult(
                database_path=database_path,
                run_id=run_id,
                inserted=False,
                prediction_row_count=(
                    len(frame)
                ),
                prediction_sha256=(
                    prediction_hash
                ),
            )

        with connection:
            connection.execute(
                """
                INSERT INTO prediction_runs (
                    run_id,
                    run_fingerprint,
                    created_at_utc,
                    artifact_type,
                    model_file_path,
                    model_file_sha256,
                    model_name,
                    candidate_id,
                    outer_fold,
                    deployment_ready,
                    non_deployment_override,
                    input_scope,
                    input_row_count,
                    feature_count,
                    feature_names_json,
                    class_mapping_json,
                    prediction_sha256,
                    run_metadata_json
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    fingerprint,
                    utc_now_iso(),
                    str(
                        metadata[
                            "artifact_type"
                        ]
                    ),
                    str(
                        metadata[
                            "model_file_path"
                        ]
                    ),
                    str(
                        metadata[
                            "model_file_sha256"
                        ]
                    ),
                    str(
                        metadata[
                            "model_name"
                        ]
                    ),
                    str(
                        metadata[
                            "candidate_id"
                        ]
                    ),
                    metadata.get(
                        "outer_fold"
                    ),
                    int(
                        metadata[
                            "deployment_ready"
                        ]
                    ),
                    int(
                        metadata[
                            "non_deployment_override"
                        ]
                    ),
                    str(
                        metadata[
                            "input_scope"
                        ]
                    ),
                    len(frame),
                    len(
                        metadata[
                            "feature_names"
                        ]
                    ),
                    canonical_json(
                        metadata[
                            "feature_names"
                        ]
                    ),
                    canonical_json(
                        metadata[
                            "class_mapping"
                        ]
                    ),
                    prediction_hash,
                    canonical_json(
                        metadata
                    ),
                ),
            )

            connection.executemany(
                INSERT_PREDICTION_ROW,
                prediction_rows_for_insert(
                    run_id=run_id,
                    predictions=frame,
                ),
            )

        foreign_key_issues = (
            connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        if foreign_key_issues:
            raise ValueError(
                "Foreign-key audit failed "
                "after prediction persistence."
            )

        return PredictionStoreResult(
            database_path=database_path,
            run_id=run_id,
            inserted=True,
            prediction_row_count=len(frame),
            prediction_sha256=(
                prediction_hash
            ),
        )

    finally:
        connection.close()


def read_prediction_run(
    database_path: Path,
    run_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    database_path = (
        Path(database_path).resolve()
    )

    if not database_path.exists():
        raise FileNotFoundError(
            database_path
        )

    connection = get_connection(
        database_path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        run = connection.execute(
            """
            SELECT *
            FROM prediction_runs
            WHERE run_id = ?
            """,
            (str(run_id),),
        ).fetchone()

        if run is None:
            raise KeyError(
                f"Prediction run not found: "
                f"{run_id}"
            )

        rows = pd.read_sql_query(
            """
            SELECT *
            FROM prediction_rows
            WHERE run_id = ?
            ORDER BY row_position
            """,
            connection,
            params=(str(run_id),),
        )

        return (
            dict(run),
            rows,
        )

    finally:
        connection.close()


def audit_prediction_database(
    database_path: Path,
) -> dict[str, Any]:
    database_path = (
        Path(database_path).resolve()
    )

    if not database_path.exists():
        raise FileNotFoundError(
            database_path
        )

    connection = get_connection(
        database_path
    )

    try:
        quick_check = [
            row[0]
            for row in connection.execute(
                "PRAGMA quick_check"
            ).fetchall()
        ]

        foreign_key_check = (
            connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
        )

        run_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM prediction_runs
            """
        ).fetchone()[0]

        row_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM prediction_rows
            """
        ).fetchone()[0]

        return {
            "quick_check": quick_check,
            "foreign_key_issue_count": len(
                foreign_key_check
            ),
            "prediction_run_count": int(
                run_count
            ),
            "prediction_row_count": int(
                row_count
            ),
        }

    finally:
        connection.close()


def synthetic_prediction_frame() -> pd.DataFrame:
    probabilities = np.array(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],
            [0.05, 0.10, 0.70, 0.10, 0.05],
        ],
        dtype=float,
    )

    argmax = np.argmax(
        probabilities,
        axis=1,
    )

    labels = np.array(
        ["Wake", "N2"],
        dtype=object,
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    entropy = -np.sum(
        probabilities
        * np.log(probabilities),
        axis=1,
    )

    return pd.DataFrame(
        {
            "subject_id": [100, 100],
            "recording_id": [
                "SMOKE",
                "SMOKE",
            ],
            "night": [1, 1],
            "epoch_id": [0, 1],
            "source_row_index": [0, 1],
            "predicted_label_encoded": (
                argmax
            ),
            "predicted_label": labels,
            "probability_argmax_label_encoded": (
                argmax
            ),
            "probability_argmax_label": (
                labels
            ),
            "predict_probability_argmax_agree": [
                True,
                True,
            ],
            "prediction_confidence": (
                sorted_probabilities[:, -1]
            ),
            "prediction_margin": (
                sorted_probabilities[:, -1]
                - sorted_probabilities[:, -2]
            ),
            "prediction_entropy": entropy,
            "prediction_normalized_entropy": (
                entropy
                / math.log(5)
            ),
            "probability_wake": (
                probabilities[:, 0]
            ),
            "probability_n1": (
                probabilities[:, 1]
            ),
            "probability_n2": (
                probabilities[:, 2]
            ),
            "probability_n3": (
                probabilities[:, 3]
            ),
            "probability_rem": (
                probabilities[:, 4]
            ),
        }
    )


def smoke_test() -> None:
    metadata = {
        "artifact_type": (
            "phase3_prediction_store_smoke"
        ),
        "model_file_path": (
            "temporary/smoke.joblib"
        ),
        "model_file_sha256": (
            "a" * 64
        ),
        "model_name": "smoke_classifier",
        "candidate_id": (
            "smoke_classifier__candidate_001"
        ),
        "outer_fold": 1,
        "deployment_ready": False,
        "non_deployment_override": True,
        "input_scope": "smoke_test",
        "feature_names": [
            "feature_a",
            "feature_b",
        ],
        "class_mapping": {
            "Wake": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "REM": 4,
        },
    }

    predictions = (
        synthetic_prediction_frame()
    )

    with tempfile.TemporaryDirectory() as directory:
        database_path = (
            Path(directory)
            / "predictions.sqlite3"
        )

        first = persist_prediction_run(
            database_path=database_path,
            run_metadata=metadata,
            predictions=predictions,
        )

        second = persist_prediction_run(
            database_path=database_path,
            run_metadata=metadata,
            predictions=predictions,
        )

        run, rows = read_prediction_run(
            database_path=database_path,
            run_id=first.run_id,
        )

        audit = audit_prediction_database(
            database_path
        )

        if not first.inserted:
            raise ValueError(
                "Initial persistence was "
                "not inserted."
            )

        if second.inserted:
            raise ValueError(
                "Idempotent persistence "
                "inserted a duplicate."
            )

        if first.run_id != second.run_id:
            raise ValueError(
                "Idempotent run IDs differ."
            )

        if len(rows) != len(
            predictions
        ):
            raise ValueError(
                "Stored prediction row count "
                "is incorrect."
            )

        if audit["quick_check"] != [
            "ok"
        ]:
            raise ValueError(
                "SQLite quick check failed."
            )

        if audit[
            "foreign_key_issue_count"
        ] != 0:
            raise ValueError(
                "Foreign-key check failed."
            )

        print(
            "=== PHASE 3 PREDICTION "
            "STORE SMOKE TEST ==="
        )
        print(
            "Run ID:",
            first.run_id,
        )
        print(
            "First insert:",
            first.inserted,
        )
        print(
            "Second insert:",
            second.inserted,
        )
        print(
            "Prediction runs:",
            audit[
                "prediction_run_count"
            ],
        )
        print(
            "Prediction rows:",
            audit[
                "prediction_row_count"
            ],
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
            "Stored model:",
            run["model_name"],
        )
        print(
            "Transactional idempotence: PASS"
        )
        print(
            "Prediction store smoke test: PASS"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist Phase 3 prediction runs "
            "transactionally in SQLite."
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

    print("Use --smoke-test.")


if __name__ == "__main__":
    main()
