# Database Schema Documentation

The repository contains two independent SQLite databases: a legacy Phase 2
feature store and a Phase 3 prediction store. The separation prevents raw or
engineered features from being confused with model-inference records.

## 1. Phase 2 Feature Database

Path:

```text
database/sleep_eeg.db
```

### `subjects`

Stores subject-level metadata.

| Column | Purpose |
|---|---|
| `subject_id` | Primary key |
| `source_dataset` | Source dataset name |
| `recording_id` | Recording identifier |
| `notes` | Optional provenance note |

### `eeg_epochs`

Stores one row per 30-second EEG epoch.

| Column group | Columns |
|---|---|
| Identity | `id`, `subject_id`, `epoch_id`, `start_time_sec`, `eeg_channel` |
| Time domain | `mean`, `std`, `min`, `max`, `signal_energy` |
| Frequency domain | `delta_power`, `theta_power`, `alpha_power`, `beta_power` |
| Labels | `sleep_stage`, `sleep_stage_raw` |

Contracts:

- `id` is an auto-increment primary key.
- `subject_id` references `subjects(subject_id)`.
- `(subject_id, epoch_id, eeg_channel)` is unique.

Validated contents in the review archive:

- 1 subject
- 2,650 epochs
- SQLite integrity check: `ok`

This database is retained for Phase 2 compatibility. The full Phase 3 model
input is stored in validated CSV artifacts rather than this legacy schema.

## 2. Phase 3 Prediction Store

Path:

```text
sqlite-db/phase3_runtime/phase3_predictions.sqlite3
```

Implementation:

```text
scripts/phase3_prediction_store.py
```

### `prediction_runs`

One row describes one deterministic prediction import.

Important fields:

- `run_id` — primary key
- `run_fingerprint` — unique idempotency key
- `model_file_path`, `model_file_sha256`
- `model_name`, `candidate_id`, `outer_fold`
- `deployment_ready`, `non_deployment_override`
- `input_scope`, `input_row_count`, `feature_count`
- `feature_names_json`, `class_mapping_json`
- `prediction_sha256`, `run_metadata_json`

### `prediction_rows`

One row stores one epoch-level prediction.

Identity fields:

- `run_id`
- `row_position`
- `subject_id`
- `recording_id`
- `night`
- `epoch_id`
- optional `source_row_index`

Prediction fields:

- predicted and probability-argmax labels
- agreement flag
- confidence and probability margin
- entropy and normalized entropy
- probabilities for Wake, N1, N2, N3, and REM
- optional ground-truth label and correctness flag

Contracts:

- Primary key: `(run_id, row_position)`
- Unique epoch identity within a run:
  `(run_id, subject_id, recording_id, night, epoch_id)`
- `run_id` references `prediction_runs(run_id)` with `ON DELETE CASCADE`
- Probability, confidence, margin, and entropy ranges are enforced by SQL checks
- Foreign keys are enabled and validated by the persistence layer
- Repeated persistence of the same deterministic run is idempotent

Indexes support subject-order retrieval, predicted-label analysis, and model
hash lookup.

Validated contents in the review archive:

- 1 prediction run
- 1,103 prediction rows
- SQLite integrity check: `ok`

The included prediction store is a local engineering example. Final
full-dataset predictions must be written only after validated full outer
evaluation and final refit.

## Query Reference

See [`sql_queries.md`](sql_queries.md) for reproducible inspection, analysis,
duplicate detection, probability validation, and integrity queries.
