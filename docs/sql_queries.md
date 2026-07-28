# SQLite Query Reference

The repository contains two separate SQLite databases. They serve different
purposes and must not be mixed when reporting results.

## 1. Phase 2 Feature Database

Database:

```text
database/sleep_eeg.db
```

Validated local contents in the review archive:

- `subjects`: 1 row
- `eeg_epochs`: 2,650 rows

### Stage distribution

```sql
SELECT
    sleep_stage,
    COUNT(*) AS epoch_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM eeg_epochs
GROUP BY sleep_stage
ORDER BY epoch_count DESC;
```

### Mean band power by stage

```sql
SELECT
    sleep_stage,
    AVG(delta_power) AS mean_delta_power,
    AVG(theta_power) AS mean_theta_power,
    AVG(alpha_power) AS mean_alpha_power,
    AVG(beta_power) AS mean_beta_power
FROM eeg_epochs
GROUP BY sleep_stage
ORDER BY sleep_stage;
```

### Epoch count by subject

```sql
SELECT
    subject_id,
    COUNT(*) AS epoch_count,
    MIN(start_time_sec) AS first_epoch_start_sec,
    MAX(start_time_sec) AS last_epoch_start_sec
FROM eeg_epochs
GROUP BY subject_id
ORDER BY subject_id;
```

### Foreign-key check

```sql
PRAGMA foreign_keys = ON;
PRAGMA foreign_key_check;
```

## 2. Phase 3 Prediction Store

Database:

```text
sqlite-db/phase3_runtime/phase3_predictions.sqlite3
```

Validated local contents in the review archive:

- `prediction_runs`: 1 row
- `prediction_rows`: 1,103 rows

The included run is an engineering-validation example and is not a final
full-dataset scientific result.

### Prediction run inventory

```sql
SELECT
    run_id,
    created_at_utc,
    model_name,
    candidate_id,
    outer_fold,
    deployment_ready,
    non_deployment_override,
    input_scope,
    input_row_count,
    model_file_sha256,
    prediction_sha256
FROM prediction_runs
ORDER BY created_at_utc DESC;
```

### Predicted-stage distribution

```sql
SELECT
    predicted_label,
    COUNT(*) AS prediction_count,
    ROUND(AVG(prediction_confidence), 4) AS mean_confidence,
    ROUND(AVG(prediction_normalized_entropy), 4) AS mean_normalized_entropy
FROM prediction_rows
GROUP BY predicted_label
ORDER BY prediction_count DESC;
```

### Evaluation metrics for rows with ground truth

```sql
SELECT
    COUNT(*) AS labeled_rows,
    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct_rows,
    ROUND(AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END), 6) AS accuracy,
    ROUND(AVG(prediction_confidence), 6) AS mean_confidence,
    ROUND(AVG(prediction_normalized_entropy), 6) AS mean_normalized_entropy
FROM prediction_rows
WHERE is_correct IS NOT NULL;
```

### Per-class recall components

```sql
SELECT
    true_label,
    COUNT(*) AS support,
    SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END) AS true_positive,
    ROUND(
        1.0 * SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END)
        / COUNT(*),
        6
    ) AS recall
FROM prediction_rows
WHERE true_label IS NOT NULL
GROUP BY true_label
ORDER BY true_label;
```

### Most confident errors

```sql
SELECT
    subject_id,
    recording_id,
    night,
    epoch_id,
    true_label,
    predicted_label,
    prediction_confidence,
    prediction_margin,
    prediction_normalized_entropy
FROM prediction_rows
WHERE is_correct = 0
ORDER BY prediction_confidence DESC, prediction_margin DESC
LIMIT 25;
```

### Probability normalization audit

```sql
SELECT
    COUNT(*) AS invalid_probability_rows
FROM prediction_rows
WHERE ABS(
    probability_wake
    + probability_n1
    + probability_n2
    + probability_n3
    + probability_rem
    - 1.0
) > 1e-8;
```

### Duplicate-identifier audit

```sql
SELECT
    run_id,
    subject_id,
    recording_id,
    night,
    epoch_id,
    COUNT(*) AS duplicate_count
FROM prediction_rows
GROUP BY
    run_id,
    subject_id,
    recording_id,
    night,
    epoch_id
HAVING COUNT(*) > 1;
```

### Database integrity checks

```sql
PRAGMA foreign_keys = ON;
PRAGMA integrity_check;
PRAGMA foreign_key_check;
```

Expected `PRAGMA integrity_check` result:

```text
ok
```

## Running Queries from PowerShell

When the SQLite CLI is installed:

```powershell
sqlite3 ".\database\sleep_eeg.db" ".read docs/sql_queries.sql"
```

Queries may also be executed through Python's standard `sqlite3` module. The
Phase 3 store implementation and validation logic are in
`scripts/phase3_prediction_store.py`.
