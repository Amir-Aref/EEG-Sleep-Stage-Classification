# Project Audit Report

## Audit Scope

This audit covers the clean review archive created from the authoritative
`EEG-Sleep-Stage-Classification-integration` worktree at repository commit:

```text
11c9ff4f2d79c16f0ac26744212a244d4604dc32
```

The received archive contained 148 files and approximately 36 MB of extracted
content. Raw EDF files, the Git worktree pointer, and the local virtual
environment were intentionally excluded.

## Executive Result

The reviewed project is structurally sound and its existing Phase 2 and Phase
3 contracts remain intact. The source compiles, the full automated test suite
passes in the review environment, both included SQLite databases pass integrity
checks, and all local Phase 3 source/model hashes validate.

The added MLflow layer is intentionally separated from training. It imports
validated artifacts, enforces scientific-scope contracts, records provenance,
and prevents the four-subject local engineering run from being presented as a
final scientific experiment.

## Key Findings and Resolutions

### 1. MLflow dependency conflict

The incoming dependency set combined `mlflow==3.14.0` with `pandas==3.0.3`.
MLflow 3.14 requires pandas below major version 3, so the installable project
environment now pins:

```text
pandas==2.3.3
mlflow==3.14.0
```

Historical model metadata still records pandas 3.0.3 because that is the
runtime in which the included local sklearn pipelines were created. The model
loader treats pandas as provenance-only and strictly validates the persistence-
critical Python major/minor, scikit-learn, NumPy, and joblib versions.

### 2. Missing experiment-tracking infrastructure

Added:

- `config/mlflow_tracking.json`
- `scripts/mlflow_tracking.py`
- `tests/test_mlflow_tracking.py`
- `docs/mlflow_tracking.md`

The implementation provides:

- SQLite tracking backend
- Local artifact store
- Parent and nested child runs
- Parameter, metric, artifact, and sklearn-model logging
- Model Registry integration
- Deterministic import fingerprints
- SHA-256 validation
- Separate `local_validation` and `full_dataset` scopes
- A protected `champion` alias reserved for a validated full-data deployment
  model

### 3. Outdated top-level documentation

`README.md` was rewritten to describe the complete Phase 2/Phase 3 repository,
subject-safe nested evaluation, saved-model inference, SQLite persistence,
MLflow, Docker, testing, and current full-data Kaggle status.

`README.docx` and `README.pdf` were retained as historical binary snapshots.
They should be regenerated only after final full-dataset metrics and figures are
available.

### 4. Docker runtime mismatch and build context

The Docker image now uses Python 3.13, matching the saved-model Python
major/minor runtime. It runs as a non-root user. A `.dockerignore` excludes raw
EDF files, virtual environments, caches, temporary review archives, and local
MLflow state.

A real Docker build was not available in the review environment and remains a
manual acceptance step.

### 5. Continuous integration coverage

GitHub Actions now:

- installs the pinned project dependencies
- verifies scientific and MLflow imports
- compiles Python sources
- validates MLflow configuration and local artifacts
- initializes an isolated temporary MLflow store
- runs the complete test suite
- validates the portable Phase 2 execution plan

A real GitHub Actions run remains pending until the updated source is committed
and pushed.

### 6. Database documentation and integrity

The repository has two independent databases:

- `database/sleep_eeg.db` — Phase 2 feature database
- `sqlite-db/phase3_runtime/phase3_predictions.sqlite3` — Phase 3 prediction
  store

Validated review-archive contents:

| Database | Tables and rows | Integrity |
|---|---|---|
| Phase 2 | `subjects`: 1; `eeg_epochs`: 2,650 | `ok` |
| Phase 3 | `prediction_runs`: 1; `prediction_rows`: 1,103 | `ok` |

The Phase 3 store also has:

- zero foreign-key violations
- zero invalid probability-sum rows
- zero duplicate epoch identifiers within a run

Database schema and query documentation were completed, and reproducible local
query outputs were added under `docs/phase3_sql_query_outputs/`.

### 7. Empty and duplicated files

Removed from the reviewed source:

- empty `docs/docker_run_success.png`
- empty `docs/github_actions_success.png`
- empty duplicate `docs/sql_queries_output/` directory
- nested `_project_inventory.zip`
- generated Python caches

Valid Docker and CI screenshots remain under `docs/screenshots/`.

### 8. Secrets and absolute-path scan

No private keys or project credentials were found. Absolute paths found in the
reviewed source are limited to explicit documentation examples for an optional
MLflow tracking-root override. Runtime source paths remain project-relative or
environment-configurable.

## Scientific Integrity Review

The included local Phase 3 artifacts are a four-subject engineering validation
run. Their machine-readable contracts correctly set scientific reporting to
false. The MLflow importer preserves this restriction and uses a separate
registry name.

The project does not create or infer missing full-data test metrics. Full
scientific reporting remains blocked until all five outer folds, outer-test
evaluation, final refit, prediction storage, and final MLflow import are
complete and validated.

## Validation Evidence

| Check | Result |
|---|---|
| Python source compilation | PASS |
| Full unittest discovery | PASS — 177 tests, 1 skipped |
| MLflow configuration validation | PASS |
| Local Phase 3 artifact validation | PASS |
| Local source hashes | PASS — 22 validated references |
| Local model hashes | PASS — 5 model files |
| Phase 2 execution plan | PASS |
| JSON parse audit | PASS |
| CSV parse audit | PASS |
| Image integrity audit | PASS after empty-file cleanup |
| DOCX container integrity | PASS |
| PDF header validation | PASS |
| Phase 2 SQLite integrity | PASS |
| Phase 3 SQLite integrity | PASS |
| Real MLflow SQLite integration | NOT RUN — package unavailable in review runtime |
| Pinned dependency dry-run | NOT VERIFIED — review package mirror lacks `numpy==2.5.1` |
| Docker build/run | PENDING local validation |
| GitHub Actions run | PENDING push |

The single skipped unit test is the real MLflow integration test. It is designed
to run automatically after MLflow is installed from `requirements.txt`.

The dependency dry-run failure in the review environment is an index-coverage
limitation: its private package mirror does not expose the requested NumPy
version. It is not evidence of a project dependency conflict. The known
MLflow/pandas conflict was removed explicitly.

## Remaining Acceptance Work

1. Install the updated pinned dependencies in the user's `.venv`.
2. Run all 177 tests with MLflow installed; the integration skip must disappear.
3. Initialize the SQLite MLflow store and import local engineering artifacts.
4. Verify runs, artifacts, and registered versions in the MLflow UI.
5. Complete Kaggle Batch 1 and Batch 2.
6. Run full outer evaluation and final refit.
7. Persist full-data predictions to SQLite.
8. Import final reportable artifacts and assign `champion` only to the validated
   deployment model.
9. Rebuild Docker, run GitHub Actions, regenerate final binary documentation,
   and complete the presentation/video.
