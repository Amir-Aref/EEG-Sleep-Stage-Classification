# Review ZIP Change Log

## Source

- Input archive: `EEG-project-review.zip`
- Source repository commit: `11c9ff4f2d79c16f0ac26744212a244d4604dc32`
- Review date: 2026-07-27

## Added Files

- `.dockerignore`
- `config/mlflow_tracking.json`
- `docs/mlflow_tracking.md`
- `docs/phase3_sql_query_outputs/01_prediction_run_inventory.csv`
- `docs/phase3_sql_query_outputs/02_predicted_stage_distribution.csv`
- `docs/phase3_sql_query_outputs/03_evaluation_summary.csv`
- `docs/phase3_sql_query_outputs/04_per_class_recall.csv`
- `docs/phase3_sql_query_outputs/05_most_confident_errors.csv`
- `docs/phase3_sql_query_outputs/06_integrity_summary.txt`
- `reports/PROJECT_AUDIT.md`
- `reports/FILE_MANIFEST.json`
- `reports/VALIDATION_SUMMARY.json`
- `reports/ZIP_CHANGELOG.md`
- `scripts/mlflow_tracking.py`
- `tests/test_mlflow_tracking.py`

## Modified Files

- `.github/workflows/data-pipeline-ci.yml`
  - expanded CI to compile, test, validate MLflow contracts, and initialize an
    isolated tracking store
- `.gitignore`
  - ignores local MLflow state, temporary archives, and logs
- `Dockerfile`
  - aligned Python runtime, added non-root execution, and retained pinned install
- `PROJECT_COMPLETION_CHECKLIST.md`
  - synchronized completed and remaining work
- `README.md`
  - replaced Phase 2-only documentation with complete project instructions
- `docs/database_schema.md`
  - documented both Phase 2 and Phase 3 databases
- `docs/sql_queries.md`
  - added reproducible Phase 2 and Phase 3 SQL queries
- `reports/FINAL_REPORT_DRAFT.md`
  - added a full report structure and explicit local/full-data separation
- `requirements.txt`
  - added MLflow and changed pandas to the MLflow-compatible 2.3.3 pin

## Removed Files and Directories

- `_project_inventory.zip` — accidental nested review artifact
- `docs/docker_run_success.png` — zero-byte duplicate
- `docs/github_actions_success.png` — zero-byte duplicate
- `docs/sql_queries_output/` — empty duplicate output directory
- generated `__pycache__/` directories and `.pyc` files

Valid screenshots remain under `docs/screenshots/`.

## Behavioral Changes

- Existing validated Phase 2 and Phase 3 training/evaluation artifacts are not
  modified.
- Existing local performance metrics are not relabeled as final results.
- MLflow imports existing artifacts without rerunning model search.
- Local and full-data model registries are separated.
- Duplicate imports are blocked by a deterministic fingerprint unless the
  explicit `--force` option is used.
- The final `champion` alias is blocked for local-validation artifacts.
- Saved-model loading validates persistence-critical runtime versions and file
  hashes before unpickling trusted artifacts.

## Validation Summary

- Python compilation: PASS
- Unit and contract tests: PASS — 177 tests, 1 MLflow integration skip
- MLflow configuration: PASS
- Local Phase 3 artifact contract: PASS
- SQLite integrity: PASS for both databases
- Structured-file and image integrity: PASS
- Real MLflow integration: pending installation in the user's environment
- Docker and GitHub Actions: pending external execution
