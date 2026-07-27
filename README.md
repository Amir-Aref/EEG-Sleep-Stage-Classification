# EEG-Based Sleep Stage Classification

A reproducible machine-learning project for classifying 30-second EEG epochs into five sleep stages:

- Wake
- N1
- N2
- N3
- REM

The project uses the Sleep-EDF Database Expanded sleep-cassette subset and enforces subject-level isolation throughout validation and testing. The primary model-selection metric is Macro-F1.

> This repository is an educational and research implementation. It is not a medical diagnostic system.

## Current Project Status

The complete Phase 2 data pipeline and the Phase 3 engineering implementation are available in the repository. Full-dataset nested model search is being executed on Kaggle in checkpointed batches.

Validated full-dataset preparation:

- 78 subjects
- 153 recordings
- 306 validated EDF files
- 195,469 thirty-second epochs
- 28 leakage-safe model features
- 5 outer folds and 3 inner folds per outer fold
- 29 model candidates

The local Phase 3 artifacts included in this repository use four subjects for engineering validation only. Their metrics must not be reported as final scientific results. Machine-readable artifacts explicitly include `scientific_reporting_allowed=false`.

Project progress is tracked in [`PROJECT_COMPLETION_CHECKLIST.md`](PROJECT_COMPLETION_CHECKLIST.md).

## Evaluation Design

The full scientific evaluation uses nested, subject-grouped cross-validation:

1. Five outer folds estimate performance on unseen subjects.
2. Three inner folds select models and hyperparameters using development subjects only.
3. The outer test partition is never used for model selection.
4. Preprocessing is fitted only on each training partition.
5. Every subject appears in exactly one outer test fold.

Random epoch-level splitting is forbidden by the evaluation protocol in [`config/phase3_evaluation_protocol.json`](config/phase3_evaluation_protocol.json).

## Model Registry

The configured model families are:

1. Dummy prior baseline
2. Logistic regression
3. SGD logistic classifier
4. Random forest
5. Extra trees

The complete registry contains 29 candidates. See [`config/phase3_model_registry.json`](config/phase3_model_registry.json).

## Repository Structure

```text
.
├── .github/workflows/              # Continuous integration
├── artifacts/models/               # Trusted local engineering model artifacts
├── config/                         # Evaluation, model, and MLflow contracts
├── data/
│   ├── interim/                    # Epoch metadata
│   ├── metadata/                   # Schemas, manifests, metrics, and provenance
│   ├── processed/                  # Feature and model-input datasets
│   └── sample/                     # Small compatibility sample
├── database/                       # Phase 2 SQLite database
├── docs/                           # Database, SQL, Docker, CI, and MLflow docs
├── notebooks/                      # Exploratory notebook
├── outputs/                        # Legacy Phase 2 outputs
├── reports/                        # EDA, audit, report draft, and change log
├── scripts/                        # Data, modeling, inference, storage, and tracking code
├── sqlite-db/                      # Local Phase 3 prediction-store example
├── tests/                          # Contract and integration tests
├── pipeline.py                     # Legacy Phase 2 compatibility pipeline
├── Dockerfile
└── requirements.txt
```

## Environment

The saved local Phase 3 models were produced with:

- Python 3.13.x
- NumPy 2.5.1
- pandas 3.0.3
- scikit-learn 1.9.0
- joblib 1.5.3

The delivery environment pins **pandas 2.3.3** because MLflow 3.14 requires
`pandas<3`. pandas is retained in model provenance, but the artifact loader
strictly gates Python major/minor, scikit-learn, NumPy, and joblib because the
saved objects are scikit-learn pipelines. Trusted joblib artifacts must still
be loaded only from this project.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validation Commands

Compile all Python sources:

```bash
python -m compileall -q pipeline.py scripts tests
```

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Validate the Phase 2 execution plan without running it:

```bash
python scripts/run_phase2_pipeline.py --plan
```

Validate local Phase 3 artifacts without creating MLflow runs:

```bash
python scripts/mlflow_tracking.py validate-config
python scripts/mlflow_tracking.py validate-local
```

## Phase 2 Pipeline

The original compatibility pipeline can be run with:

```bash
python pipeline.py
```

The production-style Phase 2 runner provides planning and provenance support:

```bash
python scripts/run_phase2_pipeline.py --plan
```

Raw Sleep-EDF recordings are intentionally excluded from Git. Runtime paths can be redirected with:

- `EEG_RUNTIME_ROOT`
- `EEG_SLEEP_EDFX_RAW_DIR`

## Phase 3 Components

The Phase 3 implementation includes:

- Dataset contract validation
- Subject-safe split generation
- Model registry and candidate enumeration
- Inner model search
- Selection artifacts without test metrics
- Outer test evaluation
- Saved end-to-end sklearn pipelines
- Final refit
- Targetless prediction pipeline
- SQLite prediction storage
- MLflow experiment tracking and model registration

## MLflow Tracking

MLflow uses a local SQLite backend and a local artifact directory. This supports durable run metadata and the open-source Model Registry while keeping all files under a configurable project-local root.

Initialize the tracking store:

```bash
python scripts/mlflow_tracking.py init
```

Import the included local engineering artifacts:

```bash
python scripts/mlflow_tracking.py import-local --git-commit 11c9ff4f2d79c16f0ac26744212a244d4604dc32
```

The importer:

- Verifies source and model SHA-256 hashes
- Enforces local versus full-dataset scientific-reporting contracts
- Refuses incompatible saved-model runtimes by default
- Logs parameters, metrics, manifests, source joblib files, and MLflow model packages
- Registers local-validation models separately from the final full-dataset model
- Uses an import fingerprint to prevent accidental duplicate imports

Print the exact MLflow server command:

```bash
python scripts/mlflow_tracking.py ui-command
```

Run the printed command and open `http://127.0.0.1:5000`.

Detailed instructions are available in [`docs/mlflow_tracking.md`](docs/mlflow_tracking.md).

## Importing Future Full-Dataset Results

After the full nested search, outer evaluation, and final refit are complete, use the generic importer:

```bash
python scripts/mlflow_tracking.py import-phase3 \
  --scope full_dataset \
  --selection data/metadata/phase3_full_inner_search_results.json \
  --outer-evaluation data/metadata/phase3_full_outer_evaluation.json \
  --model-manifest data/metadata/phase3_full_trained_model_manifest.json \
  --final-refit-manifest data/metadata/phase3_full_final_refit_manifest.json \
  --git-commit <commit-sha>
```

Full-dataset model registration is blocked unless the artifacts explicitly permit scientific reporting and the final model is deployment-ready.

## Prediction and SQLite Storage

The saved-model prediction pipeline validates:

- Model manifest and model hash
- Expected feature names and order
- Optional ground-truth columns
- Class probabilities
- Confidence, margin, and normalized entropy
- Deployment-readiness policy

The prediction store persists validated rows to SQLite with foreign-key enforcement and idempotent run handling. Reproducible query examples are in [`docs/sql_queries.md`](docs/sql_queries.md), and validated local query outputs are stored under [`docs/phase3_sql_query_outputs/`](docs/phase3_sql_query_outputs/).

## Docker

Build the image:

```bash
docker build -t eeg-sleep-stage-classification .
```

Run the default compatibility pipeline:

```bash
docker run --rm eeg-sleep-stage-classification
```

The image uses Python 3.13 to match the major/minor runtime of the included saved models. Generated MLflow state, raw EDF files, virtual environments, caches, and temporary archives are excluded from the Docker build context.

## Reproducibility and Safety

- Source artifacts and models are identified by SHA-256.
- Joblib files must only be loaded from trusted project artifacts.
- Local engineering results are separated from final full-dataset results.
- The importer does not rerun training or access new held-out data.
- Full scientific reporting remains blocked until the complete outer evaluation is validated.

## Documentation

- [`docs/database_schema.md`](docs/database_schema.md)
- [`docs/sql_queries.md`](docs/sql_queries.md)
- [`docs/mlflow_tracking.md`](docs/mlflow_tracking.md)
- [`reports/FINAL_REPORT_DRAFT.md`](reports/FINAL_REPORT_DRAFT.md)
- [`reports/PROJECT_AUDIT.md`](reports/PROJECT_AUDIT.md)
- [`reports/ZIP_CHANGELOG.md`](reports/ZIP_CHANGELOG.md)
