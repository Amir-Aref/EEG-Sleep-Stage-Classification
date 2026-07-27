# MLflow Tracking and Model Registry

## Purpose

The project uses MLflow to record Phase 3 experiment lineage without rerunning model search. Existing validated artifacts are imported into a dedicated experiment after their source hashes, model hashes, scientific-reporting flags, and runtime contracts pass validation.

The tracking implementation is in:

```text
config/mlflow_tracking.json
scripts/mlflow_tracking.py
tests/test_mlflow_tracking.py
```

## Storage Architecture

The default local layout is:

```text
mlflow/
├── mlflow.db          # SQLite backend for runs and Model Registry metadata
└── artifacts/         # Logged files and MLflow model packages
```

The SQLite backend is used instead of the legacy file-only backend because the project requires the open-source Model Registry. The artifact store remains local and portable.

Generated tracking state is ignored by Git. It may be included in the final submission archive after validation when the course deliverable requires the experiment history.

## Safety Scopes

Every imported run has one of two scopes.

### `local_validation`

- Uses the included four-subject engineering artifacts.
- Must have `scientific_reporting_allowed=false`.
- Must not be used for final performance claims.
- Registers models under `EEG_Sleep_Stage_Classifier_Local_Validation`.
- Never receives the `champion` alias.

### `full_dataset`

- Uses the complete 78-subject nested evaluation artifacts.
- Requires complete model selection and outer evaluation.
- Requires scientifically reportable outer and final artifacts.
- Requires a deployment-ready final refit.
- Registers models under `EEG_Sleep_Stage_Classifier`.
- Assigns the configured `champion` alias only to a validated final deployment model.

## Installation

Create and activate the project virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses `pandas==2.3.3` in the installable environment. Do not replace
it with the pandas 3.x version recorded in historical model metadata; that
metadata documents the original training runtime rather than the current
MLflow-compatible dependency set.

The saved local models were generated with scikit-learn 1.9.0, NumPy 2.5.1, joblib 1.5.3, and pandas 3.0.3. The delivery environment pins pandas 2.3.3 because MLflow 3.14 requires `pandas<3`. The importer records pandas for provenance but strictly validates Python major/minor, scikit-learn, NumPy, and joblib before loading trusted joblib models.

## Configuration Validation

This command does not create a database and does not require model loading:

```powershell
python scripts/mlflow_tracking.py validate-config
```

Expected ending:

```text
MLFLOW CONFIGURATION: PASS
```

## Local Artifact Validation

This verifies:

- Candidate-space completeness
- Test-isolation flags
- Scientific-reporting restrictions
- Source artifact SHA-256 values
- Model file SHA-256 values and file sizes
- Saved-model roundtrip flags

Run:

```powershell
python scripts/mlflow_tracking.py validate-local
```

Expected ending:

```text
LOCAL PHASE 3 ARTIFACT VALIDATION: PASS
```

This validation does not create MLflow runs.

## Initialize the Tracking Store

```powershell
python scripts/mlflow_tracking.py init
```

This creates the SQLite backend, artifact directory, and experiment if they do not already exist.

The root can be overridden safely:

```powershell
python scripts/mlflow_tracking.py init --tracking-root "D:\EEG-MLflow"
```

The equivalent environment variable is:

```powershell
$env:EEG_MLFLOW_ROOT = "D:\EEG-MLflow"
```

## Import Local Engineering Artifacts

From the repository root:

```powershell
python scripts/mlflow_tracking.py import-local `
  --git-commit 11c9ff4f2d79c16f0ac26744212a244d4604dc32
```

The import creates:

- One parent provenance run
- Four outer-fold child runs
- One final-refit child run
- Logged selection, evaluation, model, and configuration artifacts
- Logged sklearn model packages with signatures and input examples
- Registered local-validation model versions
- `data/metadata/mlflow_tracking_summary.json`

The import fingerprint prevents a successful import from being duplicated accidentally. Use `--force` only when intentionally creating another copy.

## Metrics-Only Diagnostic Import

For diagnosing tracking setup without loading or registering joblib models:

```powershell
python scripts/mlflow_tracking.py import-local --skip-models
```

A later model-inclusive import is treated as a different fingerprint, so it is not blocked by the diagnostic run.

## Runtime Mismatch Policy

The default policy rejects saved-model loading when the current environment differs from the recorded Python major/minor or pinned NumPy, pandas, scikit-learn, or joblib versions.

An explicit override exists:

```powershell
python scripts/mlflow_tracking.py import-local --allow-version-mismatch
```

This override is unsafe and should only be used for debugging trusted artifacts. It must not be used for final validation or scientific reporting.

## Start the MLflow UI

Print the exact command for the configured paths:

```powershell
python scripts/mlflow_tracking.py ui-command
```

Run the printed command. The default URL is:

```text
http://127.0.0.1:5000
```

Verify in the UI:

- Experiment name: `EEG Sleep Stage Classification`
- Parent and nested child runs
- Scope tags
- `scientific_reporting_allowed` tags
- Metrics and artifacts
- Registered model versions
- No `champion` alias on local-validation models

## Import Full-Dataset Results

After Batch 1, Batch 2, outer evaluation, saved models, and final refit have passed validation:

```powershell
python scripts/mlflow_tracking.py import-phase3 `
  --scope full_dataset `
  --selection data/metadata/phase3_full_inner_search_results.json `
  --outer-evaluation data/metadata/phase3_full_outer_evaluation.json `
  --model-manifest data/metadata/phase3_full_trained_model_manifest.json `
  --final-refit-manifest data/metadata/phase3_full_final_refit_manifest.json `
  --git-commit <commit-sha>
```

The command intentionally fails when:

- Selection contains test metrics or test predictions.
- Candidate search is incomplete.
- Outer evaluation is incomplete.
- Source or model hashes do not match.
- Full artifacts forbid scientific reporting.
- The final model is not deployment-ready.
- The saved-model runtime is incompatible.

## Expected Tags

Important run and model-version tags include:

```text
eeg.scope
eeg.role
eeg.git_commit
eeg.model_name
eeg.candidate_id
eeg.outer_fold
eeg.scientific_reporting_allowed
eeg.deployment_ready
eeg.import_fingerprint
```

These tags prevent local engineering evidence from being mistaken for final full-dataset evidence.

## Final Submission Procedure

1. Run local or full import in the exact pinned environment.
2. Run the full test suite.
3. Start the MLflow server.
4. Capture experiment and registry screenshots.
5. Verify `data/metadata/mlflow_tracking_summary.json`.
6. Preserve the `mlflow` directory in the final submission copy when required.
7. Keep `mlflow` ignored in the normal Git working tree unless the instructor explicitly requires tracking-state files in version control.

## References

- MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/
- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry/
- MLflow sklearn API: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html
