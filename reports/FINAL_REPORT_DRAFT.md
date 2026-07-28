# EEG-Based Sleep Stage Classification — Final Report Draft

## Document Status

This document contains the validated project background, data-processing methodology, engineering implementation, and report structure. Full-dataset model-selection and outer-test results remain pending until the checkpointed Kaggle batches and final evaluation are complete.

Local four-subject Phase 3 metrics are included only as engineering evidence. They are not final scientific results.

## 1. Executive Summary

This project develops a reproducible and leakage-safe machine-learning system for classifying 30-second EEG epochs into Wake, N1, N2, N3, and REM. The full Sleep-EDF Database Expanded sleep-cassette subset was processed with subject-level identifiers preserved throughout the pipeline. The final scientific protocol uses nested, subject-grouped cross-validation so that subjects never overlap between training, validation, and test partitions.

The implementation covers:

- Raw-data verification and provenance
- Epoch metadata generation
- EEG feature extraction and audit
- Leakage-safe model-input construction
- Exploratory data analysis
- Nested model selection
- Outer test evaluation
- Saved end-to-end sklearn pipelines
- Final refit
- Targetless prediction
- SQLite prediction storage
- MLflow experiment tracking and model registration

> Final scientific results will be inserted after the complete nested search and outer-test evaluation pass validation.

## 2. Problem Definition

- **Task:** Multiclass classification
- **Input:** One 30-second EEG epoch represented by engineered time-domain, Hjorth, band-power, ratio, and spectral features
- **Target classes:** Wake, N1, N2, N3, and REM
- **Primary metric:** Macro-F1
- **Grouping unit:** Subject
- **Critical constraint:** Subjects must never overlap between training, validation, and test partitions

The system is an educational and research implementation and is not a medical diagnostic device.

## 3. Dataset

- **Dataset:** Sleep-EDF Database Expanded
- **Subset:** Sleep Cassette
- **Subjects:** 78
- **Recordings:** 153
- **Validated EDF files:** 306
- **Epoch duration:** 30 seconds
- **Total epochs:** 195,469

### 3.1 Class Distribution

| Sleep stage | Epoch count |
|---|---:|
| Wake | 65,941 |
| N1 | 21,522 |
| N2 | 69,132 |
| N3 | 13,039 |
| REM | 25,835 |

The class distribution is imbalanced. N2 and Wake dominate the dataset, while N3 and N1 are less frequent. Macro-F1 is therefore used as the primary metric so that minority classes contribute equally to model selection.

## 4. Data Acquisition and Validation

The data pipeline validates the complete PSG and hypnogram inventory before feature processing.

Validated controls include:

- Expected PSG and hypnogram counts
- Recording pair consistency
- SHA-256 verification
- EDF inspection
- Annotation mapping
- Subject and recording identifiers
- Explicit exclusion of invalid sleep labels
- Thirty-second epoch alignment

Raw EDF recordings are not committed to Git. Runtime paths can be redirected for Kaggle or external storage without changing source-code paths.

## 5. Preprocessing and Feature Engineering

The final model input contains 28 features.

### 5.1 Time-Domain Features

- Mean
- Standard deviation
- Median
- Minimum and maximum
- Peak-to-peak amplitude
- Zero-crossing rate
- Line length
- Skewness
- Excess kurtosis

### 5.2 Hjorth Features

- Hjorth mobility
- Hjorth complexity

### 5.3 Frequency-Domain Features

- Absolute delta, theta, alpha, sigma, and beta power
- Relative delta, theta, alpha, and sigma power
- Theta-alpha ratio
- Alpha-beta ratio
- Sigma-beta ratio
- Spectral entropy
- Dominant frequency
- Spectral centroid
- Spectral edge frequency at 95%

### 5.4 Removed Features

Seven redundant or strongly dependent features were removed before modeling:

- RMS
- Mean-square amplitude
- Signal energy
- Hjorth activity
- Total band power
- Relative beta power
- Delta-theta ratio

The model-input audit confirms that scaling is not performed globally and random epoch splitting is forbidden.

## 6. Exploratory Data Analysis

The full EDA stage produced 161 figures, including:

- Class-distribution plots
- Stage-level feature distributions
- Outlier summaries
- Subject and recording summaries
- Per-recording hypnograms
- Subject-shift plots
- Relative band-power comparisons

The final report will include a curated subset of these figures rather than all generated files.

## 7. Leakage-Safe Evaluation Protocol

The final scientific evaluation uses nested, subject-grouped cross-validation:

- **Outer folds:** 5
- **Inner folds per outer fold:** 3
- **Total inner split records:** 15
- **Group column:** `subject_id`
- **Random epoch split:** Forbidden
- **Model-selection partition:** Inner training and validation only
- **Final evaluation partition:** Unseen outer-test subjects only

Every subject appears in exactly one outer-test fold. Preprocessing, feature selection, class weighting, and any resampling decisions are fitted only on the training partition of the applicable fold.

## 8. Model Registry and Search Space

The model registry contains five model families:

1. Dummy prior baseline
2. Logistic regression
3. SGD logistic classifier
4. Random forest
5. Extra trees

The full registry contains 29 candidates. Across five outer folds and three inner folds, the complete nested search requires 435 fitted pipelines.

The dummy prior model is a non-selectable baseline. The remaining four model families are eligible for model selection.

## 9. Local Engineering Validation

The repository includes a complete four-subject local Phase 3 run to validate code, artifact contracts, persistence, inference, and SQLite storage.

### 9.1 Scientific Status

```text
scope=local_validation
scientific_reporting_allowed=false
intended_use=engineering_smoke_test_only
```

These results must not be presented as final model performance.

### 9.2 Local Engineering Metrics

The pooled local evaluation contains 3,921 epochs across four outer folds.

| Metric | Local engineering value |
|---|---:|
| Pooled accuracy | 0.7488 |
| Pooled balanced accuracy | 0.6705 |
| Pooled Macro-F1 | 0.6631 |
| Pooled weighted F1 | 0.7568 |
| Pooled Cohen's kappa | 0.6478 |
| Mean fold Macro-F1 | 0.6510 |
| Standard deviation of fold Macro-F1 | 0.0288 |

The local run demonstrates that the full Phase 3 code paths work, but the sample is too small for final scientific claims.

## 10. Full-Dataset Model Selection Results

> Pending completion and validation of Search Batch 1 and Search Batch 2.

### 10.1 Selected Candidate per Outer Fold

| Outer fold | Selected model | Candidate ID | Inner validation Macro-F1 |
|---:|---|---|---:|
| 1 | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending |
| 5 | Pending | Pending | Pending |

Required validation before this table is finalized:

- 145 candidate summary rows
- 29 unique candidates
- Five selected candidates
- Complete candidate space
- No test metrics in the selection artifact

## 11. Full Outer-Test Evaluation

> Pending leakage-safe full-dataset outer evaluation.

Required outputs:

- Macro-F1
- Balanced accuracy
- Weighted F1
- Accuracy
- Cohen's kappa
- Multiclass log loss
- Per-class precision, recall, F1, and support
- Per-fold confusion matrices
- Pooled confusion matrix
- Prediction confidence
- Probability margin
- Normalized entropy
- Fold mean and standard deviation

## 12. Model Persistence and Reproducibility

The model-artifact layer saves complete sklearn pipelines that include preprocessing and classification.

Validation requirements include:

- Model file SHA-256
- Model file size
- Feature list and class mapping
- Training subject list
- Excluded test subjects
- Candidate parameters
- Prediction equivalence after reload
- Probability equivalence after reload
- Runtime metadata

The local repository currently contains four outer-fold pipelines and one local final-refit pipeline. All five files have validated hashes and roundtrip flags.

## 13. Final Refit and Deployment Model

> Full-dataset final refit remains pending.

The final deployment configuration must be selected without using outer-test results for hyperparameter tuning. The final refit must produce a complete inference-ready pipeline, a manifest, a checksum, and reload-equivalence evidence.

A deployment model is not promoted merely because it can generate predictions. Scientific-reporting and deployment-readiness flags must both pass their explicit contracts.

## 14. Prediction Pipeline and SQLite Storage

The prediction pipeline validates:

- Trusted model manifest
- Model checksum
- Feature names and order
- Optional ground-truth columns
- Class-probability alignment
- Prediction confidence
- Probability margin
- Normalized entropy
- Deployment-readiness policy

The SQLite prediction store provides:

- Foreign-key enforcement
- Canonical prediction hashes
- Idempotent persistence
- Optional ground-truth handling
- Subject, recording, and epoch identifiers
- Analytical SQL queries

Final full-dataset prediction storage remains pending.

## 15. MLflow Experiment Tracking

The project now includes an MLflow integration that imports already validated Phase 3 artifacts without rerunning training.

### 15.1 Storage

- SQLite backend for run and Model Registry metadata
- Local artifact directory for JSON, CSV, figures, joblib files, and MLflow models
- Configurable tracking root

The installable delivery environment pins pandas 2.3.3 because MLflow 3.14 requires pandas below major version 3. Historical model metadata retains pandas 3.0.3 as provenance for the original training runtime.

### 15.2 Logged Information

- Repository commit
- Dataset and protocol provenance
- Import fingerprint
- Candidate parameters
- Inner validation metrics
- Outer evaluation metrics
- Source manifests
- Source joblib files
- MLflow sklearn model packages
- Model signatures and input examples
- Registered model versions

### 15.3 Safety Scopes

Local engineering runs use:

```text
scope=local_validation
scientific_reporting_allowed=false
```

Full-dataset runs use:

```text
scope=full_dataset
scientific_reporting_allowed=true
```

The local registry name is separate from the final full-dataset registry name. A local model can never receive the `champion` alias. The full importer refuses incomplete, non-reportable, hash-invalid, runtime-incompatible, or non-deployment-ready artifacts.

### 15.4 Remaining MLflow Evidence

- Run the importer in the pinned local environment
- Verify the SQLite tracking store
- Verify registered model versions
- Start the MLflow UI
- Capture experiment and registry screenshots
- Import final full-dataset artifacts after completion

## 16. Software Quality and Continuous Integration

Repository quality controls include:

- Python source compilation
- Unit and contract tests
- Path portability tests
- Artifact determinism tests
- Hash and provenance tests
- Model persistence tests
- Prediction and SQLite tests
- MLflow configuration and import tests
- Docker build isolation
- GitHub Actions validation

The baseline repository passed 162 tests before MLflow changes. The reviewed source now passes 177 tests; one real MLflow integration test is skipped only when MLflow is not installed in the active environment.

## 17. Docker

The Docker image uses Python 3.13 to match the major/minor runtime of the included saved models. It installs pinned dependencies, runs as a non-root user, and excludes raw EDF files, virtual environments, caches, temporary review archives, and local MLflow state from the build context.

Final Docker build and run evidence must be refreshed after the new dependency set is installed.

## 18. Scientific Analysis Plan

The final analysis will address:

- Which model families generalize best to unseen subjects
- Stability of selected candidates across outer folds
- Per-class precision, recall, and F1
- N1 confusion with adjacent stages
- N2 dominance and class imbalance
- Subject-level variability
- Prediction confidence and uncertainty
- Stable feature-importance patterns
- Limitations of engineered single-channel features

## 19. Limitations

- The project uses engineered features from a selected EEG channel.
- Sleep-stage labels may contain scorer variability.
- Class imbalance is substantial, especially for N1 and N3.
- The local engineering dataset contains only four subjects.
- External-dataset generalization has not been established.
- The system is not validated for clinical diagnosis.

## 20. Future Work

- Multi-channel EEG modeling
- EOG and EMG integration
- Temporal sequence models
- Deep learning on raw EEG
- External validation
- Probability calibration
- Explainability and uncertainty analysis
- Model monitoring after deployment

## 21. Conclusion

The project currently provides a complete, contract-driven data and machine-learning implementation with explicit leakage prevention, deterministic artifacts, trusted model persistence, prediction storage, and MLflow infrastructure. The final conclusion will be completed after full-dataset model search, unbiased outer evaluation, final refit, database persistence, and MLflow evidence are validated.
