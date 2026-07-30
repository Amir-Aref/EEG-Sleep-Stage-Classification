# EEG Sleep Stage Classification - Completion Checklist

> **Overall progress:** 97.5% complete
>
> **Remaining:** 2.5%
>
> **Current blocking tasks:** final clean-clone and Docker audit,
> submission ZIP, PowerPoint, presentation video, and accessible video
> link.
>
> **Current Kaggle status:** all full-dataset compute stages completed
> and validated.

## Status Legend

- `[ ]` Not started
- `[-]` In progress
- `[x]` Completed and validated
- `[!]` Blocked or intentionally unavailable

---

## A. Repository and Data Pipeline

- [x] Select the integration worktree as the authoritative repository
- [x] Complete the Phase 2 download, preprocessing, feature, EDA, and database pipeline
- [x] Validate 78 subjects and 153 recordings
- [x] Validate 195,469 thirty-second epochs
- [x] Produce 28 leakage-safe model features
- [x] Preserve subject identifiers throughout the pipeline
- [x] Prevent random epoch-level splitting
- [x] Add deterministic manifests and SHA-256 provenance

## B. Full Nested Model Search

- [x] Run outer folds 1-3
- [x] Validate the Batch 1 checkpoint
- [x] Run outer folds 4-5
- [x] Validate the Batch 2 checkpoint
- [x] Evaluate all 29 configured candidates
- [x] Complete all 15 inner validation folds
- [x] Freeze one selected candidate per outer fold
- [x] Confirm test metrics were not used during model selection
- [x] Save complete search and selection artifacts

## C. Outer Test Evaluation

- [x] Train the selected candidate for each outer fold
- [x] Evaluate each subject exactly once in an unseen outer test fold
- [x] Generate 195,469 outer-test predictions
- [x] Save aligned probabilities for all five classes
- [x] Calculate fold-level and pooled metrics
- [x] Calculate per-class precision, recall, F1, and support
- [x] Create raw and normalized pooled confusion matrices
- [x] Confirm mean outer Macro-F1 of 0.658662
- [x] Confirm pooled Macro-F1 of 0.661936
- [x] Confirm pooled balanced accuracy of 0.670416

## D. Final Refit and Saved Model

- [x] Select `random_forest__candidate_002` without outer-test leakage
- [x] Refit the selected configuration on all 195,469 rows
- [x] Save the final end-to-end sklearn pipeline
- [x] Validate model file size and SHA-256
- [x] Validate predictions after model reload
- [x] Validate probabilities after model reload
- [x] Validate the exact Python 3.12 training runtime
- [x] Mark the final model deployment-ready
- [x] Mark the final artifacts as eligible for scientific reporting
- [!] Outer-fold fitted model files were not retained; complete
  outer-fold metrics and predictions were retained instead

## E. Prediction Pipeline and SQLite

- [x] Run full inference with the final deployment model
- [x] Generate predicted class and five class probabilities
- [x] Generate confidence, margin, and entropy fields
- [x] Preserve subject, recording, night, and epoch identifiers
- [x] Save full predictions to CSV
- [x] Save 195,469 predictions to SQLite
- [x] Validate foreign-key enforcement
- [x] Validate idempotent persistence
- [x] Validate database row counts and integrity
- [x] Record prediction-store provenance and checksums

## F. MLflow

- [x] Configure MLflow 3.14.0 with a local SQLite backend
- [x] Add local-validation and full-dataset safety scopes
- [x] Import local engineering artifacts
- [x] Import five full-dataset outer evaluation runs
- [x] Import the final full-dataset refit run
- [x] Log the final sklearn model with signature and input example
- [x] Register `EEG_Sleep_Stage_Classifier`
- [x] Register model version 1
- [x] Assign the `champion` alias to the final deployment model
- [x] Preserve a sanitized MLflow import summary
- [x] Validate MLflow configuration and import contracts
- [-] Capture final full-dataset MLflow UI screenshots for the presentation

## G. Scientific Analysis

- [x] Compare selected candidates across outer folds
- [x] Analyze fold-level variability
- [x] Analyze per-class precision, recall, and F1
- [x] Identify N1 as the weakest class
- [x] Rank frequent confusion pairs
- [x] Produce final-model feature importance
- [x] Create four final scientific figures
- [x] Discuss class imbalance
- [x] Discuss subject-level generalization
- [x] Document scientific limitations
- [x] State that the system is not a clinical diagnostic tool
- [!] Cross-fold feature-stability analysis is unavailable because
  outer-fold fitted models were not retained

## H. Documentation

- [x] Add full-dataset search, evaluation, and final-refit artifacts
- [x] Add the artifact provenance manifest
- [x] Add the final scientific Markdown report
- [x] Add the machine-readable scientific summary
- [x] Add feature-importance and confusion-pair tables
- [x] Add scientific figures
- [x] Document MLflow tracking and registry behavior
- [x] Document the database schema and SQL queries
- [x] Update README with final scientific results
- [x] Update this checklist to the completed project state
- [ ] Perform the final delivery-language and link review

## I. Validation and Delivery

- [x] Run 183 full-project tests locally
- [x] Pass GitHub Actions for the result and reporting pull requests
- [x] Validate committed files with `git diff --check`
- [x] Keep large model, prediction, input, and SQLite files outside Git
- [x] Record all external artifact sizes and SHA-256 values
- [ ] Build the final Docker image
- [ ] Run and smoke-test the final Docker image
- [ ] Validate a clean repository clone
- [ ] Review the final Git diff
- [ ] Remove temporary local validation resources
- [ ] Create and validate the final submission ZIP

## J. Presentation and Video

- [ ] Create the final PowerPoint
- [ ] Prepare a presentation script of at most 15 minutes
- [ ] Explain the dataset and prediction target
- [ ] Explain preprocessing and feature engineering
- [ ] Explain subject-safe nested evaluation
- [ ] Present final model results
- [ ] Demonstrate the MLflow UI
- [ ] Demonstrate saved-model inference
- [ ] Demonstrate SQLite prediction storage
- [ ] Discuss limitations and real-world interpretation
- [ ] Record the final video
- [ ] Upload the video to Google Drive
- [ ] Add the accessible link to `video_link.txt` or README

## Final Acceptance Gate

- [x] Required Phase 3 scientific artifacts exist
- [x] Checkpoints and final model have validated checksums
- [x] No outer-test leakage occurred
- [x] Final predictions exist in SQLite
- [x] MLflow tracking and registry import succeeded
- [x] Scientific documentation is complete
- [ ] Docker and clean-clone checks pass
- [ ] Presentation and video are complete
- [ ] Final ZIP opens and contains every required deliverable

## Progress Log

| Date | Task | Status | Evidence |
|---|---|---|---|
| 2026-07-27 | Repository and MLflow infrastructure review | Completed | PR #5 |
| 2026-07-28 | Grouped full-dataset split support | Completed | PR #6 |
| 2026-07-29 | Metrics-only full-dataset MLflow support | Completed | PR #7 |
| 2026-07-30 | Full-dataset artifacts and provenance | Completed | PR #8 |
| 2026-07-30 | Final scientific report and figures | Completed | PR #9 |
| 2026-07-30 | Full project local test suite | Completed | 183 tests passed |
