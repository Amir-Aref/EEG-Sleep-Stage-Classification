EEG-Based Sleep Stage Classification - Phase 2

Project Overview
This repository contains the Phase 2 data pipeline for an EEG Sleep Stage Classification project.The goal of this phase is to build a clean and reproducible data pipeline that prepares EEG features for machine learning.Phase 2 focuses on database implementation, data preprocessing, feature engineering, pipeline automation, and CI/CD .


Important scope notes:
Model training is NOT part of Phase 2. 
Real-time systems are not implemented in this phase.
All file paths must remain relative (no local absolute paths).


Dataset
Name: Sleep-EDF Database Expanded.
Source URL: https://www.physionet.org/content/sleep-edfx/1.0.0/.
Input Data: data/sample/sleep_edf_sample_features_subject0.csv.

This dataset contains EEG features extracted from signals. Important columns for feature engineering include delta_power, theta_power, alpha_power, beta_power, signal_energy, and sleep_stage .


Project Structure
database/: Contains the SQLite database (sleep_eeg.db).
scripts/: Contains data processing scripts like preprocess.py and feature_engineering.py.
outputs/: Contains the generated datasets (preprocessed_features.csv and model_ready_dataset.csv) .
docs/: Contains project documentation and the CI/CD success screenshot (docs/screenshots/github_actions_success.png) .
.github/workflows/: Contains the GitHub Actions CI setup (data-pipeline-ci.yml) .
pipeline.py: The main script to run the entire data pipeline.
requirements.txt: Project dependencies.


How to Install
To avoid system Python restrictions, the project should run inside a virtual environment.
Create and activate a virtual environment:
python -m venv .venv source .venv/bin/activate (On Windows use: ..venv\Scripts\Activate.ps1)
Install the required libraries:
pip install -r requirements.txt 



How to Run the Pipeline
The entire Phase 2 pipeline is automated. You can run the database processes, preprocessing, and feature engineering with a single command:
python pipeline.py After execution, the outputs/ folder will contain the final model_ready_dataset.csv .


Database Implementation
The project utilizes an SQLite database (sleep_eeg.db). The Database Lead is responsible for maintaining the schema and writing SQL queries to analyze sleep stage distributions, band power statistics by sleep stage, and row counts per subject.

Preprocessing and Feature EngineeringPreprocessing: Loads the sample EEG dataset, removes rows with invalid sleep stage labels, handles missing values, and keeps all important EEG feature columns .
Feature Engineering: Calculates additional derived features to improve model performance. This includes Total Power, relative band powers (delta, theta, alpha, beta), specific ratios (delta_theta_ratio, alpha_beta_ratio), and log features (log_signal_energy) .
Label Encoding: Sleep stages are converted from text to integers (Wake -> 0, N1 -> 1, N2 -> 2, N3 -> 3, REM -> 4) in the sleep_stage_encoded column . The final model-ready dataset is stripped of IDs and raw text labels . 



CI/CD Automation
This project uses GitHub Actions for Continuous Integration . The workflow is configured to run automatically on every push and pull request to the main branch, ensuring the data pipeline executes reliably without errors .