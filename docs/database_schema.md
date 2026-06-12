# Database Schema Documentation

## Database

SQLite database file:

`database/sleep_eeg.db`

## Main Tables

### subjects

Stores subject-level metadata.

Columns:

- `subject_id`: primary key
- `source_dataset`: original dataset name
- `recording_id`: recording identifier
- `notes`: optional description

### eeg_epochs

Stores one row per 30-second EEG epoch.

Columns:

- `id`: auto-increment primary key
- `subject_id`: foreign key to subjects
- `epoch_id`: epoch number
- `start_time_sec`: epoch start time in seconds
- `eeg_channel`: EEG channel name
- `mean`, `std`, `min`, `max`: time-domain features
- `signal_energy`: energy of EEG signal in the epoch
- `delta_power`, `theta_power`, `alpha_power`, `beta_power`: frequency-band powers
- `sleep_stage`: cleaned label for modeling
- `sleep_stage_raw`: original hypnogram annotation

## Design Rationale

This schema separates subject-level metadata from epoch-level EEG features.

It supports future expansion to multiple subjects and multiple EEG channels.

The main table, `eeg_epochs`, is designed to be directly usable by the preprocessing and modeling pipeline.
