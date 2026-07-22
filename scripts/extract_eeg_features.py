"""Extract validated EEG features from Sleep-EDF epochs.

The extractor:
- processes one recording at a time
- filters each continuous EEG recording before segmentation
- calculates time-domain and frequency-domain features
- preserves metadata separately from model features
- records quality flags without silently deleting epochs
- supports resumable per-recording feature files
- performs no machine-learning scaling
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Final

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

try:
    from .config import (
        AMPLITUDE_ARTIFACT_THRESHOLD_UV,
        DEFAULT_EEG_CHANNEL,
        EEG_FILTER_HIGH_HZ,
        EEG_FILTER_LOW_HZ,
        EEG_FREQUENCY_BANDS,
        EPOCH_DURATION_SECONDS,
        EPOCH_FEATURES_PATH,
        EPOCH_METADATA_PATH,
        FEATURE_EXTRACTION_SUMMARY_PATH,
        FEATURE_PARTS_DIR,
        FEATURE_SCHEMA_PATH,
        FLATLINE_STD_THRESHOLD_UV,
        SLEEP_EDFX_RAW_DIR,
        WELCH_OVERLAP_FRACTION,
        WELCH_WINDOW_SECONDS,
        ensure_runtime_directories,
    )
except ImportError:
    from config import (
        AMPLITUDE_ARTIFACT_THRESHOLD_UV,
        DEFAULT_EEG_CHANNEL,
        EEG_FILTER_HIGH_HZ,
        EEG_FILTER_LOW_HZ,
        EEG_FREQUENCY_BANDS,
        EPOCH_DURATION_SECONDS,
        EPOCH_FEATURES_PATH,
        EPOCH_METADATA_PATH,
        FEATURE_EXTRACTION_SUMMARY_PATH,
        FEATURE_PARTS_DIR,
        FEATURE_SCHEMA_PATH,
        FLATLINE_STD_THRESHOLD_UV,
        SLEEP_EDFX_RAW_DIR,
        WELCH_OVERLAP_FRACTION,
        WELCH_WINDOW_SECONDS,
        ensure_runtime_directories,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MICROVOLTS_PER_VOLT: Final[float] = 1_000_000.0

METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "subject_id",
    "night",
    "recording_id",
    "epoch_id",
    "start_time_sec",
    "end_time_sec",
    "start_sample",
    "stop_sample",
    "sampling_frequency_hz",
    "epoch_duration_sec",
    "eeg_channel",
    "sleep_stage_raw",
    "sleep_stage",
    "sleep_stage_encoded",
)

QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "sample_count",
    "is_finite_signal",
    "flatline_flag",
    "amplitude_artifact_flag",
    "quality_issue_flag",
)

TIME_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "mean_uv",
    "std_uv",
    "median_uv",
    "min_uv",
    "max_uv",
    "rms_uv",
    "peak_to_peak_uv",
    "mean_square_uv2",
    "signal_energy_uv2",
    "zero_crossing_rate",
    "line_length_uv",
    "skewness",
    "kurtosis_excess",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)

SPECTRAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "delta_power_uv2",
    "theta_power_uv2",
    "alpha_power_uv2",
    "sigma_power_uv2",
    "beta_power_uv2",
    "total_band_power_uv2",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_sigma_power",
    "relative_beta_power",
    "delta_theta_ratio",
    "theta_alpha_ratio",
    "alpha_beta_ratio",
    "sigma_beta_ratio",
    "spectral_entropy",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_edge_frequency_95_hz",
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *TIME_FEATURE_COLUMNS,
    *SPECTRAL_FEATURE_COLUMNS,
)

FLOAT_TOLERANCE: Final[float] = 1e-10


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract EEG features from validated Sleep-EDF epochs."
    )

    selection = parser.add_mutually_exclusive_group(
        required=True
    )

    selection.add_argument(
        "--recording-id",
        action="append",
        default=[],
        help=(
            "Recording ID such as SC4001. "
            "May be supplied more than once."
        ),
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Process all recordings present in epoch metadata.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing per-recording feature files.",
    )

    return parser.parse_args()


def load_epoch_metadata() -> pd.DataFrame:
    """Load and validate authoritative epoch metadata."""

    if not EPOCH_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Epoch metadata not found. Run "
            "`python scripts/build_epoch_metadata.py` first."
        )

    metadata = pd.read_csv(EPOCH_METADATA_PATH)

    required_columns = set(METADATA_COLUMNS).difference(
        {"eeg_channel"}
    )

    missing_columns = sorted(
        required_columns.difference(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Epoch metadata is missing columns: {missing_columns}"
        )

    if metadata[
        ["recording_id", "epoch_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate recording/epoch IDs detected in metadata."
        )

    return metadata


def select_recordings(
    metadata: pd.DataFrame,
    recording_ids: list[str],
    process_all: bool,
) -> list[str]:
    """Select recordings deterministically."""

    available = sorted(
        metadata["recording_id"].astype(str).unique()
    )

    if process_all:
        return available

    normalized = [
        recording_id.strip().upper()
        for recording_id in recording_ids
    ]

    missing = sorted(set(normalized) - set(available))

    if missing:
        raise ValueError(
            f"Recording IDs not found in epoch metadata: {missing}"
        )

    return list(dict.fromkeys(normalized))


def resolve_psg_path(recording_id: str) -> Path:
    """Resolve one PSG EDF file for a recording."""

    candidates = sorted(
        SLEEP_EDFX_RAW_DIR.glob(
            f"{recording_id}*-PSG.edf"
        )
    )

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one PSG for {recording_id}, "
            f"found {len(candidates)}."
        )

    return candidates[0]


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Calculate a ratio without introducing a fixed epsilon."""

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return float("nan")

    return float(numerator / denominator)


def calculate_hjorth(
    signal: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate Hjorth activity, mobility and complexity."""

    first_difference = np.diff(signal)
    second_difference = np.diff(first_difference)

    variance_signal = float(np.var(signal, ddof=0))
    variance_first = float(
        np.var(first_difference, ddof=0)
    )
    variance_second = float(
        np.var(second_difference, ddof=0)
    )

    activity = variance_signal

    if variance_signal <= 0.0:
        return activity, 0.0, 0.0

    mobility = math.sqrt(
        variance_first / variance_signal
    )

    if variance_first <= 0.0 or mobility <= 0.0:
        return activity, mobility, 0.0

    mobility_first = math.sqrt(
        variance_second / variance_first
    )

    complexity = mobility_first / mobility

    return (
        float(activity),
        float(mobility),
        float(complexity),
    )


def calculate_time_features(
    filtered_uv: np.ndarray,
) -> dict[str, float]:
    """Calculate time-domain EEG features."""

    mean_value = float(np.mean(filtered_uv))
    std_value = float(np.std(filtered_uv, ddof=0))
    rms_value = float(
        np.sqrt(np.mean(filtered_uv ** 2))
    )

    (
        hjorth_activity,
        hjorth_mobility,
        hjorth_complexity,
    ) = calculate_hjorth(filtered_uv)

    zero_crossings = np.diff(
        np.signbit(filtered_uv)
    ) != 0

    return {
        "mean_uv": mean_value,
        "std_uv": std_value,
        "median_uv": float(np.median(filtered_uv)),
        "min_uv": float(np.min(filtered_uv)),
        "max_uv": float(np.max(filtered_uv)),
        "rms_uv": rms_value,
        "peak_to_peak_uv": float(
            np.ptp(filtered_uv)
        ),
        "mean_square_uv2": float(
            np.mean(filtered_uv ** 2)
        ),
        "signal_energy_uv2": float(
            np.sum(filtered_uv ** 2)
        ),
        "zero_crossing_rate": float(
            np.mean(zero_crossings)
        ),
        "line_length_uv": float(
            np.sum(np.abs(np.diff(filtered_uv)))
        ),
        "skewness": float(
            skew(
                filtered_uv,
                bias=False,
                nan_policy="raise",
            )
        ),
        "kurtosis_excess": float(
            kurtosis(
                filtered_uv,
                fisher=True,
                bias=False,
                nan_policy="raise",
            )
        ),
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }


def create_welch_psd(
    filtered_uv: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Welch PSD for one epoch."""

    nperseg = int(
        round(
            WELCH_WINDOW_SECONDS
            * sampling_frequency
        )
    )

    nperseg = min(nperseg, len(filtered_uv))

    noverlap = int(
        round(
            nperseg
            * WELCH_OVERLAP_FRACTION
        )
    )

    noverlap = min(noverlap, nperseg - 1)

    frequencies, psd = welch(
        filtered_uv,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    if (
        len(frequencies) < 2
        or len(psd) != len(frequencies)
    ):
        raise ValueError("Invalid Welch PSD output.")

    if not np.isfinite(psd).all():
        raise ValueError(
            "Welch PSD contains non-finite values."
        )

    return frequencies, psd


def integrate_band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
    include_high: bool,
) -> float:
    """Integrate one non-overlapping frequency band."""

    if include_high:
        mask = (
            (frequencies >= low_hz)
            & (frequencies <= high_hz)
        )
    else:
        mask = (
            (frequencies >= low_hz)
            & (frequencies < high_hz)
        )

    selected_psd = psd[mask]

    if selected_psd.size == 0:
        raise ValueError(
            f"No PSD bins found for band {low_hz}-{high_hz} Hz."
        )

    frequency_resolution = float(
        frequencies[1] - frequencies[0]
    )

    return float(
        selected_psd.sum()
        * frequency_resolution
    )


def calculate_spectral_features(
    filtered_uv: np.ndarray,
    sampling_frequency: float,
) -> dict[str, float]:
    """Calculate spectral EEG features."""

    frequencies, psd = create_welch_psd(
        filtered_uv,
        sampling_frequency,
    )

    absolute_powers: dict[str, float] = {}

    band_items = list(
        EEG_FREQUENCY_BANDS.items()
    )

    for index, (band_name, limits) in enumerate(
        band_items
    ):
        absolute_powers[band_name] = integrate_band_power(
            frequencies=frequencies,
            psd=psd,
            low_hz=float(limits[0]),
            high_hz=float(limits[1]),
            include_high=(
                index == len(band_items) - 1
            ),
        )

    total_power = float(
        sum(absolute_powers.values())
    )

    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError(
            f"Invalid total band power: {total_power}"
        )

    relative_powers = {
        band_name: float(power / total_power)
        for band_name, power in absolute_powers.items()
    }

    relative_sum = float(
        sum(relative_powers.values())
    )

    if not math.isclose(
        relative_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            f"Relative band powers do not sum to one: "
            f"{relative_sum}"
        )

    analysis_low = min(
        limits[0]
        for limits in EEG_FREQUENCY_BANDS.values()
    )

    analysis_high = max(
        limits[1]
        for limits in EEG_FREQUENCY_BANDS.values()
    )

    analysis_mask = (
        (frequencies >= analysis_low)
        & (frequencies <= analysis_high)
    )

    analysis_frequencies = frequencies[
        analysis_mask
    ]

    analysis_psd = psd[analysis_mask]

    psd_sum = float(analysis_psd.sum())

    if psd_sum <= 0.0:
        raise ValueError(
            "PSD sum is not positive in analysis range."
        )

    probability = analysis_psd / psd_sum
    positive_probability = probability[
        probability > 0.0
    ]

    spectral_entropy = float(
        -np.sum(
            positive_probability
            * np.log2(positive_probability)
        )
        / np.log2(len(probability))
    )

    dominant_frequency = float(
        analysis_frequencies[
            int(np.argmax(analysis_psd))
        ]
    )

    spectral_centroid = float(
        np.sum(
            analysis_frequencies
            * analysis_psd
        )
        / psd_sum
    )

    cumulative_power = np.cumsum(analysis_psd)
    edge_target = 0.95 * cumulative_power[-1]

    edge_index = int(
        np.searchsorted(
            cumulative_power,
            edge_target,
            side="left",
        )
    )

    edge_index = min(
        edge_index,
        len(analysis_frequencies) - 1,
    )

    spectral_edge_frequency = float(
        analysis_frequencies[edge_index]
    )

    return {
        "delta_power_uv2": absolute_powers["delta"],
        "theta_power_uv2": absolute_powers["theta"],
        "alpha_power_uv2": absolute_powers["alpha"],
        "sigma_power_uv2": absolute_powers["sigma"],
        "beta_power_uv2": absolute_powers["beta"],
        "total_band_power_uv2": total_power,
        "relative_delta_power": relative_powers["delta"],
        "relative_theta_power": relative_powers["theta"],
        "relative_alpha_power": relative_powers["alpha"],
        "relative_sigma_power": relative_powers["sigma"],
        "relative_beta_power": relative_powers["beta"],
        "delta_theta_ratio": safe_ratio(
            absolute_powers["delta"],
            absolute_powers["theta"],
        ),
        "theta_alpha_ratio": safe_ratio(
            absolute_powers["theta"],
            absolute_powers["alpha"],
        ),
        "alpha_beta_ratio": safe_ratio(
            absolute_powers["alpha"],
            absolute_powers["beta"],
        ),
        "sigma_beta_ratio": safe_ratio(
            absolute_powers["sigma"],
            absolute_powers["beta"],
        ),
        "spectral_entropy": spectral_entropy,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_edge_frequency_95_hz": (
            spectral_edge_frequency
        ),
    }


def build_filter(
    sampling_frequency: float,
) -> np.ndarray:
    """Create a stable Butterworth band-pass filter."""

    nyquist = sampling_frequency / 2.0

    if EEG_FILTER_HIGH_HZ >= nyquist:
        raise ValueError(
            "Filter high cutoff must be below Nyquist frequency."
        )

    return butter(
        N=4,
        Wn=(
            EEG_FILTER_LOW_HZ,
            EEG_FILTER_HIGH_HZ,
        ),
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )


def extract_recording_features(
    recording_id: str,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Extract all epoch features from one recording."""

    recording_metadata = metadata[
        metadata["recording_id"] == recording_id
    ].copy()

    recording_metadata = recording_metadata.sort_values(
        "epoch_id",
        kind="stable",
    ).reset_index(drop=True)

    if recording_metadata.empty:
        raise ValueError(
            f"No epoch metadata found for {recording_id}."
        )

    psg_path = resolve_psg_path(recording_id)

    logger.info(
        "Loading channel %s from %s",
        DEFAULT_EEG_CHANNEL,
        psg_path.name,
    )

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False,
        verbose="ERROR",
    )

    if DEFAULT_EEG_CHANNEL not in raw.ch_names:
        raise ValueError(
            f"Required channel {DEFAULT_EEG_CHANNEL!r} "
            f"is missing in {recording_id}."
        )

    raw.pick([DEFAULT_EEG_CHANNEL])
    raw.load_data(verbose="ERROR")

    sampling_frequency = float(
        raw.info["sfreq"]
    )

    expected_frequencies = recording_metadata[
        "sampling_frequency_hz"
    ].unique()

    if (
        len(expected_frequencies) != 1
        or not math.isclose(
            float(expected_frequencies[0]),
            sampling_frequency,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    ):
        raise ValueError(
            f"Sampling-frequency mismatch for {recording_id}."
        )

    continuous_uv = (
        raw.get_data()[0]
        * MICROVOLTS_PER_VOLT
    )

    if not np.isfinite(continuous_uv).all():
        raise ValueError(
            f"Non-finite raw EEG samples in {recording_id}."
        )

    sos = build_filter(sampling_frequency)

    logger.info(
        "Filtering continuous EEG for %s",
        recording_id,
    )

    filtered_continuous_uv = sosfiltfilt(
        sos,
        continuous_uv,
    )

    expected_sample_count = int(
        round(
            EPOCH_DURATION_SECONDS
            * sampling_frequency
        )
    )

    records: list[dict[str, object]] = []

    for index, epoch in enumerate(
        recording_metadata.itertuples(index=False),
        start=1,
    ):
        start_sample = int(epoch.start_sample)
        stop_sample = int(epoch.stop_sample)

        raw_epoch_uv = continuous_uv[
            start_sample:stop_sample
        ]

        filtered_epoch_uv = filtered_continuous_uv[
            start_sample:stop_sample
        ]

        if len(filtered_epoch_uv) != expected_sample_count:
            raise ValueError(
                f"Unexpected sample count for "
                f"{recording_id}/{epoch.epoch_id}: "
                f"{len(filtered_epoch_uv)}"
            )

        finite_signal = bool(
            np.isfinite(raw_epoch_uv).all()
            and np.isfinite(filtered_epoch_uv).all()
        )

        if not finite_signal:
            raise ValueError(
                f"Non-finite epoch signal for "
                f"{recording_id}/{epoch.epoch_id}."
            )

        raw_max_absolute_uv = float(
            np.max(np.abs(raw_epoch_uv))
        )

        filtered_std_uv = float(
            np.std(filtered_epoch_uv, ddof=0)
        )

        flatline_flag = bool(
            filtered_std_uv
            < FLATLINE_STD_THRESHOLD_UV
        )

        amplitude_artifact_flag = bool(
            raw_max_absolute_uv
            > AMPLITUDE_ARTIFACT_THRESHOLD_UV
        )

        record: dict[str, object] = {
            "subject_id": int(epoch.subject_id),
            "night": int(epoch.night),
            "recording_id": recording_id,
            "epoch_id": int(epoch.epoch_id),
            "start_time_sec": float(epoch.start_time_sec),
            "end_time_sec": float(epoch.end_time_sec),
            "start_sample": start_sample,
            "stop_sample": stop_sample,
            "sampling_frequency_hz": sampling_frequency,
            "epoch_duration_sec": float(
                epoch.epoch_duration_sec
            ),
            "eeg_channel": DEFAULT_EEG_CHANNEL,
            "sleep_stage_raw": str(
                epoch.sleep_stage_raw
            ),
            "sleep_stage": str(epoch.sleep_stage),
            "sleep_stage_encoded": int(
                epoch.sleep_stage_encoded
            ),
            "sample_count": int(
                len(filtered_epoch_uv)
            ),
            "is_finite_signal": finite_signal,
            "flatline_flag": flatline_flag,
            "amplitude_artifact_flag": (
                amplitude_artifact_flag
            ),
            "quality_issue_flag": bool(
                flatline_flag
                or amplitude_artifact_flag
            ),
        }

        record.update(
            calculate_time_features(
                filtered_epoch_uv
            )
        )

        record.update(
            calculate_spectral_features(
                filtered_epoch_uv,
                sampling_frequency,
            )
        )

        records.append(record)

        if index % 100 == 0 or index == len(
            recording_metadata
        ):
            logger.info(
                "%s: extracted %d/%d epochs",
                recording_id,
                index,
                len(recording_metadata),
            )

    features = pd.DataFrame(records)

    validate_feature_dataframe(
        features,
        expected_rows=len(recording_metadata),
    )

    return features


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    """Validate feature-table invariants."""

    if len(dataframe) != expected_rows:
        raise ValueError(
            "Feature row count does not match epoch metadata."
        )

    expected_columns = {
        *METADATA_COLUMNS,
        *QUALITY_COLUMNS,
        *FEATURE_COLUMNS,
    }

    missing_columns = sorted(
        expected_columns.difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Feature table is missing columns: {missing_columns}"
        )

    if dataframe[
        ["recording_id", "epoch_id"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate recording/epoch IDs in feature table."
        )

    if not dataframe["is_finite_signal"].all():
        raise ValueError(
            "Feature table contains non-finite signals."
        )

    numeric_feature_values = dataframe[
        list(FEATURE_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(
        numeric_feature_values
    ).all():
        bad_columns = dataframe[
            list(FEATURE_COLUMNS)
        ].columns[
            ~np.isfinite(
                numeric_feature_values
            ).all(axis=0)
        ].tolist()

        raise ValueError(
            f"Non-finite feature values detected in: "
            f"{bad_columns}"
        )

    relative_columns = [
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_sigma_power",
        "relative_beta_power",
    ]

    relative_sum = dataframe[
        relative_columns
    ].sum(axis=1)

    if not np.allclose(
        relative_sum.to_numpy(),
        1.0,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            "Relative band-power sums are invalid."
        )


def partial_path_for(recording_id: str) -> Path:
    """Return the per-recording feature output path."""

    return (
        FEATURE_PARTS_DIR
        / f"{recording_id}_epoch_features.csv"
    )


def save_feature_schema() -> None:
    """Save explicit metadata, quality and feature column roles."""

    schema = {
        "metadata_columns": list(METADATA_COLUMNS),
        "quality_columns": list(QUALITY_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "sleep_stage_encoded",
        "target_label_column": "sleep_stage",
        "signal_channel": DEFAULT_EEG_CHANNEL,
        "signal_unit": "microvolts",
        "filter": {
            "type": "Butterworth SOS band-pass",
            "order": 4,
            "low_hz": EEG_FILTER_LOW_HZ,
            "high_hz": EEG_FILTER_HIGH_HZ,
            "application": (
                "continuous recording before epoch slicing"
            ),
        },
        "welch": {
            "window": "hann",
            "window_seconds": WELCH_WINDOW_SECONDS,
            "overlap_fraction": WELCH_OVERLAP_FRACTION,
            "detrend": "constant",
            "scaling": "density",
        },
        "frequency_bands_hz": {
            name: list(limits)
            for name, limits in EEG_FREQUENCY_BANDS.items()
        },
        "scaling_applied": False,
    }

    FEATURE_SCHEMA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def build_summary(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Build one feature-extraction summary record."""

    recording_id = str(
        dataframe["recording_id"].iloc[0]
    )

    class_counts = (
        dataframe["sleep_stage"]
        .value_counts()
        .to_dict()
    )

    summary: dict[str, object] = {
        "recording_id": recording_id,
        "subject_id": int(
            dataframe["subject_id"].iloc[0]
        ),
        "night": int(
            dataframe["night"].iloc[0]
        ),
        "epoch_count": int(len(dataframe)),
        "quality_issue_count": int(
            dataframe["quality_issue_flag"].sum()
        ),
        "flatline_count": int(
            dataframe["flatline_flag"].sum()
        ),
        "amplitude_artifact_count": int(
            dataframe[
                "amplitude_artifact_flag"
            ].sum()
        ),
        "median_std_uv": float(
            dataframe["std_uv"].median()
        ),
        "median_total_band_power_uv2": float(
            dataframe[
                "total_band_power_uv2"
            ].median()
        ),
    }

    for stage in ("Wake", "N1", "N2", "N3", "REM"):
        summary[f"count_{stage.lower()}"] = int(
            class_counts.get(stage, 0)
        )

    return summary


def main() -> None:
    """Run feature extraction."""

    arguments = parse_arguments()
    ensure_runtime_directories()
    FEATURE_PARTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_epoch_metadata()

    recording_ids = select_recordings(
        metadata=metadata,
        recording_ids=arguments.recording_id,
        process_all=arguments.all,
    )

    logger.info(
        "Selected recordings: %s",
        recording_ids,
    )

    summaries: list[dict[str, object]] = []
    selected_feature_frames: list[pd.DataFrame] = []

    for recording_id in recording_ids:
        output_path = partial_path_for(recording_id)

        expected_rows = int(
            (
                metadata["recording_id"]
                == recording_id
            ).sum()
        )

        if output_path.exists() and not arguments.overwrite:
            logger.info(
                "Loading existing feature part: %s",
                output_path,
            )

            features = pd.read_csv(output_path)

            validate_feature_dataframe(
                features,
                expected_rows=expected_rows,
            )
        else:
            features = extract_recording_features(
                recording_id,
                metadata,
            )

            features.to_csv(
                output_path,
                index=False,
                lineterminator="\n",
            )

            logger.info(
                "Saved feature part: %s",
                output_path,
            )

        selected_feature_frames.append(features)
        summaries.append(build_summary(features))

    combined_selected = pd.concat(
        selected_feature_frames,
        ignore_index=True,
    )

    combined_selected = combined_selected.sort_values(
        [
            "subject_id",
            "night",
            "recording_id",
            "epoch_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    validate_feature_dataframe(
        combined_selected,
        expected_rows=int(
            metadata[
                metadata["recording_id"].isin(
                    recording_ids
                )
            ].shape[0]
        ),
    )

    summary_dataframe = pd.DataFrame(
        summaries
    ).sort_values(
        ["subject_id", "night", "recording_id"],
        kind="stable",
    ).reset_index(drop=True)

    FEATURE_EXTRACTION_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dataframe.to_csv(
        FEATURE_EXTRACTION_SUMMARY_PATH,
        index=False,
        lineterminator="\n",
    )

    save_feature_schema()

    if arguments.all:
        combined_selected.to_csv(
            EPOCH_FEATURES_PATH,
            index=False,
            lineterminator="\n",
        )

        logger.info(
            "Saved complete feature dataset: %s",
            EPOCH_FEATURES_PATH,
        )

    display_columns = [
        "recording_id",
        "epoch_count",
        "quality_issue_count",
        "flatline_count",
        "amplitude_artifact_count",
        "median_std_uv",
        "median_total_band_power_uv2",
    ]

    print("\n=== FEATURE EXTRACTION SUMMARY ===")
    print(
        summary_dataframe[
            display_columns
        ].to_string(index=False)
    )

    relative_columns = [
        "relative_delta_power",
        "relative_theta_power",
        "relative_alpha_power",
        "relative_sigma_power",
        "relative_beta_power",
    ]

    relative_sum = combined_selected[
        relative_columns
    ].sum(axis=1)

    print("\n=== VALIDATION ===")
    print("Recordings:", len(recording_ids))
    print("Epochs:", len(combined_selected))
    print("Feature columns:", len(FEATURE_COLUMNS))
    print(
        "Relative-power sum min:",
        float(relative_sum.min()),
    )
    print(
        "Relative-power sum max:",
        float(relative_sum.max()),
    )
    print(
        "Non-finite features:",
        int(
            (
                ~np.isfinite(
                    combined_selected[
                        list(FEATURE_COLUMNS)
                    ].to_numpy(dtype=float)
                )
            ).sum()
        ),
    )
    print(
        "Quality issues:",
        int(
            combined_selected[
                "quality_issue_flag"
            ].sum()
        ),
    )

    print("\nSample feature rows:")
    print(
        combined_selected[
            [
                "recording_id",
                "epoch_id",
                "sleep_stage",
                "std_uv",
                "relative_delta_power",
                "relative_theta_power",
                "relative_alpha_power",
                "relative_sigma_power",
                "relative_beta_power",
                "spectral_entropy",
                "dominant_frequency_hz",
                "quality_issue_flag",
            ]
        ].head().to_string(index=False)
    )

    print("\nEEG feature extraction validation: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "EEG feature extraction failed."
        )
        sys.exit(1)
