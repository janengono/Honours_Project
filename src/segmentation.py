"""
segmentation.py
----------------
Implements Section 3.2.3 (Temporal Segmentation) for the DEEP LEARNING input.

Each 5.5s trial from BioVid is already one labeled sample (that part of the
proposal's "segmentation" is done for you by the dataset itself). What this
module does is different: it cuts each 5.5s trial into a SEQUENCE of shorter,
overlapping sub-windows, so the Transformer receives a time series of tokens
rather than one flat vector -- this is what Section 3.3.2's "temporal
dependencies" and "attention over time" actually operate on.

Output shape per trial: (n_subwindows, n_features_per_subwindow)
Stacked across all trials: (n_trials, n_subwindows, n_features_per_subwindow)
"""

import numpy as np
import pandas as pd

from preprocessing.ecg_pipeline import bandpass_filter_ecg
from preprocessing.eda_pipeline import lowpass_filter_eda, zscore_normalize
from preprocessing.emg_pipeline import clean_emg_channel


def make_window_indices(n_samples, sampling_rate, window_sec, overlap_sec):
    """
    Returns a list of (start_idx, end_idx) pairs for sliding sub-windows
    across a signal of length n_samples.
    """
    window_samples = int(window_sec * sampling_rate)
    step_samples = int((window_sec - overlap_sec) * sampling_rate)
    if step_samples <= 0:
        raise ValueError("overlap_sec must be smaller than window_sec")

    indices = []
    start = 0
    while start + window_samples <= n_samples:
        indices.append((start, start + window_samples))
        start += step_samples
    return indices


def window_stats(segment):
    """Lightweight per-window summary: mean, std, min, max. Keeps the sequence
    dimensionality small (4 stats x 5 channels = 20 features/timestep) which
    is a sane starting point for a Transformer trained on ~8,600 trials --
    raw high-frequency samples per window would need far more data to avoid
    overfitting the attention layers."""
    return [
        float(np.mean(segment)),
        float(np.std(segment)),
        float(np.min(segment)),
        float(np.max(segment)),
    ]


def build_trial_sequence(df, sampling_rate, cfg, subject_eda_mean=None, subject_eda_std=None):
    """
    Cleans all 5 channels for one trial, then slices them into aligned
    overlapping sub-windows and computes summary stats per window per channel.

    Returns:
      sequence: np.ndarray of shape (n_windows, 20)  [4 stats x 5 channels]
      feature_names: list of 20 column names, e.g. 'ecg_mean', 'gsr_std', ...
    """
    seg_cfg = cfg["segmentation"]
    window_sec = seg_cfg["sub_window_sec"]
    overlap_sec = seg_cfg["overlap_sec"]

    # Clean each channel using the SAME cleaning steps as the trial-level
    # feature extraction, so sub-window features and trial-level features
    # are consistent with each other.
    ecg_clean = bandpass_filter_ecg(
        df["ecg"].values, sampling_rate,
        cfg["ecg"]["bandpass_low_hz"], cfg["ecg"]["bandpass_high_hz"], cfg["ecg"]["filter_order"],
    )
    eda_filtered = lowpass_filter_eda(
        df["gsr"].values, sampling_rate, cfg["eda"]["lowpass_cutoff_hz"], cfg["eda"]["filter_order"],
    )
    eda_clean = zscore_normalize(eda_filtered, subject_eda_mean, subject_eda_std)

    channels = {
        "ecg": ecg_clean,
        "gsr": eda_clean,
        "emg_trapezius": clean_emg_channel(df["emg_trapezius"].values, sampling_rate, cfg),
        "emg_corrugator": clean_emg_channel(df["emg_corrugator"].values, sampling_rate, cfg),
        "emg_zygomaticus": clean_emg_channel(df["emg_zygomaticus"].values, sampling_rate, cfg),
    }

    n_samples = len(df)
    win_indices = make_window_indices(n_samples, sampling_rate, window_sec, overlap_sec)
    if not win_indices:
        raise ValueError(
            f"Trial too short ({n_samples} samples) for a {window_sec}s window "
            f"at {sampling_rate}Hz. Reduce sub_window_sec in config.yaml."
        )

    stat_names = ["mean", "std", "min", "max"]
    feature_names = [f"{ch}_{stat}" for ch in channels for stat in stat_names]

    sequence = []
    for (start, end) in win_indices:
        row = []
        for ch_signal in channels.values():
            row.extend(window_stats(ch_signal[start:end]))
        sequence.append(row)

    return np.array(sequence, dtype=np.float32), feature_names


if __name__ == "__main__":
    # Smoke test on one trial
    from data_loading import load_config, load_samples_index, load_trial_biosignals

    cfg = load_config()
    samples = load_samples_index(cfg)
    row = samples.iloc[0]
    df = load_trial_biosignals(cfg, row["subject_name"], row["sample_name"])

    seq, names = build_trial_sequence(df, cfg["signal"]["sampling_rate_hz"], cfg)
    print(f"Trial: {row['subject_name']}/{row['sample_name']}")
    print(f"Sequence shape: {seq.shape}  (n_subwindows, n_features)")
    print(f"Feature names ({len(names)}): {names}")
    print(f"First window:\n{seq[0]}")
