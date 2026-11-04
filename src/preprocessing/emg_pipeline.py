"""
emg_pipeline.py
----------------
Implements Section 3.2.2 (EMG Preprocessing) of the proposal:
  1. High-frequency noise filtering
  2. Rectification
  3. Signal smoothing
  4. Temporal normalization
  5. RMS activation (Eq. 3.4)

Applied independently to each of the three EMG channels Part B provides:
trapezius, corrugator, zygomaticus.
"""

import numpy as np
import neurokit2 as nk


def clean_emg_channel(raw_emg, sampling_rate, cfg):
    """
    Shared cleaning steps (filter -> rectify -> smooth) used by both the
    full-trial RMS feature extraction below and the sub-window segmentation
    pipeline, so both stages apply identical preprocessing.
    Returns the raw-scale (not z-scored) smoothed signal.
    """
    highpass_hz = cfg["emg"]["highpass_cutoff_hz"]
    order = cfg["emg"]["filter_order"]

    filtered = nk.signal_filter(
        raw_emg, sampling_rate=sampling_rate, lowcut=highpass_hz,
        order=order, method="butterworth",
    )
    rectified = np.abs(filtered)
    smooth_window_samples = max(1, int(sampling_rate * 0.05))
    smoothed = nk.signal_smooth(rectified, method="convolution", size=smooth_window_samples)
    return smoothed


def preprocess_emg_channel(raw_emg, sampling_rate, cfg):
    """
    Full EMG pipeline for a single channel in a single trial.
    Returns (features_dict_with_channel_agnostic_keys, processed_signal_array)

    IMPORTANT: RMS activation is computed from the rectified/smoothed signal
    BEFORE per-trial z-score normalization. Normalizing each 5.5s trial
    against its own mean/std would force every trial onto a similar scale,
    erasing exactly the amplitude differences between low-pain and high-pain
    trials this study needs. The z-scored version is still returned (useful
    for waveform-shape comparisons / as a raw-sequence model input) but is
    NOT used to compute the RMS summary features.
    """
    rms_window_ms = cfg["emg"]["rms_window_ms"]
    smoothed = clean_emg_channel(raw_emg, sampling_rate, cfg)

    # RMS activation (Eq. 3.4) computed on the RAW-SCALE smoothed signal,
    # preserving true amplitude differences between trials/classes
    rms_window_samples = max(1, int(sampling_rate * rms_window_ms / 1000))
    n_windows = len(smoothed) // rms_window_samples
    rms_values = [
        np.sqrt(np.mean(smoothed[i * rms_window_samples:(i + 1) * rms_window_samples] ** 2))
        for i in range(n_windows)
    ]
    rms_values = np.array(rms_values) if rms_values else np.array([0.0])

    # Temporal (z-score) normalization -- kept as a separate output for
    # waveform-shape use cases, not fed into the RMS features above
    std = np.std(smoothed)
    normalized = (smoothed - np.mean(smoothed)) / std if std > 0 else np.zeros_like(smoothed)

    features = {
        "rms_mean": float(np.mean(rms_values)),
        "rms_max": float(np.max(rms_values)),
        "rms_std": float(np.std(rms_values)),
    }
    return features, normalized


def extract_emg_features(row_signals, sampling_rate, cfg):
    """
    row_signals: dict mapping channel name -> raw array, e.g.
      {"emg_trapezius": arr, "emg_corrugator": arr, "emg_zygomaticus": arr}

    Returns a flat features dict prefixed by channel name, e.g. 'emg_trapezius_rms_mean'.
    """
    all_features = {}
    processed_signals = {}
    for channel_name in cfg["emg"]["channels"]:
        if channel_name not in row_signals:
            continue
        feats, processed = preprocess_emg_channel(row_signals[channel_name], sampling_rate, cfg)
        for k, v in feats.items():
            all_features[f"{channel_name}_{k}"] = v
        processed_signals[channel_name] = processed
    return all_features, processed_signals
