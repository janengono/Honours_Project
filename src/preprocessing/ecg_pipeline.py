

import numpy as np
import neurokit2 as nk


def bandpass_filter_ecg(raw_ecg, sampling_rate, low_hz=0.5, high_hz=40.0, order=4):
    """
    Apply the band-pass filter described in Eq. 3.2 of the proposal.
    neurokit2's ecg_clean already implements a validated Butterworth bandpass
    tuned for ECG; we pass explicit low/high cutoffs to match the config.
    """
    cleaned = nk.signal_filter(
        raw_ecg,
        sampling_rate=sampling_rate,
        lowcut=low_hz,
        highcut=high_hz,
        order=order,
        method="butterworth",
    )
    return cleaned


def extract_ecg_features(raw_ecg, sampling_rate, cfg):
    """
    Full ECG pipeline for one trial: filter -> R-peak detection -> HRV features.

    Returns a dict of scalar features summarizing this trial (5.5s window),
    plus the cleaned signal array for anyone who wants the raw waveform
    (e.g. for a raw-signal-input deep learning branch instead of handcrafted features).
    """
    low_hz = cfg["ecg"]["bandpass_low_hz"]
    high_hz = cfg["ecg"]["bandpass_high_hz"]
    order = cfg["ecg"]["filter_order"]

    cleaned = bandpass_filter_ecg(raw_ecg, sampling_rate, low_hz, high_hz, order)

    features = {"ecg_mean": np.mean(cleaned), "ecg_std": np.std(cleaned)}

    # 5.5s is very short for full HRV analysis (HRV normally wants >30s-2min),
    # so we compute what's numerically defensible here and flag the rest as
    # a documented limitation in your report rather than silently failing.
    try:
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_indices = rpeaks["ECG_R_Peaks"]

        if len(r_indices) >= 2:
            rr_intervals_ms = np.diff(r_indices) / sampling_rate * 1000
            features["hrv_mean_rr_ms"] = float(np.mean(rr_intervals_ms))
            features["hrv_sdnn_ms"] = float(np.std(rr_intervals_ms, ddof=1)) if len(rr_intervals_ms) > 1 else np.nan
            features["n_r_peaks"] = len(r_indices)
            features["heart_rate_bpm"] = float(60000 / np.mean(rr_intervals_ms))
        else:
            # Too few beats in this short window to compute RR-interval stats
            features["hrv_mean_rr_ms"] = np.nan
            features["hrv_sdnn_ms"] = np.nan
            features["n_r_peaks"] = len(r_indices)
            features["heart_rate_bpm"] = np.nan
    except Exception as e:
        # R-peak detection can fail on noisy trials -- don't crash the whole batch
        features["hrv_mean_rr_ms"] = np.nan
        features["hrv_sdnn_ms"] = np.nan
        features["n_r_peaks"] = 0
        features["heart_rate_bpm"] = np.nan
        features["ecg_error"] = str(e)

    return features, cleaned
