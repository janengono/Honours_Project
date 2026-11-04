"""
eda_pipeline.py
----------------
Implements Section 3.2.2 (EDA Preprocessing) of the proposal:
  1. Low-pass filtering
  2. Signal smoothing
  3. Tonic-phasic decomposition
  4. Z-score normalization (Eq. 3.3)

Note on the 'gsr' column: BioVid's raw GSR values are already in a
converted unit (microsiemens-like scale, based on the sample values you
shared e.g. 1.616), not raw ADC counts, so no additional unit conversion
is applied here.
"""

import numpy as np
import neurokit2 as nk


def lowpass_filter_eda(raw_gsr, sampling_rate, cutoff_hz=1.0, order=4):
    return nk.signal_filter(
        raw_gsr,
        sampling_rate=sampling_rate,
        lowcut=None,
        highcut=cutoff_hz,
        order=order,
        method="butterworth",
    )


def zscore_normalize(signal_array, subject_mean=None, subject_std=None):
    """
    Eq. 3.3: z = (x - mu) / sigma

    IMPORTANT: normalization should be computed PER SUBJECT across that
    subject's trials, not per-trial in isolation -- otherwise you lose the
    inter-trial variation you're trying to model (a 5.5s trial normalized
    against only itself will always have mean 0, std 1, destroying signal).
    Pass subject_mean/subject_std computed across all of that subject's
    raw trials (see build_dataset.py for how these get computed).
    """
    if subject_mean is None or subject_std is None:
        # fallback: normalize within-trial (only use this for a quick sanity check,
        # not for the final feature set)
        subject_mean = np.mean(signal_array)
        subject_std = np.std(signal_array)
    if subject_std == 0 or np.isnan(subject_std):
        return np.zeros_like(signal_array)
    return (signal_array - subject_mean) / subject_std


def extract_eda_features(raw_gsr, sampling_rate, cfg, subject_mean=None, subject_std=None):
    """
    Full EDA pipeline for one trial: lowpass filter -> tonic/phasic decomposition
    -> z-score normalize -> summary features.
    """
    cutoff_hz = cfg["eda"]["lowpass_cutoff_hz"]
    order = cfg["eda"]["filter_order"]

    filtered = lowpass_filter_eda(raw_gsr, sampling_rate, cutoff_hz, order)
    normalized = zscore_normalize(filtered, subject_mean, subject_std)

    features = {
        "eda_mean": float(np.mean(normalized)),
        "eda_std": float(np.std(normalized)),
        "eda_max": float(np.max(normalized)),
        "eda_min": float(np.min(normalized)),
        "eda_slope": float(np.polyfit(np.arange(len(normalized)), normalized, 1)[0]),
    }

    try:
        # Use the already per-subject-normalized signal, NOT a fresh per-trial
        # standardization (nk.standardize on `filtered` alone would re-scale
        # each trial to its own mean/std, erasing the amplitude differences
        # between low-pain and high-pain trials this study needs).
        decomposed = nk.eda_phasic(
            normalized, sampling_rate=sampling_rate,
            method=cfg["eda"]["decomposition_method"],
        )
        tonic = decomposed["EDA_Tonic"].values
        phasic = decomposed["EDA_Phasic"].values

        features["eda_tonic_mean"] = float(np.mean(tonic))
        features["eda_tonic_slope"] = float(np.polyfit(np.arange(len(tonic)), tonic, 1)[0])
        features["eda_phasic_mean"] = float(np.mean(phasic))
        features["eda_phasic_std"] = float(np.std(phasic))
        features["eda_phasic_max"] = float(np.max(phasic))  # peak SCR amplitude proxy

        # Count SCR (skin conductance response) peaks -- classic pain-response indicator
        peaks_info = nk.eda_peaks(phasic, sampling_rate=sampling_rate)[1]
        features["scr_count"] = int(len(peaks_info.get("SCR_Peaks", [])))
    except Exception as e:
        features["eda_tonic_mean"] = np.nan
        features["eda_tonic_slope"] = np.nan
        features["eda_phasic_mean"] = np.nan
        features["eda_phasic_std"] = np.nan
        features["eda_phasic_max"] = np.nan
        features["scr_count"] = np.nan
        features["eda_error"] = str(e)

    return features, normalized
