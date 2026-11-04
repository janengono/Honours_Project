"""
build_dataset.py
-----------------
Orchestrates the full Stage 1 preprocessing pipeline:
  1. Load Samples.csv index
  2. Pass 1 over the data: compute per-subject mean/std for EDA z-score
     normalization (must be computed across all of a subject's trials,
     not within a single 5.5s trial -- see eda_pipeline.zscore_normalize docstring)
  3. Pass 2: run ECG/EDA/EMG preprocessing per trial, assemble feature table
  4. Save the resulting feature matrix to data/processed/features.csv

This produces the tabular, handcrafted-feature dataset your causal analysis
(Section 3.4) will run on directly. If you also want raw/lightly-processed
waveforms for the Transformer's raw-input branch (Section 3.3.2), see the
`--save-waveforms` flag below.
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loading import load_config, load_samples_index, load_trial_biosignals, iterate_trials
from preprocessing.ecg_pipeline import extract_ecg_features
from preprocessing.eda_pipeline import extract_eda_features
from preprocessing.emg_pipeline import extract_emg_features


def compute_subject_eda_stats(cfg, samples_df):
    """
    Pass 1: for each subject, load all their raw GSR trials and compute
    a single mean/std to use for z-score normalization of every trial
    belonging to that subject.
    """
    print("Computing per-subject EDA normalization stats (pass 1/2)...")
    subject_stats = {}
    for subject_name, group in tqdm(samples_df.groupby("subject_name")):
        all_gsr = []
        for _, row in group.iterrows():
            try:
                df = load_trial_biosignals(cfg, row["subject_name"], row["sample_name"])
                all_gsr.append(df["gsr"].values)
            except (FileNotFoundError, ValueError):
                continue
        if all_gsr:
            concatenated = np.concatenate(all_gsr)
            subject_stats[subject_name] = (np.mean(concatenated), np.std(concatenated))
    return subject_stats


def build_dataset(config_path=None, limit=None, save_waveforms=False):
    cfg = load_config(config_path)
    samples_df = load_samples_index(cfg)
    if limit:
        samples_df = samples_df.head(limit)

    sampling_rate = cfg["signal"]["sampling_rate_hz"]
    subject_eda_stats = compute_subject_eda_stats(cfg, samples_df)

    rows = []
    waveform_store = {} if save_waveforms else None

    print("Extracting features per trial (pass 2/2)...")
    for subject_id, class_id, subject_name, sample_name, df in tqdm(
        iterate_trials(cfg, limit=limit), total=len(samples_df)
    ):
        row_features = {
            "subject_id": subject_id,
            "class_id": class_id,
            "subject_name": subject_name,
            "sample_name": sample_name,
        }

        # --- ECG ---
        ecg_feats, ecg_clean = extract_ecg_features(df["ecg"].values, sampling_rate, cfg)
        row_features.update(ecg_feats)

        # --- EDA ---
        subj_mean, subj_std = subject_eda_stats.get(subject_name, (None, None))
        eda_feats, eda_norm = extract_eda_features(
            df["gsr"].values, sampling_rate, cfg, subject_mean=subj_mean, subject_std=subj_std
        )
        row_features.update(eda_feats)

        # --- EMG (3 channels) ---
        emg_raw = {
            "emg_trapezius": df["emg_trapezius"].values,
            "emg_corrugator": df["emg_corrugator"].values,
            "emg_zygomaticus": df["emg_zygomaticus"].values,
        }
        emg_feats, emg_processed = extract_emg_features(emg_raw, sampling_rate, cfg)
        row_features.update(emg_feats)

        rows.append(row_features)

        if save_waveforms:
            key = f"{subject_name}_{sample_name}"
            waveform_store[key] = {
                "ecg": ecg_clean,
                "eda": eda_norm,
                **{f"emg_{ch}": sig for ch, sig in emg_processed.items()},
                "class_id": class_id,
                "subject_id": subject_id,
            }

    feature_df = pd.DataFrame(rows)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(this_dir, "..")
    out_dir = os.path.join(project_root, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "features.csv")
    feature_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(feature_df)} trials x {feature_df.shape[1]} columns to {out_path}")

    if save_waveforms:
        waveform_path = os.path.join(out_dir, "waveforms.npz")
        np.savez_compressed(waveform_path, **{
            f"{k}__{sig_name}": sig
            for k, d in waveform_store.items()
            for sig_name, sig in d.items()
            if isinstance(sig, np.ndarray)
        })
        print(f"Saved processed waveforms to {waveform_path}")

    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials, for quick testing")
    parser.add_argument("--save-waveforms", action="store_true")
    args = parser.parse_args()

    build_dataset(args.config, limit=args.limit, save_waveforms=args.save_waveforms)
