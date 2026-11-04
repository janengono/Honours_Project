"""
build_sequences.py
-------------------
Runs segmentation.build_trial_sequence() across every trial in the dataset
and saves the result as a single compressed .npz, ready to feed the
multimodal Transformer (Section 3.3).

Output: data/processed/sequences.npz containing:
  X            -> shape (n_trials, n_windows, n_features), float32
  y            -> shape (n_trials,), the class_id (0-4) per trial
  subject_ids  -> shape (n_trials,), for building splits later
  sample_names -> shape (n_trials,), for traceability back to raw files
  feature_names-> the 20 column names describing the last axis of X
"""

import os
import numpy as np
from tqdm import tqdm

from data_loading import load_config, load_samples_index, iterate_trials
from build_dataset import compute_subject_eda_stats
from segmentation import build_trial_sequence


def build_sequence_dataset(config_path=None, limit=None):
    cfg = load_config(config_path)
    samples_df = load_samples_index(cfg)
    if limit:
        samples_df = samples_df.head(limit)

    sampling_rate = cfg["signal"]["sampling_rate_hz"]
    subject_eda_stats = compute_subject_eda_stats(cfg, samples_df)

    X, y, subject_ids, sample_names = [], [], [], []
    feature_names = None
    n_skipped = 0

    print("Building sub-windowed sequences per trial...")
    for subject_id, class_id, subject_name, sample_name, df in tqdm(
        iterate_trials(cfg, limit=limit), total=len(samples_df)
    ):
        subj_mean, subj_std = subject_eda_stats.get(subject_name, (None, None))
        try:
            seq, names = build_trial_sequence(
                df, sampling_rate, cfg, subject_eda_mean=subj_mean, subject_eda_std=subj_std
            )
        except Exception as e:
            print(f"[WARN] Skipping {subject_name}/{sample_name}: {e}")
            n_skipped += 1
            continue

        if feature_names is None:
            feature_names = names

        X.append(seq)
        y.append(class_id)
        subject_ids.append(subject_id)
        sample_names.append(sample_name)

    # Sanity check: all trials should produce the same number of sub-windows
    # since they're all 5.5s at the same sampling rate. If not, something's
    # off with a specific file's length -- pad/truncate rather than crash.
    lengths = {seq.shape[0] for seq in X}
    if len(lengths) > 1:
        min_len = min(lengths)
        print(f"[WARN] Inconsistent sub-window counts {lengths} across trials -- "
              f"truncating all to {min_len} windows.")
        X = [seq[:min_len] for seq in X]

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    subject_ids = np.array(subject_ids, dtype=np.int64)
    sample_names = np.array(sample_names, dtype=object)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(this_dir, "..", cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sequences.npz")

    np.savez_compressed(
        out_path, X=X, y=y, subject_ids=subject_ids,
        sample_names=sample_names, feature_names=np.array(feature_names, dtype=object),
    )

    print(f"\nSaved sequence dataset to {out_path}")
    print(f"X shape: {X.shape}  (n_trials, n_windows, n_features)")
    print(f"y shape: {y.shape}, classes: {sorted(set(y.tolist()))}")
    print(f"Skipped {n_skipped} trials due to errors")

    return X, y, subject_ids, sample_names, feature_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_sequence_dataset(args.config, limit=args.limit)
