"""
data_loading.py
----------------
Loads the BioVid Part B sample index (Samples.csv) and provides utilities
to locate and read the corresponding raw biosignal file for each trial.

Samples.csv columns used: subject_id, class_id, subject_name, sample_name
Raw biosignal columns:    time, gsr, ecg, emg_trapezius, emg_corrugator, emg_zygomaticus
Raw time column is in MICROSECONDS relative to the start of that trial's video.
"""

import os
import glob
import pandas as pd
import yaml


def load_config(config_path=None):
    """
    Loads configs/config.yaml. If no path is given, resolves it relative to
    this file's location (project_root/configs/config.yaml), so this works
    whether you run the script from the project root or from inside src/.
    """
    if config_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "..", "configs", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find config file at: {os.path.abspath(config_path)}\n"
            f"Run this script from the project root or the src/ folder, "
            f"or pass an explicit --config path."
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_samples_index(cfg):
    """Load Samples.csv and return a DataFrame with one row per trial."""
    samples_path = os.path.join(cfg["paths"]["root"], cfg["paths"]["samples_csv"])
    df = pd.read_csv(samples_path)

    # Defensive: BioVid's Samples.csv is sometimes semicolon or tab separated.
    # If pandas parsed everything into a single column, retry with common delimiters.
    if df.shape[1] == 1:
        for sep in [";", "\t", ","]:
            try:
                df = pd.read_csv(samples_path, sep=sep)
                if df.shape[1] > 1:
                    break
            except Exception:
                continue

    required_cols = {"subject_id", "class_id", "subject_name", "sample_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Samples.csv is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Open Samples.csv and check the actual column names/delimiter."
        )
    return df


def find_raw_biosignal_file(cfg, subject_name, sample_name):
    """
    Locate the raw biosignal CSV for a given trial.
    Uses glob matching on sample_name rather than assuming an exact suffix,
    since the README only confirms the '_bio.csv' suffix for the *filtered*
    folder explicitly -- this is more robust to naming variations.
    """
    subject_dir = os.path.join(
        cfg["paths"]["root"], cfg["paths"]["biosignals_raw_dir"], str(subject_name)
    )
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"No such subject directory: {subject_dir}")

    candidates = glob.glob(os.path.join(subject_dir, f"{sample_name}*"))
    if not candidates:
        raise FileNotFoundError(
            f"No raw biosignal file found for sample '{sample_name}' in {subject_dir}"
        )
    if len(candidates) > 1:
        # Prefer a file ending in _bio.csv if multiple matches exist
        preferred = [c for c in candidates if c.endswith("_bio.csv")]
        return preferred[0] if preferred else candidates[0]
    return candidates[0]


def load_trial_biosignals(cfg, subject_name, sample_name):
    """
    Load one trial's raw biosignal file as a DataFrame with:
      - time_s: time in seconds (converted from microseconds), zeroed to start at 0
      - gsr, ecg, emg_trapezius, emg_corrugator, emg_zygomaticus
    """
    filepath = find_raw_biosignal_file(cfg, subject_name, sample_name)

    # Files are tab-separated based on the sample provided ("time\tgsr\tecg...")
    df = pd.read_csv(filepath, sep="\t")

    if df.shape[1] == 1:
        # Fallback in case delimiter differs
        df = pd.read_csv(filepath, sep=None, engine="python")

    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"time", "gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{filepath} is missing expected columns: {missing}")

    df["time_s"] = (df["time"] - df["time"].iloc[0]) / 1e6  # microseconds -> seconds
    return df


def iterate_trials(cfg, limit=None):
    """
    Generator yielding (subject_id, class_id, subject_name, sample_name, biosignal_df)
    for every trial in Samples.csv. Skips and warns on any trial whose file can't be loaded,
    rather than crashing the whole pipeline run.
    """
    samples = load_samples_index(cfg)
    if limit:
        samples = samples.head(limit)

    for _, row in samples.iterrows():
        try:
            df = load_trial_biosignals(cfg, row["subject_name"], row["sample_name"])
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {row['subject_name']}/{row['sample_name']}: {e}")
            continue
        yield row["subject_id"], row["class_id"], row["subject_name"], row["sample_name"], df


if __name__ == "__main__":
    # Quick smoke test: load config, index, and the first trial's data
    cfg = load_config()
    samples = load_samples_index(cfg)
    print(f"Loaded {len(samples)} trials from Samples.csv")
    print(samples.head())

    first = next(iterate_trials(cfg, limit=1))
    subject_id, class_id, subject_name, sample_name, df = first
    print(f"\nFirst trial: subject={subject_name}, sample={sample_name}, class={class_id}")
    print(df.head())
    print(f"Trial duration: {df['time_s'].iloc[-1]:.2f}s, {len(df)} rows")
