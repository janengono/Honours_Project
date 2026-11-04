"""
sanity_check.py
----------------
Quick validation of features.csv before moving on to modeling.
Run from the src/ folder: python sanity_check.py
"""

import os
import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(this_dir, "..", "data", "processed", "features.csv")

df = pd.read_csv(features_path)

print(f"Loaded {len(df)} rows, {df.shape[1]} columns\n")

print("=== Class distribution ===")
print(df["class_id"].value_counts().sort_index())
print()

print("=== Missing values per column (top 10) ===")
print(df.isna().sum().sort_values(ascending=False).head(10))
print()

print("=== Subjects covered ===")
print(f"{df['subject_id'].nunique()} unique subjects")
print()

print("=== Sanity check: do features differ between baseline (0) and max pain (4)? ===")
key_features = ["heart_rate_bpm", "hrv_sdnn_ms", "eda_phasic_mean", "scr_count",
                 "emg_trapezius_rms_mean", "emg_corrugator_rms_mean"]
for feat in key_features:
    if feat in df.columns:
        bl_mean = df[df["class_id"] == 0][feat].mean()
        pain_mean = df[df["class_id"] == 4][feat].mean()
        print(f"  {feat:30s}  BL: {bl_mean:10.3f}   PA4: {pain_mean:10.3f}   diff: {pain_mean - bl_mean:+.3f}")
