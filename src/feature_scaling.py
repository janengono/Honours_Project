"""
feature_scaling.py
-------------------
Standardizes the (n_trials, seq_len, n_features) sequence array so every
feature has roughly zero mean / unit variance before it reaches the model.

CRITICAL: the scaler is fit on the TRAINING split only, then applied to
train/val/test/LOSO-held-out-fold using those same training statistics.
Fitting on the full dataset (including val/test) would leak information
about the evaluation data into preprocessing, inflating your reported
performance in a way that wouldn't hold up to scrutiny.
"""

import numpy as np


def fit_scaler(X_train):
    """
    X_train: (n_trials, seq_len, n_features)
    Returns (mean, std), each of shape (n_features,), computed by pooling
    across both the trial and time-window axes (i.e. one scale factor per
    feature, not per timestep).
    """
    flattened = X_train.reshape(-1, X_train.shape[-1])  # (n_trials * seq_len, n_features)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero on any constant feature
    return mean, std


def apply_scaler(X, mean, std):
    """Applies (X - mean) / std, broadcasting mean/std across trials and timesteps."""
    return (X - mean) / std
