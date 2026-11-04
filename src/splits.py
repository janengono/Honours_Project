"""
splits.py
---------
Implements Section 3.5 (Validation Strategy) and Section 3.4.4
(Confounder-Aware Analysis / subject-independent evaluation).

Two splitting strategies, both SUBJECT-LEVEL (never trial-level):

1. subject_level_split(): a standard train/val/test split, for the main
   development loop -- tuning hyperparameters, checking the model trains
   at all, comparing architecture variants quickly.

2. loso_folds(): Leave-One-Subject-Out cross-validation, generating one
   fold per subject. This is your FINAL reported generalization number --
   the proposal explicitly commits to LOSO in Section 3.2.4/3.5, and it's
   the standard protocol in BioVid literature, so reviewers will expect it.

Why subject-level and not trial-level: a subject's 5.5s trials share that
person's baseline physiology (resting heart rate, skin conductance range,
muscle tone). If trials from the same subject appear in both train and
test, the model can partly learn to recognize *the subject*, not the pain
response -- inflating your accuracy in a way that won't generalize to a
new person. This is exactly the kind of confound your proposal's Section
3.4.4 is about, so getting this right is part of your actual contribution,
not just a technicality.
"""

import numpy as np


def subject_level_split(subject_ids, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Splits unique subjects into train/val/test sets (not trials).
    Returns three sets of subject_ids.
    """
    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_subjects = set(shuffled[:n_train].tolist())
    val_subjects = set(shuffled[n_train:n_train + n_val].tolist())
    test_subjects = set(shuffled[n_train + n_val:].tolist())

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    return train_subjects, val_subjects, test_subjects


def apply_subject_split(X, y, subject_ids, subject_set):
    """Filters X/y down to only the trials belonging to subjects in subject_set."""
    mask = np.isin(subject_ids, list(subject_set))
    return X[mask], y[mask], subject_ids[mask]


def loso_folds(subject_ids):
    """
    Generator yielding (fold_index, held_out_subject, train_mask, test_mask)
    for every unique subject. train_mask/test_mask are boolean arrays over
    the full trial array, ready to index X/y directly.

    Usage:
        for fold_i, held_out_subject, train_mask, test_mask in loso_folds(subject_ids):
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            # train model, evaluate, record result for held_out_subject
    """
    unique_subjects = np.unique(subject_ids)
    for fold_i, held_out_subject in enumerate(unique_subjects):
        test_mask = subject_ids == held_out_subject
        train_mask = ~test_mask
        yield fold_i, held_out_subject, train_mask, test_mask


if __name__ == "__main__":
    # Smoke test using the sequence dataset built by build_sequences.py
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    seq_path = os.path.join(this_dir, "..", "data", "processed", "sequences.npz")

    if not os.path.exists(seq_path):
        print(f"No sequences.npz found at {seq_path} -- run build_sequences.py first.")
    else:
        data = np.load(seq_path, allow_pickle=True)
        X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
        print(f"Loaded X={X.shape}, y={y.shape}, {len(np.unique(subject_ids))} subjects")

        train_s, val_s, test_s = subject_level_split(subject_ids)
        print(f"\nSubject-level split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test subjects")

        X_train, y_train, _ = apply_subject_split(X, y, subject_ids, train_s)
        X_val, y_val, _ = apply_subject_split(X, y, subject_ids, val_s)
        X_test, y_test, _ = apply_subject_split(X, y, subject_ids, test_s)
        print(f"Trial counts: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        n_folds = len(np.unique(subject_ids))
        print(f"\nLOSO would produce {n_folds} folds (one per subject).")
        first_fold = next(loso_folds(subject_ids))
        fold_i, held_out, train_mask, test_mask = first_fold
        print(f"Example fold 0: held-out subject={held_out}, "
              f"train trials={train_mask.sum()}, test trials={test_mask.sum()}")
