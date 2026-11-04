"""
cv_utils.py
-----------
Subject-level cross-validation helpers (numpy/pandas only).

- make_folds(): Leave-One-Subject-Out (n_folds=None) or grouped K-fold over SUBJECTS.
- split_train_val_subjects(): carve a validation set out of the TRAINING subjects,
  so model selection never touches the held-out test subjects.
- summarize_predictions(): pooled accuracy / macro-F1, per-subject accuracy,
  pooled confusion matrix, and a bootstrap 95% CI that resamples SUBJECTS
  (the unit that matters, since a subject's trials are not independent).
"""

import numpy as np


def make_folds(subject_ids, n_folds=None, seed=0):
    """List of arrays; each array holds the subject ids held out in that fold.
    n_folds=None (or >= n_subjects) -> Leave-One-Subject-Out."""
    subjects = np.unique(subject_ids)
    if n_folds is None or n_folds >= len(subjects):
        return [np.array([s]) for s in subjects]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    return [np.sort(part) for part in np.array_split(shuffled, n_folds)]


def split_train_val_subjects(train_subjects, val_frac, seed):
    """Split the training subjects into (train, val) subject arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(train_subjects))
    n_val = max(1, int(round(len(perm) * val_frac)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def summarize_predictions(df, n_classes, n_boot=2000, seed=0):
    """df columns: subject_id, y_true, y_pred (one row per test trial)."""
    y_true, y_pred = df["y_true"].values, df["y_pred"].values
    cm = confusion_matrix(y_true, y_pred, n_classes)
    pooled_acc = float(np.mean(y_true == y_pred))
    pooled_f1 = macro_f1_from_cm(cm)

    per_subj = df.assign(correct=(df["y_true"] == df["y_pred"]).astype(float)) \
                 .groupby("subject_id")["correct"].agg(["sum", "count"])
    subj_acc = per_subj["sum"] / per_subj["count"]

    rng = np.random.default_rng(seed)
    s, c = per_subj["sum"].values, per_subj["count"].values
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), size=len(s))
        boots.append(s[idx].sum() / c[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "n_subjects": int(len(per_subj)), "n_trials": int(len(df)),
        "pooled_accuracy": pooled_acc, "pooled_macro_f1": pooled_f1,
        "accuracy_ci95": (float(lo), float(hi)),
        "per_subject_acc_mean": float(subj_acc.mean()),
        "per_subject_acc_std": float(subj_acc.std(ddof=1)) if len(subj_acc) > 1 else 0.0,
        "per_subject_acc_min": float(subj_acc.min()), "per_subject_acc_max": float(subj_acc.max()),
        "confusion_matrix": cm, "chance": 1.0 / n_classes,
    }


# ----------------------------------------------------------------------
# Feature helpers used by f1_experiments.py and loso_train.py (--subject-norm)
# ----------------------------------------------------------------------
def subject_normalize(X, subject_ids):
    """Z-score every feature WITHIN each subject (label-free).
    X is (n_trials, n_features) or (n_trials, n_windows, n_features); statistics pool
    over that subject's trials (and windows), one mean/std per feature.
    NOTE: this uses each subject's own unlabeled trials, including held-out subjects'.
    No labels and no other subject's data are involved, but state it in the methodology."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X)
    for s in np.unique(subject_ids):
        m = subject_ids == s
        flat = X[m].reshape(-1, X.shape[-1])
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std[std == 0] = 1.0
        out[m] = (X[m] - mean) / std
    return out


def shape_features(X):
    """(n, T, F) window sequences -> (n, F*6): per feature the mean, std, linear slope,
    relative position of the max and of the min, and last-minus-first. The position of
    the peak/minimum and the start-to-end change were the most important features in
    Gouverneur et al. [18]; per-window mean/std/min/max only expose them indirectly."""
    n, T, F = X.shape
    t = np.arange(T, dtype=np.float64)
    tc = t - t.mean()
    slope = np.einsum("ntf,t->nf", X, tc) / np.sum(tc ** 2)
    feats = [X.mean(axis=1), X.std(axis=1), slope,
             X.argmax(axis=1) / (T - 1), X.argmin(axis=1) / (T - 1), X[:, -1, :] - X[:, 0, :]]
    return np.concatenate(feats, axis=1)


def prior_match(P, iters=200):
    """Rescale class probabilities so the predicted class distribution over this set of
    trials is uniform. Uses NO labels, but ASSUMES the classes are balanced, which holds
    for BioVid by design (20 trials per class per subject) and would not in a clinic."""
    P = np.asarray(P, dtype=np.float64) + 1e-12
    w = np.ones(P.shape[1])
    for _ in range(iters):
        Q = P * w
        Q /= Q.sum(axis=1, keepdims=True)
        w *= (1.0 / P.shape[1]) / Q.mean(axis=0)
    Q = P * w
    return Q / Q.sum(axis=1, keepdims=True)


def within_one_accuracy(cm):
    n = cm.shape[0]
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) <= 1
    return float(cm[mask].sum() / cm.sum())
