"""
compare_cv.py
-------------
Paired comparison of two cross-validated runs that used the SAME folds and subjects,
e.g. the Transformer without and with per-subject normalisation:

    python compare_cv.py ../results/cv_predictions_5class_10fold.csv ^
                         ../results/cv_predictions_5class_10fold_subjnorm.csv

Resamples SUBJECTS with replacement (the unit that matters), recomputes pooled accuracy
and macro-F1 for both runs on the same resample, and reports the difference (B - A)
with a 95% interval. Pairing removes between-subject variance, so it is a much sharper
test than comparing two separate intervals.
"""

import sys
import numpy as np
import pandas as pd
from cv_utils import confusion_matrix, macro_f1_from_cm


def load(path):
    df = pd.read_csv(path)
    df["trial"] = df.groupby("subject_id").cumcount()
    return df.set_index(["subject_id", "trial"]).sort_index()


def per_subject_cms(df, subjects, n_classes):
    return np.stack([confusion_matrix(df.loc[s]["y_true"].values, df.loc[s]["y_pred"].values, n_classes)
                     for s in subjects])


def main(path_a, path_b, n_boot=5000, seed=0):
    a, b = load(path_a), load(path_b)
    if not a.index.equals(b.index) or not (a["y_true"].values == b["y_true"].values).all():
        sys.exit("The two files do not cover the same trials in the same order; they must come from "
                 "runs with identical folds and data.")
    subjects = a.index.get_level_values("subject_id").unique().values
    n_classes = int(a["y_true"].max()) + 1
    cm_a, cm_b = per_subject_cms(a, subjects, n_classes), per_subject_cms(b, subjects, n_classes)

    def metrics(cm):
        return np.trace(cm) / cm.sum(), macro_f1_from_cm(cm)

    acc_a, f1_a = metrics(cm_a.sum(0))
    acc_b, f1_b = metrics(cm_b.sum(0))
    rng = np.random.default_rng(seed)
    d_acc, d_f1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(subjects), size=len(subjects))
        ma, mb = metrics(cm_a[idx].sum(0)), metrics(cm_b[idx].sum(0))
        d_acc.append(mb[0] - ma[0]); d_f1.append(mb[1] - ma[1])

    print(f"{len(subjects)} subjects, {len(a)} trials, {n_boot} subject-level bootstrap resamples\n")
    print(f"{'':10} {'A':>8} {'B':>8} {'B - A':>9}   95% interval for B - A      P(B > A)")
    for name, va, vb, d in (("accuracy", acc_a, acc_b, d_acc), ("macro-F1", f1_a, f1_b, d_f1)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{name:10} {va:8.4f} {vb:8.4f} {vb - va:+9.4f}   [{lo:+.4f}, {hi:+.4f}]          {np.mean(np.array(d) > 0):.3f}")
    print("\nIf the interval excludes 0, the improvement is distinguishable from subject-sampling noise.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
