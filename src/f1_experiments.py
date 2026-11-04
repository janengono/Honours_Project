"""
f1_experiments.py
-----------------
Tests, cheaply and on the SAME subject-grouped folds, which changes might raise
five-class F1. Uses classical models (a Random Forest, a gradient-boosting model),
so a full comparison takes minutes, not hours. Whatever helps can then be ported to
the Transformer (loso_train.py --subject-norm).

Variants (cumulative):
  A  window means only               (20 features: the information the earlier RF used)
  B  + shape features                (mean, std, slope, argmax, argmin, last-first per feature)
  C  B, normalised per subject       (label-free; see cv_utils.subject_normalize)
  D  C + class-prior matched decoding (assumes balanced classes; see cv_utils.prior_match)

HONEST USE (selection bias): every variant is scored on the same folds, so picking the
best one and quoting its score is optimistic. Choose the variant with --seed 0, then
confirm it ONCE on a fresh partition with --seed 1 and report the confirmation number,
listing all variants you tried.

Usage:
    python f1_experiments.py                  # RF + HistGradientBoosting, 10 folds, seed 0
    python f1_experiments.py --models rf
    python f1_experiments.py --seed 1         # confirmation run on a fresh partition
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from cv_utils import (make_folds, subject_normalize, shape_features, prior_match,
                      confusion_matrix, macro_f1_from_cm, within_one_accuracy,
                      summarize_predictions)

N_CLASSES = 5


def load_sequences():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(this_dir, "..", "data", "processed", "sequences.npz"), allow_pickle=True)
    return d["X"].astype(np.float64), d["y"], d["subject_ids"]


def build_variants(X, sids):
    Xn = subject_normalize(X, sids)
    return {
        "A means only": X.mean(axis=1),
        "B + shape": shape_features(X),
        "C + subject-norm": shape_features(Xn),
        "D + prior-match": shape_features(Xn),      # same features as C, different decoding
    }


def make_model(kind, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def cv_predict(F, y, sids, folds, kind, seed, decode_prior_match):
    pred = np.empty(len(y), dtype=int)
    for test_subj in folds:
        te = np.isin(sids, test_subj)
        model = make_model(kind, seed).fit(F[~te], y[~te])
        P = model.predict_proba(F[te])
        if decode_prior_match:
            P = prior_match(P)
        pred[te] = model.classes_[P.argmax(axis=1)]
    return pred


def main(args):
    X, y, sids = load_sequences()
    folds = make_folds(sids, args.n_folds, seed=args.seed)
    print(f"{len(X)} trials, {len(np.unique(sids))} subjects, {len(folds)} folds, partition seed {args.seed}\n")
    variants = build_variants(X, sids)

    rows = []
    for kind in args.models.split(","):
        for name, F in variants.items():
            t0 = time.time()
            pred = cv_predict(F, y, sids, folds, kind.strip(), args.seed, name.startswith("D"))
            cm = confusion_matrix(y, pred, N_CLASSES)
            s = summarize_predictions(pd.DataFrame({"subject_id": sids, "y_true": y, "y_pred": pred}), N_CLASSES)
            rows.append({"model": kind.strip(), "variant": name,
                         "accuracy": s["pooled_accuracy"],
                         "acc 95% CI": f"{s['accuracy_ci95'][0]:.3f}-{s['accuracy_ci95'][1]:.3f}",
                         "macro-F1": macro_f1_from_cm(cm), "within-one": within_one_accuracy(cm),
                         "sec": round(time.time() - t0)})
            print(f"  done {kind.strip():>3} | {name:<18} | acc {rows[-1]['accuracy']:.3f} | "
                  f"F1 {rows[-1]['macro-F1']:.3f} | {rows[-1]['sec']}s", flush=True)

    table = pd.DataFrame(rows)
    for c in ("accuracy", "macro-F1", "within-one"):
        table[c] = table[c].map(lambda v: f"{v:.3f}")
    print("\n" + "=" * 86)
    print(table.to_string(index=False))
    print("=" * 86)
    print("Reference: chance accuracy 0.200. Differences under about 0.02 are within noise.")
    print("If a variant helps, confirm it once with a different --seed before reporting it.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, f"f1_experiments_seed{args.seed}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--models", default="rf,hgb")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
