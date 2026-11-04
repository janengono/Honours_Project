"""
loso_train.py
-------------
Subject-independent evaluation (Section 3.5): Leave-One-Subject-Out (default) or
grouped K-fold over subjects (--n-folds K, much cheaper).

Per fold:
  1. hold out the test subject(s)
  2. carve a validation set out of the TRAINING subjects (--val-frac)
  3. fit the feature scaler on the training subjects only
  4. train; pick the best epoch by VALIDATION macro-F1 (test subjects are never used)
  5. evaluate once on the held-out test subject(s) and append the predictions

Results are written after every fold, so you can stop and `--resume` later.
Summary reports pooled accuracy/F1, a bootstrap 95% CI over subjects, per-subject
accuracy and the pooled confusion matrix.

Usage (time ONE fold first, then extrapolate):
    python loso_train.py --max-folds 1
    python loso_train.py --n-folds 10                 # cheap cross-validated figure
    python loso_train.py                              # full LOSO (86 folds)
    python loso_train.py --resume                     # continue an interrupted run
    python loso_train.py --binary                     # BL vs PA4
"""

import argparse
import copy
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn
from feature_scaling import fit_scaler, apply_scaler
from train import evaluate
from train_binary import filter_to_binary
from cv_utils import make_folds, split_train_val_subjects, summarize_predictions, subject_normalize


def make_loader(X, y, sids, feature_names, modality_indices, batch_size, shuffle):
    ds = PainSequenceDataset(X, y, sids, feature_names, modality_indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def run_fold(fold_id, test_subjects, X, y, sids, feature_names, modality_indices,
             mod_dims, n_classes, args, device):
    seed = args.seed + fold_id
    torch.manual_seed(seed)

    all_subjects = np.unique(sids)
    train_pool = np.setdiff1d(all_subjects, test_subjects)
    train_subj, val_subj = split_train_val_subjects(train_pool, args.val_frac, seed)

    tr, va, te = (np.isin(sids, train_subj), np.isin(sids, val_subj), np.isin(sids, test_subjects))
    mean, std = fit_scaler(X[tr])                      # TRAIN subjects only
    X_tr, X_va, X_te = (apply_scaler(X[m], mean, std) for m in (tr, va, te))

    train_loader = make_loader(X_tr, y[tr], sids[tr], feature_names, modality_indices, args.batch_size, True)
    val_loader = make_loader(X_va, y[va], sids[va], feature_names, modality_indices, args.batch_size, False)
    test_loader = make_loader(X_te, y[te], sids[te], feature_names, modality_indices, args.batch_size, False)

    model = PainTransformer(modality_dims=mod_dims, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, fusion_heads=args.fusion_heads,
                            n_classes=n_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_f1, best_state, best_epoch, patience = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for modality_tensors, labels in train_loader:
            modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
            optimizer.zero_grad()
            loss = criterion(model(modality_tensors), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, _, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best_state)
    _, te_acc, te_f1, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    preds = pd.DataFrame({"fold": fold_id, "subject_id": sids[te], "y_true": y_true, "y_pred": y_pred})
    return te_acc, te_f1, best_epoch, best_f1, preds


def print_summary(pred_df, n_classes, class_names):
    s = summarize_predictions(pred_df, n_classes)
    lines = [
        "=" * 64,
        f"SUBJECT-INDEPENDENT RESULT  ({s['n_subjects']} held-out subjects, {s['n_trials']} trials)",
        "=" * 64,
        f"Pooled accuracy : {s['pooled_accuracy']:.4f}   (chance = {s['chance']:.2f})",
        f"  95% CI (bootstrap over subjects): {s['accuracy_ci95'][0]:.4f} - {s['accuracy_ci95'][1]:.4f}",
        f"Pooled macro-F1 : {s['pooled_macro_f1']:.4f}",
        f"Per-subject accuracy: mean {s['per_subject_acc_mean']:.4f} +/- {s['per_subject_acc_std']:.4f} "
        f"(min {s['per_subject_acc_min']:.2f}, max {s['per_subject_acc_max']:.2f})",
        "", "Pooled confusion matrix (rows = true, cols = predicted):",
        "        " + " ".join(f"{n:>5}" for n in class_names),
    ]
    for i, row in enumerate(s["confusion_matrix"]):
        lines.append(f"{class_names[i]:>7} " + " ".join(f"{v:5d}" for v in row))
    text = "\n".join(lines)
    print("\n" + text)
    return text


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, sids, feature_names = load_sequence_data()
    if args.binary:
        X, y, sids = filter_to_binary(X, y, sids, 0, 4)
        n_classes, class_names, task = 2, ["BL", "PA4"], "binary"
    else:
        n_classes, class_names, task = 5, ["BL", "PA1", "PA2", "PA3", "PA4"], "5class"

    if args.subject_norm:
        # label-free per-subject z-scoring of every feature (uses each subject's own
        # unlabeled trials, including held-out subjects'); disclose this in the methodology
        X = subject_normalize(X, sids).astype(np.float32)
        print("Applied per-subject feature normalisation (label-free).")

    modality_indices = get_modality_indices(feature_names)
    mod_dims = modality_feature_dims(modality_indices)

    folds = make_folds(sids, args.n_folds, seed=args.seed)
    mode = "loso" if args.n_folds is None or args.n_folds >= len(np.unique(sids)) else f"{args.n_folds}fold"
    print(f"Task: {task} | CV: {mode} | {len(folds)} folds | {len(X)} trials, {len(np.unique(sids))} subjects")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(this_dir, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{task}_{mode}" + ("_subjnorm" if args.subject_norm else "")
    res_path = os.path.join(out_dir, f"cv_folds_{tag}.csv")
    pred_path = os.path.join(out_dir, f"cv_predictions_{tag}.csv")

    done = set()
    if args.resume and os.path.exists(res_path):
        done = set(pd.read_csv(res_path)["fold"].tolist())
        print(f"Resuming: {len(done)} folds already complete")
    elif not args.resume:
        for p in (res_path, pred_path):
            if os.path.exists(p):
                os.remove(p)

    todo = [(i, f) for i, f in enumerate(folds) if i not in done]
    if args.max_folds:
        todo = todo[: args.max_folds]

    t_start = time.time()
    for n_done, (fold_id, test_subjects) in enumerate(todo, 1):
        t0 = time.time()
        acc, f1, best_epoch, val_f1, preds = run_fold(
            fold_id, test_subjects, X, y, sids, feature_names, modality_indices,
            mod_dims, n_classes, args, device)
        secs = time.time() - t0

        pd.DataFrame([{"fold": fold_id, "test_subjects": " ".join(map(str, test_subjects)),
                       "n_test": len(preds), "test_acc": acc, "test_f1_macro": f1,
                       "best_epoch": best_epoch, "val_f1_macro": val_f1, "seconds": round(secs, 1)}]
                     ).to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
        preds.to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False)

        avg = (time.time() - t_start) / n_done
        eta_min = avg * (len(todo) - n_done) / 60
        print(f"fold {fold_id:3d} | subj {list(test_subjects)} | test_acc={acc:.3f} f1={f1:.3f} | "
              f"best_epoch={best_epoch:2d} | {secs:.0f}s | ETA {eta_min:.0f} min "
              f"({n_done}/{len(todo)})", flush=True)

    if os.path.exists(pred_path):
        text = print_summary(pd.read_csv(pred_path), n_classes, class_names)
        with open(os.path.join(out_dir, f"cv_summary_{tag}.txt"), "w") as f:
            f.write(text + "\n")
        print(f"\nSaved: {res_path}\n       {pred_path}\n       cv_summary_{tag}.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=None, help="K for grouped K-fold over subjects; omit for LOSO")
    p.add_argument("--binary", action="store_true")
    p.add_argument("--subject-norm", action="store_true", help="per-subject z-scoring of all features (label-free)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-folds", type=int, default=None, help="run only the first N pending folds (for timing)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of TRAIN subjects used for validation")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--fusion-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
