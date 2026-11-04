
"""
Priority 2: Unimodal Transformer comparisons.

Run from project root:

Pilot:
  python src/cv_unimodal_transformers.py --task five_class --max-folds 1
  python src/cv_unimodal_transformers.py --task binary --max-folds 1

Full:
  python src/cv_unimodal_transformers.py --task five_class --resume
  python src/cv_unimodal_transformers.py --task binary --resume
"""

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import PainTransformer, get_modality_indices
from pain_dataset import (
    PainSequenceDataset,
    load_sequence_data,
    collate_fn,
)
from feature_scaling import fit_scaler, apply_scaler
from train_binary import filter_to_binary
from cv_utils import (
    make_folds,
    split_train_val_subjects,
    subject_normalize,
    summarize_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "priority2_unimodal"
CHECKPOINTS = ROOT / "checkpoints" / "priority2_unimodal"
MODALITIES = ("ecg", "eda", "emg")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, sids, names, indices, batch_size, shuffle):
    dataset = PainSequenceDataset(X, y, sids, names, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def evaluate_model(model, loader, device, n_classes, criterion):
    model.eval()
    all_true, all_pred = [], []
    total_loss, total_count = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(
        all_true,
        all_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )

    return total_loss / total_count, accuracy, macro_f1, all_pred


def run_one(fold_id, test_subjects, modality,
            X, y, sids, feature_names, all_indices,
            n_classes, args, device):

    seed = args.seed + fold_id
    set_seed(seed)

    subjects = np.unique(sids)
    train_pool = np.setdiff1d(subjects, test_subjects)

    train_subj, val_subj = split_train_val_subjects(
        train_pool, args.val_frac, seed
    )

    tr = np.isin(sids, train_subj)
    va = np.isin(sids, val_subj)
    te = np.isin(sids, test_subjects)

    assert tr.any() and va.any() and te.any()
    assert not np.any(tr & va)
    assert not np.any(tr & te)
    assert not np.any(va & te)

    # Exactly the same preprocessing sequence as
    # loso_interpretability.py.
    mean, std = fit_scaler(X[tr])

    X_tr = apply_scaler(X[tr], mean, std)
    X_va = apply_scaler(X[va], mean, std)
    X_te = apply_scaler(X[te], mean, std)

    # Only the selected modality enters the network.
    indices = {modality: all_indices[modality]}
    dims = {modality: len(all_indices[modality])}

    train_loader = make_loader(
        X_tr, y[tr], sids[tr], feature_names,
        indices, args.batch_size, True
    )
    val_loader = make_loader(
        X_va, y[va], sids[va], feature_names,
        indices, args.batch_size, False
    )
    test_loader = make_loader(
        X_te, y[te], sids[te], feature_names,
        indices, args.batch_size, False
    )

    model = PainTransformer(
        modality_dims=dims,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        fusion_heads=args.fusion_heads,
        n_classes=n_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

        val_loss, _, val_f1, _ = evaluate_model(
            model, val_loader, device, n_classes, criterion
        )
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    model.load_state_dict(best_state)

    _, test_acc, test_f1, predictions = evaluate_model(
        model, test_loader, device, n_classes, criterion
    )

    ckpt_dir = CHECKPOINTS / args.task / modality
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_id:02d}.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "modality": modality,
        "modality_dims": dims,
        "model_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "fusion_heads": args.fusion_heads,
            "dropout": args.dropout,
        },
        "feature_mean": mean,
        "feature_std": std,
        "feature_names": list(feature_names),
        "test_subjects": list(test_subjects),
        "fold": fold_id,
        "subject_norm": args.task == "five_class",
        "n_classes": n_classes,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
    }, ckpt_path)

    rows = []
    for i, (sid, truth, pred) in enumerate(
        zip(sids[te], y[te], predictions)
    ):
        rows.append({
            "task": args.task,
            "model": f"transformer_{modality}",
            "fold": fold_id,
            "trial_index": i,
            "subject_id": sid,
            "y_true": int(truth),
            "y_pred": int(pred),
        })

    result = {
        "task": args.task,
        "model": f"transformer_{modality}",
        "fold": fold_id,
        "n_test": int(te.sum()),
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }

    return pd.DataFrame(rows), result


def verify_alignment(task, predictions):
    filename = (
        "cv_ablation_5class_10fold_subjnorm_interpretability.csv"
        if task == "five_class"
        else "cv_ablation_binary_10fold_interpretability.csv"
    )

    ref = pd.read_csv(ROOT / "results" / filename)
    ref = ref[ref["condition"] == "full"].copy()
    ref["trial_index"] = ref.groupby(
        "fold", sort=False
    ).cumcount()

    keys = ["fold", "trial_index"]

    for modality in MODALITIES:
        sample = predictions[
            predictions["model"] == f"transformer_{modality}"
        ].copy()

        merged = sample.merge(
            ref[keys + ["subject_id", "y_true"]],
            on=keys,
            how="left",
            suffixes=("_new", "_reference"),
            validate="one_to_one",
            indicator=True,
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                f"Missing reference trials: {modality}"
            )

        same_subjects = (
            merged["subject_id_new"].astype(str).to_numpy()
            == merged["subject_id_reference"].astype(str).to_numpy()
        ).all()

        same_labels = (
            merged["y_true_new"].to_numpy()
            == merged["y_true_reference"].to_numpy()
        ).all()

        if not (same_subjects and same_labels):
            raise ValueError(
                f"Transformer trial alignment failed: {modality}"
            )

    print(
        f"Alignment verified for all three modalities "
        f"({len(predictions) // 3} trials per model)"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", required=True,
        choices=["five_class", "binary"]
    )
    p.add_argument("--max-folds", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--fusion-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    args = p.parse_args()

    if not 1 <= args.max_folds <= 10:
        p.error("--max-folds must be between 1 and 10")

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    X, y, sids, feature_names = load_sequence_data()

    if args.task == "binary":
        X, y, sids = filter_to_binary(X, y, sids, 0, 4)
        n_classes = 2
    else:
        X = subject_normalize(
            X, sids
        ).astype(np.float32)
        n_classes = 5

    all_indices = get_modality_indices(feature_names)
    folds = make_folds(sids, n_folds=10, seed=args.seed)

    pred_path = OUT / f"{args.task}_predictions.csv"
    fold_path = OUT / f"{args.task}_fold_results.csv"

    if args.resume and pred_path.exists():
        if not fold_path.exists():
            raise ValueError("Missing fold results for resume")

        predictions = pd.read_csv(pred_path)
        fold_results = pd.read_csv(fold_path)
    else:
        predictions = pd.DataFrame()
        fold_results = pd.DataFrame()

    for fold_id, test_subjects in enumerate(
        folds[:args.max_folds]
    ):
        for modality in MODALITIES:
            model_name = f"transformer_{modality}"

            already_done = (
                not predictions.empty
                and (
                    (predictions["fold"] == fold_id)
                    & (predictions["model"] == model_name)
                ).any()
            )

            if already_done:
                # Resume assumes the previous fold was fully saved.
                expected = int(np.isin(
                    sids, test_subjects
                ).sum())
                existing = predictions[
                    (predictions["fold"] == fold_id)
                    & (predictions["model"] == model_name)
                ]

                checkpoint = (
                    CHECKPOINTS / args.task / modality
                    / f"fold_{fold_id:02d}.pt"
                )

                if (
                    len(existing) != expected
                    or not checkpoint.exists()
                ):
                    raise ValueError(
                        f"Incomplete saved run: "
                        f"fold {fold_id}, {modality}"
                    )

                print(
                    f"Skipping fold {fold_id} / {modality}"
                )
                continue

            print(
                f"\n{args.task} | fold {fold_id}/9 "
                f"| {modality.upper()} "
                f"| subjects {list(test_subjects)}",
                flush=True,
            )

            start = time.time()
            new_preds, result = run_one(
                fold_id, test_subjects, modality,
                X, y, sids, feature_names,
                all_indices, n_classes, args, device
            )

            result["seconds"] = round(
                time.time() - start, 1
            )

            predictions = pd.concat(
                [predictions, new_preds],
                ignore_index=True
            )
            fold_results = pd.concat(
                [fold_results, pd.DataFrame([result])],
                ignore_index=True
            )

            predictions.to_csv(pred_path, index=False)
            fold_results.to_csv(fold_path, index=False)

            print(
                f"  val_F1={result['val_macro_f1']:.4f} "
                f"test_acc={result['test_accuracy']:.4f} "
                f"test_F1={result['test_macro_f1']:.4f} "
                f"time={result['seconds']:.1f}s",
                flush=True,
            )

    verify_alignment(args.task, predictions)

    summary = []
    for name, group in predictions.groupby("model"):
        metrics = summarize_predictions(
            group, n_classes, n_boot=2000, seed=42
        )
        summary.append({
            "task": args.task,
            "model": name,
            "n_subjects": metrics["n_subjects"],
            "n_trials": metrics["n_trials"],
            "accuracy": metrics["pooled_accuracy"],
            "macro_f1": metrics["pooled_macro_f1"],
            "accuracy_ci_lower":
                metrics["accuracy_ci95"][0],
            "accuracy_ci_upper":
                metrics["accuracy_ci95"][1],
        })

    summary = pd.DataFrame(summary)
    summary.to_csv(
        OUT / f"{args.task}_summary.csv",
        index=False
    )

    print("\nUNIMODAL TRANSFORMER RESULTS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
