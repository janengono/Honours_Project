"""
train.py
--------
Trains PainTransformer on the sequence dataset using a subject-level
train/val split (Section 3.5). This is the fast-iteration training loop --
for the final reported result, see loso_train.py which repeats this same
training procedure across all 86 Leave-One-Subject-Out folds.

Usage:
    python train.py --epochs 30 --batch-size 64
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model import PainTransformer, get_modality_indices
from pain_dataset import (
    PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn,
)
from splits import subject_level_split, apply_subject_split
from feature_scaling import fit_scaler, apply_scaler


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for modality_tensors, labels in loader:
            modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
            labels = labels.to(device)

            logits = model(modality_tensors)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1_macro, all_labels, all_preds


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, subject_ids, feature_names = load_sequence_data()
    modality_indices = get_modality_indices(feature_names)
    mod_dims = modality_feature_dims(modality_indices)
    print(f"Modality feature dims: {mod_dims}")

    train_subj, val_subj, test_subj = subject_level_split(subject_ids, seed=args.seed)
    X_train, y_train, sid_train = apply_subject_split(X, y, subject_ids, train_subj)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    print(f"Train: {len(X_train)} trials from {len(train_subj)} subjects")
    print(f"Val:   {len(X_val)} trials from {len(val_subj)} subjects")
    print(f"(Test set of {len(test_subj)} subjects held out untouched -- "
          f"use evaluate.py only once, after model selection is finalized)")

    # Standardize features using TRAIN-ONLY statistics (see feature_scaling.py
    # docstring for why fitting on train only matters). This was previously
    # missing entirely -- ECG/EMG features are in the thousands while EDA is
    # single digits, which likely slowed/capped what the model could learn.
    feat_mean, feat_std = fit_scaler(X_train)
    X_train = apply_scaler(X_train, feat_mean, feat_std)
    X_val = apply_scaler(X_val, feat_mean, feat_std)
    print("Applied train-fitted feature standardization to train and val sets.")

    train_ds = PainSequenceDataset(X_train, y_train, sid_train, feature_names, modality_indices)
    val_ds = PainSequenceDataset(X_val, y_val, sid_val, feature_names, modality_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = PainTransformer(
        modality_dims=mod_dims, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, fusion_heads=args.fusion_heads, n_classes=5,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # Eq. 3.7 base + label smoothing
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(this_dir, "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    best_val_f1 = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for modality_tensors, labels in train_loader:
            modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(modality_tensors)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_ds)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1_macro={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "modality_dims": mod_dims,
                "model_config": {
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "fusion_heads": args.fusion_heads,
                    "dropout": args.dropout,
                },
                "feature_mean": feat_mean, "feature_std": feat_std,
                "val_f1_macro": val_f1, "val_acc": val_acc, "epoch": epoch,
            }, best_ckpt_path)
            print(f"  -> New best model saved (val_f1_macro={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{args.early_stop_patience} epochs)")
                break

    print(f"\nBest val F1-macro: {best_val_f1:.4f} (checkpoint saved to {best_ckpt_path})")

    # Final classification report on val set using the best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    print("\nValidation classification report (best checkpoint):")
    print(classification_report(y_true, y_pred, target_names=["BL", "PA1", "PA2", "PA3", "PA4"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
