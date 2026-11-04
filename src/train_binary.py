"""
train_binary.py
----------------
Trains the SAME PainTransformer architecture on a binary BL-vs-PA4 task
(no pain vs. maximum pain), directly precedented by Albahdal et al. in
Section 2.4/2.5 of the proposal's lit review, who found binary
classification consistently outperforms multi-class on BioVid.

Purpose: gives a stronger, cleaner model to anchor the SHAP/attention
explainability analysis (Section 3.4) with higher confidence, alongside
the harder (but more clinically representative) 5-class model from train.py.
Both results are worth reporting -- the gap between them IS a finding,
not just a fallback.

Usage:
    python train_binary.py --epochs 40
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn
from splits import subject_level_split, apply_subject_split
from feature_scaling import fit_scaler, apply_scaler
from train import evaluate  # reuse the same eval loop (works for any n_classes)


def filter_to_binary(X, y, subject_ids, class_a=0, class_b=4):
    """Keeps only trials from class_a (-> label 0) and class_b (-> label 1)."""
    mask = np.isin(y, [class_a, class_b])
    X_bin, y_bin, sid_bin = X[mask], y[mask], subject_ids[mask]
    y_bin = np.where(y_bin == class_a, 0, 1)
    return X_bin, y_bin, sid_bin


def train_binary(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, subject_ids, feature_names = load_sequence_data()
    X, y, subject_ids = filter_to_binary(X, y, subject_ids, args.class_a, args.class_b)
    print(f"Filtered to binary task: class {args.class_a} vs class {args.class_b} "
          f"-> {len(X)} trials")

    modality_indices = get_modality_indices(feature_names)
    mod_dims = modality_feature_dims(modality_indices)

    train_subj, val_subj, test_subj = subject_level_split(subject_ids, seed=args.seed)
    X_train, y_train, sid_train = apply_subject_split(X, y, subject_ids, train_subj)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    print(f"Train: {len(X_train)} trials, Val: {len(X_val)} trials")

    feat_mean, feat_std = fit_scaler(X_train)
    X_train = apply_scaler(X_train, feat_mean, feat_std)
    X_val = apply_scaler(X_val, feat_mean, feat_std)

    train_ds = PainSequenceDataset(X_train, y_train, sid_train, feature_names, modality_indices)
    val_ds = PainSequenceDataset(X_val, y_val, sid_val, feature_names, modality_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = PainTransformer(
        modality_dims=mod_dims, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, fusion_heads=args.fusion_heads, n_classes=2,
        dropout=args.dropout,
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(this_dir, "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_model_binary.pt")

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
                "class_a": args.class_a, "class_b": args.class_b,
                "val_f1_macro": val_f1, "val_acc": val_acc, "epoch": epoch,
            }, best_ckpt_path)
            print(f"  -> New best model saved (val_f1_macro={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val F1-macro: {best_val_f1:.4f} (checkpoint saved to {best_ckpt_path})")

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    print("\nValidation classification report (best checkpoint):")
    print(classification_report(y_true, y_pred, target_names=[f"class_{args.class_a}", f"class_{args.class_b}"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-a", type=int, default=0, help="BL, mapped to label 0")
    parser.add_argument("--class-b", type=int, default=4, help="PA4, mapped to label 1")
    parser.add_argument("--epochs", type=int, default=40)
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
    train_binary(args)
