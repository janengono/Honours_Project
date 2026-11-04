"""run_multiseed.py -- repeat 5-class training over several seeds; report mean +/- std.
Answers "run it multiple times": a narrow range means ~30% is genuine task difficulty."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn
from splits import subject_level_split, apply_subject_split
from feature_scaling import fit_scaler, apply_scaler
from train import evaluate


def run_one_seed(seed, epochs, device):
    X, y, subject_ids, feature_names = load_sequence_data()
    modality_indices = get_modality_indices(feature_names)
    mod_dims = modality_feature_dims(modality_indices)

    train_subj, val_subj, _ = subject_level_split(subject_ids, seed=seed)
    X_train, y_train, sid_train = apply_subject_split(X, y, subject_ids, train_subj)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    mean, std = fit_scaler(X_train)
    X_train, X_val = apply_scaler(X_train, mean, std), apply_scaler(X_val, mean, std)

    train_loader = DataLoader(PainSequenceDataset(X_train, y_train, sid_train, feature_names, modality_indices),
                              batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PainSequenceDataset(X_val, y_val, sid_val, feature_names, modality_indices),
                            batch_size=64, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(seed)
    model = PainTransformer(modality_dims=mod_dims, d_model=64, n_heads=4, n_layers=2,
                            fusion_heads=4, n_classes=5, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_f1, best_acc, patience = -1, 0, 0
    for _ in range(epochs):
        model.train()
        for modality_tensors, labels in train_loader:
            modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
            optimizer.zero_grad()
            criterion(model(modality_tensors), labels.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        if val_f1 > best_f1:
            best_f1, best_acc, patience = val_f1, val_acc, 0
        else:
            patience += 1
            if patience >= 10:
                break
    return best_acc, best_f1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 123, 2024])
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs, f1s = [], []
    for seed in args.seeds:
        acc, f1 = run_one_seed(seed, args.epochs, device)
        print(f"seed={seed:5d}  val_acc={acc:.4f}  val_f1_macro={f1:.4f}")
        accs.append(acc); f1s.append(f1)

    print(f"\nAccuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f} (range {min(accs):.4f}-{max(accs):.4f})")
    print(f"F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (range {min(f1s):.4f}-{max(f1s):.4f})")
