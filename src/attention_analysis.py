"""
attention_analysis.py
----------------------
Implements the interpretability claim in Section 3.3.2. Extracts the fusion
layer's attention weights on real validation trials, aggregates from
token-level (30x30) down to modality-level (3x3), per class.

Usage:
    python attention_analysis.py --checkpoint ../checkpoints/best_model.pt
    python attention_analysis.py --checkpoint ../checkpoints/best_model_binary.pt --binary
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn
from splits import subject_level_split, apply_subject_split
from feature_scaling import apply_scaler
from train_binary import filter_to_binary

SEQ_LEN = 10


def modality_block_average(attn_matrix, modality_order, seq_len=SEQ_LEN):
    n_mod = len(modality_order)
    block_avg = np.zeros((n_mod, n_mod))
    for i in range(n_mod):
        for j in range(n_mod):
            row_start, row_end = i * seq_len, (i + 1) * seq_len
            col_start, col_end = j * seq_len, (j + 1) * seq_len
            block_avg[i, j] = attn_matrix[row_start:row_end, col_start:col_end].mean()
    return block_avg


def print_modality_matrix(matrix, modality_order, title):
    print(f"\n{title}")
    header = " " * 14 + "".join(f"{name:>10}" for name in modality_order)
    print(header)
    for i, name in enumerate(modality_order):
        row = f"  attends {name:<6}" + "".join(f"{matrix[i,j]:>10.4f}" for j in range(len(modality_order)))
        print(row)


def run_attention_analysis(checkpoint_path, binary=False, class_a=0, class_b=4, n_samples_per_class=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mod_dims = checkpoint["modality_dims"]
    cfg = checkpoint["model_config"]
    feat_mean, feat_std = checkpoint["feature_mean"], checkpoint["feature_std"]
    n_classes = 2 if binary else 5
    modality_order = list(mod_dims.keys())

    model = PainTransformer(
        modality_dims=mod_dims, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], fusion_heads=cfg["fusion_heads"],
        n_classes=n_classes, dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X, y, subject_ids, feature_names = load_sequence_data()
    if binary:
        X, y, subject_ids = filter_to_binary(X, y, subject_ids, class_a, class_b)
    modality_indices = get_modality_indices(feature_names)

    _, val_subj, _ = subject_level_split(subject_ids, seed=42)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    X_val = apply_scaler(X_val, feat_mean, feat_std)

    val_ds = PainSequenceDataset(X_val, y_val, sid_val, feature_names, modality_indices)

    class_names = ([f"class_{class_a}", f"class_{class_b}"] if binary
                    else ["BL", "PA1", "PA2", "PA3", "PA4"])

    print(f"Extracting attention weights on validation set ({len(val_ds)} trials)")
    print(f"Averaging up to {n_samples_per_class} trials per class\n")

    overall_matrices = []

    for class_id, class_name in enumerate(class_names):
        indices = [i for i in range(len(val_ds)) if val_ds.y[i] == class_id][:n_samples_per_class]
        if not indices:
            continue

        subset = torch.utils.data.Subset(val_ds, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        class_matrices = []
        with torch.no_grad():
            for modality_tensors, labels in loader:
                modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
                _, attn = model(modality_tensors, return_attention=True)
                attn_avg = attn.mean(dim=(0, 1)).cpu().numpy()
                class_matrices.append(attn_avg)

        class_avg = np.mean(class_matrices, axis=0)
        block_avg = modality_block_average(class_avg, modality_order)
        overall_matrices.append(block_avg)
        print_modality_matrix(block_avg, modality_order, f"--- {class_name} (n={len(indices)}) ---")

    return overall_matrices, class_names, modality_order


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--class-a", type=int, default=0)
    parser.add_argument("--class-b", type=int, default=4)
    parser.add_argument("--n-samples-per-class", type=int, default=100)
    args = parser.parse_args()
    run_attention_analysis(
        args.checkpoint, binary=args.binary, class_a=args.class_a,
        class_b=args.class_b, n_samples_per_class=args.n_samples_per_class,
    )
