"""
modality_ablation.py
---------------------
Implements Section 3.4.3 (Causal Contribution Investigation).

Logic: for a trained model, systematically mask ONE modality's input at a
time (replace it with zeros -- the "no information" baseline after
standardization, since zero is the mean of a z-scored feature) and measure
how much validation performance drops. A modality whose removal causes a
large performance drop is one the model depends on causally for its
predictions; a modality whose removal barely changes performance is either
redundant with the others, or wasn't contributing much pain-relevant
information in the first place.

This directly implements the proposal's own description: "If removing a
modality causes a substantial degradation in pain estimation performance,
this suggests that the modality contributes meaningful causal information."

Usage:
    python modality_ablation.py --checkpoint ../checkpoints/best_model.pt
    python modality_ablation.py --checkpoint ../checkpoints/best_model_binary.pt --binary
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import PainTransformer, get_modality_indices
from pain_dataset import PainSequenceDataset, load_sequence_data, modality_feature_dims, collate_fn
from splits import subject_level_split, apply_subject_split
from feature_scaling import apply_scaler
from train import evaluate
from train_binary import filter_to_binary


def mask_modality(modality_tensors, target_modality):
    """Returns a copy of modality_tensors with target_modality zeroed out."""
    masked = {}
    for name, tensor in modality_tensors.items():
        masked[name] = torch.zeros_like(tensor) if name == target_modality else tensor
    return masked


def evaluate_with_masking(model, loader, device, target_modality=None):
    """
    Same as train.evaluate(), but optionally zeroes out one modality on
    every batch before the forward pass. target_modality=None runs the
    normal, unmasked evaluation (the baseline to compare against).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for modality_tensors, labels in loader:
            modality_tensors = {k: v.to(device) for k, v in modality_tensors.items()}
            if target_modality is not None:
                modality_tensors = mask_modality(modality_tensors, target_modality)
            labels = labels.to(device)

            logits = model(modality_tensors)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return acc, f1_macro


def run_ablation(checkpoint_path, binary=False, class_a=0, class_b=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mod_dims = checkpoint["modality_dims"]
    cfg = checkpoint["model_config"]
    feat_mean, feat_std = checkpoint["feature_mean"], checkpoint["feature_std"]
    n_classes = 2 if binary else 5

    model = PainTransformer(
        modality_dims=mod_dims, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], fusion_heads=cfg["fusion_heads"],
        n_classes=n_classes, dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_f1_macro={checkpoint['val_f1_macro']:.4f})")

    X, y, subject_ids, feature_names = load_sequence_data()
    if binary:
        X, y, subject_ids = filter_to_binary(X, y, subject_ids, class_a, class_b)

    modality_indices = get_modality_indices(feature_names)

    # Use the SAME split as training, so this evaluates on the model's own
    # validation set (consistent with how it was selected) -- for a fully
    # independent number you'd want a held-out test set instead, see
    # evaluate.py (not yet built) for that final step.
    _, val_subj, _ = subject_level_split(subject_ids, seed=42)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    X_val = apply_scaler(X_val, feat_mean, feat_std)

    val_ds = PainSequenceDataset(X_val, y_val, sid_val, feature_names, modality_indices)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print(f"\nEvaluating on {len(val_ds)} validation trials from {len(val_subj)} held-out subjects\n")

    baseline_acc, baseline_f1 = evaluate_with_masking(model, val_loader, device, target_modality=None)
    print(f"{'Condition':<25} {'Accuracy':>10} {'F1-macro':>10} {'F1 drop':>10}")
    print(f"{'-'*57}")
    print(f"{'Full model (no masking)':<25} {baseline_acc:>10.4f} {baseline_f1:>10.4f} {'--':>10}")

    results = {"full_model": {"acc": baseline_acc, "f1_macro": baseline_f1, "f1_drop": 0.0}}

    for modality_name in mod_dims.keys():
        masked_acc, masked_f1 = evaluate_with_masking(model, val_loader, device, target_modality=modality_name)
        f1_drop = baseline_f1 - masked_f1
        print(f"{'Mask ' + modality_name:<25} {masked_acc:>10.4f} {masked_f1:>10.4f} {f1_drop:>10.4f}")
        results[modality_name] = {"acc": masked_acc, "f1_macro": masked_f1, "f1_drop": f1_drop}

    print(f"\n{'='*57}")
    print("Interpretation: larger F1 drop = model relies more heavily on that")
    print("modality's information to make correct predictions. A modality with")
    print("near-zero drop when masked is either redundant with the remaining")
    print("modalities, or wasn't contributing much pain-discriminative signal.")

    ranked = sorted(
        ((name, r["f1_drop"]) for name, r in results.items() if name != "full_model"),
        key=lambda x: -x[1],
    )
    print(f"\nModality contribution ranking (most to least causally important):")
    for i, (name, drop) in enumerate(ranked, 1):
        print(f"  {i}. {name} (F1 drop: {drop:+.4f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt)")
    parser.add_argument("--binary", action="store_true", help="Set this if the checkpoint is the binary BL-vs-PA4 model")
    parser.add_argument("--class-a", type=int, default=0)
    parser.add_argument("--class-b", type=int, default=4)
    args = parser.parse_args()
    run_ablation(args.checkpoint, binary=args.binary, class_a=args.class_a, class_b=args.class_b)
