"""
shap_analysis.py
-----------------
Implements Section 3.4.2 (SHAP-Based Explainability Analysis, Equation 3.8).

PainTransformer takes a dict of per-modality tensors, but the `shap` library
expects a model that's a plain callable over a single tensor. FlattenedInputModel
below wraps the trained model so SHAP can treat it as a black box while the
wrapper handles splitting the input back into modalities internally.

Uses shap.GradientExplainer, which works directly with PyTorch models via
backpropagated gradients (faster and more stable for a differentiable model
like this one than KernelExplainer, which treats the model as a true black
box and needs many more samples to converge).

Usage:
    python shap_analysis.py --checkpoint ../checkpoints/best_model.pt
    python shap_analysis.py --checkpoint ../checkpoints/best_model_binary.pt --binary

NOTE: this was written without a local `shap` install to test against (no
internet access in the environment this was developed in). The GradientExplainer
API has been stable across recent shap versions, but if you hit an error on
`explainer.shap_values(...)`, paste it back and it can be patched quickly --
the model/wrapper code itself doesn't depend on shap's specifics.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from model import PainTransformer, get_modality_indices
from pain_dataset import load_sequence_data
from splits import subject_level_split, apply_subject_split
from feature_scaling import apply_scaler
from train_binary import filter_to_binary


class FlattenedInputModel(nn.Module):
    """Wraps PainTransformer so it accepts one (batch, seq_len, n_features)
    tensor instead of a dict, splitting it back into modalities internally
    using the same index mapping get_modality_indices() produces."""

    def __init__(self, base_model, modality_indices):
        super().__init__()
        self.base_model = base_model
        self.modality_indices = modality_indices

    def forward(self, x):
        modality_tensors = {
            name: x[:, :, idxs] for name, idxs in self.modality_indices.items()
        }
        return self.base_model(modality_tensors)


def run_shap_analysis(checkpoint_path, binary=False, class_a=0, class_b=4,
                       n_background=100, n_explain=200):
    import shap  # imported here so the rest of the file is importable even if shap isn't installed yet

    device = torch.device("cpu")  # SHAP + small models: CPU is fine and avoids GPU/driver issues here

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mod_dims = checkpoint["modality_dims"]
    cfg = checkpoint["model_config"]
    feat_mean, feat_std = checkpoint["feature_mean"], checkpoint["feature_std"]
    n_classes = 2 if binary else 5

    base_model = PainTransformer(
        modality_dims=mod_dims, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], fusion_heads=cfg["fusion_heads"],
        n_classes=n_classes, dropout=cfg["dropout"],
    ).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    X, y, subject_ids, feature_names = load_sequence_data()
    if binary:
        X, y, subject_ids = filter_to_binary(X, y, subject_ids, class_a, class_b)
    modality_indices = get_modality_indices(feature_names)

    _, val_subj, _ = subject_level_split(subject_ids, seed=42)
    X_val, y_val, sid_val = apply_subject_split(X, y, subject_ids, val_subj)
    X_val = apply_scaler(X_val, feat_mean, feat_std)

    wrapped_model = FlattenedInputModel(base_model, modality_indices)
    wrapped_model.eval()

    # Background set: a random sample of validation trials, used by SHAP as
    # the "reference" distribution each explained prediction is compared against
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_val), size=min(n_background, len(X_val)), replace=False)
    background = torch.tensor(X_val[bg_idx], dtype=torch.float32)

    explain_idx = rng.choice(len(X_val), size=min(n_explain, len(X_val)), replace=False)
    to_explain = torch.tensor(X_val[explain_idx], dtype=torch.float32)
    y_explain = y_val[explain_idx]

    print(f"Background set: {len(background)} trials, explaining {len(to_explain)} trials")
    print("Running SHAP GradientExplainer (this may take a few minutes on CPU)...")

    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(to_explain)

    # shap_values shape handling: GradientExplainer on a multi-output model
    # typically returns a list of arrays (one per class) or a single array
    # with an extra trailing class dimension, depending on shap version.
    if isinstance(shap_values, list):
        per_class_values = shap_values  # list of (n_explain, seq_len, n_features)
    else:
        # assume shape (n_explain, seq_len, n_features, n_classes)
        per_class_values = [shap_values[..., c] for c in range(n_classes)]

    class_names = ([f"class_{class_a}", f"class_{class_b}"] if binary
                    else ["BL", "PA1", "PA2", "PA3", "PA4"])

    # --- Global feature importance, per class ---
    print(f"\n{'='*70}\nGLOBAL FEATURE IMPORTANCE (mean |SHAP value| across trials and sub-windows)\n{'='*70}")

    for class_id, class_name in enumerate(class_names):
        class_mask = y_explain == class_id
        if class_mask.sum() == 0:
            continue
        vals = per_class_values[class_id][class_mask]  # (n_this_class, seq_len, n_features)
        mean_abs_per_feature = np.mean(np.abs(vals), axis=(0, 1))  # (n_features,)

        top_idx = np.argsort(-mean_abs_per_feature)[:5]
        print(f"\n--- {class_name} (n={class_mask.sum()}) top 5 features ---")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {feature_names[idx]:<28} mean|SHAP| = {mean_abs_per_feature[idx]:.5f}")

    # --- Global modality-level importance (summed across that modality's features) ---
    print(f"\n{'='*70}\nMODALITY-LEVEL IMPORTANCE (summed feature importance per modality)\n{'='*70}")
    all_vals = np.concatenate(per_class_values, axis=0)  # pool across classes for an overall ranking
    overall_mean_abs = np.mean(np.abs(all_vals), axis=(0, 1))  # (n_features,)

    for modality_name, idx_list in modality_indices.items():
        modality_importance = overall_mean_abs[idx_list].sum()
        print(f"  {modality_name:<8} total importance: {modality_importance:.5f}")

    return per_class_values, feature_names, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--class-a", type=int, default=0)
    parser.add_argument("--class-b", type=int, default=4)
    parser.add_argument("--n-background", type=int, default=100)
    parser.add_argument("--n-explain", type=int, default=200)
    args = parser.parse_args()
    run_shap_analysis(
        args.checkpoint, binary=args.binary, class_a=args.class_a, class_b=args.class_b,
        n_background=args.n_background, n_explain=args.n_explain,
    )
