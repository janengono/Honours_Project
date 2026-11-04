
"""
Cross-validated SHAP for the final PainTransformer models.

Run from the project root:
python src/cv_shap_analysis.py --task five_class --max-folds 1
python src/cv_shap_analysis.py --task binary --max-folds 1

Then run all folds without --max-folds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import shap

from model import PainTransformer, get_modality_indices
from pain_dataset import load_sequence_data
from train_binary import filter_to_binary
from feature_scaling import apply_scaler
from cv_utils import (
    make_folds,
    split_train_val_subjects,
    subject_normalize,
)
from shap_analysis import FlattenedInputModel


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "cv_shap"
OUTPUT.mkdir(parents=True, exist_ok=True)


def get_model(checkpoint, feature_names, n_classes):
    cfg = checkpoint["model_config"]

    model = PainTransformer(
        modality_dims=checkpoint["modality_dims"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        fusion_heads=cfg["fusion_heads"],
        n_classes=n_classes,
        dropout=cfg["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices = get_modality_indices(feature_names)
    wrapper = FlattenedInputModel(model, indices)
    wrapper.eval()

    return wrapper, indices


def stratified_sample(y, candidate_indices, per_class, rng):
    """Sample held-out trials with balanced class representation."""
    chosen = []

    for label in np.unique(y[candidate_indices]):
        available = candidate_indices[
            y[candidate_indices] == label
        ]

        selected = rng.choice(
            available,
            size=min(per_class, len(available)),
            replace=False,
        )
        chosen.extend(selected.tolist())

    return np.asarray(chosen, dtype=int)


def normalise_shap_output(values, n_samples, seq_len,
                          n_features, n_classes):
    """Support list and ndarray SHAP multi-output formats."""
    if isinstance(values, list):
        if len(values) != n_classes:
            raise ValueError("Unexpected number of SHAP outputs")

        arr = np.stack(
            [np.asarray(v) for v in values], axis=-1
        )
    else:
        arr = np.asarray(values)

    expected = (n_samples, seq_len, n_features, n_classes)

    if arr.shape != expected:
        raise ValueError(
            f"Unexpected SHAP shape {arr.shape}; expected {expected}"
        )

    if not np.isfinite(arr).all():
        raise ValueError("Non-finite SHAP values detected")

    return arr


def analyse_fold(task, fold_id, X, y, sids, feature_names,
                 test_subjects, args):

    binary = task == "binary"
    n_classes = 2 if binary else 5

    checkpoint_dir = (
        ROOT / "checkpoints" / "interpretability" /
        ("binary_standard" if binary else "5class_subjnorm")
    )

    path = checkpoint_dir / f"fold_{fold_id:02d}.pt"

    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    saved_subjects = np.asarray(checkpoint["test_subjects"])

    if set(saved_subjects.tolist()) != set(test_subjects.tolist()):
        raise ValueError(
            f"Fold {fold_id}: checkpoint subjects do not match CV"
        )

    if checkpoint["n_classes"] != n_classes:
        raise ValueError("Checkpoint class count mismatch")

    if bool(checkpoint["subject_norm"]) != (not binary):
        raise ValueError("Checkpoint preprocessing mismatch")

    if list(checkpoint["feature_names"]) != list(feature_names):
        raise ValueError("Feature order mismatch")

    model, modality_indices = get_model(
        checkpoint, feature_names, n_classes
    )

    # Reconstruct training and held-out subjects.
    all_subjects = np.unique(sids)
    train_pool = np.setdiff1d(all_subjects, test_subjects)

    train_subjects, val_subjects = split_train_val_subjects(
        train_pool, val_frac=0.1, seed=42 + fold_id
    )

    train_idx = np.flatnonzero(
        np.isin(sids, train_subjects)
    )

    test_idx = np.flatnonzero(
        np.isin(sids, test_subjects)
    )

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Empty training or test set")

    rng = np.random.default_rng(args.seed + fold_id)

    # Background: training subjects only.
    bg_idx = rng.choice(
        train_idx,
        size=min(args.background, len(train_idx)),
        replace=False,
    )

    # Explanations: held-out subjects only.
    explain_idx = stratified_sample(
        y, test_idx, args.per_class, rng
    )

    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]

    X_bg = apply_scaler(X[bg_idx], mean, std)
    X_explain = apply_scaler(X[explain_idx], mean, std)

    background = torch.tensor(
        X_bg, dtype=torch.float32
    )

    explained = torch.tensor(
        X_explain, dtype=torch.float32
    )

    print(
        f"\n{task} fold {fold_id}: "
        f"{len(bg_idx)} background trials, "
        f"{len(explain_idx)} held-out explanations",
        flush=True,
    )

    explainer = shap.GradientExplainer(
        model, background
    )

    values = explainer.shap_values(
        explained,
        nsamples=args.nsamples,
    )

    values = normalise_shap_output(
        values,
        len(explain_idx),
        X.shape[1],
        X.shape[2],
        n_classes,
    )

    with torch.no_grad():
        logits = model(explained).numpy()
        probabilities = torch.softmax(
            torch.tensor(logits), dim=1
        ).numpy()

    predictions = probabilities.argmax(axis=1)

    # Extract attribution to each trial's TRUE class logit.
    true_values = np.stack([
        values[i, :, :, int(y[idx])]
        for i, idx in enumerate(explain_idx)
    ])

    # Also preserve attribution to the PREDICTED class logit.
    predicted_values = np.stack([
        values[i, :, :, int(predictions[i])]
        for i in range(len(explain_idx))
    ])

    fold_dir = OUTPUT / task
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Store raw explanations for later plots and local analysis.
    np.savez_compressed(
        fold_dir / f"fold_{fold_id:02d}.npz",
        shap_values=values.astype(np.float32),
        true_class_shap=true_values.astype(np.float32),
        predicted_class_shap=predicted_values.astype(np.float32),
        inputs=X_explain.astype(np.float32),
        y_true=y[explain_idx],
        y_pred=predictions,
        probabilities=probabilities.astype(np.float32),
        subject_ids=sids[explain_idx],
        dataset_indices=explain_idx,
        feature_names=np.asarray(feature_names),
    )

    # Feature rankings from true-class logit attribution.
    importance = np.abs(true_values).mean(axis=(0, 1))

    feature_rows = []

    for feature_id, feature_name in enumerate(feature_names):
        feature_rows.append({
            "task": task,
            "fold": fold_id,
            "feature": feature_name,
            "importance": float(importance[feature_id]),
            "n_trials": len(explain_idx),
        })

    pd.DataFrame(feature_rows).to_csv(
        fold_dir / f"features_fold_{fold_id:02d}.csv",
        index=False,
    )

    modality_rows = []

    for name, indices in modality_indices.items():
        vals = importance[indices]

        modality_rows.append({
            "task": task,
            "fold": fold_id,
            "modality": name.upper(),
            "total_importance": float(vals.sum()),
            "mean_per_feature": float(vals.mean()),
            "n_features": len(indices),
            "n_trials": len(explain_idx),
        })

    pd.DataFrame(modality_rows).to_csv(
        fold_dir / f"modalities_fold_{fold_id:02d}.csv",
        index=False,
    )

    # Metadata for representative individual explanations.
    local_rows = []

    for i, idx in enumerate(explain_idx):
        local_rows.append({
            "task": task,
            "fold": fold_id,
            "local_index": i,
            "dataset_index": int(idx),
            "subject_id": int(sids[idx]),
            "y_true": int(y[idx]),
            "y_pred": int(predictions[i]),
            "correct": bool(predictions[i] == y[idx]),
            "predicted_confidence": float(
                probabilities[i, predictions[i]]
            ),
        })

    pd.DataFrame(local_rows).to_csv(
        fold_dir / f"individuals_fold_{fold_id:02d}.csv",
        index=False,
    )

    print("Top five features:")
    for j in np.argsort(-importance)[:5]:
        print(f"  {feature_names[j]:<28} {importance[j]:.5f}")

    print("Modality importance:")
    for row in modality_rows:
        print(
            f"  {row['modality']:<5} "
            f"total={row['total_importance']:.5f} "
            f"per_feature={row['mean_per_feature']:.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True,
        choices=["five_class", "binary"]
    )
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--background", type=int, default=30)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(2)

    X, y, sids, feature_names = load_sequence_data()

    if args.task == "binary":
        X, y, sids = filter_to_binary(X, y, sids, 0, 4)
    else:
        X = subject_normalize(X, sids).astype(np.float32)

    folds = make_folds(sids, n_folds=10, seed=42)

    for fold_id, test_subjects in enumerate(
        folds[:args.max_folds]
    ):
        analyse_fold(
            args.task, fold_id, X, y, sids,
            feature_names, test_subjects, args
        )

    print("\nSHAP analysis completed.")


if __name__ == "__main__":
    main()
