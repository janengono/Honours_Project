
"""
Random Forest baselines for Priority 2.

Run from project root:
    python src/cv_rf_baselines.py --task five_class --max-folds 1
    python src/cv_rf_baselines.py --task binary --max-folds 1

Full experiments:
    python src/cv_rf_baselines.py --task five_class
    python src/cv_rf_baselines.py --task binary

Resume interrupted experiments:
    python src/cv_rf_baselines.py --task five_class --resume
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from pain_dataset import load_sequence_data
from train_binary import filter_to_binary
from model import get_modality_indices
from feature_scaling import fit_scaler, apply_scaler
from cv_utils import (
    make_folds,
    split_train_val_subjects,
    subject_normalize,
    summarize_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "priority2_rf"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["multimodal", "ecg", "eda", "emg"]

# Small, fixed candidate set; no test-set tuning.
PARAM_GRID = [
    {"n_estimators": 200, "min_samples_leaf": 1,
     "max_features": "sqrt"},
    {"n_estimators": 300, "min_samples_leaf": 3,
     "max_features": "sqrt"},
    {"n_estimators": 300, "min_samples_leaf": 5,
     "max_features": 0.5},
    {"n_estimators": 300, "min_samples_leaf": 2,
     "max_features": 0.5},
]


def score(y_true, y_pred, n_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        ),
    }


def feature_columns(feature_names, indices):
    """Flatten (N,T,F) after selecting the relevant modality."""
    names = [feature_names[i] for i in indices]
    return names


def make_rf(params, seed):
    return RandomForestClassifier(
        **params,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def run_fold(
    fold_id, test_subjects, X, y, sids,
    feature_names, modality_indices,
    n_classes, args,
):
    seed = args.seed + fold_id
    all_subjects = np.unique(sids)
    train_pool = np.setdiff1d(all_subjects, test_subjects)

    train_subj, val_subj = split_train_val_subjects(
        train_pool, args.val_frac, seed
    )

    tr = np.isin(sids, train_subj)
    va = np.isin(sids, val_subj)
    te = np.isin(sids, test_subjects)

    if not (tr.any() and va.any() and te.any()):
        raise ValueError("Empty train/validation/test split")

    if (
        np.any(tr & va)
        or np.any(tr & te)
        or np.any(va & te)
    ):
        raise ValueError("Subject split overlap")

    # Match the Transformer training-fold scaler.
    mean, std = fit_scaler(X[tr])

    X_train = apply_scaler(X[tr], mean, std)
    X_val = apply_scaler(X[va], mean, std)
    X_test = apply_scaler(X[te], mean, std)

    predictions = []
    selections = []

    for config in CONFIGS:
        if config == "multimodal":
            ids = np.arange(X.shape[-1])
        else:
            ids = np.asarray(
                modality_indices[config], dtype=int
            )

        # Select modality, then flatten 10 time windows.
        train_flat = X_train[:, :, ids].reshape(
            X_train.shape[0], -1
        )
        val_flat = X_val[:, :, ids].reshape(
            X_val.shape[0], -1
        )
        test_flat = X_test[:, :, ids].reshape(
            X_test.shape[0], -1
        )

        best_f1 = -1.0
        best_params = None
        best_model = None

        for params in PARAM_GRID:
            model = make_rf(params, seed)
            model.fit(train_flat, y[tr])

            val_pred = model.predict(val_flat)
            val_f1 = score(
                y[va], val_pred, n_classes
            )["macro_f1"]

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = params.copy()
                best_model = model

        # Important: the selected model remains fitted
        # ONLY on training subjects, matching the
        # Transformer's selected-epoch protocol.
        test_pred = best_model.predict(test_flat)
        test_scores = score(
            y[te], test_pred, n_classes
        )

        for trial_index, (sid, truth, pred) in enumerate(
            zip(sids[te], y[te], test_pred)
        ):
            predictions.append({
                "task": args.task,
                "model": f"rf_{config}",
                "fold": fold_id,
                "trial_index": trial_index,
                "subject_id": sid,
                "y_true": int(truth),
                "y_pred": int(pred),
            })

        selections.append({
            "task": args.task,
            "model": f"rf_{config}",
            "fold": fold_id,
            "n_train_subjects": len(train_subj),
            "n_val_subjects": len(val_subj),
            "n_test_subjects": len(test_subjects),
            "n_input_features": train_flat.shape[1],
            "val_macro_f1": best_f1,
            "test_accuracy": test_scores["accuracy"],
            "test_macro_f1": test_scores["macro_f1"],
            **best_params,
        })

        print(
            f"  {config:<11} "
            f"val_F1={best_f1:.4f} "
            f"test_acc={test_scores['accuracy']:.4f} "
            f"test_F1={test_scores['macro_f1']:.4f}",
            flush=True,
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(selections),
    )


def validate_transformer_alignment(task, rf_predictions):
    """
    Compare trial order and labels with the final
    Transformer ablation CSV.

    This does not require rerunning the Transformer.
    """
    if task == "five_class":
        name = (
            "cv_ablation_5class_10fold_"
            "subjnorm_interpretability.csv"
        )
    else:
        name = (
            "cv_ablation_binary_10fold_"
            "interpretability.csv"
        )

    path = ROOT / "results" / name

    if not path.exists():
        print(
            f"WARNING: Transformer file missing: {path}"
        )
        return

    old = pd.read_csv(path)
    old = old[old["condition"] == "full"].copy()

    # The Transformer and RF evaluation both use
    # the original test-mask trial order.
    old["trial_index"] = old.groupby(
        "fold", sort=False
    ).cumcount()

    ref = old[
        ["fold", "trial_index", "subject_id", "y_true"]
    ].copy()

    ref["subject_id"] = ref["subject_id"].astype(str)

    sample = rf_predictions[
        rf_predictions["model"] == "rf_multimodal"
    ][
        ["fold", "trial_index", "subject_id", "y_true"]
    ].copy()

    sample["subject_id"] = (
        sample["subject_id"].astype(str)
    )

    merged = sample.merge(
        ref,
        on=["fold", "trial_index"],
        how="left",
        suffixes=("_rf", "_transformer"),
        validate="one_to_one",
        indicator=True,
    )

    if not (
        (merged["_merge"] == "both").all()
        and (
            merged["subject_id_rf"]
            == merged["subject_id_transformer"]
        ).all()
        and (
            merged["y_true_rf"]
            == merged["y_true_transformer"]
        ).all()
    ):
        raise ValueError(
            "RF and Transformer held-out trials "
            "do not align. Do not compare results."
        )

    print(
        "Transformer alignment verified:",
        len(sample), "held-out trials"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True,
        choices=["five_class", "binary"],
    )
    parser.add_argument(
        "--max-folds", type=int, default=10
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1
    )
    args = parser.parse_args()

    X, y, sids, feature_names = load_sequence_data()

    if args.task == "binary":
        X, y, sids = filter_to_binary(
            X, y, sids, 0, 4
        )
        n_classes = 2
    else:
        X = subject_normalize(
            X, sids
        ).astype(np.float32)
        n_classes = 5

    modality_indices = get_modality_indices(
        feature_names
    )

    folds = make_folds(
        sids, n_folds=10, seed=args.seed
    )

    pred_path = OUT / f"{args.task}_predictions.csv"
    tuning_path = OUT / f"{args.task}_tuning.csv"

    if args.resume and pred_path.exists():
        previous = pd.read_csv(pred_path)
        if tuning_path.exists():
            previous_tuning = pd.read_csv(tuning_path)
        else:
            raise ValueError(
                "Resume requires the tuning CSV too"
            )

        completed = set(
            previous.groupby("fold")["model"]
            .nunique()
            .loc[lambda x: x == len(CONFIGS)]
            .index.tolist()
        )
    else:
        previous = pd.DataFrame()
        previous_tuning = pd.DataFrame()
        completed = set()

    pred_parts = (
        [previous] if not previous.empty else []
    )
    tuning_parts = (
        [previous_tuning]
        if not previous_tuning.empty else []
    )

    for fold_id, test_subjects in enumerate(folds):
        if fold_id >= args.max_folds:
            break

        if fold_id in completed:
            print(f"Skipping completed fold {fold_id}")
            continue

        print(
            f"\n{args.task} | fold {fold_id}/9 | "
            f"test subjects: {list(test_subjects)}",
            flush=True,
        )

        pred_df, tuning_df = run_fold(
            fold_id,
            test_subjects,
            X, y, sids,
            feature_names,
            modality_indices,
            n_classes,
            args,
        )

        pred_parts.append(pred_df)
        tuning_parts.append(tuning_df)

        # Write after each fold for safe resumption.
        pd.concat(
            pred_parts, ignore_index=True
        ).to_csv(pred_path, index=False)

        pd.concat(
            tuning_parts, ignore_index=True
        ).to_csv(tuning_path, index=False)

    if not pred_parts:
        raise ValueError("No results produced")

    predictions = pd.concat(
        pred_parts, ignore_index=True
    )

    validate_transformer_alignment(
        args.task, predictions
    )

    summary_rows = []

    for model_name, group in predictions.groupby("model"):
        metrics = summarize_predictions(
            group, n_classes, n_boot=2000, seed=42
        )

        summary_rows.append({
            "task": args.task,
            "model": model_name,
            "n_subjects": metrics["n_subjects"],
            "n_trials": metrics["n_trials"],
            "accuracy": metrics["pooled_accuracy"],
            "macro_f1": metrics["pooled_macro_f1"],
            "accuracy_ci_lower":
                metrics["accuracy_ci95"][0],
            "accuracy_ci_upper":
                metrics["accuracy_ci95"][1],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT / f"{args.task}_summary.csv",
        index=False,
    )

    print("\nPOOLED RANDOM FOREST RESULTS")
    print(summary.to_string(index=False))
    print("\nSaved results in:", OUT)


if __name__ == "__main__":
    main()
