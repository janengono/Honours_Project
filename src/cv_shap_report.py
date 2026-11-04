
"""
Final cross-validated SHAP reporting.

Reads previously saved SHAP arrays; does NOT rerun SHAP.

Run from project root:
    python src/cv_shap_report.py

Outputs:
    results/cv_shap/final_report/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "cv_shap"
DEST = SOURCE / "final_report"
DEST.mkdir(parents=True, exist_ok=True)

TASKS = {
    "five_class": ["BL", "PA1", "PA2", "PA3", "PA4"],
    "binary": ["BL", "PA4"],
}

COLORS = {
    "ECG": "#64748b",
    "EDA": "#d946a6",
    "EMG": "#8b5cf6",
}


def modality_of(feature):
    name = str(feature).lower()
    if name.startswith(("gsr", "eda")):
        return "EDA"
    if name.startswith("ecg"):
        return "ECG"
    if name.startswith("emg"):
        return "EMG"
    raise ValueError(f"Unknown feature modality: {feature}")


def load_task(task):
    files = sorted((SOURCE / task).glob("fold_*.npz"))

    if len(files) != 10:
        raise ValueError(
            f"{task}: expected 10 SHAP folds, found {len(files)}"
        )

    records = []
    names = None
    seen_folds = set()

    for path in files:
        fold = int(path.stem.split("_")[-1])

        if fold in seen_folds:
            raise ValueError("Duplicate fold")
        seen_folds.add(fold)

        with np.load(path, allow_pickle=False) as d:
            v = d["shap_values"].copy()
            y = d["y_true"].copy().astype(int)
            pred = d["y_pred"].copy().astype(int)
            sids = d["subject_ids"].copy()
            probs = d["probabilities"].copy()
            x = d["inputs"].copy()
            features = d["feature_names"].astype(str).tolist()

        if names is None:
            names = features
        elif features != names:
            raise ValueError("Feature order differs between folds")

        n = len(y)
        n_classes = len(TASKS[task])

        if v.shape != (n, x.shape[1], len(names), n_classes):
            raise ValueError(f"Unexpected SHAP shape in {path}")

        if probs.shape != (n, n_classes):
            raise ValueError("Probability shape mismatch")

        if not np.isfinite(v).all():
            raise ValueError("Non-finite SHAP values")

        if np.any((y < 0) | (y >= n_classes)):
            raise ValueError("Invalid true labels")

        if np.any((pred < 0) | (pred >= n_classes)):
            raise ValueError("Invalid predictions")

        if not np.array_equal(probs.argmax(axis=1), pred):
            raise ValueError("Saved probabilities/predictions disagree")

        records.append({
            "fold": fold,
            "values": v,
            "inputs": x,
            "true": y,
            "pred": pred,
            "sids": sids,
            "probs": probs,
        })

    records.sort(key=lambda r: r["fold"])
    return records, names


def build_tables(task, records, names):
    feature_rows = []
    modality_rows = []
    trial_rows = []
    class_rows = []

    for r in records:
        v = r["values"]
        y = r["true"]
        pred = r["pred"]
        fold = r["fold"]

        # Select the logit corresponding to each trial's true class.
        true_attr = np.take_along_axis(
            v,
            y[:, None, None, None],
            axis=-1,
        )[..., 0]

        # Separately select the predicted-class logit.
        pred_attr = np.take_along_axis(
            v,
            pred[:, None, None, None],
            axis=-1,
        )[..., 0]

        # Each trial receives equal weight.
        # Time-window dimension is averaged.
        true_abs = np.abs(true_attr).mean(axis=1)
        pred_abs = np.abs(pred_attr).mean(axis=1)

        for i in range(len(y)):
            trial_rows.append({
                "task": task,
                "fold": fold,
                "subject_id": str(r["sids"][i]),
                "local_index": i,
                "true_class": TASKS[task][y[i]],
                "predicted_class": TASKS[task][pred[i]],
                "correct": int(y[i] == pred[i]),
                "confidence": float(r["probs"][i, pred[i]]),
            })

            for j, feature in enumerate(names):
                feature_rows.append({
                    "task": task,
                    "fold": fold,
                    "subject_id": str(r["sids"][i]),
                    "true_class": TASKS[task][y[i]],
                    "feature": feature,
                    "modality": modality_of(feature),
                    "true_class_importance": float(true_abs[i, j]),
                    "predicted_class_importance": float(
                        pred_abs[i, j]
                    ),
                })

            for modality in ("ECG", "EDA", "EMG"):
                ids = [
                    j for j, name in enumerate(names)
                    if modality_of(name) == modality
                ]

                modality_rows.append({
                    "task": task,
                    "fold": fold,
                    "subject_id": str(r["sids"][i]),
                    "true_class": TASKS[task][y[i]],
                    "modality": modality,
                    "n_features": len(ids),
                    "total_importance": float(
                        true_abs[i, ids].sum()
                    ),
                    "mean_per_feature": float(
                        true_abs[i, ids].mean()
                    ),
                    "predicted_total_importance": float(
                        pred_abs[i, ids].sum()
                    ),
                })

        # Class-specific rankings:
        # explain the class logit on trials bearing that true label.
        for c, class_name in enumerate(TASKS[task]):
            mask = y == c
            if not mask.any():
                continue

            scores = np.abs(v[mask, :, :, c]).mean(axis=(0, 1))

            for j, name in enumerate(names):
                class_rows.append({
                    "task": task,
                    "fold": fold,
                    "class": class_name,
                    "feature": name,
                    "importance": float(scores[j]),
                    "n_trials": int(mask.sum()),
                })

    feature_df = pd.DataFrame(feature_rows)
    modality_df = pd.DataFrame(modality_rows)
    trial_df = pd.DataFrame(trial_rows)
    class_df = pd.DataFrame(class_rows)

    # Aggregate from trial-level values, not printed fold rankings.
    global_features = (
        feature_df.groupby(
            ["task", "feature", "modality"], as_index=False
        )[[
            "true_class_importance",
            "predicted_class_importance",
        ]]
        .mean()
        .sort_values(
            "true_class_importance", ascending=False
        )
    )

    global_modalities = (
        modality_df.groupby(
            ["task", "modality", "n_features"], as_index=False
        )[[
            "total_importance",
            "mean_per_feature",
            "predicted_total_importance",
        ]]
        .mean()
        .sort_values(
            "total_importance", ascending=False
        )
    )

    # Weight fold-specific class importance by the number
    # of sampled trials from that class.
    class_df["weighted_importance"] = (
        class_df["importance"] * class_df["n_trials"]
    )

    grouped = class_df.groupby(
        ["task", "class", "feature"], as_index=False
    ).agg(
        weighted_sum=("weighted_importance", "sum"),
        n_trials=("n_trials", "sum"),
    )

    grouped["importance"] = (
        grouped["weighted_sum"] / grouped["n_trials"]
    )

    global_classes = grouped[
        ["task", "class", "feature", "importance", "n_trials"]
    ].sort_values(
        ["class", "importance"],
        ascending=[True, False],
    )

    return (
        global_features,
        global_modalities,
        global_classes,
        trial_df,
        modality_df,
    )


def plot_features(task, df):
    top = df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top["feature"],
        top["true_class_importance"],
        color=[
            COLORS[m] for m in top["modality"]
        ],
    )

    ax.set_xlabel("Mean absolute SHAP attribution")
    ax.set_title(
        f"{task.replace('_', ' ').title()}: "
        "global feature importance\n"
        "True-class logits; held-out trials"
    )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=COLORS[m], label=m)
            for m in ("ECG", "EDA", "EMG")
        ],
        loc="lower right",
    )

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(
        DEST / f"{task}_top20_features.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_modalities(task, df):
    order = ["ECG", "EDA", "EMG"]
    df = df.set_index("modality").loc[order]

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )

    for ax, col, title in zip(
        axes,
        ["total_importance", "mean_per_feature"],
        [
            "Summed attribution per modality",
            "Mean attribution per feature",
        ],
    ):
        ax.bar(
            order,
            df[col].values,
            color=[COLORS[m] for m in order],
        )
        ax.set_title(title)
        ax.set_ylabel("Mean absolute SHAP attribution")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{task.replace('_', ' ').title()} — "
        "cross-validated SHAP"
    )
    fig.tight_layout()
    fig.savefig(
        DEST / f"{task}_modalities.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_class_features(task, class_df):
    classes = TASKS[task]
    fig, axes = plt.subplots(
        1, len(classes),
        figsize=(4.5 * len(classes), 5),
        squeeze=False,
    )

    for ax, label in zip(axes[0], classes):
        sub = (
            class_df[class_df["class"] == label]
            .nlargest(8, "importance")
            .iloc[::-1]
        )

        ax.barh(
            sub["feature"],
            sub["importance"],
            color=[
                COLORS[modality_of(f)]
                for f in sub["feature"]
            ],
        )
        ax.set_title(label)
        ax.set_xlabel("Mean |SHAP|")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{task.replace('_', ' ').title()}: "
        "class-specific feature importance"
    )
    fig.tight_layout()
    fig.savefig(
        DEST / f"{task}_class_specific.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_individuals(task, records, names):
    """
    Select one high-confidence correct and one high-confidence
    incorrect prediction per task, if available.

    Signed contributions are summed over time windows,
    then ranked by absolute magnitude.
    """
    candidates = []

    for r in records:
        for i in range(len(r["true"])):
            candidates.append((
                float(r["probs"][i, r["pred"][i]]),
                r,
                i,
            ))

    examples = []

    for correct in (True, False):
        eligible = [
            item for item in candidates
            if (
                item[1]["true"][item[2]]
                == item[1]["pred"][item[2]]
            ) == correct
        ]

        if eligible:
            examples.append((
                correct,
                max(eligible, key=lambda z: z[0]),
            ))

    for correct, (confidence, r, i) in examples:
        predicted_class = int(r["pred"][i])

        # Signed attribution to the predicted-class logit.
        # Sum over time to preserve net direction.
        signed = r["values"][
            i, :, :, predicted_class
        ].sum(axis=0)

        top = np.argsort(-np.abs(signed))[:12]
        top = top[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            [names[j] for j in top],
            signed[top],
            color=[
                "#16a34a" if signed[j] >= 0 else "#dc2626"
                for j in top
            ],
        )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Signed SHAP contribution to predicted-class logit"
        )

        true_name = TASKS[task][int(r["true"][i])]
        pred_name = TASKS[task][predicted_class]

        ax.set_title(
            f"{task.replace('_', ' ').title()} | "
            f"{'Correct' if correct else 'Incorrect'}\n"
            f"True: {true_name} | Predicted: {pred_name} | "
            f"Confidence: {confidence:.2f} | Fold: {r['fold']}"
        )

        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()

        label = "correct" if correct else "incorrect"
        fig.savefig(
            DEST / f"{task}_individual_{label}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    combined_features = []
    combined_modalities = []
    combined_classes = []
    combined_trials = []

    for task in TASKS:
        print(f"\nProcessing {task}...", flush=True)

        records, names = load_task(task)

        (
            features,
            modalities,
            classes,
            trials,
            modality_trials,
        ) = build_tables(task, records, names)

        combined_features.append(features)
        combined_modalities.append(modalities)
        combined_classes.append(classes)
        combined_trials.append(trials)

        plot_features(task, features)
        plot_modalities(task, modalities)
        plot_class_features(task, classes)
        plot_individuals(task, records, names)

        modality_trials.to_csv(
            DEST / f"{task}_trial_modality_importance.csv",
            index=False,
        )

        print(f"Explained trials: {len(trials)}")
        print("\nTop five features:")
        print(
            features[
                ["feature", "true_class_importance"]
            ].head(5).to_string(index=False)
        )

        print("\nModality ranking:")
        print(
            modalities[
                [
                    "modality",
                    "total_importance",
                    "mean_per_feature",
                ]
            ].to_string(index=False)
        )

    pd.concat(combined_features).to_csv(
        DEST / "global_feature_rankings.csv",
        index=False,
    )

    pd.concat(combined_modalities).to_csv(
        DEST / "global_modality_rankings.csv",
        index=False,
    )

    pd.concat(combined_classes).to_csv(
        DEST / "class_specific_rankings.csv",
        index=False,
    )

    pd.concat(combined_trials).to_csv(
        DEST / "explained_trial_metadata.csv",
        index=False,
    )

    print("\nAll final SHAP reports saved to:")
    print(DEST)


if __name__ == "__main__":
    main()
