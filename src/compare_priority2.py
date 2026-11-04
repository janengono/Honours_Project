
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RF_DIR = ROOT / "results" / "priority2_rf"
OUT = RF_DIR / "paired_comparisons.csv"

PAIRS = [
    ("transformer", "rf_multimodal"),
    ("rf_multimodal", "rf_emg"),
    ("rf_multimodal", "rf_eda"),
    ("rf_emg", "rf_eda"),
]

def load_aligned(task):
    rf = pd.read_csv(RF_DIR / f"{task}_predictions.csv")

    filename = (
        "cv_ablation_5class_10fold_subjnorm_interpretability.csv"
        if task == "five_class"
        else "cv_ablation_binary_10fold_interpretability.csv"
    )
    tf = pd.read_csv(ROOT / "results" / filename)
    tf = tf[tf["condition"] == "full"].copy()
    tf["trial_index"] = tf.groupby("fold", sort=False).cumcount()
    tf["model"] = "transformer"

    cols = [
        "model", "fold", "trial_index",
        "subject_id", "y_true", "y_pred"
    ]
    combined = pd.concat(
        [rf[cols], tf[cols]], ignore_index=True
    )

    combined["subject_id"] = (
        combined["subject_id"].astype(str)
    )

    keys = ["fold", "trial_index"]
    labels = combined[
        combined["model"] == "transformer"
    ][keys + ["subject_id", "y_true"]].copy()

    if labels.duplicated(keys).any():
        raise ValueError("Duplicate Transformer trial keys")

    wide = labels.copy()

    for name, group in combined.groupby("model"):
        if group.duplicated(keys).any():
            raise ValueError(f"Duplicate keys for {name}")

        if len(group) != len(labels):
            raise ValueError(f"Trial count mismatch: {name}")

        merged = labels.merge(
            group[keys + ["subject_id", "y_true", "y_pred"]],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("_ref", "_model"),
            indicator=True,
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(f"Missing trials: {name}")

        if not (
            merged["subject_id_ref"].astype(str).to_numpy()
            == merged["subject_id_model"].astype(str).to_numpy()
        ).all():
            raise ValueError(f"Subject mismatch: {name}")

        if not (
            merged["y_true_ref"].to_numpy()
            == merged["y_true_model"].to_numpy()
        ).all():
            raise ValueError(f"Label mismatch: {name}")

        wide[name] = merged["y_pred"].to_numpy()

    return wide


def paired_bootstrap(wide, first, second, n_classes,
                     n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    y = wide["y_true"].to_numpy()
    a = wide[first].to_numpy()
    b = wide[second].to_numpy()

    def macro_f1(indices):
        f1_a = f1_score(
            y[indices], a[indices],
            labels=labels, average="macro", zero_division=0
        )
        f1_b = f1_score(
            y[indices], b[indices],
            labels=labels, average="macro", zero_division=0
        )
        return f1_a - f1_b

    observed = macro_f1(np.arange(len(wide)))

    subjects = wide["subject_id"].unique()
    subject_array = wide["subject_id"].to_numpy()
    subject_indices = {
        s: np.flatnonzero(subject_array == s)
        for s in subjects
    }

    draws = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(
            subjects, size=len(subjects), replace=True
        )
        indices = np.concatenate(
            [subject_indices[s] for s in sampled]
        )
        draws[i] = macro_f1(indices)

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return observed, lo, hi


def main():
    rows = []

    for task, n_classes, expected in [
        ("five_class", 5, 8600),
        ("binary", 2, 3440),
    ]:
        wide = load_aligned(task)

        if len(wide) != expected:
            raise ValueError(
                f"{task}: expected {expected}, got {len(wide)}"
            )

        if wide["subject_id"].nunique() != 86:
            raise ValueError("Expected 86 subjects")

        print(f"\n{task}: {len(wide)} aligned trials")

        for first, second in PAIRS:
            diff, lo, hi = paired_bootstrap(
                wide, first, second, n_classes
            )

            rows.append({
                "task": task,
                "model_a": first,
                "model_b": second,
                "macro_f1_difference": diff,
                "ci_lower": lo,
                "ci_upper": hi,
            })

            print(
                f"{first} - {second}: "
                f"{diff:+.4f} "
                f"[{lo:+.4f}, {hi:+.4f}]"
            )

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
