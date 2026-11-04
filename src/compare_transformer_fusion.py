
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
UNI_DIR = ROOT / "results" / "priority2_unimodal"
OUTPUT = UNI_DIR / "paired_fusion_comparisons.csv"

N_BOOT = 2000
SEED = 42


def load_predictions(task):
    uni = pd.read_csv(UNI_DIR / f"{task}_predictions.csv")

    reference_name = (
        "cv_ablation_5class_10fold_subjnorm_interpretability.csv"
        if task == "five_class"
        else "cv_ablation_binary_10fold_interpretability.csv"
    )

    multi = pd.read_csv(ROOT / "results" / reference_name)
    multi = multi[multi["condition"] == "full"].copy()

    multi["trial_index"] = (
        multi.groupby("fold", sort=False).cumcount()
    )
    multi["model"] = "transformer_multimodal"

    columns = [
        "model", "fold", "trial_index",
        "subject_id", "y_true", "y_pred"
    ]

    combined = pd.concat(
        [multi[columns], uni[columns]],
        ignore_index=True
    )

    keys = ["fold", "trial_index"]
    names = [
        "transformer_multimodal",
        "transformer_ecg",
        "transformer_eda",
        "transformer_emg",
    ]

    reference = combined[
        combined["model"] == names[0]
    ].copy()

    if reference.duplicated(keys).any():
        raise ValueError("Duplicate reference trial keys")

    wide = reference[
        keys + ["subject_id", "y_true"]
    ].copy()

    for name in names:
        subset = combined[
            combined["model"] == name
        ].copy()

        if len(subset) != len(reference):
            raise ValueError(f"Trial count mismatch: {name}")

        if subset.duplicated(keys).any():
            raise ValueError(f"Duplicate trial keys: {name}")

        merged = wide[
            keys + ["subject_id", "y_true"]
        ].merge(
            subset[
                keys + ["subject_id", "y_true", "y_pred"]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("_ref", "_new"),
            indicator=True,
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(f"Missing trials: {name}")

        if not (
            merged["subject_id_ref"].astype(str).to_numpy()
            == merged["subject_id_new"].astype(str).to_numpy()
        ).all():
            raise ValueError(f"Subject mismatch: {name}")

        if not (
            merged["y_true_ref"].to_numpy()
            == merged["y_true_new"].to_numpy()
        ).all():
            raise ValueError(f"Label mismatch: {name}")

        wide[name] = merged["y_pred"].to_numpy()

    return wide


def bootstrap_difference(
    wide, first, second, n_classes,
    n_boot=N_BOOT, seed=SEED
):
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))

    y = wide["y_true"].to_numpy()
    pred_a = wide[first].to_numpy()
    pred_b = wide[second].to_numpy()

    subjects = wide["subject_id"].unique()
    subject_ids = wide["subject_id"].to_numpy()

    subject_rows = {
        sid: np.flatnonzero(subject_ids == sid)
        for sid in subjects
    }

    def difference(indices):
        a = f1_score(
            y[indices], pred_a[indices],
            labels=labels,
            average="macro",
            zero_division=0
        )
        b = f1_score(
            y[indices], pred_b[indices],
            labels=labels,
            average="macro",
            zero_division=0
        )
        return a - b

    observed = difference(np.arange(len(wide)))
    draws = np.empty(n_boot)

    for i in range(n_boot):
        sampled = rng.choice(
            subjects,
            size=len(subjects),
            replace=True
        )

        indices = np.concatenate(
            [subject_rows[sid] for sid in sampled]
        )
        draws[i] = difference(indices)

    lower, upper = np.percentile(draws, [2.5, 97.5])
    return observed, lower, upper


def main():
    rows = []

    comparisons = [
        ("transformer_multimodal", "transformer_emg"),
        ("transformer_multimodal", "transformer_eda"),
        ("transformer_multimodal", "transformer_ecg"),
        ("transformer_emg", "transformer_eda"),
    ]

    for task, n_classes, expected in [
        ("five_class", 5, 8600),
        ("binary", 2, 3440),
    ]:
        wide = load_predictions(task)

        if len(wide) != expected:
            raise ValueError(
                f"{task}: expected {expected} trials"
            )

        if wide["subject_id"].nunique() != 86:
            raise ValueError("Expected 86 subjects")

        print(
            f"\n{task.upper()} — "
            f"{len(wide)} aligned held-out trials"
        )

        for first, second in comparisons:
            diff, lower, upper = bootstrap_difference(
                wide, first, second, n_classes
            )

            rows.append({
                "task": task,
                "model_a": first,
                "model_b": second,
                "macro_f1_difference": diff,
                "ci_lower": lower,
                "ci_upper": upper,
            })

            print(
                f"{first} - {second}: "
                f"{diff:+.4f} "
                f"[{lower:+.4f}, {upper:+.4f}]"
            )

    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
