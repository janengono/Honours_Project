
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "interpretability_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["full", "mask_ecg", "mask_eda", "mask_emg"]
MODALITIES = ["ECG", "EDA", "EMG"]
N_BOOTSTRAP = 2000
SEED = 42

TASKS = {
    "five_class": {
        "file": RESULTS / "cv_ablation_5class_10fold_subjnorm_interpretability.csv",
        "n_classes": 5,
        "title": "Five-class stimulus classification",
    },
    "binary": {
        "file": RESULTS / "cv_ablation_binary_10fold_interpretability.csv",
        "n_classes": 2,
        "title": "Binary BL vs PA4 classification",
    },
}


def score(y_true, y_pred, n_classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        ),
    }


def load_and_validate(path, n_classes):
    df = pd.read_csv(path)

    required = {
        "fold", "subject_id", "condition", "y_true", "y_pred"
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    if set(df["condition"]) != set(CONDITIONS):
        raise ValueError("Unexpected or missing masking conditions")

    # Check each subject has equal numbers of trials per condition.
    counts = df.groupby(
        ["subject_id", "condition"]
    ).size().unstack()

    if counts.isna().any().any():
        raise ValueError("Some subjects are missing conditions")

    if not counts.eq(counts["full"], axis=0).all().all():
        raise ValueError("Unequal trial counts between conditions")

    if df["y_true"].min() < 0 or df["y_true"].max() >= n_classes:
        raise ValueError("Unexpected class labels")

    # Check that every subject occurs in only one test fold.
    if df.groupby("subject_id")["fold"].nunique().max() != 1:
        raise ValueError("A subject appears in multiple test folds")

    # Reconstruct trial alignment from within-subject row order.
    # This is valid for the supplied script because all four
    # conditions use the same non-shuffled test loader.
    df["trial_index"] = df.groupby(
        ["subject_id", "condition"]
    ).cumcount()

    wide = df.pivot(
        index=["subject_id", "trial_index"],
        columns="condition",
        values="y_pred",
    )

    truth = df[df["condition"] == "full"].set_index(
        ["subject_id", "trial_index"]
    )["y_true"]

    if wide.isna().any().any():
        raise ValueError("Missing trial-level predictions")

    for condition in CONDITIONS:
        other = df[df["condition"] == condition].set_index(
            ["subject_id", "trial_index"]
        )["y_true"]
        if not truth.equals(other.reindex(truth.index)):
            raise ValueError(f"Label alignment mismatch: {condition}")

    wide = wide.loc[truth.index].astype(int)
    return truth, wide


def bootstrap_drops(truth, wide, n_classes):
    subjects = truth.index.get_level_values("subject_id").unique()
    sid_array = truth.index.get_level_values("subject_id").to_numpy()

    y = truth.to_numpy()
    preds = {c: wide[c].to_numpy() for c in CONDITIONS}

    subject_rows = {
        sid: np.flatnonzero(sid_array == sid)
        for sid in subjects
    }

    rng = np.random.default_rng(SEED)
    samples = {c: [] for c in CONDITIONS[1:]}

    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([subject_rows[s] for s in chosen])

        full_f1 = score(
            y[indices], preds["full"][indices], n_classes
        )["macro_f1"]

        for condition in CONDITIONS[1:]:
            masked_f1 = score(
                y[indices], preds[condition][indices], n_classes
            )["macro_f1"]
            samples[condition].append(full_f1 - masked_f1)

    return {
        c: np.asarray(values)
        for c, values in samples.items()
    }


def analyse_task(task_key, config):
    truth, wide = load_and_validate(
        config["file"], config["n_classes"]
    )
    n_classes = config["n_classes"]

    y = truth.to_numpy()
    performance = {
        c: score(y, wide[c].to_numpy(), n_classes)
        for c in CONDITIONS
    }

    full_f1 = performance["full"]["macro_f1"]
    bootstrap = bootstrap_drops(truth, wide, n_classes)

    rows = []
    for condition in CONDITIONS:
        result = performance[condition]
        drop = full_f1 - result["macro_f1"]

        if condition == "full":
            lo, hi = np.nan, np.nan
        else:
            lo, hi = np.quantile(
                bootstrap[condition], [0.025, 0.975]
            )

        rows.append({
            "task": task_key,
            "condition": condition,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "f1_drop": drop,
            "drop_ci_lower": lo,
            "drop_ci_upper": hi,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULTS / f"cv_ablation_summary_{task_key}.csv",
        index=False,
    )

    # Per-subject performance and masking differences.
    subject_rows = []
    subjects = truth.index.get_level_values("subject_id").unique()

    for sid in subjects:
        mask = truth.index.get_level_values("subject_id") == sid
        ys = y[mask]
        full = score(
            ys, wide.loc[mask, "full"].to_numpy(), n_classes
        )["macro_f1"]

        for condition in CONDITIONS[1:]:
            masked = score(
                ys, wide.loc[mask, condition].to_numpy(), n_classes
            )["macro_f1"]
            subject_rows.append({
                "task": task_key,
                "subject_id": sid,
                "modality": condition.replace("mask_", "").upper(),
                "f1_drop": full - masked,
            })

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(
        RESULTS / f"cv_ablation_per_subject_{task_key}.csv",
        index=False,
    )

    # Figure 1/2: Pooled F1 drop with bootstrap confidence intervals.
    plot_df = summary[summary["condition"] != "full"]
    drops = plot_df["f1_drop"].to_numpy()
    lower = plot_df["drop_ci_lower"].to_numpy()
    upper = plot_df["drop_ci_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(3)

    ax.bar(x, drops, color=["#9BA8C5", "#E8A0BF", "#9076B5"])
    ax.errorbar(
        x, drops,
        yerr=np.vstack([drops - lower, upper - drops]),
        fmt="none", ecolor="black", capsize=5,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, MODALITIES)
    ax.set_ylabel("Decrease in pooled macro-F1")
    ax.set_title(config["title"])
    ax.text(
        0.01, -0.20,
        "Error bars: paired subject-bootstrap 95% confidence intervals",
        transform=ax.transAxes, fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{task_key}_ablation_ci.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3/4: Variation between subjects.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [
        subject_df.loc[
            subject_df["modality"] == m, "f1_drop"
        ].to_numpy()
        for m in MODALITIES
    ]
    ax.boxplot(values, tick_labels=MODALITIES, showmeans=True)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Per-subject macro-F1 decrease")
    ax.set_title(f"{config['title']} — subject variability")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{task_key}_subject_variability.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\n{'=' * 65}")
    print(config["title"])
    print(f"{'=' * 65}")
    print(summary.round(4).to_string(index=False))
    print(f"Subjects: {len(subjects)}")
    print(f"Trials: {len(truth)}")

    return summary


def main():
    all_summaries = []

    for task_key, config in TASKS.items():
        all_summaries.append(analyse_task(task_key, config))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(
        RESULTS / "cv_ablation_combined_summary.csv",
        index=False,
    )

    print("\nAnalysis completed successfully.")
    print(f"Tables saved to: {RESULTS}")
    print(f"Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
