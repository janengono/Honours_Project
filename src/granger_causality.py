"""
granger_causality.py
---------------------
Implements a genuine causal-DISCOVERY method (as distinct from modality
ablation's causal-NECESSITY test and SHAP's correlational ATTRIBUTION),
following the taxonomy in Sanchez et al. [17]: Granger causality tests
whether the past of one time series helps predict the future of another.

ADAPTATION NOTE (read before interpreting results): pain intensity is
constant within a single 5.5s trial, so we cannot Granger-test "does a
modality cause pain intensity" directly -- there is no time-varying label
to test against. Instead, we test DIRECTIONAL DEPENDENCE BETWEEN MODALITIES
across the 10 sub-windows within a trial (e.g. does EDA's trajectory
Granger-cause EMG's trajectory?), and compare whether that directional
structure differs between baseline and high-pain trials. This is a
legitimate and distinct question from ablation/SHAP: it asks about causal
structure IN THE PHYSIOLOGICAL DATA ITSELF, independent of what the trained
model happened to learn.

SECOND ADAPTATION: 10 time points per trial is short for a single reliable
Granger test (which wants a longer series to fit a stable VAR model). Rather
than concatenating trials end-to-end (which would introduce artificial
discontinuities at trial boundaries), each trial is tested individually at
a single lag (lag=1, the maximum defensible given only 10 points), and we
report the PROPORTION of trials per class showing significant directional
dependence (p<0.05) -- treating many short, individually underpowered tests
as a population-level vote, rather than trusting any single trial's result.

Usage:
    python granger_causality.py --limit 500   # quick test first
    python granger_causality.py               # full run (slower, see note below)
"""
import io, contextlib
import argparse
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from pain_dataset import load_sequence_data
from model import get_modality_indices

MODALITIES = ["ecg", "eda", "emg"]


def extract_modality_series(X_trial, modality_indices, feature_names):
    """
    X_trial: (seq_len=10, n_features=20) for one trial.
    Returns dict {'ecg': (10,), 'eda': (10,), 'emg': (10,)} -- one
    representative scalar series per modality per sub-window, built from
    the '_mean' statistic (EMG's three channels are averaged together into
    a single combined series, since Granger causality here concerns
    modality-level, not channel-level, dependence).
    """
    series = {}

    ecg_mean_idx = feature_names.index("ecg_mean")
    series["ecg"] = X_trial[:, ecg_mean_idx]

    eda_mean_idx = feature_names.index("gsr_mean")
    series["eda"] = X_trial[:, eda_mean_idx]

    emg_mean_indices = [
        feature_names.index(f"emg_{ch}_mean")
        for ch in ["trapezius", "corrugator", "zygomaticus"]
    ]
    series["emg"] = X_trial[:, emg_mean_indices].mean(axis=1)

    return series


def granger_test_pair(cause_series, effect_series, maxlag=1):
    """
    Tests whether cause_series Granger-causes effect_series at the given lag.
    statsmodels' grangercausalitytests expects a 2-column array where column
    0 is the "effect" series and column 1 is the candidate "cause" series.
    Returns the p-value (ssr_ftest) at the requested lag, or None if the
    test fails (e.g. a constant/degenerate series, which breaks the
    underlying VAR fit -- not uncommon with only 10 points).
    """
    data = np.column_stack([effect_series, cause_series])
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            # NOTE: statsmodels >=0.15 removed the `verbose` kwarg (it no longer
            # prints by default). If you're on an older statsmodels (<0.15) that
            # still prints per-test output by default, pass verbose=False here
            # instead -- check your installed version if this line errors.
            result = grangercausalitytests(data, maxlag=maxlag)
        p_value = result[maxlag][0]["ssr_ftest"][1]
        f_stat = result[maxlag][0]["ssr_ftest"][0]
        return p_value, f_stat
    except Exception:
        return None, None


def run_granger_analysis(limit=None, alpha=0.05):
    X, y, subject_ids, feature_names = load_sequence_data()
    if limit:
        X, y, subject_ids = X[:limit], y[:limit], subject_ids[:limit]

    directed_pairs = [(a, b) for a, b in itertools.permutations(MODALITIES, 2)]
    class_names = {0: "BL", 1: "PA1", 2: "PA2", 3: "PA3", 4: "PA4"}

    records = []
    n_failed = 0

    print(f"Running Granger causality on {len(X)} trials x {len(directed_pairs)} directed pairs...")
    print("(This can take a few minutes -- each trial/pair is a separate VAR fit)\n")

    for i in range(len(X)):
        series = extract_modality_series(X[i], None, feature_names)
        class_id = int(y[i])

        for cause, effect in directed_pairs:
            p_val, f_stat = granger_test_pair(series[cause], series[effect], maxlag=1)
            if p_val is None:
                n_failed += 1
                continue
            records.append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "cause": cause,
                "effect": effect,
                "pair": f"{cause}->{effect}",
                "p_value": p_val,
                "f_stat": f_stat,
                "significant": p_val < alpha,
            })

    results_df = pd.DataFrame(records)
    print(f"Completed {len(results_df)} tests ({n_failed} trials/pairs failed to fit, skipped)\n")

    # --- Summary: proportion of trials significant per class per directed pair ---
    summary = (
        results_df.groupby(["class_name", "pair"])
        .agg(n_trials=("significant", "count"),
             pct_significant=("significant", "mean"),
             mean_f_stat=("f_stat", "mean"))
        .reset_index()
    )
    summary["pct_significant"] = (summary["pct_significant"] * 100).round(1)
    summary["mean_f_stat"] = summary["mean_f_stat"].round(3)

    print("=" * 70)
    print("GRANGER CAUSALITY: % of trials with significant directional dependence (p<0.05)")
    print("=" * 70)
    pivot = summary.pivot(index="pair", columns="class_name", values="pct_significant")
    class_order = [c for c in ["BL", "PA1", "PA2", "PA3", "PA4"] if c in pivot.columns]
    print(pivot[class_order].to_string())

    print("\n" + "=" * 70)
    print("BL vs PA4 comparison (the clearest contrast, matching earlier analyses)")
    print("=" * 70)
    if "BL" in pivot.columns and "PA4" in pivot.columns:
        comparison = pivot[["BL", "PA4"]].copy()
        comparison["change"] = comparison["PA4"] - comparison["BL"]
        comparison = comparison.sort_values("change", ascending=False)
        print(comparison.to_string())

    return results_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials, for a quick test run")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_granger_analysis(limit=args.limit, alpha=args.alpha)
