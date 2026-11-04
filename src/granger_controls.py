"""
granger_controls.py
-------------------
Completes the Granger causality analysis by addressing the two limitations
stated in the methodology:

  (1) NO SURROGATE CONTROL. The baseline rate for EMG->EDA was already ~22%,
      far above the nominal 5%. A surrogate null estimates what rate to expect
      when the cause series of trial i is paired with the effect series of a
      DIFFERENT trial from the SAME subject and SAME class. This keeps each
      series' own autocorrelation and any stimulus-locked shared time course,
      but breaks trial-specific coupling. So:
          real rate       = trial-specific coupling + shared evoked time course
          surrogate rate  = shared evoked time course (and chance)
          excess          = real - surrogate  -> trial-specific coupling

  (2) TRIALS ARE NESTED IN SUBJECTS. Pooling trials treats 8,600 trials as
      independent. Here the per-class rate is computed per SUBJECT first, and
      classes are compared across subjects (paired Wilcoxon BL vs PA4,
      Friedman across the five classes), with Holm correction over the six pairs.

The lag-1 Granger F-test is implemented directly in numpy (restricted vs
unrestricted OLS) instead of calling statsmodels thousands of times. It is
equivalent to statsmodels' 'ssr_ftest' at lag 1; run `python granger_controls.py
--selftest` to verify that against statsmodels on your machine.

Usage:
    python granger_controls.py --selftest
    python granger_controls.py                      # full analysis
    python granger_controls.py --n-surrogates 5     # steadier null estimate
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
from scipy import stats

MODALITIES = ["ecg", "eda", "emg"]
CLASS_NAMES = {0: "BL", 1: "PA1", 2: "PA2", 3: "PA3", 4: "PA4"}


# ----------------------------------------------------------------------
# Data loading (numpy only -- no torch needed)
# ----------------------------------------------------------------------
def load_sequences():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "..", "data", "processed", "sequences.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run build_sequences.py first.")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["subject_ids"], list(d["feature_names"])


def build_modality_series(X, feature_names):
    """Returns array (n_trials, 3, 10): window-mean series for ecg, eda, emg
    (emg = average of the three channel window means), same as granger_causality.py."""
    ecg = X[:, :, feature_names.index("ecg_mean")]
    eda = X[:, :, feature_names.index("gsr_mean")]
    emg_idx = [feature_names.index(f"emg_{c}_mean")
               for c in ("trapezius", "corrugator", "zygomaticus")]
    emg = X[:, :, emg_idx].mean(axis=2)
    return np.stack([ecg, eda, emg], axis=1).astype(np.float64)


# ----------------------------------------------------------------------
# Fast lag-1 Granger F-test
# ----------------------------------------------------------------------
def granger_p_lag1(cause, effect):
    """p-value for 'cause Granger-causes effect' at lag 1 (nan if degenerate).
    Restricted:   x_t = a0 + a1 x_{t-1}
    Unrestricted: x_t = a0 + a1 x_{t-1} + b1 y_{t-1}      F-test of b1 = 0."""
    x, y = effect, cause
    target = x[1:]
    ones = np.ones_like(target)
    Xr = np.column_stack([ones, x[:-1]])
    Xu = np.column_stack([ones, x[:-1], y[:-1]])
    br, *_ = np.linalg.lstsq(Xr, target, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, target, rcond=None)
    rss_r = float(np.sum((target - Xr @ br) ** 2))
    rss_u = float(np.sum((target - Xu @ bu) ** 2))
    df_den = len(target) - 3
    if df_den <= 0 or rss_u <= 1e-14 * max(1.0, float(np.sum(target ** 2))):
        return np.nan
    f_stat = (rss_r - rss_u) / rss_u * df_den
    if not np.isfinite(f_stat) or f_stat < 0:
        return np.nan
    return float(stats.f.sf(f_stat, 1, df_den))


DIRECTED_PAIRS = list(itertools.permutations(range(3), 2))   # (cause_idx, effect_idx)
PAIR_LABELS = [f"{MODALITIES[c]}->{MODALITIES[e]}" for c, e in DIRECTED_PAIRS]


def pvalues_real(series):
    n = series.shape[0]
    out = np.full((n, len(DIRECTED_PAIRS)), np.nan)
    for i in range(n):
        for k, (c, e) in enumerate(DIRECTED_PAIRS):
            out[i, k] = granger_p_lag1(series[i, c], series[i, e])
    return out


def pvalues_surrogate(series, y, sids, rng):
    """Pair trial i's cause series with another trial's effect series, drawn from
    the same subject AND the same class (cyclic shift by a random non-zero offset,
    so no trial is ever paired with itself)."""
    n = series.shape[0]
    out = np.full((n, len(DIRECTED_PAIRS)), np.nan)
    groups = {}
    for i in range(n):
        groups.setdefault((sids[i], y[i]), []).append(i)
    for idx in groups.values():
        idx = np.array(idx)
        m = len(idx)
        if m < 2:
            continue
        for k, (c, e) in enumerate(DIRECTED_PAIRS):
            shift = int(rng.integers(1, m))
            partner = np.roll(idx, -shift)
            for a, b in zip(idx, partner):
                out[a, k] = granger_p_lag1(series[a, c], series[b, e])
    return out


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def per_subject_rates(p, y, sids, alpha):
    """DataFrame indexed by (subject, class) with % significant per pair (nan-aware)."""
    sig = np.where(np.isnan(p), np.nan, (p < alpha).astype(float))
    df = pd.DataFrame(sig, columns=PAIR_LABELS)
    df["subject"], df["class"] = sids, y
    return df.groupby(["subject", "class"]).mean() * 100.0


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wide(rates_df, pair, classes=(0, 1, 2, 3, 4)):
    """subjects x classes matrix for one pair, dropping subjects missing any class."""
    w = rates_df[pair].unstack("class")[list(classes)]
    return w.dropna()


def safe_wilcoxon(a, b):
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:        # all differences zero
        return 1.0


def main(alpha=0.05, n_surrogates=3, seed=0):
    X, y, sids, feature_names = load_sequences()
    series = build_modality_series(X, feature_names)
    n = len(X)
    print(f"{n} trials, {len(np.unique(sids))} subjects, {len(DIRECTED_PAIRS)} directed pairs\n")

    print("Computing real p-values ...")
    p_real = pvalues_real(series)
    print(f"  {int(np.isnan(p_real).sum())} of {p_real.size} tests degenerate (skipped)")

    rng = np.random.default_rng(seed)
    surr_rates = []
    for r in range(n_surrogates):
        print(f"Computing surrogate null {r + 1}/{n_surrogates} ...")
        p_s = pvalues_surrogate(series, y, sids, rng)
        surr_rates.append(per_subject_rates(p_s, y, sids, alpha))
    real_rates = per_subject_rates(p_real, y, sids, alpha)
    surr_mean = sum(surr_rates) / len(surr_rates)
    excess = real_rates - surr_mean

    # ---- Table A: pooled per class, real vs surrogate -------------------
    def pooled(rates):          # mean across subjects, per class
        return rates.groupby("class").mean()

    pr, ps = pooled(real_rates), pooled(surr_mean)
    print("\n" + "=" * 78)
    print(f"A. Percent of trials significant (p<{alpha:.2f}): REAL vs SURROGATE NULL (mean over subjects)")
    print("=" * 78)
    rows = []
    for pair in PAIR_LABELS:
        rows.append([pair, "real"] + [f"{pr.loc[c, pair]:5.1f}" for c in range(5)])
        rows.append(["", "surrogate"] + [f"{ps.loc[c, pair]:5.1f}" for c in range(5)])
        rows.append(["", "excess"] + [f"{pr.loc[c, pair] - ps.loc[c, pair]:+5.1f}" for c in range(5)])
    print(pd.DataFrame(rows, columns=["pair", "", "BL", "PA1", "PA2", "PA3", "PA4"]).to_string(index=False))

    # ---- Table B/C: subject-level tests ---------------------------------
    def subject_tests(rates, title):
        res = []
        for pair in PAIR_LABELS:
            w = wide(rates, pair)
            if len(w) < 5:
                continue
            p_w = safe_wilcoxon(w[4].values, w[0].values)
            try:
                p_f = float(stats.friedmanchisquare(*[w[c].values for c in range(5)]).pvalue)
            except ValueError:
                p_f = 1.0
            res.append({"pair": pair, "n_subj": len(w), "BL": w[0].mean(), "PA4": w[4].mean(),
                        "PA4-BL": w[4].mean() - w[0].mean(), "p_wilcoxon": p_w, "p_friedman": p_f})
        out = pd.DataFrame(res)
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].values)
        out["p_friedman_holm"] = holm(out["p_friedman"].values)
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        show = out.copy()
        for col in ["BL", "PA4", "PA4-BL"]:
            show[col] = show[col].map(lambda v: f"{v:+.1f}" if col == "PA4-BL" else f"{v:.1f}")
        for col in ["p_wilcoxon", "p_friedman", "p_wilcoxon_holm", "p_friedman_holm"]:
            show[col] = show[col].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
        print(show.to_string(index=False))
        return out

    subject_tests(real_rates,
                  "B. SUBJECT-LEVEL: real rate, BL vs PA4 (paired Wilcoxon) and across 5 classes (Friedman)")
    subject_tests(excess,
                  "C. SUBJECT-LEVEL: EXCESS over surrogate (real - surrogate), same tests")

    print("\nHow to read this:")
    print(" - A: if 'surrogate' is close to 'real', the dependence comes from the shared")
    print("   stimulus-locked time course, not trial-specific coupling. 'excess' is the")
    print("   trial-specific part.")
    print(" - B/C: units are SUBJECTS (not trials), so p-values respect the nesting.")
    print("   Holm columns correct for testing six pairs. Use those when claiming a change.")


def selftest():
    """Check granger_p_lag1 against statsmodels' ssr_ftest (lag 1)."""
    from statsmodels.tsa.stattools import grangercausalitytests
    import io, contextlib, warnings
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(200):
        n = 10
        a, b = rng.normal(size=n), rng.normal(size=n)
        if trial % 2 == 0:                      # plant a real dependence half the time
            b[1:] += 0.8 * a[:-1]
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(np.column_stack([b, a]), maxlag=1)
        p_sm = res[1][0]["ssr_ftest"][1]
        p_me = granger_p_lag1(a, b)
        worst = max(worst, abs(p_sm - p_me))
    print(f"Max |p_statsmodels - p_numpy| over 200 random tests: {worst:.2e}")
    print("PASS" if worst < 1e-8 else "MISMATCH -- do not use the numbers until this is resolved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-surrogates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        main(alpha=args.alpha, n_surrogates=args.n_surrogates, seed=args.seed)
