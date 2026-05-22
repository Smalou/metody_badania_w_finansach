"""Robustness checks dla cross-sectional regresji.

(1) Leave-one-out per spolka (M3 = baseline) - rozklad beta1.
(2) Excluding extreme firms (NVDA, top1/top3 AI, top3 AR).
(3) Alternative AI scores (det_main, det_main_gpu, det_extended).
(4) Alternative windows Y (3 / 6 / 12 miesiecy).
(5) Alternative scaling X (percentile / z-score / raw / log).

Output:
  data_cache/robustness_loo.csv
  data_cache/robustness_exclude.csv
  data_cache/robustness_alt_scores.csv
  data_cache/robustness_alt_windows.csv
  data_cache/robustness_alt_scaling.csv
  figures/leave_one_out_beta1.pdf
  figures/robustness_summary.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from common import CACHE
from plot_utils import setup_style, savefig
from cross_sectional import MODEL_SPECS, fit_model

FEATURES = CACHE / "cross_sectional_features.csv"
OUT_LOO = CACHE / "robustness_loo.csv"
OUT_EXCL = CACHE / "robustness_exclude.csv"
OUT_ALT_SCORES = CACHE / "robustness_alt_scores.csv"
OUT_ALT_WINDOWS = CACHE / "robustness_alt_windows.csv"
OUT_ALT_SCALING = CACHE / "robustness_alt_scaling.csv"


def fit_simple(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    sub = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    if len(sub) < len(x_cols) + 3:
        return {"n": len(sub), "beta1": np.nan, "se_hc1": np.nan, "p_hc1": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "r2": np.nan}
    m = sm.OLS(y, X).fit(cov_type="HC1")
    b1 = m.params[x_cols[0]]
    se = m.bse[x_cols[0]]
    return {
        "n": len(sub), "beta1": float(b1), "se_hc1": float(se),
        "p_hc1": float(m.pvalues[x_cols[0]]),
        "ci_low": float(b1 - 1.96 * se), "ci_high": float(b1 + 1.96 * se),
        "r2": float(m.rsquared),
    }


def leave_one_out(features: pd.DataFrame, y_col="ar_6m", baseline_model="M3") -> pd.DataFrame:
    x_cols = MODEL_SPECS[baseline_model]
    rows = []
    for excluded in features["ticker"]:
        sub = features[features["ticker"] != excluded]
        res = fit_simple(sub, y_col, x_cols)
        res["excluded_ticker"] = excluded
        res["excluded_group"] = features.loc[features["ticker"] == excluded, "group"].iloc[0]
        rows.append(res)
    return pd.DataFrame(rows)


def excluding_firms(features: pd.DataFrame, y_col="ar_6m") -> pd.DataFrame:
    """Excluding NVDA, top1 AI, top3 AI, top3 AR (dla glownego M1 i M3)."""
    rows = []
    top1_ai = features.nlargest(1, "ai_exposure_det_main")["ticker"].tolist()
    top3_ai = features.nlargest(3, "ai_exposure_det_main")["ticker"].tolist()
    top3_ar = features.nlargest(3, y_col)["ticker"].tolist()
    bottom3_ar = features.nsmallest(3, y_col)["ticker"].tolist()

    variants = {
        "full": [],
        "no_nvda": ["NVDA"],
        "no_top1_ai": top1_ai,
        "no_top3_ai": top3_ai,
        "no_top3_ar": top3_ar,
        "no_bottom3_ar": bottom3_ar,
    }
    for name, excl in variants.items():
        sub = features[~features["ticker"].isin(excl)]
        for model in ("M1", "M3"):
            x_cols = MODEL_SPECS[model]
            res = fit_simple(sub, y_col, x_cols)
            res["variant"] = name
            res["excluded"] = ",".join(excl) if excl else ""
            res["model"] = model
            rows.append(res)
    return pd.DataFrame(rows)


def alternative_scores(features: pd.DataFrame, y_col="ar_6m") -> pd.DataFrame:
    scores = [
        "ai_exposure_det_main",
        "ai_exposure_det_main_gpu",
        "ai_exposure_det_main_no_automation",  # P6: bez "automation"
        "ai_exposure_det_extended",
        "ai_exposure_llm",
        "ai_exposure_llm_missing0",            # P4: NaN -> 0 (test samo-selekcji)
    ]
    rows = []
    for s in scores:
        if s not in features.columns:
            continue
        for model in ("M1", "M3"):
            x_cols = [s] + MODEL_SPECS[model][1:]
            res = fit_simple(features, y_col, x_cols)
            res["x_main"] = s
            res["model"] = model
            rows.append(res)
    return pd.DataFrame(rows)


def alternative_windows(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y_col in ("ar_3m", "ar_6m", "ar_12m"):
        for model in ("M1", "M3"):
            x_cols = MODEL_SPECS[model]
            res = fit_simple(features, y_col, x_cols)
            res["y_window"] = y_col
            res["model"] = model
            rows.append(res)
    return pd.DataFrame(rows)


def alternative_scaling(features: pd.DataFrame, y_col="ar_6m") -> pd.DataFrame:
    """percentile (main) vs z-score vs raw vs log."""
    rows = []
    f = features.copy()
    raw = f["mentions_per_10k_main"]
    f["ai_exp_raw"] = raw
    f["ai_exp_log"] = np.log1p(raw)
    f["ai_exp_zscore"] = (raw - raw.mean()) / raw.std(ddof=1)
    f["ai_exp_pct"] = f["ai_exposure_det_main"]
    for name in ("ai_exp_pct", "ai_exp_zscore", "ai_exp_log", "ai_exp_raw"):
        for model in ("M1", "M3"):
            x_cols = [name] + MODEL_SPECS[model][1:]
            res = fit_simple(f, y_col, x_cols)
            res["scaling"] = name
            res["model"] = model
            rows.append(res)
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================
def plot_loo(loo_df: pd.DataFrame, baseline_beta1: float | None = None):
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    sorted_loo = loo_df.sort_values("beta1").reset_index(drop=True)
    colors = {"AI_infra": "#1f77b4", "AI_platforms": "#9467bd",
              "Broad_tech": "#2ca02c", "Defensive": "#d62728"}
    bar_colors = [colors.get(g, "gray") for g in sorted_loo["excluded_group"]]
    xpos = np.arange(len(sorted_loo))
    ax.bar(xpos, sorted_loo["beta1"], color=bar_colors,
           edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.errorbar(xpos, sorted_loo["beta1"],
                yerr=[sorted_loo["beta1"] - sorted_loo["ci_low"],
                      sorted_loo["ci_high"] - sorted_loo["beta1"]],
                fmt="none", ecolor="black", elinewidth=0.5, capsize=2, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    if baseline_beta1 is not None:
        ax.axhline(baseline_beta1, color="red", linestyle="--", linewidth=1.0,
                   label=f"baseline M3 (full sample) beta1 = {baseline_beta1:+.4f}")
    ax.set_xticks(xpos)
    ax.set_xticklabels(sorted_loo["excluded_ticker"], rotation=90, fontsize=8)
    ax.set_xlabel("Wyeliminowana spolka")
    ax.set_ylabel("beta1 (AI Exposure) z M3 z 95% CI HC1")
    ax.set_title("Leave-one-out: rozklad beta1 z modelu M3 (41 regresji)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[g]) for g in colors]
    ax.legend(handles + [ax.get_lines()[-1]] if baseline_beta1 is not None else handles,
              list(colors.keys()) + [f"baseline beta1={baseline_beta1:+.4f}"]
              if baseline_beta1 is not None else list(colors.keys()),
              loc="upper left", fontsize=8, framealpha=0.95)
    savefig(fig, "leave_one_out_beta1.pdf")


def plot_robustness_summary(loo_df, excl_df, alt_scores, alt_windows, alt_scaling):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # excluding firms
    ax = axes[0, 0]
    sub = excl_df[excl_df["model"] == "M3"]
    ys = np.arange(len(sub))
    ax.errorbar(sub["beta1"], ys,
                xerr=[sub["beta1"] - sub["ci_low"], sub["ci_high"] - sub["beta1"]],
                fmt="o", color="#1f4e79", capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(sub["variant"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(a) Excluding extreme firms - beta1 z M3 (95% CI HC1)")
    ax.set_xlabel("beta1 (AI Exposure)")

    # alternative scores
    ax = axes[0, 1]
    sub = alt_scores[alt_scores["model"] == "M3"]
    ys = np.arange(len(sub))
    ax.errorbar(sub["beta1"], ys,
                xerr=[sub["beta1"] - sub["ci_low"], sub["ci_high"] - sub["beta1"]],
                fmt="o", color="#9467bd", capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(sub["x_main"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(b) Alternative AI scores - beta1 z M3")
    ax.set_xlabel("beta1")

    # alternative windows
    ax = axes[1, 0]
    sub = alt_windows[alt_windows["model"] == "M3"]
    ys = np.arange(len(sub))
    ax.errorbar(sub["beta1"], ys,
                xerr=[sub["beta1"] - sub["ci_low"], sub["ci_high"] - sub["beta1"]],
                fmt="o", color="#2ca02c", capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(sub["y_window"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(c) Alternative Y windows - beta1 z M3")
    ax.set_xlabel("beta1")

    # alternative scaling
    ax = axes[1, 1]
    sub = alt_scaling[alt_scaling["model"] == "M3"]
    ys = np.arange(len(sub))
    ax.errorbar(sub["beta1"], ys,
                xerr=[sub["beta1"] - sub["ci_low"], sub["ci_high"] - sub["beta1"]],
                fmt="o", color="#d62728", capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels(sub["scaling"], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("(d) Alternative X scaling - beta1 z M3")
    ax.set_xlabel("beta1")

    fig.tight_layout()
    savefig(fig, "robustness_summary.pdf")


def main():
    features = pd.read_csv(FEATURES)
    print(f"Wczytano {len(features)} spolek")

    # baseline M3 z pelnej proby
    baseline = fit_simple(features, "ar_6m", MODEL_SPECS["M3"])
    print(f"\nBaseline M3 (full): beta1={baseline['beta1']:+.4f}  p_HC1={baseline['p_hc1']:.4f}")

    print("\n--- (1) Leave-one-out (M3) ---")
    loo = leave_one_out(features)
    loo.to_csv(OUT_LOO, index=False)
    sign_changes = (np.sign(loo["beta1"]) != np.sign(baseline["beta1"])).sum()
    print(f"  Sign changes wzgledem baseline: {sign_changes}/{len(loo)}")
    print(f"  beta1 LOO: min={loo['beta1'].min():+.4f}, max={loo['beta1'].max():+.4f}, median={loo['beta1'].median():+.4f}")
    most_influential = loo.iloc[(loo["beta1"] - baseline["beta1"]).abs().idxmax()]
    print(f"  Najbardziej wplywowa: {most_influential['excluded_ticker']} ({most_influential['excluded_group']}), "
          f"beta1 po wykluczeniu = {most_influential['beta1']:+.4f}")

    print("\n--- (2) Excluding extreme firms ---")
    excl = excluding_firms(features)
    excl.to_csv(OUT_EXCL, index=False)
    print(excl[excl["model"] == "M3"][["variant", "n", "beta1", "p_hc1", "ci_low", "ci_high"]].round(4).to_string(index=False))

    print("\n--- (3) Alternative AI scores ---")
    alt_s = alternative_scores(features)
    alt_s.to_csv(OUT_ALT_SCORES, index=False)
    print(alt_s[alt_s["model"] == "M3"][["x_main", "beta1", "p_hc1", "ci_low", "ci_high"]].round(4).to_string(index=False))

    print("\n--- (4) Alternative Y windows ---")
    alt_w = alternative_windows(features)
    alt_w.to_csv(OUT_ALT_WINDOWS, index=False)
    print(alt_w[alt_w["model"] == "M3"][["y_window", "beta1", "p_hc1", "ci_low", "ci_high"]].round(4).to_string(index=False))

    print("\n--- (5) Alternative X scaling ---")
    alt_sc = alternative_scaling(features)
    alt_sc.to_csv(OUT_ALT_SCALING, index=False)
    print(alt_sc[alt_sc["model"] == "M3"][["scaling", "beta1", "p_hc1", "ci_low", "ci_high"]].round(4).to_string(index=False))

    print("\n--- Wykresy ---")
    plot_loo(loo, baseline["beta1"])
    plot_robustness_summary(loo, excl, alt_s, alt_w, alt_sc)
    print("\nGotowe.")


if __name__ == "__main__":
    main()
