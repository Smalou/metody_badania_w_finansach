"""Cross-sectional regression: AI Premium ~ AI Exposure + controls.

Zmienna zalezna (Y, glowna): 6-miesieczny post-event abnormal return (vs SPY),
2022-11-30 -> 2023-05-31.
Zmienna glowna (X): ai_exposure_det_main (pre-event, z 10-K przed 2022-11-30).
Kontrole: Beta_Pre, Vol_Pre, Momentum_12M_Pre, log_Market_Cap.

Modele:
  M1: AR ~ AI_Exp
  M2: + Beta + Vol
  M3: + Momentum_12M
  M4: + log_MktCap

Inference: OLS standard SE, White (HC1) robust SE, bootstrap 95% CI (B=10k).

Output:
  data_cache/cross_sectional_regression.csv
  data_cache/cross_sectional_features.csv
  figures/cross_sectional_scatter.pdf
  figures/cross_sectional_coefs.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf

from common import CACHE, BENCHMARK, EVENT_DATE, load_prices, load_metadata, log_returns, daily_rf
from plot_utils import setup_style, savefig

PRICES_PARQUET = CACHE / "prices.parquet"
SCORES_CSV = CACHE / "ai_exposure_scores.csv"
OUT_REG = CACHE / "cross_sectional_regression.csv"
OUT_FEATURES = CACHE / "cross_sectional_features.csv"

PRE_START = pd.Timestamp("2021-01-01")
PRE_END = pd.Timestamp("2022-11-29")
POST_START = pd.Timestamp("2022-11-30")
POST_END = pd.Timestamp("2023-05-31")

ALPHA = 0.05
N_BOOT = 10_000
SEED = 42


# ============================================================================
# Konstrukcja zmiennych
# ============================================================================
def compute_post_event_abnormal_return(rets: pd.DataFrame, bench: pd.Series,
                                        tickers: list[str],
                                        start=POST_START, end=POST_END) -> pd.Series:
    """6-mies. kumulatywny log-zwrot spolki minus benchmark."""
    mask = (rets.index >= start) & (rets.index <= end)
    sub = rets.loc[mask, tickers]
    bench_sub = bench.loc[mask]
    cum_stock = sub.sum(axis=0, min_count=1)
    cum_bench = bench_sub.sum()
    return cum_stock - cum_bench


def compute_pre_event_features(rets: pd.DataFrame, bench: pd.Series, rf: pd.Series,
                                 tickers: list[str]) -> pd.DataFrame:
    rf = rf.reindex(rets.index).ffill()
    mask = (rets.index >= PRE_START) & (rets.index <= PRE_END)
    pre = rets.loc[mask].copy()
    bench_pre = bench.loc[mask]
    rf_pre = rf.loc[mask]
    rows = []
    for t in tickers:
        if t not in pre.columns:
            continue
        r = pre[t].dropna()
        if len(r) < 100:
            continue
        rf_a = rf_pre.reindex(r.index)
        bench_a = bench_pre.reindex(r.index)
        excess = r - rf_a
        bench_excess = bench_a - rf_a
        var_b = bench_excess.var(ddof=1)
        beta = float(np.cov(excess, bench_excess, ddof=1)[0, 1] / var_b) if var_b > 0 else np.nan
        ann_vol = float(r.std(ddof=1) * np.sqrt(252))
        # 12-mies momentum: od 2021-11-30 do 2022-11-29 (~252 sesje)
        mom_start = PRE_END - pd.Timedelta(days=365)
        mom_window = r.loc[(r.index >= mom_start) & (r.index <= PRE_END)]
        momentum_12m = float(mom_window.sum()) if len(mom_window) > 100 else np.nan
        rows.append({
            "ticker": t, "beta_pre": beta, "vol_pre": ann_vol, "momentum_12m_pre": momentum_12m
        })
    return pd.DataFrame(rows)


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    """Snapshot market cap (z daty pobrania) - cache aby uniknac powtornego API."""
    cache = CACHE / "market_caps.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for t in tickers:
        try:
            mc = yf.Ticker(t).fast_info.market_cap
            rows.append({"ticker": t, "market_cap": float(mc) if mc else np.nan})
        except Exception as e:
            print(f"  [{t}] market_cap fetch error: {e}")
            rows.append({"ticker": t, "market_cap": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ============================================================================
# Regresje
# ============================================================================
MODEL_SPECS = {
    "M1": ["ai_exposure_det_main"],
    "M2": ["ai_exposure_det_main", "beta_pre", "vol_pre"],
    "M3": ["ai_exposure_det_main", "beta_pre", "vol_pre", "momentum_12m_pre"],
    "M4": ["ai_exposure_det_main", "beta_pre", "vol_pre", "momentum_12m_pre", "log_market_cap"],
}


def fit_model(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    sub = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    model_std = sm.OLS(y, X).fit()
    model_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
    res = {"n": len(sub), "r2": model_std.rsquared, "r2_adj": model_std.rsquared_adj}
    for col in ["const"] + x_cols:
        coef = model_std.params[col]
        se_std = model_std.bse[col]
        se_hc1 = model_hc1.bse[col]
        t_std = coef / se_std if se_std > 0 else np.nan
        p_std = model_std.pvalues[col]
        p_hc1 = model_hc1.pvalues[col]
        res[f"{col}_coef"] = coef
        res[f"{col}_se_std"] = se_std
        res[f"{col}_se_hc1"] = se_hc1
        res[f"{col}_p_std"] = p_std
        res[f"{col}_p_hc1"] = p_hc1
        res[f"{col}_ci_std_low"] = coef - 1.96 * se_std
        res[f"{col}_ci_std_high"] = coef + 1.96 * se_std
        res[f"{col}_ci_hc1_low"] = coef - 1.96 * se_hc1
        res[f"{col}_ci_hc1_high"] = coef + 1.96 * se_hc1
    return res, model_std, sub


def bootstrap_beta1(df: pd.DataFrame, y_col: str, x_cols: list[str],
                     n_boot=N_BOOT, seed=SEED) -> tuple[float, float, np.ndarray]:
    sub = df[[y_col] + x_cols].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(sub)
    boot_b1 = np.empty(n_boot)
    X_full = sm.add_constant(sub[x_cols])
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = X_full.iloc[idx]
        yb = sub[y_col].iloc[idx]
        try:
            m = sm.OLS(yb, Xb).fit()
            boot_b1[i] = m.params[x_cols[0]]
        except Exception:
            boot_b1[i] = np.nan
    boot_b1 = boot_b1[~np.isnan(boot_b1)]
    lo, hi = np.percentile(boot_b1, [2.5, 97.5])
    return float(lo), float(hi), boot_b1


# ============================================================================
# Main
# ============================================================================
def main():
    prices = load_prices()
    meta = load_metadata()
    rets = log_returns(prices)
    rf = daily_rf(prices)
    bench = rets[BENCHMARK]
    tickers = meta["ticker"].tolist()

    print(f"Spolki: {len(tickers)}; pre-event: {PRE_START.date()} -> {PRE_END.date()}")
    print(f"Post-event main window: {POST_START.date()} -> {POST_END.date()}")

    print("\nObliczam Y: 6-miesieczny abnormal return vs SPY")
    y6m = compute_post_event_abnormal_return(rets, bench, tickers)
    y3m = compute_post_event_abnormal_return(rets, bench, tickers,
                                               start=POST_START,
                                               end=POST_START + pd.Timedelta(days=90))
    y12m = compute_post_event_abnormal_return(rets, bench, tickers,
                                                start=POST_START,
                                                end=POST_START + pd.Timedelta(days=365))

    print("Obliczam kontrole z okna pre-event (Beta, Vol, Momentum_12m)")
    features = compute_pre_event_features(rets, bench, rf, tickers)

    print("Pobieram market caps (cache)")
    mc = fetch_market_caps(tickers)
    features = features.merge(mc, on="ticker", how="left")
    features["log_market_cap"] = np.log(features["market_cap"].replace(0, np.nan))

    scores = pd.read_csv(SCORES_CSV)
    score_cols = [c for c in scores.columns
                  if c.startswith("ai_exposure_") or c.startswith("mentions_") or c == "group"]
    features = features.merge(
        scores[["ticker"] + [c for c in score_cols if c != "group"] + ["group"]],
        on="ticker", how="left",
    )

    features["ar_3m"] = features["ticker"].map(y3m.to_dict())
    features["ar_6m"] = features["ticker"].map(y6m.to_dict())
    features["ar_12m"] = features["ticker"].map(y12m.to_dict())
    # P4: wariant LLM z missing=0 (zamiast NaN) - kontrola samo-selekcji 7 spolek
    if "ai_exposure_llm" in features.columns:
        features["ai_exposure_llm_missing0"] = features["ai_exposure_llm"].fillna(0.0)
    features.to_csv(OUT_FEATURES, index=False)
    print(f"Zapisano features: {OUT_FEATURES}")

    # P11: macierz korelacji glownych zmiennych regresji
    corr_vars = ["ai_exposure_det_main", "beta_pre", "vol_pre",
                 "momentum_12m_pre", "log_market_cap", "ar_6m"]
    corr_vars = [c for c in corr_vars if c in features.columns]
    corr = features[corr_vars].corr(method="pearson")
    corr.to_csv(CACHE / "cross_sectional_corr_matrix.csv")
    print(f"\nMacierz korelacji Pearsona ({len(corr_vars)} zmiennych):")
    print(corr.round(3).to_string())

    # P12: VIF dla M4 (full specification)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        m4_vars = MODEL_SPECS["M4"]
        sub = features[m4_vars].dropna()
        X = sm.add_constant(sub)
        vif = pd.DataFrame({
            "variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        })
        vif.to_csv(CACHE / "cross_sectional_vif.csv", index=False)
        print(f"\nVIF dla M4 (n={len(sub)}):")
        print(vif.round(3).to_string(index=False))
    except Exception as e:
        print(f"VIF nie policzony: {e}")

    print("\nStatystyki opisowe Y i X:")
    print(features[["ai_exposure_det_main", "beta_pre", "vol_pre", "momentum_12m_pre",
                     "log_market_cap", "ar_3m", "ar_6m", "ar_12m"]].describe().round(3))

    print("\n========== REGRESJE M1-M4 (Y=ar_6m, X_main=ai_exposure_det_main) ==========\n")
    results = []
    for name, x_cols in MODEL_SPECS.items():
        res, model_std, sub = fit_model(features, "ar_6m", x_cols)
        b1_lo, b1_hi, _ = bootstrap_beta1(features, "ar_6m", x_cols)
        res["model"] = name
        res["x_cols"] = ",".join(x_cols)
        res["beta1_boot_ci_low"] = b1_lo
        res["beta1_boot_ci_high"] = b1_hi
        results.append(res)
        b1 = res["ai_exposure_det_main_coef"]
        se_h = res["ai_exposure_det_main_se_hc1"]
        p_h = res["ai_exposure_det_main_p_hc1"]
        ci_l_h = res["ai_exposure_det_main_ci_hc1_low"]
        ci_h_h = res["ai_exposure_det_main_ci_hc1_high"]
        print(f"[{name}] n={res['n']}  R^2={res['r2']:.3f}")
        print(f"   beta1 (AI_Exp) = {b1:+.4f}  HC1 SE={se_h:.4f}  p_HC1={p_h:.4f}  "
              f"95% CI HC1 [{ci_l_h:+.4f}; {ci_h_h:+.4f}]")
        print(f"   bootstrap 95% CI dla beta1: [{b1_lo:+.4f}; {b1_hi:+.4f}]")
        print()

    pd.DataFrame(results).to_csv(OUT_REG, index=False)
    print(f"Zapisano regresje: {OUT_REG}")

    # ============================================================================
    # Wykresy
    # ============================================================================
    setup_style()

    # Scatter: AI_Exp vs ar_6m + linia OLS z M1
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    palette = {"AI_infra": "#1f77b4", "AI_platforms": "#9467bd",
               "Broad_tech": "#2ca02c", "Defensive": "#d62728"}
    for g, sub in features.groupby("group"):
        ax.scatter(sub["ai_exposure_det_main"], sub["ar_6m"],
                   c=palette.get(g, "gray"), s=70, alpha=0.85, edgecolor="black",
                   linewidth=0.6, label=g, zorder=3)
    # adnotacje topowych
    top_ai = features.nlargest(5, "ai_exposure_det_main")
    top_ar = features.nlargest(5, "ar_6m")
    annotated = set(top_ai["ticker"]) | set(top_ar["ticker"])
    for _, r in features[features["ticker"].isin(annotated)].iterrows():
        ax.annotate(r["ticker"], (r["ai_exposure_det_main"], r["ar_6m"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    # OLS line z M1
    m1_sub = features[["ai_exposure_det_main", "ar_6m"]].dropna()
    X1 = sm.add_constant(m1_sub["ai_exposure_det_main"])
    m1 = sm.OLS(m1_sub["ar_6m"], X1).fit()
    xs = np.linspace(0, 1, 100)
    ys = m1.params["const"] + m1.params["ai_exposure_det_main"] * xs
    ax.plot(xs, ys, "--", color="black", linewidth=1.2,
            label=f"OLS M1 (beta1={m1.params['ai_exposure_det_main']:+.3f})")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Pre-event AI Exposure Score (percentile rank, deterministic main)")
    ax.set_ylabel("Post-event 6-mies. abnormal return vs SPY (log)")
    ax.set_title("Cross-sectional: AI Exposure vs 6-mies. abnormal return (2022-11-30 -> 2023-05-31)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    savefig(fig, "cross_sectional_scatter.pdf")

    # Coefficient plot: beta1 per model z HC1 CI
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))
    res_df = pd.DataFrame(results)
    ys = np.arange(len(res_df))
    b1 = res_df["ai_exposure_det_main_coef"]
    lo_h = res_df["ai_exposure_det_main_ci_hc1_low"]
    hi_h = res_df["ai_exposure_det_main_ci_hc1_high"]
    lo_b = res_df["beta1_boot_ci_low"]
    hi_b = res_df["beta1_boot_ci_high"]
    ax2.errorbar(b1, ys - 0.12, xerr=[b1 - lo_h, hi_h - b1], fmt="o",
                 color="#1f4e79", capsize=4, label="HC1 95% CI")
    ax2.errorbar(b1, ys + 0.12, xerr=[b1 - lo_b, hi_b - b1], fmt="s",
                 color="#d62728", capsize=4, label="Bootstrap 95% CI")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(ys); ax2.set_yticklabels(res_df["model"])
    ax2.invert_yaxis()
    ax2.set_xlabel("beta1 (AI Exposure) - wspolczynnik w cross-sectional regression")
    ax2.set_title("Coefficient plot: beta1 (AI Exposure) per model M1-M4")
    ax2.legend(loc="best")
    savefig(fig2, "cross_sectional_coefs.pdf")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
