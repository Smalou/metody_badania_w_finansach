"""Zadanie 3: A/B testing na probach zaleznych - 21 spolek AI (infra + platforms)
wokol zdarzenia rozpoczecia boomu generatywnej AI (2022-11-30, premiera ChatGPT).

Wariant A: sredni dzienny spread vs SPY w oknie PRZED zdarzeniem.
Wariant B: sredni dzienny spread vs SPY w oknie PO zdarzeniu.
Jednostka pary: ta sama spolka.

Test glowny: paired t-test (jezeli roznice normalne).
Test odpornosci: Wilcoxon signed-rank.
Wielkosc efektu: Cohen's dz.
Analiza wrazliwosci: okna 30, 60, 120, 250 sesji.
Placebo: ta sama procedura dla daty 2021-11-30 (przed boomem AI).
Kontrola odpornosci: beta-adjusted excess return under CAPM assumption,
bez estymowanego interceptu alpha.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as st

from common import (
    CACHE, BENCHMARK, EVENT_DATE, PLACEBO_DATE,
    load_prices, load_metadata, log_returns, daily_rf, group_tickers, abnormal_returns,
)
from plot_utils import setup_style, savefig

ALPHA = 0.05
WINDOWS = [30, 60, 120, 250]
PRIMARY_WINDOW = 120


def select_windows(returns_idx: pd.DatetimeIndex, event: pd.Timestamp, k: int):
    days = returns_idx[returns_idx != event]
    before = days[days < event][-k:]
    after  = days[days > event][:k]
    return before, after


def per_stock_paired(ar_df: pd.DataFrame, event: pd.Timestamp, k: int) -> pd.DataFrame:
    before, after = select_windows(ar_df.index, event, k)
    rows = []
    for t in ar_df.columns:
        mb = ar_df.loc[before, t].dropna().mean()
        ma = ar_df.loc[after,  t].dropna().mean()
        rows.append({"ticker": t, "mean_A_before": mb, "mean_B_after": ma, "diff_B_minus_A": ma - mb})
    df = pd.DataFrame(rows)
    df["window_sessions"] = k
    df.attrs["window_before"] = (before[0], before[-1]) if len(before) else None
    df.attrs["window_after"] = (after[0], after[-1]) if len(after) else None
    return df


def paired_summary(diffs: np.ndarray, alpha=ALPHA) -> dict:
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    se_d = sd_d / np.sqrt(n)
    W_sh, p_sh = st.shapiro(diffs)
    t_crit = st.t.ppf(1 - alpha / 2, n - 1)
    ci_low, ci_high = mean_d - t_crit * se_d, mean_d + t_crit * se_d
    cohen_dz = mean_d / sd_d if sd_d > 0 else np.nan
    t_stat, t_p = st.ttest_1samp(diffs, 0.0)
    w_stat, w_p = st.wilcoxon(diffs, alternative="two-sided")
    use_t = p_sh > alpha
    main_test = "Paired t-test" if use_t else "Wilcoxon signed-rank"
    main_stat = float(t_stat) if use_t else float(w_stat)
    main_p = float(t_p) if use_t else float(w_p)
    return {
        "n": n, "mean_diff": mean_d, "sd_diff": sd_d, "se_diff": se_d,
        "ci_low": ci_low, "ci_high": ci_high, "cohen_dz": cohen_dz,
        "shapiro_W": float(W_sh), "shapiro_p": float(p_sh), "normal_diffs": bool(use_t),
        "t_stat": float(t_stat), "t_p_value": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p_value": float(w_p),
        "main_test": main_test, "main_stat": main_stat, "main_p_value": main_p,
    }


def run_sensitivity(ar_df, event, windows, label):
    rows = []
    pairs_by_window = {}
    for k in windows:
        pair = per_stock_paired(ar_df, event, k)
        pairs_by_window[k] = pair
        s = paired_summary(pair["diff_B_minus_A"].values)
        s["window_sessions"] = k
        s["event_date"] = event.date()
        s["analysis"] = label
        rows.append(s)
        print(f"  [{label} window={k:>3d}] n={s['n']:>2d}  mean_d={s['mean_diff']:+.5f}  "
              f"95% CI=[{s['ci_low']:+.5f}; {s['ci_high']:+.5f}]  "
              f"d_z={s['cohen_dz']:+.3f}  p={s['main_p_value']:.4f}  ({s['main_test']})")
    return pd.DataFrame(rows), pairs_by_window


def beta_pre_event(asset_ret: pd.Series, bench_excess: pd.Series, rf: pd.Series,
                   event: pd.Timestamp) -> float:
    """Beta estymowana tylko na danych sprzed analizowanej daty."""
    aligned = pd.concat([asset_ret - rf, bench_excess], axis=1, join="inner").dropna()
    pre = aligned[aligned.index < event]
    return float(np.cov(pre.iloc[:, 0], pre.iloc[:, 1], ddof=1)[0, 1] / np.var(pre.iloc[:, 1], ddof=1))


def capm_adjusted_returns(rets: pd.DataFrame, rf: pd.Series, bench_excess: pd.Series,
                          tickers: list[str], event: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, float]]:
    """Beta-adjusted excess returns: (r_i-r_f) - beta_pre*(r_SPY-r_f).

    To nie jest pelny market model z estymowanym interceptem alpha.
    """
    cols = {}
    betas = {}
    for t in tickers:
        beta = beta_pre_event(rets[t], bench_excess, rf, event)
        betas[t] = beta
        aligned = pd.concat([rets[t] - rf, bench_excess], axis=1, join="inner").dropna()
        cols[t] = aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]
    return pd.DataFrame(cols).dropna(how="all"), betas


def capm_primary_summary(rets: pd.DataFrame, rf: pd.Series, bench_excess: pd.Series,
                         tickers: list[str], event: pd.Timestamp, label: str):
    capm_df, betas = capm_adjusted_returns(rets, rf, bench_excess, tickers, event)
    pair = per_stock_paired(capm_df, event, PRIMARY_WINDOW)
    summary = paired_summary(pair["diff_B_minus_A"].dropna().values)
    summary["analysis"] = label
    summary["event_date"] = event.date()
    summary["window_sessions"] = PRIMARY_WINDOW
    summary["beta_mean_pre"] = float(np.mean(list(betas.values())))
    summary["beta_min_pre"] = float(np.min(list(betas.values())))
    summary["beta_max_pre"] = float(np.max(list(betas.values())))
    return summary, pair


def main():
    prices = load_prices()
    meta = load_metadata()
    rets = log_returns(prices)
    rf = daily_rf(prices).reindex(rets.index).ffill()
    bench = rets[BENCHMARK]
    bench_excess = bench - rf
    ai_tickers = group_tickers(meta, "AI_infra") + group_tickers(meta, "AI_platforms")
    print(f"Liczba spolek w grupie AI (infra + platforms): {len(ai_tickers)}")

    spread = pd.DataFrame({t: abnormal_returns(rets[t], bench) for t in ai_tickers}).dropna(how="all")
    print(f"Zakres dat spreadow vs SPY: {spread.index.min().date()} -> {spread.index.max().date()}, n={len(spread)} sesji")

    print(f"\n========== EVENT: rozpoczecie boomu AI ({EVENT_DATE.date()}) ==========")
    event_summary, event_pairs = run_sensitivity(spread, EVENT_DATE, WINDOWS, label="event_spread")

    print(f"\n========== PLACEBO: data o rok wczesniej ({PLACEBO_DATE.date()}) ==========")
    placebo_summary, placebo_pairs = run_sensitivity(spread, PLACEBO_DATE, WINDOWS, label="placebo_spread")

    print(f"\n========== CAPM-ADJUSTED CHECK (okno {PRIMARY_WINDOW}) ==========")
    capm_event, capm_event_pair = capm_primary_summary(
        rets, rf, bench_excess, ai_tickers, EVENT_DATE, "event_capm")
    capm_placebo, capm_placebo_pair = capm_primary_summary(
        rets, rf, bench_excess, ai_tickers, PLACEBO_DATE, "placebo_capm")
    for s in (capm_event, capm_placebo):
        print(f"  [{s['analysis']}] beta_pre_mean={s['beta_mean_pre']:.3f}  "
              f"mean_d={s['mean_diff']:+.5f}  95% CI=[{s['ci_low']:+.5f}; {s['ci_high']:+.5f}]  "
              f"d_z={s['cohen_dz']:+.3f}  p={s['main_p_value']:.4f} ({s['main_test']})")

    full = pd.concat([event_summary, placebo_summary], ignore_index=True)
    full.to_csv(CACHE / "task3_sensitivity.csv", index=False)
    pd.DataFrame([capm_event, capm_placebo]).to_csv(CACHE / "task3_capm_robustness.csv", index=False)
    event_pairs[PRIMARY_WINDOW].to_csv(CACHE / f"task3_pairs_w{PRIMARY_WINDOW}.csv", index=False)
    capm_event_pair.to_csv(CACHE / f"task3_capm_event_pairs_w{PRIMARY_WINDOW}.csv", index=False)
    capm_placebo_pair.to_csv(CACHE / f"task3_capm_placebo_pairs_w{PRIMARY_WINDOW}.csv", index=False)

    primary = event_summary[event_summary["window_sessions"] == PRIMARY_WINDOW].iloc[0]
    print(f"\n[GLOWNY WYNIK - okno {PRIMARY_WINDOW} sesji wokol 2022-11-30]")
    for k in ("n", "mean_diff", "ci_low", "ci_high", "cohen_dz",
              "shapiro_p", "t_p_value", "wilcoxon_p_value", "main_test", "main_p_value"):
        print(f"  {k}: {primary[k]}")
    decyzja = "ODRZUCONA" if primary['main_p_value'] < ALPHA else "NIE odrzucona"
    print(f"  H0 (brak roznicy spreadow vs SPY przed/po) {decyzja} przy alfa={ALPHA}")

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    pair_main = event_pairs[PRIMARY_WINDOW]
    diffs = pair_main["diff_B_minus_A"].values
    axes[0].hist(diffs, bins=10, color="#1f77b4", alpha=0.75, edgecolor="black")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].axvline(diffs.mean(), color="navy", linewidth=1.4,
                    label=f"srednia = {diffs.mean():+.4f}")
    axes[0].set_xlabel("Roznica srednich spreadow vs SPY (B - A)")
    axes[0].set_ylabel("Liczba spolek")
    axes[0].set_title(f"Histogram roznic - okno {PRIMARY_WINDOW} sesji")
    axes[0].legend()

    x = np.arange(len(WINDOWS))
    w = 0.35
    event_means = event_summary.set_index("window_sessions").loc[WINDOWS, "mean_diff"].values
    event_lo = event_summary.set_index("window_sessions").loc[WINDOWS, "ci_low"].values
    event_hi = event_summary.set_index("window_sessions").loc[WINDOWS, "ci_high"].values
    plac_means = placebo_summary.set_index("window_sessions").loc[WINDOWS, "mean_diff"].values
    plac_lo = placebo_summary.set_index("window_sessions").loc[WINDOWS, "ci_low"].values
    plac_hi = placebo_summary.set_index("window_sessions").loc[WINDOWS, "ci_high"].values

    axes[1].errorbar(x - w/2, event_means,
                     yerr=[event_means - event_lo, event_hi - event_means],
                     fmt="o", color="#1f77b4", capsize=4, label="EVENT 2022-11-30")
    axes[1].errorbar(x + w/2, plac_means,
                     yerr=[plac_means - plac_lo, plac_hi - plac_means],
                     fmt="s", color="#888888", capsize=4, label="PLACEBO 2021-11-30")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{k}" for k in WINDOWS])
    axes[1].set_xlabel("Dlugosc okna (sesje)")
    axes[1].set_ylabel("Srednia roznica spreadow vs SPY (B - A)")
    axes[1].set_title("Analiza wrazliwosci + placebo")
    axes[1].legend()
    savefig(fig, "task3_sensitivity_placebo.pdf")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
