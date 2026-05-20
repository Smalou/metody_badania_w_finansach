"""Wykresy zaawansowane: cumulative spread vs SPY per grupa, forest plot,
heatmapa korelacji, event window spread wokol 2022-11-30, rolling vol per grupa."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from common import (
    CACHE, BENCHMARK, EVENT_DATE,
    load_prices, load_metadata, log_returns,
    ew_portfolio_returns, abnormal_returns, group_tickers,
)
from plot_utils import setup_style, savefig

GROUP_ORDER = ["AI_infra", "AI_platforms", "Broad_tech", "Defensive"]
GROUP_PAL = {
    "AI_infra":     "#1f77b4",
    "AI_platforms": "#9467bd",
    "Broad_tech":   "#2ca02c",
    "Defensive":    "#d62728",
}


def plot_cumulative_AR(prices, meta):
    rets = log_returns(prices)
    bench_d = rets[BENCHMARK]
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for g in GROUP_ORDER:
        tickers = group_tickers(meta, g)
        port = ew_portfolio_returns(rets, tickers)
        ar = abnormal_returns(port, bench_d)
        cum = ar.cumsum()
        ax.plot(cum.index, cum.values, color=GROUP_PAL[g], linewidth=1.6, label=g)
    ax.axvline(EVENT_DATE, color="black", linestyle="--", linewidth=1.0,
               label=f"event {EVENT_DATE.date()}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Data")
    ax.set_ylabel("Skumulowany spread log-zwrotow vs SPY")
    ax.set_title("Cumulative spread vs SPY per grupa (portfel EW)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    savefig(fig, "extra_cumulative_AR.pdf")


def plot_forest(prices, meta):
    """Forest plot - srednia roznica metryki vs grupa Defensive (baseline) z 95% CI."""
    from metrics import compute_all_metrics
    from common import daily_rf
    rets = log_returns(prices)
    rf_d = daily_rf(prices)
    bench = rets[BENCHMARK]
    frames = []
    for g in GROUP_ORDER:
        tickers = group_tickers(meta, g)
        frames.append(compute_all_metrics(rets, bench, rf_d, tickers, g))
    metrics_df = pd.concat(frames, ignore_index=True)

    baseline = "Defensive"
    rows = []
    for col, nice in [("sharpe", "Sharpe"), ("alpha", "CAPM alpha"),
                      ("ann_return", "Ann. return"), ("sortino", "Sortino")]:
        base_vals = metrics_df.loc[metrics_df["group"] == baseline, col].values
        base_mean = base_vals.mean()
        base_n = len(base_vals)
        base_var = base_vals.var(ddof=1)
        for g in GROUP_ORDER:
            if g == baseline:
                continue
            g_vals = metrics_df.loc[metrics_df["group"] == g, col].values
            n = len(g_vals)
            diff = g_vals.mean() - base_mean
            se = np.sqrt(g_vals.var(ddof=1) / n + base_var / base_n)
            ci = 1.96 * se
            rows.append({"metric": nice, "group": g, "diff": diff,
                         "ci_low": diff - ci, "ci_high": diff + ci})
    forest = pd.DataFrame(rows)

    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    metrics_list = forest["metric"].unique()
    y = 0
    yticks = []
    yticklabels = []
    for m in metrics_list:
        sub = forest[forest["metric"] == m]
        for _, r in sub.iterrows():
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                        fmt="o", color=GROUP_PAL[r["group"]], capsize=4, markersize=7, linewidth=1.5)
            yticks.append(y)
            yticklabels.append(f"{m}: {r['group']}")
            y += 1
        y += 0.6
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Roznica srednich vs grupa {baseline} (z 95% CI)")
    ax.set_title("Forest plot - roznice metryk wzgledem grupy Defensive (baseline)")
    savefig(fig, "extra_forest_diff.pdf")


def plot_event_window_CAR(prices, meta, days_around=120):
    """Skumulowany spread vs SPY od T-days do T+days dla portfeli per grupa."""
    rets = log_returns(prices)
    bench = rets[BENCHMARK]
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for g in GROUP_ORDER:
        tickers = group_tickers(meta, g)
        port = ew_portfolio_returns(rets, tickers)
        ar = abnormal_returns(port, bench)
        idx = ar.index
        i_event = idx.get_indexer([EVENT_DATE], method="nearest")[0]
        lo = max(0, i_event - days_around)
        hi = min(len(idx), i_event + days_around + 1)
        window = ar.iloc[lo:hi].copy()
        relative = np.arange(lo - i_event, hi - i_event)
        car = window.cumsum() - window.cumsum().iloc[(i_event - lo) if (i_event - lo) < len(window) else -1]
        ax.plot(relative, car.values, color=GROUP_PAL[g], linewidth=1.6, label=g)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label=f"event {EVENT_DATE.date()}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Dni sesyjne wzgledem zdarzenia (T-0 = 2022-11-30)")
    ax.set_ylabel("Skumulowany spread vs SPY (re-bazowany na T-0)")
    ax.set_title(f"Event-window spread vs SPY (+/- {days_around} sesji)")
    ax.legend(loc="upper left", framealpha=0.95)
    savefig(fig, "extra_event_window_CAR.pdf")


def plot_correlation_heatmap(prices, meta):
    rets = log_returns(prices).drop(columns=[BENCHMARK, "^IRX"], errors="ignore")
    cols = []
    for g in GROUP_ORDER:
        cols.extend(group_tickers(meta, g))
    cols = [c for c in cols if c in rets.columns]
    corr = rets[cols].corr()
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.6, "label": "korelacja Pearsona"},
                annot=False)
    boundaries = []
    cum = 0
    for g in GROUP_ORDER:
        cum += len(group_tickers(meta, g))
        boundaries.append(cum)
    for b in boundaries[:-1]:
        ax.axhline(b, color="black", linewidth=1.2)
        ax.axvline(b, color="black", linewidth=1.2)
    ax.set_title("Heatmapa korelacji dziennych log-zwrotow (sektorowanie: AI infra | AI platforms | Broad tech | Defensive)")
    savefig(fig, "extra_corr_heatmap.pdf")


def plot_rolling_vol(prices, meta, window=63):
    rets = log_returns(prices)
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for g in GROUP_ORDER:
        tickers = group_tickers(meta, g)
        port = ew_portfolio_returns(rets, tickers)
        roll_vol = port.rolling(window).std(ddof=1) * np.sqrt(252)
        ax.plot(roll_vol.index, roll_vol.values, color=GROUP_PAL[g], linewidth=1.4, label=g)
    ax.axvline(EVENT_DATE, color="black", linestyle="--", linewidth=1.0,
               label=f"event {EVENT_DATE.date()}")
    ax.set_xlabel("Data")
    ax.set_ylabel(f"Rolling ann. volatility ({window} sesji)")
    ax.set_title("Rolling volatility per grupa (portfele EW, okno 63 sesji)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    savefig(fig, "extra_rolling_vol.pdf")


def main():
    prices = load_prices()
    meta = load_metadata()
    plot_cumulative_AR(prices, meta)
    plot_forest(prices, meta)
    plot_event_window_CAR(prices, meta)
    plot_correlation_heatmap(prices, meta)
    plot_rolling_vol(prices, meta)
    print("\nGotowe.")


if __name__ == "__main__":
    main()
