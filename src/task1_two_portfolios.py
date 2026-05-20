"""Zadanie 1: porownanie srednich miesiecznych spreadow vs SPY dwoch portfeli EW
po rozpoczeciu boomu AI (2022-11-30): AI infrastructure vs Defensive control.

Test glowny: Welch t-test (rownosc srednich, wariancje niekoniecznie rowne).
Analiza odpornosci: bootstrap 95% CI dla roznicy srednich.
Test pomocniczy: Mann-Whitney U (roznica rang / dominacja stochastyczna,
NIE test srednich; przy nierownych wariancjach nie interpretowac jako test median).
Wielkosc efektu: Hedges' g (poprawiona Cohen's d dla malych prob).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

from common import (
    CACHE, BENCHMARK, EVENT_DATE,
    load_prices, load_metadata, log_returns,
    ew_portfolio_returns, to_monthly, abnormal_returns,
    group_tickers, filter_after,
)
from plot_utils import setup_style, savefig

ALPHA = 0.05
SEED = 42
N_BOOT = 10_000


def hedges_g(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * J)


def welch_with_ci(x, y, alpha=ALPHA):
    n1, n2 = len(x), len(y)
    m1, m2 = float(x.mean()), float(y.mean())
    v1, v2 = float(x.var(ddof=1)), float(y.var(ddof=1))
    diff = m1 - m2
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_stat = diff / se
    p = 2 * (1 - st.t.cdf(abs(t_stat), df))
    t_crit = st.t.ppf(1 - alpha / 2, df)
    return {
        "t": float(t_stat), "df": float(df), "p_value": float(p),
        "mean_x": m1, "mean_y": m2, "mean_diff": diff, "se_diff": float(se),
        "ci_low": float(diff - t_crit * se), "ci_high": float(diff + t_crit * se),
    }


def bootstrap_ci_diff(x, y, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    rng = np.random.default_rng(seed)
    xa, ya = np.asarray(x), np.asarray(y)
    n1, n2 = len(xa), len(ya)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(xa, size=n1, replace=True).mean() - rng.choice(ya, size=n2, replace=True).mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"boot_mean_diff": float(boot.mean()), "boot_ci_low": float(lo),
            "boot_ci_high": float(hi), "n_boot": n_boot}


def main():
    prices = load_prices()
    meta = load_metadata()
    rets = log_returns(prices)

    bench_d = rets[BENCHMARK]
    ai = group_tickers(meta, "AI_infra")
    defn = group_tickers(meta, "Defensive")

    port_ai_d   = ew_portfolio_returns(rets, ai)
    port_def_d  = ew_portfolio_returns(rets, defn)

    port_ai_m   = to_monthly(port_ai_d)
    port_def_m  = to_monthly(port_def_d)
    bench_m     = to_monthly(bench_d)

    ar_ai_m  = abnormal_returns(port_ai_m,  bench_m)
    ar_def_m = abnormal_returns(port_def_m, bench_m)

    ar_ai_post  = filter_after(ar_ai_m,  EVENT_DATE)
    ar_def_post = filter_after(ar_def_m, EVENT_DATE)

    print(f"Okres po {EVENT_DATE.date()}:")
    print(f"  AI infra:  n={len(ar_ai_post)}, mean={ar_ai_post.mean():.5f}, std={ar_ai_post.std(ddof=1):.5f}")
    print(f"  Defensive: n={len(ar_def_post)}, mean={ar_def_post.mean():.5f}, std={ar_def_post.std(ddof=1):.5f}")

    print("\n--- Test normalnosci (Shapiro-Wilk) ---")
    for name, s in [("AI_infra_AR", ar_ai_post), ("Defensive_AR", ar_def_post)]:
        W, p = st.shapiro(s)
        print(f"  {name}: W={W:.4f}, p={p:.4f}, normal={p>ALPHA}")

    print("\n--- Test homogenicznosci wariancji (Levene, median) ---")
    lev_stat, lev_p = st.levene(ar_ai_post, ar_def_post, center="median")
    print(f"  stat={lev_stat:.4f}, p={lev_p:.4f}, homo={lev_p>ALPHA}")

    print("\n--- Test glowny: Welch t-test ---")
    welch = welch_with_ci(ar_ai_post, ar_def_post)
    for k, v in welch.items():
        print(f"  {k}: {v}")
    print(f"  H0 (rownosc srednich spreadow vs SPY) {'ODRZUCONA' if welch['p_value']<ALPHA else 'NIE odrzucona'}")

    g = hedges_g(ar_ai_post, ar_def_post)
    print(f"  Hedges g = {g:.4f}")

    print(f"\n--- Bootstrap CI dla roznicy srednich (B={N_BOOT}, seed={SEED}) ---")
    boot = bootstrap_ci_diff(ar_ai_post, ar_def_post)
    for k, v in boot.items():
        print(f"  {k}: {v}")

    print("\n--- Test pomocniczy: Mann-Whitney U (roznica rang / dominacja stochastyczna, NIE w srednich) ---")
    mw_stat, mw_p = st.mannwhitneyu(ar_ai_post, ar_def_post, alternative="two-sided")
    print(f"  U={mw_stat:.2f}, p={mw_p:.4f}")

    pd.DataFrame([{"test": "Welch t-test", **welch, "hedges_g": g}]).to_csv(
        CACHE / "task1_welch.csv", index=False)
    pd.DataFrame([boot]).to_csv(CACHE / "task1_bootstrap.csv", index=False)
    pd.DataFrame([{"test": "Mann-Whitney U", "stat": mw_stat, "p_value": mw_p}]).to_csv(
        CACHE / "task1_mannwhitney.csv", index=False)
    pd.DataFrame({
        "month": ar_ai_post.index,
        "AI_infra_AR": ar_ai_post.values,
        "Defensive_AR": ar_def_post.reindex(ar_ai_post.index).values,
    }).to_csv(CACHE / "task1_monthly_AR.csv", index=False)

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    sns.kdeplot(ar_ai_post,  ax=ax1, label="AI infrastructure", color="#1f77b4", fill=True, alpha=0.35)
    sns.kdeplot(ar_def_post, ax=ax1, label="Defensive control", color="#d62728", fill=True, alpha=0.35)
    ax1.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("Miesieczny spread vs SPY")
    ax1.set_ylabel("Gestosc")
    ax1.set_title("Zad 1 - rozklad miesiecznych spreadow vs SPY")
    ax1.legend()

    melt = pd.DataFrame({
        "AI infrastructure": ar_ai_post.values,
        "Defensive control": ar_def_post.values,
    }).melt(var_name="grupa", value_name="spread")
    sns.boxplot(data=melt, x="grupa", y="spread", hue="grupa",
                palette={"AI infrastructure": "#1f77b4", "Defensive control": "#d62728"},
                ax=ax2, legend=False, width=0.5)
    sns.stripplot(data=melt, x="grupa", y="spread", color="black", size=3.5, alpha=0.5, ax=ax2)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.set_xlabel("")
    ax2.set_ylabel("Miesieczny spread vs SPY")
    ax2.set_title("Zad 1 - boxplot porownawczy")
    savefig(fig, "task1_distributions.pdf")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
