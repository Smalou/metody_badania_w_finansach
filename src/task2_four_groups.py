"""Zadanie 2: porownanie srednich metryk efektywnosci miedzy 4 grupami spolek.

Dla kazdej spolki liczymy: ann. return, ann. vol, Sharpe, Sortino, MaxDD, CAPM beta/alpha.
Glowna zmienna testowana: Sharpe Ratio (najszerzej akceptowana w finansach).
Druga zmienna (secondary): CAPM alpha (bezposrednio mierzy 'AI premium' po kontroli rynku).

Wybor testu glownego zalezy od zalozen:
- normalne + homogeniczne -> ANOVA + Tukey HSD
- normalne + nierowne wariancje -> Welch ANOVA + Games-Howell
- nienormalne -> Kruskal-Wallis + Dunn (Bonferroni)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from common import CACHE, BENCHMARK, load_prices, load_metadata, log_returns, daily_rf
from metrics import compute_all_metrics
from plot_utils import setup_style, savefig

ALPHA = 0.05
GROUP_ORDER = ["AI_infra", "AI_platforms", "Broad_tech", "Defensive"]
GROUP_PAL = {
    "AI_infra":     "#1f77b4",
    "AI_platforms": "#9467bd",
    "Broad_tech":   "#2ca02c",
    "Defensive":    "#d62728",
}


def compute_metrics_df(prices, meta) -> pd.DataFrame:
    rets = log_returns(prices)
    rf_d = daily_rf(prices)
    bench = rets[BENCHMARK]
    frames = []
    for g in GROUP_ORDER:
        tickers = meta.loc[meta["group"] == g, "ticker"].tolist()
        frames.append(compute_all_metrics(rets, bench, rf_d, tickers, g))
    return pd.concat(frames, ignore_index=True)


def normality_per_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby("group"):
        W, p = st.shapiro(sub[col].dropna())
        rows.append({"group": g, "n": len(sub), "W": W, "p": p, "normal": p > ALPHA})
    return pd.DataFrame(rows)


def variance_homogeneity(df: pd.DataFrame, col: str):
    groups = [sub[col].dropna().values for _, sub in df.groupby("group")]
    stat, p = st.levene(*groups, center="median")
    return {"stat": float(stat), "p_value": float(p), "homogeneous": p > ALPHA}


def anova_with_tukey(df, col):
    groups = [sub[col].dropna().values for _, sub in df.groupby("group")]
    F, p = st.f_oneway(*groups)
    all_vals = np.concatenate(groups)
    N, k = len(all_vals), len(groups)
    grand = all_vals.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_t = ss_b + ss_w
    eta2 = ss_b / ss_t
    ms_w = ss_w / (N - k)
    omega2 = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)
    tuk = pairwise_tukeyhsd(endog=df[col].values, groups=df["group"].values, alpha=ALPHA)
    return {"test": "ANOVA", "F": float(F), "df_b": k - 1, "df_w": N - k,
            "p_value": float(p), "eta_squared": float(eta2), "omega_squared": float(omega2)}, tuk


def welch_anova_with_games_howell(df, col):
    res = anova_oneway(df[col].values, df["group"].values, use_var="unequal", welch_correction=True)
    gh = sp.posthoc_games_howell(df, val_col=col, group_col="group")
    return {"test": "Welch ANOVA", "F": float(res.statistic), "df_b": float(res.df_num),
            "df_w": float(res.df_denom), "p_value": float(res.pvalue)}, gh


def kruskal_with_dunn(df, col):
    groups = [sub[col].dropna().values for _, sub in df.groupby("group")]
    H, p = st.kruskal(*groups)
    dunn = sp.posthoc_dunn(df, val_col=col, group_col="group", p_adjust="bonferroni")
    N = len(df)
    eta2_h = (H - len(groups) + 1) / (N - len(groups))
    return {"test": "Kruskal-Wallis", "H": float(H), "df": len(groups) - 1,
            "p_value": float(p), "eta_squared_H": float(eta2_h)}, dunn


def choose_and_run(df, col):
    norm_df = normality_per_group(df, col)
    homo = variance_homogeneity(df, col)
    all_norm = bool(norm_df["normal"].all())
    print(f"\n[{col}] normalne we wszystkich grupach: {all_norm}; "
          f"homogeniczne wariancje (Levene p={homo['p_value']:.4f}): {homo['homogeneous']}")
    if not all_norm:
        print(f"[{col}] -> Kruskal-Wallis + Dunn (Bonferroni)")
        main, post = kruskal_with_dunn(df, col)
    elif not homo["homogeneous"]:
        print(f"[{col}] -> Welch ANOVA + Games-Howell")
        main, post = welch_anova_with_games_howell(df, col)
    else:
        print(f"[{col}] -> ANOVA + Tukey HSD")
        main, post = anova_with_tukey(df, col)
    return norm_df, homo, main, post


def main():
    prices = load_prices()
    meta = load_metadata()
    metrics_df = compute_metrics_df(prices, meta)
    metrics_df.to_csv(CACHE / "task2_metrics.csv", index=False)
    print(f"Wczytano {len(metrics_df)} spolek (4 grupy x ~10).")
    print("\nSrednie metryk per grupa:")
    print(metrics_df.groupby("group")[["ann_return", "ann_vol", "sharpe", "sortino", "max_dd", "beta", "alpha"]]
          .mean().round(4).reindex(GROUP_ORDER).to_string())

    print("\n========== GLOWNA ZMIENNA: Sharpe Ratio ==========")
    norm_s, homo_s, main_s, post_s = choose_and_run(metrics_df, "sharpe")
    print(f"\n[Wynik glowny - Sharpe]")
    for k, v in main_s.items():
        print(f"  {k}: {v}")
    print(f"  Decyzja: H0 (rownosc srednich) {'ODRZUCONA' if main_s['p_value']<ALPHA else 'NIE odrzucona'}")
    print("\n[Post-hoc - Sharpe]")
    print(post_s)

    print("\n========== ZMIENNA DODATKOWA: CAPM alpha ==========")
    norm_a, homo_a, main_a, post_a = choose_and_run(metrics_df, "alpha")
    print(f"\n[Wynik glowny - alpha]")
    for k, v in main_a.items():
        print(f"  {k}: {v}")
    print(f"  Decyzja: H0 {'ODRZUCONA' if main_a['p_value']<ALPHA else 'NIE odrzucona'}")
    print("\n[Post-hoc - alpha]")
    print(post_a)

    norm_s.to_csv(CACHE / "task2_normality_sharpe.csv", index=False)
    pd.DataFrame([homo_s]).to_csv(CACHE / "task2_levene_sharpe.csv", index=False)
    pd.DataFrame([main_s]).to_csv(CACHE / "task2_main_sharpe.csv", index=False)
    norm_a.to_csv(CACHE / "task2_normality_alpha.csv", index=False)
    pd.DataFrame([homo_a]).to_csv(CACHE / "task2_levene_alpha.csv", index=False)
    pd.DataFrame([main_a]).to_csv(CACHE / "task2_main_alpha.csv", index=False)
    if hasattr(post_s, "_results_table"):
        pd.DataFrame(post_s._results_table.data[1:], columns=post_s._results_table.data[0]).to_csv(
            CACHE / "task2_posthoc_sharpe.csv", index=False)
    else:
        post_s.to_csv(CACHE / "task2_posthoc_sharpe.csv")
    if hasattr(post_a, "_results_table"):
        pd.DataFrame(post_a._results_table.data[1:], columns=post_a._results_table.data[0]).to_csv(
            CACHE / "task2_posthoc_alpha.csv", index=False)
    else:
        post_a.to_csv(CACHE / "task2_posthoc_alpha.csv")

    setup_style()
    for col, title in [("sharpe", "Sharpe Ratio"), ("alpha", "CAPM alpha (rocznie)")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.violinplot(data=metrics_df, x="group", y=col, order=GROUP_ORDER,
                       hue="group", palette=GROUP_PAL, ax=ax1, legend=False, inner="quartile")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax1.set_xlabel(""); ax1.set_ylabel(title)
        ax1.set_title(f"Violin plot - {title} per grupa")

        sns.boxplot(data=metrics_df, x="group", y=col, order=GROUP_ORDER,
                    hue="group", palette=GROUP_PAL, ax=ax2, legend=False, width=0.5)
        sns.stripplot(data=metrics_df, x="group", y=col, order=GROUP_ORDER,
                      color="black", size=3.5, alpha=0.5, ax=ax2)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax2.set_xlabel(""); ax2.set_ylabel(title)
        ax2.set_title(f"Boxplot - {title} per grupa")
        savefig(fig, f"task2_{col}_per_group.pdf")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
