"""CAPM / FF3 / Carhart 4-factor regressions per spolka i per portfel grupowy.

Czynniki: Kenneth French Data Library (daily Fama-French Research + Momentum).
Inference: Newey-West HAC SE (lag=5).

Pliki:
  data_cache/ff_factors.csv (cache pobranych czynnikow)
  data_cache/factor_model_results.csv (alphy/bety per spolka i per grupa)
  figures/factor_model_alpha_per_group.pdf
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm

from common import CACHE, BENCHMARK, load_prices, load_metadata, log_returns, ew_portfolio_returns, group_tickers
from plot_utils import setup_style, savefig

FF_OUT = CACHE / "ff_factors.csv"
RESULTS_OUT = CACHE / "factor_model_results.csv"

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

GROUP_ORDER = ["AI_infra", "AI_platforms", "Broad_tech", "Defensive"]
GROUP_PAL = {
    "AI_infra": "#1f77b4", "AI_platforms": "#9467bd",
    "Broad_tech": "#2ca02c", "Defensive": "#d62728",
}


def fetch_french_csv(url: str) -> pd.DataFrame:
    """Pobiera ZIP z French Data Library, ekstraktuje CSV. Czynniki w PROCENTACH per dzien."""
    r = requests.get(url, headers={"User-Agent": "Sylwia Malinowska research"})
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    raw = z.read(name).decode("latin-1")
    # FF CSV ma header w pierwszych ~5 liniach, potem daily data, potem footer
    lines = raw.splitlines()
    # znajdz pierwsza linie zaczynajaca sie od 8-cyfrowej daty
    start = None
    for i, ln in enumerate(lines):
        if len(ln) > 8 and ln[:8].isdigit():
            start = i
            break
    end = None
    for i in range(start, len(lines)):
        if not lines[i] or not lines[i][:8].isdigit():
            end = i
            break
    if end is None:
        end = len(lines)
    data_lines = lines[start:end]
    headers_line = None
    for i in range(start - 1, -1, -1):
        if "," in lines[i] and not lines[i][:8].isdigit():
            headers_line = lines[i]
            break
    cols = ["Date"] + [c.strip() for c in headers_line.split(",") if c.strip()]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")
    # konwersja z procentow na ulamki
    for c in df.columns:
        df[c] = df[c].astype(float) / 100.0
    return df


def load_ff_factors() -> pd.DataFrame:
    if FF_OUT.exists():
        return pd.read_csv(FF_OUT, parse_dates=["Date"]).set_index("Date")
    print("Pobieram Fama-French 3-factor + Momentum z Dartmouth library")
    ff3 = fetch_french_csv(FF3_URL)
    mom = fetch_french_csv(MOM_URL)
    factors = ff3.join(mom, how="inner")
    factors.columns = [c.strip() for c in factors.columns]
    # standaryzacja nazw
    rename_map = {}
    for c in factors.columns:
        if "Mkt" in c or "RF" in c and "-" in c:
            rename_map[c] = "MKT_RF"
        if c == "RF":
            rename_map[c] = "RF"
        if c.upper() == "MOM":
            rename_map[c] = "MOM"
    factors = factors.rename(columns=rename_map)
    print(f"  zakres: {factors.index.min().date()} -> {factors.index.max().date()}")
    print(f"  kolumny: {list(factors.columns)}")
    factors.to_csv(FF_OUT)
    return factors


def run_factor_regression(excess_ret: pd.Series, factors: pd.DataFrame, model: str = "FF3") -> dict:
    df = pd.concat([excess_ret, factors], axis=1, join="inner").dropna()
    if len(df) < 50:
        return {"n": len(df)}
    y = df.iloc[:, 0]
    if model == "CAPM":
        X = sm.add_constant(df[["MKT_RF"]])
    elif model == "FF3":
        X = sm.add_constant(df[["MKT_RF", "SMB", "HML"]])
    elif model == "Carhart4":
        X = sm.add_constant(df[["MKT_RF", "SMB", "HML", "MOM"]])
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    res = {"n": len(df), "r2": float(m.rsquared), "alpha_daily": float(m.params["const"]),
           "alpha_annual": float(m.params["const"] * 252),
           "alpha_se_hac": float(m.bse["const"]), "alpha_p_hac": float(m.pvalues["const"])}
    for c in [x for x in X.columns if x != "const"]:
        res[f"{c}_coef"] = float(m.params[c])
        res[f"{c}_se_hac"] = float(m.bse[c])
        res[f"{c}_p_hac"] = float(m.pvalues[c])
    return res


def main():
    factors = load_ff_factors()
    cols = list(factors.columns)
    print(f"Kolumny FF: {cols}")
    assert "MKT_RF" in cols and "RF" in cols, f"Brak wymaganych kolumn; mamy: {cols}"
    assert "MOM" in cols, "Brak Momentum"

    prices = load_prices()
    meta = load_metadata()
    rets = log_returns(prices)
    bench = rets[BENCHMARK]

    rows = []
    print("\n========== PER SPOLKA ==========")
    for _, mrow in meta.iterrows():
        ticker = mrow["ticker"]
        if ticker not in rets.columns:
            continue
        r = rets[ticker].dropna()
        excess = r - factors["RF"].reindex(r.index).ffill()
        for model in ("CAPM", "FF3", "Carhart4"):
            res = run_factor_regression(excess, factors, model=model)
            res["ticker"] = ticker
            res["group"] = mrow["group"]
            res["unit"] = "stock"
            res["model"] = model
            rows.append(res)

    print("\n========== PER GRUPA (EW portfel) ==========")
    for g in GROUP_ORDER:
        tickers = group_tickers(meta, g)
        port = ew_portfolio_returns(rets, tickers)
        excess_port = port - factors["RF"].reindex(port.index).ffill()
        for model in ("CAPM", "FF3", "Carhart4"):
            res = run_factor_regression(excess_port, factors, model=model)
            res["ticker"] = f"PORT_{g}"
            res["group"] = g
            res["unit"] = "portfolio"
            res["model"] = model
            rows.append(res)
            print(f"  [{g} - {model}] n={res.get('n')} alpha_annual={res.get('alpha_annual', np.nan):+.4f}  "
                  f"p_HAC={res.get('alpha_p_hac', np.nan):.4f}  R^2={res.get('r2', np.nan):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_OUT, index=False)
    print(f"\nZapisano: {RESULTS_OUT}")

    # ============================================================================
    # Wykres: alpha per grupa, per model (porfele)
    # ============================================================================
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    port = df[df["unit"] == "portfolio"].copy()
    models = ["CAPM", "FF3", "Carhart4"]
    width = 0.25
    x = np.arange(len(GROUP_ORDER))
    for i, m in enumerate(models):
        sub = port[port["model"] == m].set_index("group").reindex(GROUP_ORDER)
        alphas = sub["alpha_annual"].values
        se = sub["alpha_se_hac"].values * 252
        ax.bar(x + (i - 1) * width, alphas, width=width,
               label=m, color=["#1f4e79", "#9467bd", "#d62728"][i],
               edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.errorbar(x + (i - 1) * width, alphas, yerr=1.96 * se,
                    fmt="none", ecolor="black", capsize=3, linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel("Annualized alpha (95% CI HAC)")
    ax.set_title("Factor model alphas per grupa - portfele EW; CAPM / FF3 / Carhart 4")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    savefig(fig, "factor_model_alpha_per_group.pdf")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
