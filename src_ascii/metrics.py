"""Metryki per spolka: ann. return, ann. vol, Sharpe, Sortino, MaxDD, CAPM beta, CAPM alpha."""

import numpy as np
import pandas as pd

from common import TRADING_DAYS


def annualized_return(daily_log_ret: pd.Series) -> float:
    return daily_log_ret.mean() * TRADING_DAYS


def annualized_volatility(daily_log_ret: pd.Series) -> float:
    return daily_log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_log_ret: pd.Series, daily_rf: pd.Series) -> float:
    excess = (daily_log_ret - daily_rf).dropna()
    if len(excess) < 20 or excess.std(ddof=1) == 0:
        return np.nan
    return excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_log_ret: pd.Series, daily_rf: pd.Series, mar: float = 0.0) -> float:
    excess = (daily_log_ret - daily_rf).dropna()
    if len(excess) < 20:
        return np.nan
    downside = excess[excess < mar]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else np.nan
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return excess.mean() / downside_dev * np.sqrt(TRADING_DAYS)


def max_drawdown(daily_log_ret: pd.Series) -> float:
    cum = daily_log_ret.cumsum()
    wealth = np.exp(cum)
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def capm_beta_alpha(daily_log_ret: pd.Series, bench_ret: pd.Series, daily_rf: pd.Series) -> tuple[float, float]:
    df = pd.concat([daily_log_ret, bench_ret, daily_rf], axis=1).dropna()
    df.columns = ["r", "m", "rf"]
    er = df["r"] - df["rf"]
    em = df["m"] - df["rf"]
    if len(df) < 20 or em.var(ddof=1) == 0:
        return np.nan, np.nan
    beta = float(np.cov(er, em, ddof=1)[0, 1] / em.var(ddof=1))
    alpha_daily = float(er.mean() - beta * em.mean())
    alpha_annual = alpha_daily * TRADING_DAYS
    return beta, alpha_annual


def compute_all_metrics(rets: pd.DataFrame, bench_ret: pd.Series, daily_rf: pd.Series,
                         tickers: list[str], group: str) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in rets.columns:
            continue
        r = rets[t].dropna()
        rf_aligned = daily_rf.reindex(r.index)
        bench_aligned = bench_ret.reindex(r.index)
        beta, alpha = capm_beta_alpha(r, bench_aligned, rf_aligned)
        rows.append({
            "ticker": t, "group": group,
            "ann_return": annualized_return(r),
            "ann_vol":    annualized_volatility(r),
            "sharpe":     sharpe_ratio(r, rf_aligned),
            "sortino":    sortino_ratio(r, rf_aligned),
            "max_dd":     max_drawdown(r),
            "beta":       beta,
            "alpha":      alpha,
            "n_obs":      len(r),
        })
    return pd.DataFrame(rows)
