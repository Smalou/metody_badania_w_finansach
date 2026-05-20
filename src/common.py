"""Wspolne funkcje: log returns, monthly aggregation, EW portfele, abnormal returns, RF."""

from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
TRADING_DAYS = 252
BENCHMARK = "SPY"
RF_TICKER = "^IRX"

EVENT_DATE = pd.Timestamp("2022-11-30")
PLACEBO_DATE = pd.Timestamp("2021-11-30")


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "prices.parquet")


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(CACHE / "metadata_tickers.csv")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def daily_rf(prices: pd.DataFrame) -> pd.Series:
    return prices[RF_TICKER].ffill() / 100.0 / TRADING_DAYS


def ew_portfolio_returns(rets: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in rets.columns]
    return rets[cols].mean(axis=1, skipna=True)


def to_monthly(daily_log_ret: pd.Series) -> pd.Series:
    return daily_log_ret.resample("ME").sum()


def abnormal_returns(asset_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    aligned = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    return aligned.iloc[:, 0] - aligned.iloc[:, 1]


def group_tickers(meta: pd.DataFrame, group: str) -> list[str]:
    return meta.loc[meta["group"] == group, "ticker"].tolist()


def filter_after(s: pd.Series, date: pd.Timestamp) -> pd.Series:
    return s[s.index > date]
