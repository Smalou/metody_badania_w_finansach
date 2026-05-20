"""Pobranie cen z yfinance: 41 spolek w 4 grupach + benchmark SPY + RF ^IRX.

Cache do data_cache/prices.parquet i data_cache/metadata_tickers.csv.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
CACHE.mkdir(exist_ok=True)

GROUPS = {
    "AI_infra": [
        "NVDA", "AMD", "AVGO", "TSM", "ASML", "AMAT",
        "LRCX", "KLAC", "MU", "MRVL", "SMCI",
    ],
    "AI_platforms": [
        "MSFT", "GOOGL", "AMZN", "META", "ORCL",
        "CRM", "NOW", "ADBE", "IBM", "SNOW",
    ],
    "Broad_tech": [
        "AAPL", "CSCO", "INTC", "TXN", "QCOM",
        "INTU", "ACN", "PANW", "CDNS", "ADSK",
    ],
    "Defensive": [
        "KO", "PEP", "PG", "WMT", "JNJ",
        "MRK", "MCD", "COST", "CL", "GIS",
    ],
}

BENCHMARK = "SPY"
RF = "^IRX"

START = "2021-01-01"
END   = "2026-05-15"


def fetch_close(tickers, start, end):
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = pd.concat(
            {t: data[t]["Close"] for t in tickers if t in data.columns.get_level_values(0)},
            axis=1,
        )
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close


def main():
    all_tickers = [t for ts in GROUPS.values() for t in ts] + [BENCHMARK, RF]
    print(f"Pobieram {len(all_tickers)} tickerow: {START} -> {END}")
    prices = fetch_close(all_tickers, START, END)
    out_prices = CACHE / "prices.parquet"
    prices.to_parquet(out_prices)
    print(f"  -> {out_prices.name}: shape={prices.shape}, missing_total={int(prices.isna().sum().sum())}")
    print(f"  zakres dat: {prices.index.min().date()} -> {prices.index.max().date()}")

    print("\nLiczba spolek per grupa:")
    for g, ts in GROUPS.items():
        available = [t for t in ts if t in prices.columns]
        missing = set(ts) - set(available)
        print(f"  {g}: {len(available)}/{len(ts)}{' (brak: ' + ', '.join(sorted(missing)) + ')' if missing else ''}")

    print(f"\nBenchmark {BENCHMARK}: NaN={int(prices[BENCHMARK].isna().sum())}")
    print(f"Risk-free {RF}: mean={prices[RF].mean():.3f}%, NaN={int(prices[RF].isna().sum())}")


if __name__ == "__main__":
    main()
