"""Read the daily stock CSV and calculate trailing indicators."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = {
    "Ticker": "ticker", "DlyCalDt": "date", "DlyOpen": "open",
    "DlyHigh": "high", "DlyLow": "low", "DlyClose": "close",
    "DlyVol": "volume", "DlyRet": "return", "sprtrn": "market_return",
}


def stock_path(folder):
    local = Path(folder) / "data/stocks.csv"
    shared = Path(folder).parent / "data.csv"
    return local if local.is_file() or not shared.is_file() else shared


def fingerprint(path):
    try:
        stat = Path(path).expanduser().stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def load_stocks(path):
    try:
        source = BytesIO(path) if isinstance(path, bytes) else Path(path).expanduser()
        raw = pd.read_csv(source, keep_default_na=False)
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError("Cannot read the stock CSV. Check the path in Data settings.") from error
    required = {"Ticker", "DlyCalDt", "DlyClose", "DlyRet", "sprtrn"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing stock columns: " + ", ".join(missing))
    df = raw.reindex(columns=COLUMNS).rename(columns=COLUMNS)
    df["ticker"] = df.ticker.astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df.date, errors="coerce", format="mixed", utc=True).dt.tz_localize(None).dt.normalize()
    if df.empty or df.date.isna().any() or df.ticker.isna().any() or df.ticker.eq("").any():
        raise ValueError("Stock data is empty or has invalid dates/tickers.")
    if df.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate stock/date rows found. Resolve them in the source file.")
    for column in ["open", "high", "low", "close", "volume", "return", "market_return"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if df[["close", "return", "market_return"]].isna().any().any():
        raise ValueError("Missing or invalid close/return values. No missing returns are filled in.")
    if df.close.le(0).any() or df[["return", "market_return"]].lt(-1).any().any():
        raise ValueError("Prices must be positive and returns cannot be below −100%.")
    if df.groupby("date").market_return.nunique().gt(1).any():
        raise ValueError("S&P 500 returns disagree across stocks on the same date.")
    if df.groupby("date").ticker.nunique().ne(df.ticker.nunique()).any():
        raise ValueError("Stocks have different date coverage. Check the missing stock/date rows.")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = df[["open", "high", "low", "close"]]
    invalid_candles = prices.notna().all(axis=1) & (
        prices.min(axis=1).le(0) | df.high.lt(prices.max(axis=1)) | df.low.gt(prices.min(axis=1)))
    df.loc[invalid_candles, ["open", "high", "low"]] = np.nan
    df.loc[df.volume.lt(0), "volume"] = np.nan
    groups = df.groupby("ticker")
    for window in [20, 50, 200]:
        df[f"SMA {window}"] = groups.close.transform(lambda s: s.rolling(window).mean())
    df["Volatility"] = groups["return"].transform(lambda s: s.rolling(20).std(ddof=1) * np.sqrt(252))
    differences = groups.close.pct_change(fill_method=None).sub(df["return"]).abs().gt(.1)
    audit = {"Rows": len(df), "Stocks": df.ticker.nunique(), "Trading dates": df.date.nunique(),
             "Duplicate stock/date rows": 0, "Invalid candles hidden": int(invalid_candles.sum()),
             "Price/return differences over 10 percentage points": int(differences.sum())}
    return df, audit


def comparison(df, tickers, start, end, benchmark=True):
    selected = df[df.ticker.isin(tickers) & df.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    returns = selected.pivot(index="date", columns="ticker", values="return")
    prices = selected.pivot(index="date", columns="ticker", values="close")
    if benchmark:
        returns["S&P 500"] = selected.groupby("date").market_return.first()
    return returns, prices


def historical_index(returns):
    periods = returns.copy()
    periods.iloc[0] = 0  # The first date is the starting point, not an earned return.
    return 100 * (1 + periods).cumprod(skipna=False)
