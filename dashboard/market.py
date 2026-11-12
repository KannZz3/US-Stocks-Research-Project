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
    return_column = "RET_1" if "RET_1" in raw else "DlyRet"
    required = {"Ticker", "DlyCalDt", "DlyClose", return_column, "sprtrn"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing stock columns: " + ", ".join(missing))
    columns = {k: v for k, v in COLUMNS.items() if k != "DlyRet"}
    columns[return_column] = "return"
    df = raw.reindex(columns=columns).rename(columns=columns)
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


SECTORS = ("Tech", "Consumption", "Finance", "Utilities", "Energy")


def load_sectors(folder):
    """Explicit allowlist excludes the master data_preML.csv and other files."""
    sectors = {}
    for sector in SECTORS:
        path = Path(folder) / f"data_preML_{sector}.csv"
        try:
            frame, _ = load_stocks(path)
            if frame.ticker.nunique() != 9:
                raise ValueError("Expected exactly nine stocks.")
            sectors[sector] = frame
        except ValueError as error:
            raise ValueError(f"{sector}: {error}") from error
    return sectors


def buy_and_hold(frame):
    """Equal initial capital, fixed adjusted-share units; never rebalance or fill."""
    prices = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    if prices.empty or prices.isna().any().any():
        raise ValueError("Portfolio needs complete prices for every constituent/date.")
    if not np.isfinite(prices.to_numpy()).all() or prices.le(0).any().any():
        raise ValueError("Portfolio prices must be finite and positive.")
    return (prices / prices.iloc[0]).mean(axis=1) * 100


def preview_upload(content):
    """Validate only; preserve original columns and do not infer price adjustment."""
    try:
        raw = pd.read_csv(BytesIO(content), keep_default_na=False)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Cannot read this CSV. Use a UTF-8 stock-data CSV.") from error
    required = {"Ticker", "DlyCalDt", "DlyClose"}
    if not required.issubset(raw):
        raise ValueError("Required columns: Ticker, DlyCalDt, DlyClose.")
    tickers = raw.Ticker.astype("string").str.strip().str.upper()
    dates = pd.to_datetime(raw.DlyCalDt, errors="coerce", format="mixed", utc=True).dt.normalize()
    close = pd.to_numeric(raw.DlyClose, errors="coerce")
    if raw.empty or tickers.eq("").any() or dates.isna().any():
        raise ValueError("CSV must contain rows with valid tickers and dates.")
    if close.isna().any() or not np.isfinite(close).all() or close.le(0).any():
        raise ValueError("DlyClose must contain finite positive prices.")
    if pd.DataFrame({"ticker": tickers, "date": dates}).duplicated().any():
        raise ValueError("Duplicate stock/date rows found.")
    return raw.head(50), {"Rows": len(raw), "Stocks": tickers.nunique()}
