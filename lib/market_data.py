"""Market data fetching with yfinance and local caching."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = dt.timedelta(hours=6)

_TICKER_MAP = {"BRKB": "BRK-B", "BRK.B": "BRK-B", "BRKA": "BRK-A"}


def _normalize_ticker(symbol: str) -> str:
    return _TICKER_MAP.get(symbol.upper(), symbol)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _is_cache_fresh(path: Path, min_rows: int = 100) -> bool:
    if not path.exists():
        return False
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    if (dt.datetime.now() - mtime) >= CACHE_TTL:
        return False
    try:
        df = pd.read_parquet(path)
        if len(df) < min_rows:
            return False
    except Exception:
        return False
    return True


def fetch_prices(ticker: str, period: str = "2y", force_refresh: bool = False) -> pd.DataFrame:
    """Fetch daily OHLCV for a ticker."""
    ticker = _normalize_ticker(ticker)
    cache = _cache_path(ticker)

    if not force_refresh and _is_cache_fresh(cache):
        return pd.read_parquet(cache)

    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        if cache.exists():
            return pd.read_parquet(cache)
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()]
    data = data[["Open", "High", "Low", "Close", "Volume"]]

    if len(data) > 0 and pd.isna(data["Close"].iloc[-1]):
        try:
            info = yf.Ticker(ticker).fast_info
            live_price = getattr(info, "last_price", None)
            if live_price and not pd.isna(live_price):
                data.loc[data.index[-1], "Close"] = live_price
        except Exception:
            pass

    data = data.dropna(subset=["Close"])
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    data.to_parquet(cache)
    return data


def fetch_multiple(tickers: list[str], period: str = "2y", force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    result = {}
    for t in tickers:
        try:
            df = fetch_prices(t, period=period, force_refresh=force_refresh)
            if not df.empty:
                result[t] = df
        except Exception:
            continue
    return result


def get_latest_price(ticker: str) -> Optional[float]:
    """Get the most recent closing price for a ticker."""
    df = fetch_prices(ticker)
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_latest_prices(tickers: list[str]) -> Dict[str, float]:
    """Get latest closing prices for multiple tickers."""
    prices = {}
    for t in tickers:
        p = get_latest_price(t)
        if p is not None:
            prices[t] = p
    return prices


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 10) -> pd.Series:
    """Wilder RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
