"""Relative Rotation Graph (RRG) engine.

Implements JdK RS-Ratio and RS-Momentum for sector rotation analysis.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

RRG_BASKET = [
    "XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV",
    "XLU", "XLB", "XLRE", "XLC",
]

RRG_BENCHMARK = "SPY"

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLI": "Industrials", "XLY": "Cons. Discr.", "XLP": "Cons. Staples",
    "XLV": "Health Care", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication",
}

QUADRANT_NAMES = {
    (True, True): "Leading",
    (True, False): "Weakening",
    (False, True): "Improving",
    (False, False): "Lagging",
}

SECTOR_SPDR = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC"]


def compute_rrg(
    weekly_closes: Dict[str, pd.Series],
    benchmark: str = RRG_BENCHMARK,
    ratio_period: int = 10,
    momentum_period: int = 10,
    tail_length: int = 5,
) -> pd.DataFrame:
    """Compute RRG RS-Ratio and RS-Momentum for all tickers."""
    if benchmark not in weekly_closes:
        raise ValueError(f"Benchmark {benchmark} not in data")

    bench = weekly_closes[benchmark]
    results = []

    for ticker, prices in weekly_closes.items():
        if ticker == benchmark:
            continue
        if len(prices) < ratio_period + momentum_period + tail_length + 5:
            continue

        common = prices.index.intersection(bench.index).sort_values()
        if len(common) < ratio_period + momentum_period + tail_length + 5:
            continue
        p = prices.reindex(common)
        b = bench.reindex(common)

        rs = p / b
        rs_sma = rs.rolling(ratio_period).mean()
        rs_ratio_raw = 100.0 + (rs / rs_sma - 1.0) * 100.0
        rs_ratio = rs_ratio_raw.ewm(span=ratio_period, adjust=False).mean()

        rs_mom_raw = 100.0 + (rs_ratio / rs_ratio.shift(momentum_period) - 1.0) * 100.0
        rs_momentum = rs_mom_raw.ewm(span=momentum_period, adjust=False).mean()

        valid = rs_ratio.dropna().index.intersection(rs_momentum.dropna().index)
        if len(valid) < tail_length + 1:
            continue

        recent = valid[-tail_length - 1:]
        for j, d in enumerate(recent):
            rr = float(rs_ratio.loc[d])
            rm = float(rs_momentum.loc[d])
            q = QUADRANT_NAMES[(rr >= 100, rm >= 100)]
            results.append({
                "ticker": ticker,
                "label": SECTOR_MAP.get(ticker, ticker),
                "date": d,
                "rs_ratio": rr,
                "rs_momentum": rm,
                "quadrant": q,
                "is_current": j == len(recent) - 1,
                "tail_idx": j,
            })

    return pd.DataFrame(results)


def get_rrg_ranking(rrg_df: pd.DataFrame) -> pd.DataFrame:
    """Get current ranking sorted by RS-Ratio descending."""
    current = rrg_df[rrg_df["is_current"]].copy()
    current = current.sort_values("rs_ratio", ascending=False).reset_index(drop=True)
    current["rank"] = range(1, len(current) + 1)
    return current


def detect_signals(rrg_df: pd.DataFrame) -> List[Dict]:
    """Detect key quadrant transitions."""
    current = rrg_df[rrg_df["is_current"]].set_index("ticker")
    tails = rrg_df[~rrg_df["is_current"]]
    if tails.empty:
        return []

    max_tail = tails.groupby("ticker")["tail_idx"].max()
    prev_rows = []
    for t, max_t in max_tail.items():
        row = tails[(tails["ticker"] == t) & (tails["tail_idx"] == max_t)]
        if not row.empty:
            prev_rows.append(row.iloc[0])
    if not prev_rows:
        return []

    prev = pd.DataFrame(prev_rows).set_index("ticker")

    signals = []
    for ticker in current.index:
        if ticker not in prev.index:
            continue
        curr_q = current.loc[ticker, "quadrant"]
        prev_q = prev.loc[ticker, "quadrant"]
        if curr_q == prev_q:
            continue

        transition = f"{prev_q} -> {curr_q}"
        if curr_q == "Leading" and prev_q == "Improving":
            sig_type = "BREAKOUT"
        elif curr_q == "Improving" and prev_q == "Lagging":
            sig_type = "RECOVER"
        elif curr_q == "Weakening" and prev_q == "Leading":
            sig_type = "FADING"
        elif curr_q == "Lagging" and prev_q == "Weakening":
            sig_type = "BREAKDOWN"
        else:
            sig_type = "ROTATION"

        signals.append({
            "ticker": ticker,
            "label": SECTOR_MAP.get(ticker, ticker),
            "signal": sig_type,
            "transition": transition,
            "rs_ratio": float(current.loc[ticker, "rs_ratio"]),
            "rs_momentum": float(current.loc[ticker, "rs_momentum"]),
        })

    return signals


def _resample_close(daily_closes: Dict[str, pd.DataFrame], rule: str) -> Dict[str, pd.Series]:
    """Convert daily OHLCV data to a fixed-frequency close series."""
    weekly = {}
    for ticker, df in daily_closes.items():
        if df.empty:
            continue
        close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        w = close.resample(rule).last().dropna()
        if len(w) > 10:
            weekly[ticker] = w
    return weekly


def resample_weekly(daily_closes: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Convert daily OHLCV data to weekly close series."""
    return _resample_close(daily_closes, "W-FRI")


def resample_biweekly(daily_closes: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Convert daily OHLCV data to the biweekly Friday closes used by TMT idle RRG."""
    return _resample_close(daily_closes, "2W-FRI")


def compute_idle_signal(daily_closes: Dict[str, pd.DataFrame], benchmark: str = "SPY") -> Dict:
    """Compute TMT idle signal: RRG breakout + SGOV safety net."""
    close = {}
    for t in SECTOR_SPDR + [benchmark]:
        if t in daily_closes and not daily_closes[t].empty:
            df = daily_closes[t]
            c = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
            close[t] = c

    if benchmark not in close:
        return {"hold": "SGOV", "breadth": 0, "breakout_ticker": None, "quadrants": {}, "reason": "Benchmark data unavailable"}

    spy_bw = close[benchmark].resample("2W-FRI").last().dropna()
    sector_rr = {}
    sector_rm = {}
    period = 7

    for t in SECTOR_SPDR:
        if t not in close:
            continue
        tw = close[t].resample("2W-FRI").last().dropna()
        cw = tw.index.intersection(spy_bw.index)
        if len(cw) <= period * 2 + 5:
            continue
        rs = tw.loc[cw] / spy_bw.loc[cw]
        rs_sma = rs.rolling(period).mean()
        rr_raw = 100.0 + (rs / rs_sma - 1.0) * 100.0
        rr = rr_raw.ewm(span=period, adjust=False).mean()
        rm_raw = 100.0 + (rr / rr.shift(period) - 1.0) * 100.0
        rm = rm_raw.ewm(span=period, adjust=False).mean()
        sector_rr[t] = rr
        sector_rm[t] = rm

    if not sector_rr:
        return {"hold": "SGOV", "breadth": 0, "breakout_ticker": None, "quadrants": {}, "reason": "Insufficient sector data"}

    quadrants = {}
    bullish = 0
    total = 0
    for t in SECTOR_SPDR:
        if t not in sector_rr or t not in sector_rm:
            continue
        rr_val = float(sector_rr[t].iloc[-1]) if pd.notna(sector_rr[t].iloc[-1]) else np.nan
        rm_val = float(sector_rm[t].iloc[-1]) if pd.notna(sector_rm[t].iloc[-1]) else np.nan
        if not (np.isfinite(rr_val) and np.isfinite(rm_val)):
            continue
        total += 1
        q = QUADRANT_NAMES[(rr_val >= 100, rm_val >= 100)]
        quadrants[t] = {"quadrant": q, "rs_ratio": round(rr_val, 2), "rs_momentum": round(rm_val, 2)}
        if q in ("Leading", "Improving"):
            bullish += 1

    breadth = bullish / total if total > 0 else 0.5

    breakout_ticker = None
    for t in SECTOR_SPDR:
        if t not in sector_rr or t not in sector_rm:
            continue
        rr = sector_rr[t]
        rm = sector_rm[t]
        if len(rr) < 2 or len(rm) < 2:
            continue
        rr_now = float(rr.iloc[-1]) if pd.notna(rr.iloc[-1]) else np.nan
        rm_now = float(rm.iloc[-1]) if pd.notna(rm.iloc[-1]) else np.nan
        rr_prev = float(rr.iloc[-2]) if pd.notna(rr.iloc[-2]) else np.nan
        if np.isfinite(rr_now) and np.isfinite(rm_now) and np.isfinite(rr_prev):
            if rr_now >= 100 and rm_now >= 100 and rr_prev < 100:
                breakout_ticker = t
                break

    if breadth < 0.40:
        hold = "SGOV"
        reason = f"Breadth {breadth:.0%} < 40% -> SGOV"
    elif breakout_ticker:
        hold = breakout_ticker
        label = SECTOR_MAP.get(breakout_ticker, breakout_ticker)
        reason = f"Breakout: {breakout_ticker} ({label}) Improving -> Leading"
    else:
        hold = "SPY"
        reason = f"No breakout this period (breadth {breadth:.0%}) -> keep broad market"

    return {
        "hold": hold,
        "breadth": breadth,
        "breakout_ticker": breakout_ticker,
        "quadrants": quadrants,
        "reason": reason,
    }
