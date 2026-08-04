"""TMT RSI(2) mean-reversion signal engine.

Pure strategy logic. No personal trade data.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.market_data import fetch_prices, compute_sma


@dataclass
class TMTSignal:
    """Result of the TMT RSI(2) signal computation."""
    signal_date: str
    action: str  # 'BUY TQQQ' | 'SELL TQQQ' | 'WAIT'
    qqq_rsi2: float
    qqq_price: float
    qqq_sma200: float
    above_200: bool
    tqqq_price: float
    reason: str


def compute_rsi2(series: pd.Series) -> pd.Series:
    """RSI(2) — Connors-style with EWM alpha=0.5."""
    delta = series.diff()
    up = delta.where(delta > 0, 0.0)
    dn = -delta.where(delta < 0, 0.0)
    avg_gain = up.ewm(alpha=1.0 / 2, adjust=False).mean()
    avg_loss = dn.ewm(alpha=1.0 / 2, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_tmt_signal() -> TMTSignal:
    """Compute TMT RSI(2) signal using live data.

    Decision tree:
    ┌─ QQQ > 200SMA?
    │  ├─ RSI(2) < 15 → BUY TQQQ (hold up to 10 days or RSI > 80)
    │  ├─ RSI(2) > 80 → SELL TQQQ
    │  └─ Otherwise → WAIT
    └─ QQQ ≤ 200SMA → WAIT (no entries below trend)
    """
    qqq = fetch_prices("QQQ", period="2y")
    tqqq = fetch_prices("TQQQ", period="1y")

    if qqq.empty or len(qqq) < 252 or tqqq.empty:
        return TMTSignal(
            signal_date=dt.date.today().isoformat(),
            action="WAIT",
            qqq_rsi2=0, qqq_price=0, qqq_sma200=0,
            above_200=False, tqqq_price=0,
            reason="Insufficient data",
        )

    close_qqq = qqq["Close"]
    rsi2 = compute_rsi2(close_qqq)
    sma200 = compute_sma(close_qqq, 200)

    curr_rsi2 = float(rsi2.iloc[-1])
    curr_qqq = float(close_qqq.iloc[-1])
    curr_sma200 = float(sma200.iloc[-1])
    curr_tqqq = float(tqqq["Close"].iloc[-1])
    above_200 = curr_qqq > curr_sma200

    if curr_rsi2 < 15 and above_200:
        action = "BUY TQQQ"
        reason = f"RSI(2)={curr_rsi2:.1f} < 15, QQQ ({curr_qqq:.1f}) > 200SMA ({curr_sma200:.1f})"
    elif curr_rsi2 > 80:
        action = "SELL TQQQ"
        reason = f"RSI(2)={curr_rsi2:.1f} > 80 → exit"
    else:
        action = "WAIT"
        trend = "above" if above_200 else "BELOW"
        reason = f"RSI(2)={curr_rsi2:.1f} | QQQ {trend} 200SMA"

    return TMTSignal(
        signal_date=dt.date.today().isoformat(),
        action=action,
        qqq_rsi2=round(curr_rsi2, 1),
        qqq_price=round(curr_qqq, 2),
        qqq_sma200=round(curr_sma200, 2),
        above_200=above_200,
        tqqq_price=round(curr_tqqq, 2),
        reason=reason,
    )
