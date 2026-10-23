"""Frozen HOLDEM chop-mode rules, mirrored in strategy-monitor/lib/holdem_chop.py.

In the bull TQQQ branch, a high-volatility QQQ regime replaces continuous TQQQ with the
TMT RSI(2) dip-buy leg (TQQQ while the leg is active, otherwise the safe asset).
Research: Report/rescue_study.py, Good to learn/strategies/HOLDEM_TMT_救援研究.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "holdem-chop-v1"
VOL_WINDOW = 20
VOL_ON = 0.25
VOL_OFF = 0.20
RSI2_ENTRY = 15
RSI2_EXIT = 80
TREND_SMA = 200
STOP_PCT = 0.15
MAX_HOLD_SESSIONS = 21


def realized_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def chop_regime(vol: pd.Series, on: float = VOL_ON, off: float = VOL_OFF) -> pd.Series:
    """Hysteresis: switches on above `on`, stays on until vol drops below `off`."""
    state, out = False, []
    for value in vol.to_numpy():
        if np.isfinite(value):
            state = value >= off if state else value > on
        out.append(state)
    return pd.Series(out, index=vol.index)


def rsi2(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0).ewm(alpha=0.5, adjust=False).mean()
    losses = (-delta.where(delta < 0, 0.0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + gains / losses)


def rsi2_leg(qqq_close: pd.Series, tqqq_open: pd.Series, tqqq_close: pd.Series) -> pd.Series:
    """Rule replay of the TMT RSI(2) leg: True where the rule wants TQQQ at the next open."""
    rsi = rsi2(qqq_close).to_numpy()
    above = (qqq_close > qqq_close.rolling(TREND_SMA).mean()).to_numpy()
    opens, closes = tqqq_open.to_numpy(), tqqq_close.to_numpy()
    state, entry_idx, entry_px, out = False, None, np.nan, []
    for k in range(len(rsi)):
        if state and k >= entry_idx:
            if (closes[k] / entry_px - 1 <= -STOP_PCT or not above[k]
                    or rsi[k] > RSI2_EXIT or k - entry_idx >= MAX_HOLD_SESSIONS):
                state = False
        elif not state and np.isfinite(rsi[k]) and rsi[k] < RSI2_ENTRY and above[k]:
            state, entry_idx = True, k + 1
            entry_px = opens[k + 1] if k + 1 < len(opens) else np.nan
        out.append(state)
    return pd.Series(out, index=qqq_close.index)


def chop_snapshot(qqq_close: pd.Series, tqqq_open: pd.Series, tqqq_close: pd.Series, position: int) -> dict:
    vol = realized_vol(qqq_close)
    rsi = rsi2(qqq_close)
    sma = qqq_close.rolling(TREND_SMA).mean()
    value = float(vol.iloc[position])
    return {
        "version": VERSION,
        "qqq_vol20": value if np.isfinite(value) else 0.0,
        "chop_on": bool(chop_regime(vol).iloc[position]),
        "leg_active": bool(rsi2_leg(qqq_close, tqqq_open, tqqq_close).iloc[position]),
        "qqq_rsi2": float(rsi.iloc[position]),
        "qqq_above_200": bool(qqq_close.iloc[position] > sma.iloc[position]),
    }


def apply_chop(next_hold: str, condition: str, snapshot: dict, safe_asset: str) -> tuple[str, str, bool]:
    """Only call for the bull phase-0 TQQQ branch. Returns (hold, condition, overridden)."""
    if next_hold != "TQQQ" or not snapshot["chop_on"]:
        return next_hold, condition, False
    vol = f"QQQ vol20 {snapshot['qqq_vol20']:.1%} (on >{VOL_ON:.0%}, off <{VOL_OFF:.0%})"
    if snapshot["leg_active"]:
        return "TQQQ", f"Chop mode TQQQ: {vol}; RSI(2) leg active (RSI2 {snapshot['qqq_rsi2']:.1f})", True
    return safe_asset, (f"Chop mode {safe_asset}: {vol}; RSI(2) leg inactive "
                        f"(RSI2 {snapshot['qqq_rsi2']:.1f}, entry < {RSI2_ENTRY} & QQQ > {TREND_SMA}SMA)"), True
