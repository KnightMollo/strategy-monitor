"""TMT RSI(2) mean-reversion signal engine.

Pure strategy logic. No personal trade data.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.market_data import CACHE_DIR, fetch_multiple, fetch_prices, compute_sma
from lib.rrg import compute_idle_signal
from lib.tmt_rules import HISTORY_START, SECTORS, VERSION, completed_data, format_idle_evaluation, rule_clock, rsi_snapshot, select_action

IDLE_STATE_PATH = CACHE_DIR / f"tmt_idle_{VERSION}.json"


def compute_tmt_snapshot(*, now=None) -> dict:
    moment = rule_clock(now)
    prices = completed_data({"QQQ": fetch_prices("QQQ", start=HISTORY_START),
                             "TQQQ": fetch_prices("TQQQ", start=HISTORY_START)}, moment)
    idle = compute_idle_signal(fetch_multiple(list(SECTORS) + ["SPY"], start=HISTORY_START),
                               now=moment, state_path=IDLE_STATE_PATH)
    evaluation = format_idle_evaluation(idle, moment)
    qqq, tqqq = prices["QQQ"], prices["TQQQ"]
    if qqq.empty or len(qqq) < 200 or tqqq.empty or qqq.index[-1] != tqqq.index[-1]:
        return {"ready": False, "reason": "Completed QQQ/TQQQ data unavailable", "idle_evaluation": evaluation}
    snapshot = rsi_snapshot(qqq, tqqq)
    decision = select_action(snapshot, idle, in_tqqq=None, now=moment)
    if decision["action_type"].startswith("IDLE"):
        decision["reason"] += f" | {evaluation['reason']}"
    labels = {"TQQQ_ENTRY": "BUY TQQQ if flat", "TQQQ_EXIT": "EXIT TQQQ if held",
              "IDLE_REBALANCE": "ROTATE if currently idle"}
    return {**snapshot, **decision, "ready": True, "position_aware": False,
            "action": labels.get(decision["action_type"], "NO NEW ORDER"),
            "dm_hold": evaluation["target"], "idle_reason": evaluation["reason"],
            "dm_reason": evaluation["reason"], "breadth": round(idle.get("breadth", 0) * 100),
            "breakout_ticker": idle.get("breakout_ticker"), "rrg_asof": idle.get("rrg_asof"),
            "idle_evaluation": evaluation}


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
    """Backward-compatible RSI view of the frozen public snapshot."""
    snapshot = compute_tmt_snapshot()
    actions = {"TQQQ_ENTRY": "BUY TQQQ", "TQQQ_EXIT": "SELL TQQQ"}
    return TMTSignal(
        signal_date=snapshot.get("signal_asof", dt.date.today().isoformat()),
        action=actions.get(snapshot.get("action_type"), "WAIT"),
        qqq_rsi2=round(snapshot.get("qqq_rsi2", 0), 1),
        qqq_price=round(snapshot.get("qqq_price", 0), 2),
        qqq_sma200=round(snapshot.get("qqq_sma200", 0), 2),
        above_200=snapshot.get("above_200", False),
        tqqq_price=round(snapshot.get("tqqq_price", 0), 2),
        reason=snapshot["reason"],
    )
