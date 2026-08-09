"""Hold'em signal engine — RSI-based 3-phase state machine.

Pure strategy logic. No personal trade data.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lib.market_data import compute_rsi, compute_sma, fetch_multiple


@dataclass
class HoldemSignal:
    """Result of the Hold'em signal computation."""
    signal_date: str
    next_hold: str
    key_condition: str
    phase: int  # 0=normal bull, 1=UVXY hedge, 2=SGOV cooldown
    uvxy_lock_until: str = ""
    cooldown_remaining: int = 0
    spy_price: float = 0.0
    spy_ma100: float = 0.0
    spy_ma200: float = 0.0
    spy_ma50: float = 0.0
    tqqq_price: float = 0.0
    tqqq_ma20: float = 0.0
    rsi_tqqq: float = 0.0
    rsi_upro: float = 0.0
    rsi_spy: float = 0.0
    rsi_sqqq: float = 0.0
    rsi_tlt: float = 0.0
    is_bull: bool = True


@dataclass
class HoldemConfig:
    """Configurable parameters for Hold'em strategy."""
    tickers: list = field(default_factory=lambda: ["SPY", "QQQ", "TQQQ", "UPRO", "SQQQ", "TLT", "UVXY"])
    safe_asset: str = "SGOV"
    rsi_period: int = 10
    rsi_overbought: int = 79
    rsi_exit: int = 75
    cooldown_days: int = 10
    pullback_pct: float = 0.02
    max_uvxy_days: int = 5


def compute_holdem_signal(config: HoldemConfig = None) -> HoldemSignal:
    """Compute the current Hold'em signal using live market data.

     Decision tree:
     ┌─ SPY > MA100? (Bull market)
     │  ├─ Phase 0: Hold TQQQ, with 50MA 2-day cash gate
     │  ├─ Phase 1: Hold UVXY (max 5 days)
     │  └─ Phase 2: Hold SGOV cooldown
     └─ SPY ≤ MA100 (Bear market)
         ├─ Rebound checks: RSI(TQQQ)/RSI(SPY)
         ├─ Defense switch: SQQQ vs TLT by relative RSI
         └─ Bear default: TQQQ
    """
    if config is None:
        config = HoldemConfig()

    price_data = fetch_multiple(config.tickers, period="2y")

    if not price_data or "SPY" not in price_data or "TQQQ" not in price_data:
        return HoldemSignal(
            signal_date=dt.date.today().isoformat(),
            next_hold=config.safe_asset,
            key_condition="Insufficient data to compute signal.",
            phase=0,
        )

    # Build aligned close series
    all_indices = [price_data[t].index for t in config.tickers if t in price_data]
    common_idx = all_indices[0]
    for idx in all_indices[1:]:
        common_idx = common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    if len(common_idx) < 220:
        return HoldemSignal(
            signal_date=dt.date.today().isoformat(),
            next_hold=config.safe_asset,
            key_condition=f"Insufficient aligned data ({len(common_idx)} bars).",
            phase=0,
        )

    close = {}
    for t in config.tickers:
        if t in price_data:
            close[t] = price_data[t]["Close"].reindex(common_idx)

    # Compute indicators
    ma100_spy = compute_sma(close["SPY"], 100)
    ma200_spy = compute_sma(close["SPY"], 200)
    ma50_spy = compute_sma(close["SPY"], 50)
    ma20_tqqq = compute_sma(close["TQQQ"], 20)
    rsi_tqqq = compute_rsi(close["TQQQ"], config.rsi_period)
    rsi_upro = compute_rsi(close["UPRO"], config.rsi_period)
    rsi_spy = compute_rsi(close["SPY"], config.rsi_period)
    rsi_sqqq = compute_rsi(close["SQQQ"], config.rsi_period)
    rsi_tlt = compute_rsi(close["TLT"], config.rsi_period)

    # Use last completed bar
    i = len(common_idx) - 1
    today = pd.Timestamp(dt.date.today())
    now = dt.datetime.now()
    if common_idx[i] == today and now.hour < 16 and i > 0:
        i -= 1

    # Run state machine from start to current bar
    phase = 0
    tqqq_peak = np.nan
    last_cooldown_start_date = None
    phase2_start_idx = -999
    uvxy_start_idx = -999

    RSI_EXIT = config.rsi_exit
    COOLDOWN_DAYS = config.cooldown_days
    PULLBACK_PCT = config.pullback_pct
    MAX_UVXY_DAYS = config.max_uvxy_days

    for k in range(len(common_idx)):
        if k > i:
            break

        s = float(close["SPY"].iloc[k])
        s_ma = float(ma100_spy.iloc[k]) if pd.notna(ma100_spy.iloc[k]) else np.nan
        r_tqqq = float(rsi_tqqq.iloc[k]) if pd.notna(rsi_tqqq.iloc[k]) else np.nan
        r_upro = float(rsi_upro.iloc[k]) if pd.notna(rsi_upro.iloc[k]) else np.nan
        tq = float(close["TQQQ"].iloc[k])

        if not np.isfinite(s_ma) or s <= s_ma:
            phase = 0
            tqqq_peak = np.nan
            last_cooldown_start_date = None
            phase2_start_idx = -999
            continue

        if phase == 1:
            tqqq_peak = max(tqqq_peak, tq) if np.isfinite(tqqq_peak) else tq
            days_in_uvxy = k - uvxy_start_idx
            rsi_exit_triggered = np.isfinite(r_tqqq) and np.isfinite(r_upro) and r_tqqq < RSI_EXIT and r_upro < RSI_EXIT
            max_days_triggered = days_in_uvxy >= MAX_UVXY_DAYS
            if rsi_exit_triggered or max_days_triggered:
                phase = 2
                phase2_start_idx = k
                if k + 1 < len(common_idx):
                    last_cooldown_start_date = common_idx[k + 1]
                else:
                    last_cooldown_start_date = common_idx[k] + pd.Timedelta(days=1)
                tqqq_peak = tq
            continue

        # Check cooldown lock
        if last_cooldown_start_date is not None:
            days_since = (common_idx[k] - last_cooldown_start_date).days
        else:
            days_since = float("inf")
        can_enter_uvxy = days_since > COOLDOWN_DAYS

        overbought = (np.isfinite(r_tqqq) and r_tqqq > config.rsi_overbought) or \
                     (np.isfinite(r_upro) and r_upro > config.rsi_overbought)

        if overbought and can_enter_uvxy:
            tqqq_peak = tq
            phase = 1
            uvxy_start_idx = k
            continue

        if phase == 2:
            tqqq_peak = max(tqqq_peak, tq) if np.isfinite(tqqq_peak) else tq
            drawdown = tq / tqqq_peak - 1 if np.isfinite(tqqq_peak) and tqqq_peak > 0 else 0
            tq_ma20 = float(ma20_tqqq.iloc[k]) if pd.notna(ma20_tqqq.iloc[k]) else np.nan
            if drawdown <= -PULLBACK_PCT and np.isfinite(tq_ma20) and tq > tq_ma20:
                phase = 0
                phase2_start_idx = -999

    # Compute signal at bar i
    spy_val = float(close["SPY"].iloc[i])
    spy_ma_val = float(ma100_spy.iloc[i]) if pd.notna(ma100_spy.iloc[i]) else np.nan
    tqqq_val = float(close["TQQQ"].iloc[i])
    tqqq_ma20_val = float(ma20_tqqq.iloc[i]) if pd.notna(ma20_tqqq.iloc[i]) else np.nan
    v_rsi_tqqq = float(rsi_tqqq.iloc[i]) if pd.notna(rsi_tqqq.iloc[i]) else np.nan
    v_rsi_upro = float(rsi_upro.iloc[i]) if pd.notna(rsi_upro.iloc[i]) else np.nan
    v_rsi_spy = float(rsi_spy.iloc[i]) if pd.notna(rsi_spy.iloc[i]) else np.nan
    v_rsi_sqqq = float(rsi_sqqq.iloc[i]) if pd.notna(rsi_sqqq.iloc[i]) else np.nan
    v_rsi_tlt = float(rsi_tlt.iloc[i]) if pd.notna(rsi_tlt.iloc[i]) else np.nan

    signal_date = common_idx[i].strftime("%Y-%m-%d")
    is_bull = np.isfinite(spy_ma_val) and spy_val > spy_ma_val

    tqqq_retrace = tqqq_peak * (1 - PULLBACK_PCT) if np.isfinite(tqqq_peak) else np.nan

    # Determine next hold
    if not np.isfinite(spy_ma_val):
        next_hold = config.safe_asset
        condition = "MA100 data insufficient, holding SGOV."
    elif is_bull:
        if phase == 1:
            next_hold = "UVXY"
            days_held_so_far = i - uvxy_start_idx
            days_left = max(0, MAX_UVXY_DAYS - days_held_so_far)
            condition = (
                f"Bull hedge UVXY: TQQQ RSI {v_rsi_tqqq:.1f}, UPRO RSI {v_rsi_upro:.1f}. "
                f"Exit when both < {RSI_EXIT} or max {MAX_UVXY_DAYS}d cap "
                f"({days_held_so_far}d held, {days_left}d remain)"
            )
        elif phase == 2:
            next_hold = config.safe_asset
            if cooldown_remaining > 0:
                ma20_gate = f" AND TQQQ > MA20 ({tqqq_ma20_val:.2f})" if np.isfinite(tqqq_ma20_val) else ""
                condition = f"UVXY lock until {uvxy_lock_until}. Hold {config.safe_asset} or early exit if TQQQ < {tqqq_retrace:.2f}{ma20_gate}"
            else:
                condition = f"Cooldown complete. Waiting for RSI > {config.rsi_overbought} to re-enter UVXY or TQQQ < {tqqq_retrace:.2f}"
        else:
            spy_ma50_val = float(ma50_spy.iloc[i]) if pd.notna(ma50_spy.iloc[i]) else np.nan
            days_below_50 = 0
            if np.isfinite(spy_ma50_val):
                for lookback in range(min(5, i)):
                    idx = i - lookback
                    prev_spy = float(close["SPY"].iloc[idx])
                    prev_ma50 = float(ma50_spy.iloc[idx]) if pd.notna(ma50_spy.iloc[idx]) else np.nan
                    if np.isfinite(prev_ma50) and prev_spy < prev_ma50:
                        days_below_50 += 1
                    else:
                        break

            if np.isfinite(spy_ma50_val) and days_below_50 >= 2:
                next_hold = config.safe_asset
                condition = f"Bull cash (50MA gate): SPY below MA50 for {days_below_50}d ({spy_val:.2f} < {spy_ma50_val:.2f})"
            elif np.isfinite(spy_ma50_val) and spy_val > spy_ma50_val:
                next_hold = "TQQQ"
                condition = f"Bull TQQQ: SPY {spy_val:.2f} > MA50 {spy_ma50_val:.2f} & MA100 {spy_ma_val:.2f}"
            else:
                next_hold = "TQQQ"
                condition = f"Bull TQQQ: SPY near MA50 ({spy_val:.2f} vs {spy_ma50_val:.2f}), awaiting 2d confirm"
    else:
        if np.isfinite(v_rsi_tqqq) and v_rsi_tqqq < 30:
            next_hold = "TQQQ"
            condition = f"Bear rebound: TQQQ RSI {v_rsi_tqqq:.1f} < 30"
        elif np.isfinite(v_rsi_spy) and v_rsi_spy < 31:
            next_hold = "TQQQ"
            condition = f"Bear rebound: SPY RSI {v_rsi_spy:.1f} < 31"
        elif np.isfinite(tqqq_ma20_val) and tqqq_val < tqqq_ma20_val:
            if np.isfinite(v_rsi_sqqq) and np.isfinite(v_rsi_tlt) and v_rsi_sqqq > v_rsi_tlt:
                next_hold = "SQQQ"
                condition = (
                    f"Bear defense SQQQ: TQQQ {tqqq_val:.2f} < MA20 {tqqq_ma20_val:.2f}, "
                    f"SQQQ RSI {v_rsi_sqqq:.1f} > TLT RSI {v_rsi_tlt:.1f}"
                )
            else:
                next_hold = "TLT"
                condition = (
                    f"Bear defense TLT: TQQQ {tqqq_val:.2f} < MA20 {tqqq_ma20_val:.2f}, "
                    f"TLT RSI {v_rsi_tlt:.1f} >= SQQQ RSI {v_rsi_sqqq:.1f}"
                )
        elif np.isfinite(v_rsi_sqqq) and v_rsi_sqqq < 31:
            next_hold = "SQQQ"
            condition = f"Bear continuation SQQQ: SQQQ RSI {v_rsi_sqqq:.1f} < 31"
        else:
            next_hold = "TQQQ"
            condition = "Bear default TQQQ: No rebound or defense signal"

    # Cooldown remaining
    cooldown_remaining = 0
    uvxy_lock_until = ""
    if last_cooldown_start_date is not None and phase == 2:
        elapsed = (common_idx[i] - last_cooldown_start_date).days
        cooldown_remaining = max(0, COOLDOWN_DAYS - elapsed)
        lock_end = last_cooldown_start_date + pd.Timedelta(days=COOLDOWN_DAYS)
        uvxy_lock_until = lock_end.strftime("%Y-%m-%d")

    return HoldemSignal(
        signal_date=signal_date,
        next_hold=next_hold,
        key_condition=condition,
        phase=phase,
        uvxy_lock_until=uvxy_lock_until,
        cooldown_remaining=cooldown_remaining,
        spy_price=spy_val,
        spy_ma100=spy_ma_val if np.isfinite(spy_ma_val) else 0,
        spy_ma200=float(ma200_spy.iloc[i]) if pd.notna(ma200_spy.iloc[i]) else 0,
        spy_ma50=float(ma50_spy.iloc[i]) if pd.notna(ma50_spy.iloc[i]) else 0,
        tqqq_price=tqqq_val,
        tqqq_ma20=tqqq_ma20_val if np.isfinite(tqqq_ma20_val) else 0,
        rsi_tqqq=v_rsi_tqqq if np.isfinite(v_rsi_tqqq) else 0,
        rsi_upro=v_rsi_upro if np.isfinite(v_rsi_upro) else 0,
        rsi_spy=v_rsi_spy if np.isfinite(v_rsi_spy) else 0,
        rsi_sqqq=v_rsi_sqqq if np.isfinite(v_rsi_sqqq) else 0,
        rsi_tlt=v_rsi_tlt if np.isfinite(v_rsi_tlt) else 0,
        is_bull=is_bull,
    )
