"""Frozen public TMT rules, mirrored in strategy-monitor/lib/tmt_rules.py."""
from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import exchange_calendars as xc
import numpy as np
import pandas as pd

VERSION = "tmt-fixed-2w-spy-v1"
HISTORY_START = "2019-06-03"
ANCHOR = pd.Timestamp("2019-06-07")
SECTORS = ("XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU", "XLB", "XLRE", "XLC")
SECTOR_NAMES = dict(zip(SECTORS, ("Technology", "Financials", "Energy", "Industrials", "Cons. Discr.", "Cons. Staples", "Health Care", "Utilities", "Materials", "Real Estate", "Communication")))
_STATE_LOCK = RLock()


@lru_cache(maxsize=1)
def market_calendar():
    return xc.get_calendar("XNYS", start="2019-01-01", end="2040-12-31")


def rule_clock(now=None) -> pd.Timestamp:
    moment = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if moment.tzinfo is None:
        moment = moment.tz_localize("America/New_York")
    return moment.tz_convert("UTC")


def completed_data(data: dict, now=None) -> dict:
    moment = rule_clock(now)
    calendar = market_calendar()
    completed = calendar.schedule.index[calendar.schedule["close"] <= moment]
    if completed.empty:
        return {ticker: frame.iloc[:0] for ticker, frame in data.items()}
    cutoff = completed[-1]
    return {
        ticker: frame.loc[frame.index.normalize().isin(completed) & (frame.index >= pd.Timestamp(HISTORY_START))].copy()
        for ticker, frame in data.items()
    }


def next_session(date) -> pd.Timestamp:
    calendar = market_calendar()
    session = calendar.date_to_session(pd.Timestamp(date).normalize(), direction="previous")
    return calendar.next_session(session)


def held_sessions(entry_date, signal_date) -> int:
    if pd.Timestamp(entry_date).normalize() > pd.Timestamp(signal_date).normalize():
        return 0
    sessions = market_calendar().sessions_in_range(
        pd.Timestamp(entry_date).normalize(), pd.Timestamp(signal_date).normalize()
    )
    return max(len(sessions) - 1, 0)


def idle_dates(period_end=None, now=None) -> dict:
    moment = rule_clock(now)
    calendar = market_calendar()
    today = moment.tz_convert("America/New_York").tz_localize(None).normalize()
    expected = ANCHOR + pd.Timedelta(days=((today - ANCHOR).days // 14) * 14)
    expected_session = calendar.date_to_session(expected, direction="previous")
    if calendar.session_close(expected_session) > moment:
        expected -= pd.Timedelta(days=14)
    following_candidate = expected + pd.Timedelta(days=14)
    if calendar.session_close(calendar.date_to_session(following_candidate, direction="previous")) <= moment:
        expected = following_candidate
    period = expected if period_end is None else pd.Timestamp(period_end).normalize()
    evaluation_session = calendar.date_to_session(period, direction="previous")
    execution_session = calendar.next_session(evaluation_session)
    execution_open = calendar.session_open(execution_session)
    evaluation_close = calendar.session_close(evaluation_session)
    following = expected + pd.Timedelta(days=14)
    following_session = calendar.date_to_session(following, direction="previous")
    return {
        "rule_version": VERSION,
        "expected_period": expected.strftime("%Y-%m-%d"),
        "evaluation_date": period.strftime("%Y-%m-%d"),
        "evaluation_session": evaluation_session.strftime("%Y-%m-%d"),
        "evaluation_close": evaluation_close.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M %Z"),
        "execution_date": execution_session.strftime("%Y-%m-%d"),
        "execution_open": execution_open.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M %Z"),
        "next_evaluation_date": following.strftime("%Y-%m-%d"),
        "next_evaluation_close": calendar.session_close(following_session).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M %Z"),
        "next_execution_date": calendar.next_session(following_session).strftime("%Y-%m-%d"),
        "rebalance_window": bool(evaluation_close <= moment < execution_open),
        "stale": period != expected,
    }


def frozen_rrg_frame(data: dict, tail_length: int = 2, benchmark: str = "SPY", now=None) -> pd.DataFrame:
    data = completed_data(data, now)
    required = list(SECTORS) + [benchmark]
    if any(ticker not in data or data[ticker].empty for ticker in required):
        return pd.DataFrame()
    if any(data[ticker].index.min() > pd.Timestamp(HISTORY_START) for ticker in required):
        return pd.DataFrame()
    period = pd.Timestamp(idle_dates(now=now)["expected_period"])
    latest_common = min(data[ticker].index.max() for ticker in required)
    calendar = market_calendar()
    while calendar.date_to_session(period, direction="previous") > latest_common:
        period -= pd.Timedelta(days=14)
    sampled = {
        ticker: data[ticker]["Close"].resample("336h", origin=ANCHOR, closed="right", label="right").last().dropna().loc[:period]
        for ticker in required
    }
    rows = []
    for ticker in SECTORS:
        common = sampled[ticker].index.intersection(sampled[benchmark].index)
        strength = sampled[ticker].loc[common] / sampled[benchmark].loc[common]
        ratio = (100.0 + (strength / strength.rolling(7).mean() - 1.0) * 100.0).ewm(span=7, adjust=False).mean()
        momentum = (100.0 + (ratio / ratio.shift(7) - 1.0) * 100.0).ewm(span=7, adjust=False).mean()
        valid = ratio.dropna().index.intersection(momentum.dropna().index)
        if len(valid) < 3:
            return pd.DataFrame()
        recent = valid[-max(tail_length, 2) - 1:]
        for position, date in enumerate(recent):
            ratio_value, momentum_value = float(ratio.loc[date]), float(momentum.loc[date])
            quadrant = (
                "Leading" if ratio_value >= 100 and momentum_value >= 100 else
                "Weakening" if ratio_value >= 100 else
                "Improving" if momentum_value >= 100 else "Lagging"
            )
            rows.append({"ticker": ticker, "date": date, "rs_ratio": ratio_value,
                         "rs_momentum": momentum_value, "quadrant": quadrant,
                         "is_current": position == len(recent) - 1, "tail_idx": position})
    return pd.DataFrame(rows)


def idle_decision(frame: pd.DataFrame, period) -> dict:
    current = frame.loc[frame["date"] == pd.Timestamp(period)]
    if set(current["ticker"]) != set(SECTORS) or len(current) != len(SECTORS):
        return {"ready": False, "hold": None, "reason": "All 11 sectors are required"}
    quadrants, breakouts = {}, []
    for ticker in SECTORS:
        row = current.loc[current["ticker"] == ticker].iloc[0]
        previous = frame.loc[(frame["ticker"] == ticker) & (frame["date"] < pd.Timestamp(period))].sort_values("date")
        previous_quadrant = str(previous.iloc[-1]["quadrant"]) if not previous.empty else "N/A"
        quadrants[ticker] = {"quadrant": str(row["quadrant"]), "previous_quadrant": previous_quadrant,
                             "rs_ratio": round(float(row["rs_ratio"]), 2),
                             "rs_momentum": round(float(row["rs_momentum"]), 2)}
        if previous_quadrant == "Improving" and row["quadrant"] == "Leading":
            breakouts.append(ticker)
    breadth = sum(values["quadrant"] in ("Leading", "Improving") for values in quadrants.values()) / 11
    target = "SGOV" if breadth < 0.40 else breakouts[0] if breakouts else "SPY"
    return {"ready": True, "hold": target, "breadth": breadth,
            "breakout_ticker": breakouts[0] if breakouts else None,
            "breakout_tickers": breakouts, "quadrants": quadrants,
            "rrg_asof": pd.Timestamp(period).strftime("%Y-%m-%d"),
            "reason": f"Breadth {breadth:.1%}; target {target}"}


def compute_frozen_idle(data: dict, benchmark: str = "SPY", now=None) -> dict:
    frame = frozen_rrg_frame(data, benchmark=benchmark, now=now)
    return idle_from_frame(frame, now)


def idle_from_frame(frame: pd.DataFrame, now=None) -> dict:
    if frame.empty:
        return {"ready": False, "hold": None, "breadth": 0, "breakout_ticker": None,
                "breakout_tickers": [], "quadrants": {}, "rrg_asof": None,
                "reason": "Fixed-start history for all 11 sectors and benchmark is required"}
    dates = sorted(frame["date"].unique())
    signal = idle_decision(frame, dates[-1])
    previous = idle_decision(frame, dates[-2])
    signal["previous_target"] = previous.get("hold")
    signal["target_changed"] = signal.get("hold") != previous.get("hold")
    signal.update(idle_dates(signal.get("rrg_asof"), now))
    return signal


@contextmanager
def _locked_state(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK, path.with_suffix(".lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lock_idle_evaluation(signal: dict, state_path: Path, now=None) -> dict:
    with _locked_state(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"version": VERSION, "evaluations": {}}
        if state.get("version") != VERSION:
            raise ValueError("Idle state rule version mismatch; explicit migration required")
        evaluations = state["evaluations"]
        period = signal.get("rrg_asof")
        if signal.get("ready") and period not in evaluations:
            record = dict(signal)
            record["recorded_at"] = rule_clock(now).isoformat()
            record["initial_sync"] = not evaluations and not bool(record.get("rebalance_window"))
            prior_periods = [date for date in evaluations if date < period]
            if prior_periods:
                record["previous_target"] = evaluations[max(prior_periods)]["hold"]
                record["target_changed"] = record["previous_target"] != record["hold"]
            evaluations[period] = record
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=state_path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(state, handle, indent=2)
                temporary.replace(state_path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        if not evaluations:
            return signal
        latest = max(evaluations)
        locked = dict(evaluations[latest])
        locked.update(idle_dates(latest, now))
        if not signal.get("ready") or signal.get("stale"):
            locked["data_warning"] = signal.get("reason", "New evaluation data unavailable")
            locked["rebalance_window"] = False
        locked["history"] = [{"evaluation_date": date, "target": record["hold"],
                              "previous_target": record.get("previous_target"),
                              "target_changed": bool(record.get("target_changed")),
                              "recorded_at": record.get("recorded_at"),
                              "initial_sync": bool(record.get("initial_sync")),
                              "execution_date": record.get("execution_date")}
                             for date, record in sorted(evaluations.items(), reverse=True)]
        return locked


def rsi_snapshot(qqq: pd.DataFrame, tqqq: pd.DataFrame) -> dict:
    close = qqq["Close"]
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0).ewm(alpha=0.5, adjust=False).mean()
    losses = (-delta.where(delta < 0, 0.0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + gains / losses)
    sma = close.rolling(200).mean()
    return {"qqq_rsi2": float(rsi.iloc[-1]), "qqq_price": float(close.iloc[-1]),
            "qqq_sma200": float(sma.iloc[-1]), "tqqq_price": float(tqqq["Close"].iloc[-1]),
            "signal_asof": qqq.index[-1].strftime("%Y-%m-%d"),
            "above_200": bool(close.iloc[-1] > sma.iloc[-1])}


def select_action(snapshot: dict, idle: dict, *, in_tqqq: bool | None = False,
                  current_idle: str | None = None, entry_price=None, entry_date=None, now=None) -> dict:
    moment = rule_clock(now)
    execution = next_session(snapshot["signal_asof"])
    active_rsi = moment < market_calendar().session_open(execution)
    days = held_sessions(entry_date, snapshot["signal_asof"]) if entry_date is not None else 0
    position_has_close = entry_date is None or pd.Timestamp(entry_date).normalize() <= pd.Timestamp(snapshot["signal_asof"])
    drawdown = snapshot["tqqq_price"] / entry_price - 1 if entry_price and position_has_close else None
    stop = drawdown is not None and drawdown <= -0.15
    target = idle.get("hold") if idle.get("ready") else None
    result = {"next_date": None, "next_hold": "TQQQ" if in_tqqq else current_idle,
              "action_type": "HOLD", "days_held": days,
              "tqqq_dd_pct": drawdown * 100 if drawdown is not None else None,
              "stop_hit": stop, "rule_version": VERSION}
    if in_tqqq and not position_has_close:
        result["reason"] = "New TQQQ position; await its first completed market close"
        return result
    exit_reason = None
    if in_tqqq:
        if stop:
            exit_reason = f"Stop {drawdown:+.1%} <= -15%"
        elif not snapshot["above_200"]:
            exit_reason = "QQQ <= 200SMA: trend exit"
        elif snapshot["qqq_rsi2"] > 80:
            exit_reason = f"RSI(2)={snapshot['qqq_rsi2']:.1f} > 80"
        elif days >= 21:
            exit_reason = f"Day {days} safety cap (21d actual market sessions)"
    elif in_tqqq is None and (snapshot["qqq_rsi2"] > 80 or not snapshot["above_200"]):
        exit_reason = "RSI/trend exit if TQQQ is held; position stop/time cap unavailable"
    if exit_reason:
        result.update(action_type="TQQQ_EXIT" if active_rsi else "RECONCILE_RSI",
                      next_hold=target if active_rsi else result["next_hold"],
                      next_date=execution.strftime("%Y-%m-%d") if active_rsi else None,
                      reason=exit_reason + ("; idle target unavailable" if target is None else f"; return to {target}"))
        return result
    if in_tqqq:
        result["reason"] = f"Hold TQQQ; RSI(2)={snapshot['qqq_rsi2']:.1f}, day {days}"
        return result
    if snapshot["qqq_rsi2"] < 15 and snapshot["above_200"]:
        result.update(action_type="TQQQ_ENTRY" if active_rsi else "RECONCILE_RSI",
                      next_hold="TQQQ" if active_rsi else current_idle,
                      next_date=execution.strftime("%Y-%m-%d") if active_rsi else None,
                      reason=f"RSI(2)={snapshot['qqq_rsi2']:.1f} < 15 and QQQ > 200SMA")
        return result
    if target is None:
        result.update(action_type="IDLE_UNAVAILABLE", reason="Idle data unavailable; no new idle order")
    elif idle.get("rebalance_window") and not idle.get("stale") and idle.get("target_changed") and current_idle != target:
        result.update(action_type="IDLE_REBALANCE", next_hold=target,
                      next_date=idle["execution_date"],
                      reason=f"Fixed idle update: {idle.get('previous_target')} -> {target}")
    elif current_idle is not None and current_idle != target:
        result.update(action_type="RECONCILE_IDLE",
                      reason=f"Recorded holding {current_idle} differs from evaluated {target}; no new midweek order")
    else:
        result.update(action_type="IDLE_HOLD", next_hold=current_idle or target,
                      reason=f"No new idle rotation; last evaluated target {target}")
    return result


def format_idle_evaluation(signal: dict, now=None) -> dict:
    quadrants = signal.get("quadrants", {})
    as_of = signal.get("rrg_asof")
    available = bool(as_of and set(quadrants) == set(SECTORS) and signal.get("ready", True))
    breadth = float(signal.get("breadth", 0))
    target = signal.get("hold") or ("SPY" if available else None)
    breakouts = signal.get("breakout_tickers")
    if breakouts is None:
        selected = signal.get("breakout_ticker")
        breakouts = [selected] if selected else []
    bullish = sorted(ticker for ticker, values in quadrants.items() if values.get("quadrant") in ("Leading", "Improving"))
    breadth_text = f"{len(bullish)}/{len(quadrants)} sectors Leading/Improving ({breadth:.1%})"
    if not available:
        reason = "Completed RRG data unavailable; all 11 sectors required; no new idle order."
    elif breadth < 0.40:
        reason = f"{breadth_text} < 40%; select SGOV."
        if breakouts:
            reason += f" Breakouts {', '.join(breakouts)} blocked by breadth."
    elif breakouts:
        reason = f"{breadth_text} >= 40%; select {target}. Breakouts: {', '.join(breakouts)} (strict Improving -> Leading)."
        if len(breakouts) > 1:
            reason += " Multiple candidates; fixed sector-order selection applies."
    else:
        reason = f"{breadth_text} >= 40%; no breakout. Select SPY (fixed fallback)."
    timing = idle_dates(as_of, now) if as_of is not None else idle_dates(now=now)
    actionable = bool(available and timing["rebalance_window"] and signal.get("target_changed")
                      and not timing["stale"] and not signal.get("data_warning"))
    return {
        **timing, "available": available, "as_of": as_of, "target": target,
        "reason": reason, "breadth": breadth, "breadth_pass": available and breadth >= 0.40,
        "bullish_tickers": bullish, "breakout_tickers": breakouts,
        "previous_target": signal.get("previous_target"),
        "target_changed": bool(signal.get("target_changed")), "actionable": actionable,
        "recorded_at": signal.get("recorded_at"), "data_warning": signal.get("data_warning"),
        "initial_sync": bool(signal.get("initial_sync")), "history": signal.get("history", []),
        "sectors": [{"ticker": ticker, "sector": SECTOR_NAMES.get(ticker, ticker),
                     "previous_quadrant": values.get("previous_quadrant", "N/A"),
                     "quadrant": values.get("quadrant", "N/A"), "rs_ratio": values.get("rs_ratio"),
                     "rs_momentum": values.get("rs_momentum"), "breadth_eligible": ticker in bullish,
                     "breakout": ticker in breakouts, "selected": ticker == target}
                    for ticker, values in sorted(quadrants.items())],
    }