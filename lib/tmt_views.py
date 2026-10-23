"""Shared TMT idle evaluation display."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.aa_theme import banner, esc, holding_tone
from lib.rrg import SECTOR_SPDR

ACTION_STYLES = {
    "TQQQ_ENTRY": ("Buy", "up"),
    "TQQQ_EXIT": ("Exit", "warn"),
    "IDLE_REBALANCE": ("Rotate", "info"),
}


def render_tmt_action(signal: dict) -> None:
    action = signal.get("action_type", "HOLD")
    date = signal.get("next_date")
    target = signal.get("next_hold") or "Unallocated"
    if date:
        conditional = " (conditional on position)" if not signal.get("position_aware", True) else ""
        label, tone = ACTION_STYLES.get(action, ("Action", holding_tone(target)))
        if action == "TQQQ_EXIT" and signal.get("stop_hit"):
            label, tone = "Stop", "down"
        banner(f"{date} open → {target}{conditional}", signal["reason"], tone=tone, label=label)
    elif action.startswith("RECONCILE"):
        banner("Position reconciliation required; no new order", signal["reason"], tone="warn", label="Check")
    elif action == "IDLE_UNAVAILABLE":
        banner("Idle target unavailable", signal["reason"], tone="warn", label="Data")
    elif action.startswith("IDLE"):
        banner(f"Hold {target}", "No new idle rotation.", tone=holding_tone(target), label="Hold")
    else:
        banner(f"Hold {target}", signal["reason"], tone=holding_tone(target), label="Hold")
    if not action.startswith("IDLE") and action != "RECONCILE_IDLE":
        st.caption(f"Daily signal: {signal.get('signal_asof', 'N/A')}")


def render_tmt_idle_evaluation(signal: dict, *, expanded: bool = False) -> None:
    evaluation = signal.get("idle_evaluation", {})
    if not evaluation.get("available"):
        st.warning(evaluation.get("reason", "Completed RRG evaluation unavailable."))
        return
    standby = " (standby)" if signal.get("next_hold") == "TQQQ" else ""
    breadth_text = f"{len(evaluation['bullish_tickers'])}/{len(evaluation['sectors'])} bullish ({evaluation['breadth']:.1%})"
    if not evaluation["breadth_pass"]:
        summary = f"{breadth_text} < 40%."
    elif evaluation["breakout_tickers"]:
        summary = f"{breadth_text}; breakout {evaluation['target']}."
    else:
        summary = f"{breadth_text}; no breakout."
    st.markdown(
        f"<div style='font-size:0.9rem'><b>Idle: {esc(evaluation['target'])}{standby}</b> "
        f"<span style='color:#6B6B6B'>· {esc(evaluation['as_of'])} · {esc(summary)}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Next evaluation: {evaluation['next_evaluation_date']} close (ET) | Every 2 weeks")
    if evaluation.get("data_warning"):
        st.warning(evaluation["data_warning"])
    sectors = evaluation["sectors"]
    if len(sectors) < len(SECTOR_SPDR):
        st.warning(f"Sector data: {len(sectors)}/{len(SECTOR_SPDR)} available.")
    with st.expander("Idle details", expanded=expanded):
        st.write(evaluation["reason"])
        st.caption(f"Daily signal: {signal.get('signal_asof', 'N/A')}")
        st.caption(f"Evaluation close (ET): {evaluation['evaluation_close']}")
        st.caption(f"Original execution (ET): {evaluation['execution_open']}")
        st.caption(f"Next evaluation close (ET): {evaluation['next_evaluation_close']}")
        st.caption(f"Next planned execution: {evaluation['next_execution_date']} open")
        if evaluation.get("recorded_at"):
            locked_at = pd.Timestamp(evaluation["recorded_at"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M %Z")
            st.caption(f"Locked at (ET): {locked_at}")
        if evaluation.get("initial_sync"):
            st.caption("Initial migration; past execution is not reissued.")
        st.caption(f"Leading / Improving: {', '.join(evaluation['bullish_tickers']) or 'None'}")
        st.caption(f"Breakout candidates: {', '.join(evaluation['breakout_tickers']) or 'None'}")
        st.caption("No midweek idle rotations; daily RSI/TQQQ entry and exit rules remain separate.")
        if not signal.get("position_aware", True):
            st.caption("Public snapshot: position-dependent stop/time cap cannot be assessed.")
        st.dataframe(
            pd.DataFrame(sectors), width="stretch", hide_index=True,
            column_config={
                "ticker": "Ticker", "sector": "Sector",
                "previous_quadrant": "Previous quadrant", "quadrant": "Current quadrant",
                "rs_ratio": st.column_config.NumberColumn("RS-Ratio", format="%.2f"),
                "rs_momentum": st.column_config.NumberColumn("RS-Momentum", format="%.2f"),
                "breadth_eligible": st.column_config.CheckboxColumn("Breadth eligible", disabled=True),
                "breakout": st.column_config.CheckboxColumn("Breakout", disabled=True),
                "selected": st.column_config.CheckboxColumn("Idle selected", disabled=True),
            },
        )
        history = evaluation.get("history", [])
        if history:
            st.markdown("**Update history**")
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)