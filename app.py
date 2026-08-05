"""Strategy Monitor — Public dashboard for sharing strategy signals & metrics.

No personal trade data. Safe to share with friends.
Run: streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies.holdem_signal import compute_holdem_signal, HoldemConfig
from strategies.tmt_signal import compute_tmt_signal
from lib.market_data import fetch_prices, compute_sma, compute_rsi
from lib.metrics import compute_metrics

st.set_page_config(page_title="Strategy Monitor", page_icon="📊", layout="wide")

st.title("📊 Strategy Monitor")
st.caption("Live signals & strategy metrics — no personal trade data")

def _next_trading_day() -> str:
    """Return the next trading day (skip weekends)."""
    d = dt.date.today() + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


NEXT_DATE = _next_trading_day()

# ─── Hold'em Signal ───────────────────────────────────────────────────────────
st.header("Hold'em — 3x Leveraged Rotation")

sig_config = HoldemConfig()

with st.spinner("Calculating signal..."):
    holdem = compute_holdem_signal(sig_config)

# Tomorrow's Action card
st.markdown(f"### 📅 {NEXT_DATE} → 持有 **{holdem.next_hold}**")
st.info(f"**Condition:** {holdem.key_condition}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Signal Date", holdem.signal_date)
    st.metric("Next Hold", holdem.next_hold)
with col2:
    phase_labels = {0: "🟢 Phase 0 (Normal → TQQQ)", 1: "🟡 Phase 1 (UVXY Hedge)", 2: "🔵 Phase 2 (SGOV Cooldown)"}
    st.metric("Phase", phase_labels.get(holdem.phase, "Unknown"))
    if holdem.uvxy_lock_until:
        st.metric("UVXY Lock Until", holdem.uvxy_lock_until)
with col3:
    st.metric("Market Regime", "🐂 Bull" if holdem.is_bull else "🐻 Bear")
    if holdem.cooldown_remaining > 0:
        st.metric("Cooldown Remaining", f"{holdem.cooldown_remaining} days")

with st.expander("📊 Indicator Details"):
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    with ind_col1:
        st.markdown(f"- SPY: {holdem.spy_price:.2f} (MA100: {holdem.spy_ma100:.2f}, MA50: {holdem.spy_ma50:.2f})")
        st.markdown(f"- TQQQ: {holdem.tqqq_price:.2f} (MA20: {holdem.tqqq_ma20:.2f})")
    with ind_col2:
        st.markdown(f"- RSI TQQQ: {holdem.rsi_tqqq:.1f}")
        st.markdown(f"- RSI UPRO: {holdem.rsi_upro:.1f}")
        st.markdown(f"- RSI SPY: {holdem.rsi_spy:.1f}")
    with ind_col3:
        st.markdown(f"- RSI SQQQ: {holdem.rsi_sqqq:.1f}")
        st.markdown(f"- RSI TLT: {holdem.rsi_tlt:.1f}")

with st.expander("🌳 Hold'em Decision Tree"):
    regime_color = "#16a34a" if holdem.is_bull else "#dc2626"
    regime_text = "Bull Regime" if holdem.is_bull else "Bear Regime"
    st.markdown(
        f"<div style='padding:8px 12px;border-radius:10px;background:{regime_color};color:white;display:inline-block;font-weight:600'>{regime_text} | Next Hold: {holdem.next_hold}</div>",
        unsafe_allow_html=True,
    )

    bull_overbought = (holdem.rsi_tqqq > sig_config.rsi_overbought) or (holdem.rsi_upro > sig_config.rsi_overbought)
    cooldown_done = holdem.cooldown_remaining == 0
    tqqq_below_ma20 = holdem.tqqq_ma20 > 0 and holdem.tqqq_price < holdem.tqqq_ma20
    sqqq_gt_tlt = holdem.rsi_sqqq > holdem.rsi_tlt
    spy_above_ma50 = holdem.spy_ma50 > 0 and holdem.spy_price > holdem.spy_ma50
    is_in_cash_gate = holdem.next_hold == sig_config.safe_asset and holdem.is_bull and holdem.phase == 0

    st.markdown("**Live Path Check**")
    if holdem.is_bull:
        st.markdown(f"- {'✅' if holdem.is_bull else '⬜'} SPY > MA100")
        if is_in_cash_gate:
            st.markdown("- ❌ SPY < MA50 for 2+ days → Cash (SGOV)")
        elif not spy_above_ma50:
            st.markdown("- ⚠️ SPY < MA50 (1 day, awaiting 2d confirm) → still TQQQ")
        else:
            st.markdown("- ✅ SPY > MA50 → TQQQ (3x)")
        st.markdown(f"- {'✅' if holdem.phase == 1 else '⬜'} Phase 1 (UVXY Hedge)")
        st.markdown(f"- {'✅' if holdem.phase == 2 else '⬜'} Phase 2 (Cooldown)")
        st.markdown(f"- {'✅' if bull_overbought else '⬜'} RSI overbought trigger (> {sig_config.rsi_overbought})")
        st.markdown(f"- {'✅' if cooldown_done else '⬜'} Cooldown complete")
    else:
        st.markdown(f"- {'✅' if not holdem.is_bull else '⬜'} SPY <= MA100")
        st.markdown(f"- {'✅' if holdem.rsi_tqqq < 30 else '⬜'} TQQQ RSI < 30")
        st.markdown(f"- {'✅' if holdem.rsi_spy < 31 else '⬜'} SPY RSI < 31")
        st.markdown(f"- {'✅' if tqqq_below_ma20 else '⬜'} TQQQ < MA20")
        st.markdown(f"- {'✅' if sqqq_gt_tlt else '⬜'} SQQQ RSI > TLT RSI")

    uvxy_fill = "#fde68a" if holdem.next_hold == "UVXY" else "#f8fafc"
    tqqq_bull_fill = "#bfdbfe" if holdem.next_hold == "TQQQ" and holdem.is_bull else "#f8fafc"
    cash_gate_fill = "#bbf7d0" if is_in_cash_gate else "#f8fafc"
    tqqq_bear_fill = "#bfdbfe" if holdem.next_hold == "TQQQ" and not holdem.is_bull else "#f8fafc"
    sqqq_fill = "#fecaca" if holdem.next_hold == "SQQQ" else "#f8fafc"
    tlt_fill = "#ddd6fe" if holdem.next_hold == "TLT" else "#f8fafc"
    safe_fill = "#bbf7d0" if holdem.next_hold == sig_config.safe_asset else "#f8fafc"

    tree_dot = f"""
digraph HoldemTree {{
    rankdir=LR;
    graph [bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Arial", fontsize=10];
    edge [color="#64748b", fontname="Arial", fontsize=9];

    start [label="Start", fillcolor="#e2e8f0", color="#64748b"];
    bull [label="SPY > MA100?", fillcolor="#dbeafe", color="#2563eb"];

    subgraph cluster_bull {{
        label="Bull Branch";
        color="#93c5fd";
        p1 [label="Phase 1\\nHold UVXY\\nExit: RSI<{sig_config.rsi_exit} OR {sig_config.max_uvxy_days}d cap", fillcolor="{uvxy_fill}"];
        p2 [label="Phase 2\\nHold {sig_config.safe_asset}\\nCooldown {sig_config.cooldown_days}d\\nEarly exit: pullback {sig_config.pullback_pct:.0%}\\n+ TQQQ > MA20", fillcolor="{safe_fill}"];
        overbought [label="Overbought?\\nTQQQ or UPRO RSI > {sig_config.rsi_overbought}"];
        cooldown [label="Cooldown complete?"];
        uvxy [label="Enter UVXY", fillcolor="{uvxy_fill}"];
        leverage [label="SPY < MA50\\n2 days?", fillcolor="#e2e8f0"];
        tqqq_bull [label="TQQQ (3x)", fillcolor="{tqqq_bull_fill}"];
        cash_gate [label="Cash (SGOV)\\n(50MA gate)", fillcolor="{cash_gate_fill}"];
    }}

    subgraph cluster_bear {{
        label="Bear Branch";
        color="#fca5a5";
        bear_rsi_tqqq [label="TQQQ RSI < 30?"];
        bear_rsi_spy [label="SPY RSI < 31?"];
        bear_ma20 [label="TQQQ < MA20?"];
        sqqq_vs_tlt [label="SQQQ RSI > TLT RSI?"];
        hold_sqqq [label="Hold SQQQ", fillcolor="{sqqq_fill}"];
        hold_tlt [label="Hold TLT", fillcolor="{tlt_fill}"];
        sqqq_cont [label="SQQQ RSI < 31?"];
        tqqq_bear [label="Bear default: TQQQ", fillcolor="{tqqq_bear_fill}"];
    }}

    start -> bull;
    bull -> p1 [label="Yes"];
    bull -> bear_rsi_tqqq [label="No"];

    p1 -> p2 [label="No"];
    p1 -> p1 [label="Yes"];

    p2 -> overbought [label="No"];
    p2 -> p2 [label="Yes"];

    overbought -> cooldown [label="Yes"];
    overbought -> leverage [label="No"];
    leverage -> tqqq_bull [label="No (<2d below)"];
    leverage -> cash_gate [label="Yes (2d+ below)"];
    cooldown -> uvxy [label="Yes"];
    cooldown -> p2 [label="No"];

    bear_rsi_tqqq -> tqqq_bull [label="Yes"];
    bear_rsi_tqqq -> bear_rsi_spy [label="No"];
    bear_rsi_spy -> tqqq_bull [label="Yes"];
    bear_rsi_spy -> bear_ma20 [label="No"];
    bear_ma20 -> sqqq_vs_tlt [label="Yes"];
    bear_ma20 -> sqqq_cont [label="No"];
    sqqq_vs_tlt -> hold_sqqq [label="Yes"];
    sqqq_vs_tlt -> hold_tlt [label="No"];
    sqqq_cont -> hold_sqqq [label="Yes"];
    sqqq_cont -> tqqq_bear [label="No"];
}}
"""
    st.graphviz_chart(tree_dot, use_container_width=True)
    st.caption("Styled tree + live path based on current signal values.")

st.divider()

# ─── TMT RSI(2) Signal ────────────────────────────────────────────────────────
st.header("TMT — RSI(2) Mean Reversion")

with st.spinner("Computing TMT signal..."):
    tmt = compute_tmt_signal()

# Tomorrow's Action card
action = tmt.action
if action == "BUY TQQQ":
    st.warning(f"### 📅 {NEXT_DATE} → **{action}**\n_{tmt.reason}_")
elif action == "SELL TQQQ":
    st.warning(f"### 📅 {NEXT_DATE} → **{action}**\n_{tmt.reason}_")
else:
    st.info(f"### 📅 {NEXT_DATE} → **WAIT**\n_{tmt.reason}_")

m1, m2, m3 = st.columns(3)
with m1:
    rsi_val = tmt.qqq_rsi2
    st.metric("QQQ RSI(2)", f"{rsi_val:.1f}",
              delta="OVERSOLD" if rsi_val < 15 else ("EXIT" if rsi_val > 80 else "neutral"),
              delta_color="inverse" if rsi_val < 15 else ("normal" if rsi_val > 80 else "off"))
with m2:
    st.metric("TQQQ", f"${tmt.tqqq_price:.2f}")
with m3:
    st.metric("QQQ", f"${tmt.qqq_price:.2f}", f"200SMA: {tmt.qqq_sma200:.2f}")

with st.expander("🌳 TMT Decision Tree"):
    qqq_r = tmt.qqq_rsi2
    qqq_above = tmt.above_200

    st.markdown("**Live Path Check:**")
    st.markdown(f"- {'✅' if qqq_r < 15 and qqq_above else '⬜'} QQQ RSI(2) < 15 (current: {qqq_r:.1f}) AND > 200SMA → BUY TQQQ")
    st.markdown(f"- {'✅' if qqq_r > 80 else '⬜'} QQQ RSI(2) > 80 → SELL TQQQ")
    st.markdown(f"- {'✅' if not qqq_above else '⬜'} QQQ below 200SMA → no entries")

    buy_fill = "#bbf7d0" if action == "BUY TQQQ" else "#f8fafc"
    sell_fill = "#fecaca" if action == "SELL TQQQ" else "#f8fafc"
    wait_fill = "#bfdbfe" if action == "WAIT" and not qqq_above else "#f8fafc"

    tree_dot = f"""
digraph TMTTree {{
    rankdir=TB;
    graph [bgcolor="white", pad="0.3", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Arial", fontsize=11];
    edge [color="#64748b", fontname="Arial", fontsize=10];

    start [label="QQQ RSI(2) = {qqq_r:.1f}", fillcolor="#f8fafc", color="#94a3b8"];
    oversold [label="RSI(2) < 15?", fillcolor="#f8fafc", color="#94a3b8"];
    trend [label="QQQ > 200MA?\\n{'YES' if qqq_above else 'NO'}", fillcolor="#f8fafc", color="#94a3b8"];
    overbought [label="RSI(2) > 80?", fillcolor="#f8fafc", color="#94a3b8"];
    buy [label="BUY TQQQ\\n(soft 7d, cap 21d)", fillcolor="{buy_fill}", color="#94a3b8"];
    sell [label="SELL TQQQ\\nreturn to idle", fillcolor="{sell_fill}", color="#94a3b8"];
    wait_notrend [label="WAIT\\n(no trend support)", fillcolor="{wait_fill}", color="#94a3b8"];
    idle [label="WAIT / Idle\\n(sector DM rotation)", fillcolor="#e2e8f0", color="#94a3b8"];

    start -> oversold;
    start -> overbought;
    oversold -> trend [label="Yes"];
    oversold -> idle [label="No"];
    trend -> buy [label="Yes"];
    trend -> wait_notrend [label="No"];
    overbought -> sell [label="Yes"];
    overbought -> idle [label="No"];
}}
"""
    st.graphviz_chart(tree_dot, use_container_width=True)
    st.caption("Priority: RSI(2) signal > idle rotation. Idle capital uses sector dual-momentum.")

st.divider()

# ─── Backtest Metrics (simulated from price data) ────────────────────────────
st.header("📈 Strategy Backtests")
st.caption("Simulated equity curves using historical data (no real trades)")

# Date range picker (default: YTD)
today = dt.date.today()
ytd_start = dt.date(today.year, 1, 1)
date_cols = st.columns([1, 1, 2])
with date_cols[0]:
    bt_start = st.date_input("Start Date", value=ytd_start, min_value=dt.date(2018, 1, 1), max_value=today)
with date_cols[1]:
    bt_end = st.date_input("End Date", value=today, min_value=dt.date(2018, 1, 1), max_value=today)

@st.cache_data(ttl=3600)
def run_holdem_backtest(start: str, end: str) -> pd.Series:
    """Simple Hold'em backtest: TQQQ buy & hold when SPY > MA100, else cash."""
    from lib.market_data import fetch_multiple
    data = fetch_multiple(["SPY", "TQQQ", "SQQQ"], period="5y")
    if "SPY" not in data or "TQQQ" not in data:
        return pd.Series(dtype=float)

    spy = data["SPY"]["Close"]
    tqqq = data["TQQQ"]["Close"]
    sqqq = data["SQQQ"]["Close"] if "SQQQ" in data else None
    ma100 = spy.rolling(100).mean()

    # Align
    idx = spy.index.intersection(tqqq.index)
    spy = spy.reindex(idx)
    tqqq = tqqq.reindex(idx)
    ma100 = ma100.reindex(idx)
    if sqqq is not None:
        sqqq = sqqq.reindex(idx)

    # Filter to date range
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    idx = idx[mask]
    spy = spy.reindex(idx)
    tqqq = tqqq.reindex(idx)
    ma100 = ma100.reindex(idx)
    if sqqq is not None:
        sqqq = sqqq.reindex(idx)

    nav = 100000.0
    curve = []
    for i in range(1, len(idx)):
        if pd.isna(ma100.iloc[i]):
            curve.append(nav)
            continue
        if spy.iloc[i - 1] > ma100.iloc[i - 1]:
            ret = tqqq.iloc[i] / tqqq.iloc[i - 1] - 1
        elif sqqq is not None:
            ret = sqqq.iloc[i] / sqqq.iloc[i - 1] - 1
        else:
            ret = 0
        nav *= (1 + ret)
        curve.append(nav)

    return pd.Series([100000.0] + curve, index=idx)


@st.cache_data(ttl=3600)
def run_tmt_backtest(start: str, end: str) -> pd.Series:
    """Simple RSI(2) backtest on QQQ/TQQQ."""
    from strategies.tmt_signal import compute_rsi2
    from lib.market_data import fetch_prices, compute_sma

    qqq = fetch_prices("QQQ", period="5y")
    tqqq = fetch_prices("TQQQ", period="5y")
    if qqq.empty or tqqq.empty:
        return pd.Series(dtype=float)

    idx = qqq.index.intersection(tqqq.index)
    qqq_c = qqq["Close"].reindex(idx)
    tqqq_c = tqqq["Close"].reindex(idx)
    rsi2 = compute_rsi2(qqq_c)
    sma200 = compute_sma(qqq_c, 200)

    # Filter to date range
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    idx = idx[mask]
    qqq_c = qqq_c.reindex(idx)
    tqqq_c = tqqq_c.reindex(idx)
    rsi2 = rsi2.reindex(idx)
    sma200 = sma200.reindex(idx)

    nav = 100000.0
    in_trade = False
    days_held = 0
    curve = []

    for i in range(1, len(idx)):
        if pd.isna(sma200.iloc[i]) or pd.isna(rsi2.iloc[i]):
            curve.append(nav)
            continue

        if in_trade:
            ret = tqqq_c.iloc[i] / tqqq_c.iloc[i - 1] - 1
            nav *= (1 + ret)
            days_held += 1
            if rsi2.iloc[i] > 80 or days_held >= 10:
                in_trade = False
                days_held = 0
        else:
            if rsi2.iloc[i] < 15 and qqq_c.iloc[i] > sma200.iloc[i]:
                in_trade = True
                days_held = 0

        curve.append(nav)

    return pd.Series([100000.0] + curve, index=idx)


bt_col1, bt_col2 = st.columns(2)

bt_start_str = str(bt_start)
bt_end_str = str(bt_end)

with bt_col1:
    st.subheader("Hold'em (simplified)")
    holdem_curve = run_holdem_backtest(bt_start_str, bt_end_str)
    if not holdem_curve.empty:
        st.line_chart(holdem_curve, use_container_width=True)
        m = compute_metrics(holdem_curve)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Ann. Return", f"{m['annualized_return']*100:.1f}%")
        mc2.metric("Max DD", f"{m['max_drawdown']*100:.1f}%")
        mc3.metric("Sharpe", f"{m['sharpe']:.2f}")

with bt_col2:
    st.subheader("TMT RSI(2)")
    tmt_curve = run_tmt_backtest(bt_start_str, bt_end_str)
    if not tmt_curve.empty:
        st.line_chart(tmt_curve, use_container_width=True)
        m = compute_metrics(tmt_curve)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Ann. Return", f"{m['annualized_return']*100:.1f}%")
        mc2.metric("Max DD", f"{m['max_drawdown']*100:.1f}%")
        mc3.metric("Sharpe", f"{m['sharpe']:.2f}")

st.divider()
st.caption(f"Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | Data from Yahoo Finance")
