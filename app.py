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


# ─── Hold'em Signal ───────────────────────────────────────────────────────────
st.header("Hold'em — 3x Leveraged Rotation")

with st.expander("📖 Strategy Rules & Decision Tree", expanded=False):
    st.markdown("""
**Core Logic:** RSI(10)-based 3-phase state machine

| Phase | Condition | Holding | Exit Trigger |
|-------|-----------|---------|--------------|
| **0 — Bull** | SPY > MA100 | TQQQ (3x Nasdaq) | RSI(TQQQ/UPRO) > 79 |
| **1 — Hedge** | RSI overbought | UVXY (1.5x VIX) | RSI < 75 OR 5 days |
| **2 — Cooldown** | Post-hedge | SGOV (T-bills) | TQQQ pullback ≥2% + close > MA20 |

**Bear Market (SPY ≤ MA100):** SQQQ (3x short Nasdaq), rotate to TLT when SQQQ RSI > 79

**Backtest (2022–2026):** 10.2x return, 85% CAGR, -58% max drawdown, 1.57 Sharpe
""")
    st.markdown("**Decision Tree:**")
    st.code("""
SPY vs MA100?
├─ SPY > MA100 (Bull Market)
│   ├─ Phase 0 (Normal): Hold TQQQ
│   │   └─ RSI(TQQQ) > 79 OR RSI(UPRO) > 79?
│   │       └─ YES → Enter Phase 1
│   │
│   ├─ Phase 1 (Hedge): Hold UVXY
│   │   └─ RSI(TQQQ) < 75 AND RSI(UPRO) < 75? OR held ≥ 5 days?
│   │       └─ YES → Enter Phase 2
│   │
│   └─ Phase 2 (Cooldown): Hold SGOV
│       └─ TQQQ pullback ≥ 2% from peak AND TQQQ > MA20?
│           └─ YES → Return to Phase 0
│
└─ SPY ≤ MA100 (Bear Market)
    └─ RSI(SQQQ) > 79?
        ├─ YES → Hold TLT
        └─ NO  → Hold SQQQ
    """, language=None)

with st.spinner("Computing Hold'em signal..."):
    holdem = compute_holdem_signal()

col1, col2, col3 = st.columns(3)
with col1:
    phase_labels = {0: "🟢 Phase 0 — Bull", 1: "🟡 Phase 1 — Hedge", 2: "🔵 Phase 2 — Cooldown"}
    st.metric("Current Phase", phase_labels.get(holdem.phase, "Unknown"))
with col2:
    hold_colors = {"TQQQ": "🟢", "UVXY": "🟡", "SGOV": "🔵", "SQQQ": "🔴", "TLT": "🟣"}
    icon = hold_colors.get(holdem.next_hold, "⚪")
    st.metric("Next Hold", f"{icon} {holdem.next_hold}")
with col3:
    st.metric("Signal Date", holdem.signal_date)

# Indicator table
st.markdown("**Key Indicators:**")
ind_cols = st.columns(5)
with ind_cols[0]:
    st.metric("SPY", f"${holdem.spy_price:.1f}", f"MA100: {holdem.spy_ma100:.1f}")
with ind_cols[1]:
    st.metric("TQQQ", f"${holdem.tqqq_price:.1f}", f"MA20: {holdem.tqqq_ma20:.1f}")
with ind_cols[2]:
    st.metric("RSI TQQQ", f"{holdem.rsi_tqqq:.1f}")
with ind_cols[3]:
    st.metric("RSI UPRO", f"{holdem.rsi_upro:.1f}")
with ind_cols[4]:
    st.metric("Market", "🐂 Bull" if holdem.is_bull else "🐻 Bear")

st.info(f"**Condition:** {holdem.key_condition}")

if holdem.cooldown_remaining > 0:
    st.warning(f"UVXY locked until {holdem.uvxy_lock_until} ({holdem.cooldown_remaining} days remaining)")

st.divider()

# ─── TMT RSI(2) Signal ────────────────────────────────────────────────────────
st.header("TMT — RSI(2) Mean Reversion")

with st.expander("📖 Strategy Rules & Decision Tree", expanded=False):
    st.markdown("""
**Core Logic:** Larry Connors RSI(2) oversold-bounce on leveraged ETF

| Condition | Action |
|-----------|--------|
| QQQ RSI(2) < 15 AND QQQ > 200SMA | BUY TQQQ |
| QQQ RSI(2) > 80 | SELL TQQQ |
| Held > 10 days | Safety exit |

**Idle Capital:** Sector dual-momentum rotation (12-month price momentum across 8 sectors)

**Backtest (2018–2026):** +1,782% total, 52% annualized, 82 trades, 72% win rate
""")
    st.markdown("**Decision Tree:**")
    st.code("""
QQQ > 200SMA? (Trend filter)
├─ NO → WAIT (no entries below long-term trend)
│
└─ YES
    └─ Currently holding TQQQ?
        ├─ NO
        │   └─ QQQ RSI(2) < 15?
        │       ├─ YES → BUY TQQQ
        │       └─ NO  → WAIT (idle capital → sector DM rotation)
        │
        └─ YES (in trade)
            ├─ QQQ RSI(2) > 80?       → SELL TQQQ
            ├─ Held ≥ 10 days?         → SELL TQQQ (safety cap)
            ├─ Day 7+ AND RSI(2) < 15? → HOLD (soft extension)
            └─ Otherwise               → HOLD
    """, language=None)

with st.spinner("Computing TMT signal..."):
    tmt = compute_tmt_signal()

col1, col2, col3, col4 = st.columns(4)
with col1:
    action_icons = {"BUY TQQQ": "🟢", "SELL TQQQ": "🔴", "WAIT": "⏸️"}
    st.metric("Signal", f"{action_icons.get(tmt.action, '')} {tmt.action}")
with col2:
    st.metric("QQQ RSI(2)", f"{tmt.qqq_rsi2:.1f}")
with col3:
    st.metric("QQQ Price", f"${tmt.qqq_price:.2f}", f"200SMA: {tmt.qqq_sma200:.2f}")
with col4:
    st.metric("Trend", "✅ Above 200SMA" if tmt.above_200 else "❌ Below 200SMA")

st.info(f"**Reason:** {tmt.reason}")

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
