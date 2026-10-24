"""Strategy Monitor - stripped local dashboard (no personal transaction/account data)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.aa_theme import (
    STRATEGY_COLORS,
    SERIES,
    apply_theme,
    banner,
    card,
    chart,
    holding_tone,
    kpis,
    page_header,
    rows,
)
from lib.gex import GEX_TICKERS, apply_filters, compute_gex
from lib.market_data import compute_sma, fetch_multiple, fetch_prices
from lib.rrg import (
    RRG_BASKET,
    RRG_BENCHMARK,
    SECTOR_MAP,
    SECTOR_SPDR,
    compute_idle_signal,
    compute_idle_rrg,
    compute_rrg,
    detect_signals,
    get_rrg_ranking,
    resample_biweekly,
)
from strategies.holdem_signal import HoldemConfig, compute_holdem_signal
from strategies.tmt_signal import IDLE_STATE_PATH, compute_tmt_snapshot
from lib.tmt_rules import HISTORY_START
from lib.tmt_views import render_tmt_action, render_tmt_idle_evaluation


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Strategy Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
apply_theme("Strategy Monitor")


def next_trading_day() -> str:
    d = dt.date.today() + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def get_tmt_signal_snapshot() -> dict:
    """Frozen public snapshot, without personal position data."""
    return compute_tmt_snapshot()


def holdem_state_label(signal) -> str:
    if not signal.is_bull:
        return "Bear regime"
    if signal.phase == 1:
        return "UVXY hedge"
    if signal.phase == 2:
        return "SGOV cooldown"
    if signal.chop_override:
        return "Chop mode"
    if signal.next_hold == "SGOV":
        return "Cash gate"
    return "Bull trend"


def render_rrg_map(expander_title: str, key_prefix: str) -> None:
    with st.expander(expander_title, expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            tail_weeks = st.selectbox(
                "Tail",
                [0, 4, 8, 13],
                index=2,
                format_func=lambda x: f"{x}w" if x > 0 else "Now",
                key=f"{key_prefix}_tail",
            )
        with c2:
            show_filter = st.selectbox("Show", ["All", "Top 10", "Bottom 10"], index=0, key=f"{key_prefix}_show")

        raw_data = fetch_multiple(list(set(RRG_BASKET + [RRG_BENCHMARK])), start=HISTORY_START)
        if not raw_data or RRG_BENCHMARK not in raw_data:
            st.info("RRG data unavailable now")
            return

        tail_len = max(tail_weeks, 1)
        rrg_df = compute_idle_rrg(raw_data, RRG_BENCHMARK, tail_length=tail_len)
        if rrg_df.empty:
            st.info("RRG has insufficient data for plotting")
            return

        ranking = get_rrg_ranking(rrg_df)
        if show_filter == "Top 10":
            show_tickers = ranking.head(10)["ticker"].tolist()
        elif show_filter == "Bottom 10":
            show_tickers = ranking.tail(10)["ticker"].tolist()
        else:
            show_tickers = ranking["ticker"].tolist()

        plot_df = rrg_df[rrg_df["ticker"].isin(show_tickers)].copy()
        quadrant_colors = {
            "Leading": SERIES["green"],
            "Improving": SERIES["blue"],
            "Weakening": SERIES["yellow"],
            "Lagging": "#D43C3C",
        }

        fig = go.Figure()
        fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
        fig.add_vline(x=100, line_dash="dot", line_color="#94a3b8", line_width=1)

        for ticker in show_tickers:
            tdf = plot_df[plot_df["ticker"] == ticker].sort_values("tail_idx")
            if tdf.empty:
                continue
            current = tdf[tdf["is_current"]].iloc[0]
            color = quadrant_colors.get(current["quadrant"], "#64748b")
            if tail_weeks > 0 and len(tdf) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=tdf["rs_ratio"],
                        y=tdf["rs_momentum"],
                        mode="lines+markers",
                        line=dict(color=color, width=1.5),
                        opacity=0.5,
                        showlegend=False,
                        marker=dict(size=4, color=color),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[current["rs_ratio"]],
                    y=[current["rs_momentum"]],
                    mode="markers+text",
                    marker=dict(size=9, color=color, line=dict(width=1, color="white")),
                    text=[ticker],
                    textposition="top center",
                    name=ticker,
                )
            )

        fig.update_layout(margin=dict(t=20, b=20, l=30, r=30), xaxis_title="RS-Ratio", yaxis_title="RS-Momentum", showlegend=False)
        chart(fig, height=500)

        st.dataframe(
            ranking[["rank", "ticker", "label", "rs_ratio", "rs_momentum", "quadrant"]].style.format(
                {"rs_ratio": "{:.2f}", "rs_momentum": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
            height=300,
        )

        signals = detect_signals(rrg_df)
        if signals:
            for sig in signals[:5]:
                st.caption(f"{sig['signal']}: {sig['ticker']} {sig['transition']}")


page = st.sidebar.radio("Navigation", ["Overview", "Hold'em", "TMT", "GEX Filter", "SPX BWB"], label_visibility="collapsed")

if st.sidebar.button("Refresh prices", icon=":material/refresh:", width="stretch"):
    st.cache_data.clear()
    cache_dir = BASE_DIR / ".cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            f.unlink()
    st.rerun()


if page == "Overview":
    page_header("Overview", "Public strategy signals · no personal positions", f"Updated {dt.datetime.now():%Y-%m-%d %H:%M}")
    st.subheader("Next session")
    c1, c2 = st.columns(2)

    with c1, card("Hold'em", STRATEGY_COLORS["Hold'em"], "SPY regime state machine · chop mode"):
        try:
            sig = compute_holdem_signal(HoldemConfig())
            banner(f"Hold {sig.next_hold}", sig.key_condition, tone=holding_tone(sig.next_hold), label=holdem_state_label(sig))
            rows([
                ("Signal date", sig.signal_date),
                ("Regime", "Bull" if sig.is_bull else "Bear"),
                ("Chop mode", f"{'ON' if sig.chop_on else 'OFF'} · QQQ vol20 {sig.qqq_vol20:.1%}"),
                ("RSI(2) leg", "Active" if sig.rsi2_leg else "Inactive"),
            ])
        except Exception as e:
            st.warning(f"Hold'em unavailable: {e}")

    with c2, card("TMT", STRATEGY_COLORS["TMT"], "RSI(2) dip-buy · fixed biweekly idle rotation"):
        try:
            tmt = get_tmt_signal_snapshot()
            if tmt["ready"]:
                render_tmt_action(tmt)
            else:
                st.warning(tmt["reason"])
            render_tmt_idle_evaluation(tmt)
        except Exception as e:
            st.warning(f"TMT unavailable: {e}")

    st.subheader("Sector rotation")
    try:
        rrg_raw = fetch_multiple(list(set(RRG_BASKET + [RRG_BENCHMARK])), start=HISTORY_START)
        rrg_df = compute_idle_rrg(rrg_raw, RRG_BENCHMARK, tail_length=2)
        ranking = get_rrg_ranking(rrg_df)
        if not ranking.empty:
            qc = ranking["quadrant"].value_counts()
            top = ranking.iloc[0]
            bottom = ranking.iloc[-1]
            kpis([
                {"label": "Leading", "value": str(qc.get("Leading", 0))},
                {"label": "Improving", "value": str(qc.get("Improving", 0))},
                {"label": "Weakening", "value": str(qc.get("Weakening", 0))},
                {"label": "Lagging", "value": str(qc.get("Lagging", 0))},
                {"label": "Top / bottom", "value": f"{top['ticker']} / {bottom['ticker']}", "sub": f"{top['label']} / {bottom['label']}"},
            ])
            st.caption("Completed fixed biweekly RRG vs SPY")
    except Exception as e:
        st.caption(f"RRG unavailable: {e}")

elif page == "Hold'em":
    page_header("Hold'em", "SPY regime state machine · TQQQ / UVXY / SGOV rotation")
    try:
        signal = compute_holdem_signal(HoldemConfig())
        state = holdem_state_label(signal)
        color = STRATEGY_COLORS["Hold'em"]
        left, right = st.columns([3, 2], gap="medium")
        with left, card("Next session", color, f"Signal from {signal.signal_date} close · executes at next open"):
            banner(f"Hold {signal.next_hold}", signal.key_condition, tone=holding_tone(signal.next_hold), label=state)
            facts = [
                ("Regime", f"{'Bull' if signal.is_bull else 'Bear'} · SPY {signal.spy_price:,.2f} vs MA100 {signal.spy_ma100:,.2f}"),
                ("Cash gate", f"SPY {signal.spy_price:,.2f} vs MA50 {signal.spy_ma50:,.2f}"),
                ("Phase", f"{signal.phase} · {state}"),
            ]
            if signal.phase == 2:
                facts.append(("Cooldown", f"{signal.cooldown_remaining} days left"))
            if signal.uvxy_lock_until:
                facts.append(("UVXY lock until", signal.uvxy_lock_until))
            facts.append(("Chop mode", f"{'ON' if signal.chop_on else 'OFF'} · QQQ vol20 {signal.qqq_vol20:.1%} (on >25%, off <20%)"))
            facts.append(("RSI(2) leg", f"{'Active' if signal.rsi2_leg else 'Inactive'} · QQQ RSI2 {signal.qqq_rsi2:.1f}"))
            rows(facts)
        with right, card("Indicators", color, "Daily close values driving the state machine"):
            kpis(
                [
                    {"label": "TQQQ RSI(10)", "value": f"{signal.rsi_tqqq:.1f}", "sub": "Hedge > 79"},
                    {"label": "UPRO RSI(10)", "value": f"{signal.rsi_upro:.1f}", "sub": "Hedge > 79"},
                    {"label": "QQQ vol20", "value": f"{signal.qqq_vol20:.1%}", "sub": "Chop > 25% · off < 20%"},
                    {"label": "QQQ RSI(2)", "value": f"{signal.qqq_rsi2:.1f}", "sub": "Leg entry < 15"},
                    {"label": "TQQQ", "value": f"${signal.tqqq_price:,.2f}", "sub": f"MA20 ${signal.tqqq_ma20:,.2f}"},
                    {"label": "SPY RSI(10)", "value": f"{signal.rsi_spy:.1f}", "sub": "Bear rebound < 31"},
                ],
                columns=2,
                flat=True,
                compact=True,
            )

        with st.expander("Indicator details"):
            rows([
                ("SPY MA100 / MA50", f"{signal.spy_ma100:.2f} / {signal.spy_ma50:.2f}"),
                ("TQQQ MA20", f"{signal.tqqq_ma20:.2f}"),
                ("RSI SQQQ / TLT", f"{signal.rsi_sqqq:.1f} / {signal.rsi_tlt:.1f}"),
            ])

        with st.expander("Decision path", expanded=False):
            bull_overbought = signal.rsi_tqqq > 79 or signal.rsi_upro > 79
            tqqq_below_ma20 = signal.tqqq_ma20 > 0 and signal.tqqq_price < signal.tqqq_ma20
            sqqq_gt_tlt = signal.rsi_sqqq > signal.rsi_tlt
            spy_above_ma50 = signal.spy_ma50 > 0 and signal.spy_price > signal.spy_ma50
            is_cash_gate = signal.next_hold == "SGOV" and signal.is_bull and signal.phase == 0 and not signal.chop_override

            st.markdown("**Live Path Check**")
            if signal.is_bull:
                st.markdown("- {} SPY > MA100".format("OK" if signal.is_bull else "NO"))
                if is_cash_gate:
                    st.markdown("- NO SPY < MA50 for 2+ days -> SGOV cash gate")
                elif not spy_above_ma50:
                    st.markdown("- WARN SPY < MA50 for 1 day -> await confirmation")
                else:
                    st.markdown("- OK SPY > MA50 -> TQQQ")
                if signal.chop_override:
                    st.markdown(f"- CHOP QQQ vol20 {signal.qqq_vol20:.1%} -> {signal.next_hold} (RSI(2) leg {'active' if signal.rsi2_leg else 'inactive'})")
                st.markdown(f"- {'OK' if signal.phase == 1 else 'NO'} Phase 1 UVXY hedge")
                st.markdown(f"- {'OK' if signal.phase == 2 else 'NO'} Phase 2 SGOV cooldown")
                st.markdown(f"- {'OK' if bull_overbought else 'NO'} RSI overbought trigger (>79)")
            else:
                st.markdown("- OK SPY <= MA100")
                st.markdown(f"- {'OK' if signal.rsi_tqqq < 30 else 'NO'} TQQQ RSI < 30")
                st.markdown(f"- {'OK' if signal.rsi_spy < 31 else 'NO'} SPY RSI < 31")
                st.markdown(f"- {'OK' if tqqq_below_ma20 else 'NO'} TQQQ < MA20")
                st.markdown(f"- {'OK' if sqqq_gt_tlt else 'NO'} SQQQ RSI > TLT RSI")

            tree_dot = f"""
digraph HoldemTree {{
    rankdir=LR;
    graph [bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Arial", fontsize=10];
    edge [color="#64748b", fontname="Arial", fontsize=9];

    start [label="Start", fillcolor="#e2e8f0"];
    bull [label="SPY > MA100?", fillcolor="#dbeafe", color="#2563eb"];
    p1 [label="Phase 1\\nHold UVXY\\nExit: RSI < 75 or 5d cap"];
    p2 [label="Phase 2\\nHold SGOV\\nCooldown 10d\\nEarly exit: pullback 2% + TQQQ > MA20"];
    overbought [label="Overbought?\\nTQQQ or UPRO RSI > 79"];
    cooldown [label="Cooldown complete?"];
    uvxy [label="Enter UVXY"];
    gate [label="SPY < MA50\\n2 days?"];
    chop [label="Chop mode?\\nQQQ vol20 >25% on, <20% off"];
    leg [label="RSI(2) leg active?\\nQQQ RSI2 < 15 & > 200SMA"];
    chop_cash [label="Cash (SGOV)\\nchop mode"];
    tqqq [label="TQQQ (3x)"];
    cash_gate [label="Cash (SGOV)\\n50MA gate"];
    bear_rsi_tqqq [label="TQQQ RSI < 30?"];
    bear_rsi_spy [label="SPY RSI < 31?"];
    bear_ma20 [label="TQQQ < MA20?"];
    sqqq_vs_tlt [label="SQQQ RSI > TLT RSI?"];
    hold_sqqq [label="Hold SQQQ"];
    hold_tlt [label="Hold TLT"];
    sqqq_cont [label="SQQQ RSI < 31?"];
    tqqq_bear [label="Bear default: TQQQ"];

    start -> bull;
    bull -> p1 [label="Yes, phase 1"];
    bull -> bear_rsi_tqqq [label="No"];
    p1 -> p2 [label="Exit"];
    p1 -> p1 [label="Hold"];
    p2 -> overbought [label="No"];
    p2 -> p2 [label="Yes"];
    overbought -> cooldown [label="Yes"];
    overbought -> gate [label="No"];
    gate -> chop [label="No (<2d below)"];
    chop -> tqqq [label="No"];
    chop -> leg [label="Yes"];
    leg -> tqqq [label="Yes"];
    leg -> chop_cash [label="No"];
    gate -> cash_gate [label="Yes (2d+ below)"];
    cooldown -> uvxy [label="Yes"];
    cooldown -> p2 [label="No"];
    bear_rsi_tqqq -> tqqq_bear [label="Yes"];
    bear_rsi_tqqq -> bear_rsi_spy [label="No"];
    bear_rsi_spy -> tqqq_bear [label="Yes"];
    bear_rsi_spy -> bear_ma20 [label="No"];
    bear_ma20 -> sqqq_vs_tlt [label="Yes"];
    bear_ma20 -> sqqq_cont [label="No"];
    sqqq_vs_tlt -> hold_sqqq [label="Yes"];
    sqqq_vs_tlt -> hold_tlt [label="No"];
    sqqq_cont -> hold_sqqq [label="Yes"];
    sqqq_cont -> tqqq_bear [label="No"];
}}
"""
            st.graphviz_chart(tree_dot, width="stretch")

    except Exception as e:
        st.error(f"Hold'em signal failed: {e}")

elif page == "TMT":
    page_header("TMT", "QQQ RSI(2) dip-buy into TQQQ · fixed biweekly RRG idle rotation · public snapshot")
    try:
        tmt = get_tmt_signal_snapshot()
        if not tmt["ready"]:
            st.warning(tmt["reason"])
            render_tmt_idle_evaluation(tmt)
        else:
            color = STRATEGY_COLORS["TMT"]
            left, right = st.columns([3, 2], gap="medium")
            with left, card("Next session", color, f"Daily signal {tmt.get('signal_asof', 'N/A')} · executes at next open"):
                render_tmt_action(tmt)
                render_tmt_idle_evaluation(tmt)
            with right, card("Indicators", color, "Bull: entry < 15 · exit > 80 · Bear sniper: entry < 5 · exit > 60"):
                kpis(
                    [
                        {"label": "QQQ RSI(2)", "value": f"{tmt['qqq_rsi2']:.1f}"},
                        {"label": "QQQ vs 200SMA", "value": f"${tmt['qqq_price']:,.2f}",
                         "sub": f"SMA ${tmt['qqq_sma200']:,.2f} · {'above (bull leg)' if tmt['above_200'] else 'below (bear sniper)'}"},
                        {"label": "TQQQ", "value": f"${tmt['tqqq_price']:,.2f}"},
                        {"label": f"Idle {tmt['dm_hold']}", "value": f"{tmt['breadth']}%", "sub": "Breadth"},
                    ],
                    columns=2,
                    flat=True,
                    compact=True,
                )

            with st.expander("Decision path", expanded=False):
                st.markdown("**Public signal path (position state is intentionally omitted)**")
                st.markdown(
                    "- RSI(2) < 15 and QQQ > 200SMA -> BUY TQQQ (bull leg)\n"
                    "- RSI(2) > 80 or QQQ <= 200SMA -> EXIT a bull-leg TQQQ if held\n"
                    "- Bear sniper: RSI(2) < 5 and QQQ <= 200SMA -> BUY TQQQ; exit RSI(2) > 60, 5 market sessions or -10%, no trend exit\n"
                    "- Idle rotation only after a fixed 2-week evaluation with a changed target\n"
                    "- Position-dependent stops (-15% / -10%) and caps (21 / 5 sessions) require holding data"
                )
                st.caption(f"Idle hold: {tmt['dm_hold']} | {tmt['idle_reason']}")

                qqq_r = tmt["qqq_rsi2"]
                qqq_above = tmt["above_200"]
                breadth_pct = tmt["breadth"]
                breakout_ticker = tmt["breakout_ticker"]
                breakout_label = breakout_ticker or "-"
                buy_fill = "#bbf7d0" if tmt["next_hold"] == "TQQQ" else "#f8fafc"
                sell_fill = "#fecaca" if qqq_r > 80 else "#f8fafc"
                idle_fill = "#ddd6fe" if tmt["next_hold"] == "SGOV" else "#f8fafc"
                sector_fill = "#bfdbfe" if tmt["next_hold"] not in ("SGOV", "SPY", "TQQQ") else "#f8fafc"
                notrend_fill = "#bfdbfe" if not qqq_above and 5 <= qqq_r < 15 else "#f8fafc"
                bear_fill = "#bbf7d0" if not qqq_above and qqq_r < 5 else "#f8fafc"
                tree_dot = f"""
digraph TMTTree {{
    rankdir=TB;
    graph [bgcolor="white", pad="0.3", nodesep="0.5", ranksep="0.6"];
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Arial", fontsize=11];
    edge [color="#64748b", fontname="Arial", fontsize=10];

    start [label="QQQ RSI(2) = {qqq_r:.1f}"];
    oversold [label="RSI(2) < 15?"];
    trend [label="QQQ > 200MA?\\n{'YES' if qqq_above else 'NO'}"];
    overbought [label="RSI(2) > 80?"];
    buy [label="BUY TQQQ if flat\\n(cap 21 market sessions)", fillcolor="{buy_fill}"];
    sell [label="EXIT TQQQ\\nreturn to idle", fillcolor="{sell_fill}"];
    wait_notrend [label="WAIT\\n(no trend support)", fillcolor="{notrend_fill}"];
    bear_check [label="RSI(2) < 5?\\n(bear sniper)"];
    bear_buy [label="BUY TQQQ if flat\\nexit RSI>60 / 5 sessions / -10%", fillcolor="{bear_fill}"];

    subgraph cluster_idle {{
        label="RRG Idle (fixed 2-week close)\\nBreadth: {breadth_pct}%";
        color="#94a3b8";
        breadth_check [label="Breadth >= 40%?"];
        breakout_check [label="Sector breakout?\\n(Improving -> Leading)"];
        hold_sector [label="HOLD {breakout_label}\\n(breakout sector)", fillcolor="{sector_fill}"];
        hold_sgov [label="HOLD SGOV\\n(risk-off)", fillcolor="{idle_fill}"];
        hold_idle [label="HOLD {tmt['dm_hold']}\\n(no breakout)"];
        breadth_check -> hold_sgov [label="No (<40%)"];
        breadth_check -> breakout_check [label="Yes"];
        breakout_check -> hold_sector [label="Yes"];
        breakout_check -> hold_idle [label="No breakout"];
    }}

    start -> oversold;
    start -> overbought;
    oversold -> trend [label="Yes"];
    oversold -> breadth_check [label="No (idle)"];
    trend -> buy [label="Yes"];
    trend -> bear_check [label="No"];
    bear_check -> bear_buy [label="Yes"];
    bear_check -> wait_notrend [label="No"];
    overbought -> sell [label="Yes"];
    overbought -> breadth_check [label="No"];
}}
"""
                st.graphviz_chart(tree_dot, width="stretch")

    except Exception as e:
        st.error(f"TMT signal failed: {e}")

    st.subheader("Sector rotation")
    render_rrg_map("Sector rotation map (RRG)", "tmt_rrg")

elif page == "GEX Filter":
    page_header("GEX Filter", "Gamma exposure scan · 38 tickers · 8 filters · public, no account inputs")

    rrg_quadrants = {}
    try:
        idle_data = fetch_multiple(list(set(SECTOR_SPDR + ["SPY"])), start=HISTORY_START)
        idle_result = compute_idle_signal(idle_data, "SPY", state_path=IDLE_STATE_PATH)
        rrg_quadrants = idle_result.get("quadrants", {})
    except Exception:
        rrg_quadrants = {}

    if st.button("Run GEX scan", type="primary", icon=":material/radar:"):
        results = []
        failed = []
        progress = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(GEX_TICKERS):
            status_text.text(f"Scanning {ticker}... ({i + 1}/{len(GEX_TICKERS)})")
            progress.progress((i + 1) / len(GEX_TICKERS))
            try:
                gex = compute_gex(ticker)
                if gex:
                    filt = apply_filters(gex, rrg_quadrants)
                    results.append({**gex, **filt})
                else:
                    failed.append({"ticker": ticker, "reason": "No options data"})
            except Exception as e:
                failed.append({"ticker": ticker, "reason": str(e)[:80]})

        progress.empty()
        status_text.empty()

        confirmed = [r for r in results if r["status"] == "CONFIRMED"]
        blocked = [r for r in results if r["status"] == "BLOCKED"]

        kpis([
            {"label": "Confirmed", "value": str(len(confirmed)), "tone": "up" if confirmed else None},
            {"label": "Blocked", "value": str(len(blocked))},
            {"label": "Failed", "value": str(len(failed)), "tone": "down" if failed else None},
        ])

        if confirmed:
            st.subheader("Confirmed Setups")
            for r in confirmed:
                with st.expander(f"{r['ticker']} · R/R {r['rr']:.2f} · +{r['upside_pct']:.1f}%"):
                    kpis([
                        {"label": "Spot", "value": f"${r['spot']}"},
                        {"label": "MaxGEX", "value": f"${r['max_gex']}"},
                        {"label": "Put wall", "value": f"{r['put_wall']}"},
                        {"label": "Call wall", "value": f"{r['call_wall']}"},
                        {"label": "Net GEX", "value": f"{r['net_gex_sign']}"},
                    ], compact=True)
                    rows([(f"{'✓' if f['pass'] else '✗'} {f['label']}", f["detail"], "up" if f["pass"] else "down")
                          for f in r["filters"].values()])

        table_rows = []
        for r in sorted(results, key=lambda x: x["passed"], reverse=True):
            table_rows.append(
                {
                    "Ticker": r["ticker"],
                    "Status": r["status"],
                    "Filters": f"{r['passed']}/{r['total']}",
                    "Spot": r["spot"],
                    "MaxGEX": r["max_gex"],
                    "Upside%": r["upside_pct"],
                    "R/R": r["rr"],
                    "NetGEX": r["net_gex_sign"],
                }
            )
        for f in failed:
            table_rows.append(
                {
                    "Ticker": f["ticker"],
                    "Status": "FAILED",
                    "Filters": "-",
                    "Spot": None,
                    "MaxGEX": None,
                    "Upside%": None,
                    "R/R": None,
                    "NetGEX": f["reason"],
                }
            )

        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True, height=550)
    else:
        st.caption("Run the scan to evaluate all tickers with 8 filters.")

elif page == "SPX BWB":
    page_header("SPX BWB Daily Playbook", "Signal and structure guidance only · no position tracking")

    import yfinance as yf

    @st.cache_data(ttl=300)
    def get_spx_vix() -> dict:
        spx_t = yf.Ticker("^GSPC")
        vix_t = yf.Ticker("^VIX")
        spx_info = spx_t.fast_info
        vix_info = vix_t.fast_info
        spx_price = spx_info.get("lastPrice", spx_info.get("regularMarketPrice", 0))
        vix_price = vix_info.get("lastPrice", vix_info.get("regularMarketPrice", 0))
        spx_prev = spx_info.get("previousClose", spx_price)
        vix_prev = vix_info.get("previousClose", vix_price)
        return {"spx": spx_price, "vix": vix_price, "spx_prev": spx_prev, "vix_prev": vix_prev}

    m = get_spx_vix()
    spx = float(m["spx"])
    vix = float(m["vix"])
    spx_chg = (spx - float(m["spx_prev"])) / float(m["spx_prev"]) * 100 if m["spx_prev"] else 0
    vix_chg = (vix - float(m["vix_prev"])) / float(m["vix_prev"]) * 100 if m["vix_prev"] else 0

    if vix < 14:
        regime = "Low Vol"
    elif vix < 20:
        regime = "Normal"
    elif vix < 28:
        regime = "Elevated"
    elif vix < 35:
        regime = "Panic"
    else:
        regime = "Extreme"

    can_enter = 13 <= vix <= 28
    kpis([
        {"label": "SPX", "value": f"{spx:,.0f}", "sub": f"{spx_chg:+.2f}%"},
        {"label": "VIX", "value": f"{vix:.1f}", "sub": f"{vix_chg:+.1f}%"},
        {"label": "VIX regime", "value": regime, "tone": "down" if vix >= 28 else None},
        {"label": "Entry", "value": "Allowed" if can_enter else "Avoid", "tone": "up" if can_enter else "down"},
    ])

    vix_scale = min(1.5, max(0.75, vix / 18.0))
    dte = 40 if vix > 25 else 50

    k1 = int(spx * (1 - 0.035 * vix_scale) / 5) * 5
    k2 = int(spx * (1 - 0.065 * vix_scale) / 5) * 5
    k3 = int(spx * (1 - 0.12 * vix_scale) / 5) * 5
    cvs = int(spx * (1 + 0.015) / 5) * 5
    cvl = cvs + 30
    exp = dt.date.today() + dt.timedelta(days=dte)

    st.subheader("Suggested structure")
    if can_enter:
        banner("Entry window is open", f"VIX {vix:.1f} inside the 13–28 band", tone="up", label="Open")
    else:
        banner("No entry", f"VIX {vix:.1f}: wait for the 13–28 band", tone="down", label="Wait")
    with st.columns(2)[0]:
        rows([
            ("Expiry", f"{exp.isoformat()} (DTE {dte})"),
            ("Put BWB", f"Buy {k1} / Sell 2x {k2} / Buy {k3}"),
            ("Call vertical", f"Sell {cvs} / Buy {cvl}"),
        ])

st.caption(f"Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
