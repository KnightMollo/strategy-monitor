"""Strategy Monitor - stripped local dashboard (no personal transaction/account data)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.gex import GEX_TICKERS, apply_filters, compute_gex
from lib.market_data import compute_sma, fetch_multiple, fetch_prices
from lib.rrg import (
    RRG_BASKET,
    RRG_BENCHMARK,
    SECTOR_MAP,
    SECTOR_SPDR,
    compute_idle_signal,
    compute_rrg,
    detect_signals,
    get_rrg_ranking,
    resample_biweekly,
)
from strategies.holdem_signal import HoldemConfig, compute_holdem_signal
from strategies.tmt_signal import compute_rsi2


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Strategy Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def next_trading_day() -> str:
    d = dt.date.today() + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def get_tmt_signal_snapshot() -> dict:
    """RSI(2) priority signal + idle RRG fallback, without position dependence."""
    qqq = fetch_prices("QQQ", period="2y")
    tqqq = fetch_prices("TQQQ", period="1y")
    if qqq.empty or tqqq.empty or len(qqq) < 220:
        return {"ready": False, "reason": "Insufficient QQQ/TQQQ data"}

    qqq_close = qqq["Close"]
    rsi2 = compute_rsi2(qqq_close)
    sma200 = compute_sma(qqq_close, 200)

    curr_rsi = float(rsi2.iloc[-1])
    curr_qqq = float(qqq_close.iloc[-1])
    curr_tqqq = float(tqqq["Close"].iloc[-1])
    curr_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else np.nan
    above_200 = bool(np.isfinite(curr_sma200) and curr_qqq > curr_sma200)

    idle_data = fetch_multiple(list(set(SECTOR_SPDR + ["SPY", "SGOV"])), period="2y")
    idle_sig = compute_idle_signal(idle_data, benchmark="SPY")
    idle_hold = idle_sig.get("hold", "SGOV")

    if curr_rsi < 15 and above_200:
        action = "BUY TQQQ"
        next_hold = "TQQQ"
        reason = f"RSI(2)={curr_rsi:.1f} < 15 and QQQ > 200SMA"
    elif curr_rsi > 80:
        action = "EXIT TQQQ"
        next_hold = idle_hold
        reason = f"RSI(2)={curr_rsi:.1f} > 80, rotate to idle"
    else:
        action = "WAIT"
        next_hold = idle_hold
        reason = f"RSI(2)={curr_rsi:.1f}, no trigger; idle -> {idle_hold}"

    return {
        "ready": True,
        "action": action,
        "next_hold": next_hold,
        "reason": reason,
        "qqq_rsi2": round(curr_rsi, 1),
        "qqq_price": round(curr_qqq, 2),
        "tqqq_price": round(curr_tqqq, 2),
        "qqq_sma200": round(curr_sma200, 2) if np.isfinite(curr_sma200) else 0.0,
        "above_200": above_200,
        "next_date": next_trading_day(),
        "dm_hold": idle_hold,
        "breadth": round(idle_sig.get("breadth", 0.0) * 100),
        "breakout_ticker": idle_sig.get("breakout_ticker"),
        "idle_reason": idle_sig.get("reason", ""),
    }


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

        raw_data = fetch_multiple(list(set(RRG_BASKET + [RRG_BENCHMARK])), period="2y")
        if not raw_data or RRG_BENCHMARK not in raw_data:
            st.info("RRG data unavailable now")
            return

        biweekly = resample_biweekly(raw_data)
        if RRG_BENCHMARK not in biweekly:
            st.info("RRG benchmark series unavailable")
            return

        tail_len = max(tail_weeks, 1)
        rrg_df = compute_rrg(
            biweekly,
            RRG_BENCHMARK,
            ratio_period=7,
            momentum_period=7,
            tail_length=tail_len,
        )
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
            "Leading": "#16a34a",
            "Improving": "#2563eb",
            "Weakening": "#eab308",
            "Lagging": "#dc2626",
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

        fig.update_layout(height=500, margin=dict(t=20, b=20, l=30, r=30), xaxis_title="RS-Ratio", yaxis_title="RS-Momentum", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            ranking[["rank", "ticker", "label", "rs_ratio", "rs_momentum", "quadrant"]].style.format(
                {"rs_ratio": "{:.2f}", "rs_momentum": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        signals = detect_signals(rrg_df)
        if signals:
            sig_icons = {"BREAKOUT": "🚀", "RECOVER": "🔄", "FADING": "📉", "BREAKDOWN": "⬇", "ROTATION": "🔀"}
            for sig in signals[:5]:
                st.caption(f"{sig_icons.get(sig['signal'], '•')} {sig['signal']}: {sig['ticker']} {sig['transition']}")


st.sidebar.title("Strategy Monitor")
page = st.sidebar.radio("Navigation", ["Overview", "Hold'em", "TMT", "GEX Filter", "SPX BWB"])

st.sidebar.markdown("---")
if st.sidebar.button("Refresh Prices"):
    st.cache_data.clear()
    cache_dir = BASE_DIR / ".cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            f.unlink()
    st.rerun()


if page == "Overview":
    st.title("📊 Strategy Monitor")
    st.caption(f"Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | Public mode (no personal data)")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Hold'em Signal")
        try:
            sig = compute_holdem_signal(HoldemConfig())
            st.metric("Next Hold", sig.next_hold)
            st.caption(sig.key_condition)
            st.caption(f"Phase {sig.phase} | Regime: {'Bull' if sig.is_bull else 'Bear'}")
        except Exception as e:
            st.warning(f"Hold'em unavailable: {e}")

    with c2:
        st.subheader("TMT Signal")
        try:
            tmt = get_tmt_signal_snapshot()
            if tmt["ready"]:
                st.metric("Next Hold", tmt["next_hold"])
                st.caption(f"{tmt['next_date']} -> {tmt['action']}")
                st.caption(tmt["reason"])
            else:
                st.warning(tmt["reason"])
        except Exception as e:
            st.warning(f"TMT unavailable: {e}")

    st.markdown("---")
    st.markdown("**Sector Rotation (RRG Weekly)**")
    try:
        rrg_raw = fetch_multiple(list(set(RRG_BASKET + [RRG_BENCHMARK])), period="2y")
        rrg_biweekly = resample_biweekly(rrg_raw)
        rrg_df = compute_rrg(
            rrg_biweekly,
            RRG_BENCHMARK,
            ratio_period=7,
            momentum_period=7,
            tail_length=2,
        )
        ranking = get_rrg_ranking(rrg_df)
        if not ranking.empty:
            qc = ranking["quadrant"].value_counts()
            st.caption(
                f"Leading {qc.get('Leading', 0)} | Improving {qc.get('Improving', 0)} | "
                f"Weakening {qc.get('Weakening', 0)} | Lagging {qc.get('Lagging', 0)}"
            )
            top = ranking.iloc[0]
            bottom = ranking.iloc[-1]
            st.caption(f"Top: {top['ticker']} ({top['label']}) | Bottom: {bottom['ticker']} ({bottom['label']})")
    except Exception as e:
        st.caption(f"RRG unavailable: {e}")

elif page == "Hold'em":
    st.title("🃏 Hold'em Strategy")
    try:
        signal = compute_holdem_signal(HoldemConfig())
        phase_labels = {0: "Phase 0 (Normal)", 1: "Phase 1 (UVXY Hedge)", 2: "Phase 2 (SGOV Cooldown)"}

        col1, col2, col3 = st.columns(3)
        col1.metric("Signal Date", signal.signal_date)
        col1.metric("Next Hold", signal.next_hold)
        col2.metric("Phase", phase_labels.get(signal.phase, "Unknown"))
        col2.metric("Market Regime", "Bull" if signal.is_bull else "Bear")
        col3.metric("SPY", f"{signal.spy_price:.2f}")
        col3.metric("TQQQ", f"{signal.tqqq_price:.2f}")

        st.info(signal.key_condition)

        with st.expander("Indicator Details", expanded=True):
            i1, i2, i3 = st.columns(3)
            i1.markdown(f"- SPY MA100: {signal.spy_ma100:.2f}")
            i1.markdown(f"- SPY MA50: {signal.spy_ma50:.2f}")
            i1.markdown(f"- TQQQ MA20: {signal.tqqq_ma20:.2f}")
            i2.markdown(f"- RSI TQQQ: {signal.rsi_tqqq:.1f}")
            i2.markdown(f"- RSI UPRO: {signal.rsi_upro:.1f}")
            i2.markdown(f"- RSI SPY: {signal.rsi_spy:.1f}")
            i3.markdown(f"- RSI SQQQ: {signal.rsi_sqqq:.1f}")
            i3.markdown(f"- RSI TLT: {signal.rsi_tlt:.1f}")

        with st.expander("Decision Tree", expanded=True):
            bull_overbought = signal.rsi_tqqq > 79 or signal.rsi_upro > 79
            tqqq_below_ma20 = signal.tqqq_ma20 > 0 and signal.tqqq_price < signal.tqqq_ma20
            sqqq_gt_tlt = signal.rsi_sqqq > signal.rsi_tlt
            spy_above_ma50 = signal.spy_ma50 > 0 and signal.spy_price > signal.spy_ma50
            is_cash_gate = signal.next_hold == "SGOV" and signal.is_bull and signal.phase == 0

            st.markdown("**Live Path Check**")
            if signal.is_bull:
                st.markdown("- {} SPY > MA100".format("OK" if signal.is_bull else "NO"))
                if is_cash_gate:
                    st.markdown("- NO SPY < MA50 for 2+ days -> SGOV cash gate")
                elif not spy_above_ma50:
                    st.markdown("- WARN SPY < MA50 for 1 day -> await confirmation")
                else:
                    st.markdown("- OK SPY > MA50 -> TQQQ")
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
    gate -> tqqq [label="No (<2d below)"];
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
            st.graphviz_chart(tree_dot, use_container_width=True)

    except Exception as e:
        st.error(f"Hold'em signal failed: {e}")

elif page == "TMT":
    st.title("📈 TMT - RSI(2) + Idle Rotation")
    try:
        tmt = get_tmt_signal_snapshot()
        if not tmt["ready"]:
            st.warning(tmt["reason"])
        else:
            st.warning(f"{tmt['next_date']} -> {tmt['next_hold']} ({tmt['action']})")
            st.caption(tmt["reason"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("QQQ RSI(2)", f"{tmt['qqq_rsi2']:.1f}")
            m2.metric("QQQ", f"${tmt['qqq_price']:.2f}")
            m3.metric("TQQQ", f"${tmt['tqqq_price']:.2f}")
            m4.metric("Idle Breadth", f"{tmt['breadth']}%")

            with st.expander("Decision Tree", expanded=True):
                st.markdown("**Public signal path (position state is intentionally omitted)**")
                st.markdown(
                    "- RSI(2) < 15 and QQQ > 200SMA -> BUY TQQQ\n"
                    "- RSI(2) > 80 -> EXIT TQQQ and rotate to Idle\n"
                    "- Otherwise -> Idle rotation (RRG breadth + breakout)"
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
                notrend_fill = "#bfdbfe" if not qqq_above and qqq_r < 15 else "#f8fafc"
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
    buy [label="BUY TQQQ\\n(soft 7d, cap 21d)", fillcolor="{buy_fill}"];
    sell [label="EXIT TQQQ\\nreturn to idle", fillcolor="{sell_fill}"];
    wait_notrend [label="WAIT\\n(no trend support)", fillcolor="{notrend_fill}"];

    subgraph cluster_idle {{
        label="RRG Idle (biweekly)\\nBreadth: {breadth_pct}%";
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
    trend -> wait_notrend [label="No"];
    overbought -> sell [label="Yes"];
    overbought -> breadth_check [label="No"];
}}
"""
                st.graphviz_chart(tree_dot, use_container_width=True)

    except Exception as e:
        st.error(f"TMT signal failed: {e}")

    render_rrg_map("Sector Rotation Map (RRG)", "tmt_rrg")

elif page == "GEX Filter":
    st.title("⚡ GEX Filter")
    st.caption("Gamma exposure scan (public, no private account inputs)")

    rrg_quadrants = {}
    try:
        idle_data = fetch_multiple(list(set(SECTOR_SPDR + ["SPY", "SGOV"])), period="2y")
        idle_result = compute_idle_signal(idle_data, "SPY")
        rrg_quadrants = idle_result.get("quadrants", {})
    except Exception:
        rrg_quadrants = {}

    if st.button("Run GEX Scan", type="primary"):
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

        s1, s2, s3 = st.columns(3)
        s1.metric("Confirmed", len(confirmed))
        s2.metric("Blocked", len(blocked))
        s3.metric("Failed", len(failed))

        if confirmed:
            st.subheader("Confirmed Setups")
            for r in confirmed:
                with st.expander(f"{r['ticker']} | R/R {r['rr']:.2f} | +{r['upside_pct']:.1f}%"):
                    st.markdown(
                        f"Spot ${r['spot']} | MaxGEX ${r['max_gex']} | PutWall {r['put_wall']} | "
                        f"CallWall {r['call_wall']} | NetGEX {r['net_gex_sign']}"
                    )
                    for f in r["filters"].values():
                        icon = "OK" if f["pass"] else "NO"
                        st.caption(f"{icon} {f['label']}: {f['detail']}")

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
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=550)
    else:
        st.info("Run scan to evaluate all tickers with 8 filters")

elif page == "SPX BWB":
    st.title("🦋 SPX BWB Daily Playbook")
    st.caption("Signal and structure guidance only (no position tracking)")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPX", f"{spx:,.0f}", f"{spx_chg:+.2f}%")
    c2.metric("VIX", f"{vix:.1f}", f"{vix_chg:+.1f}%")

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
    c3.metric("VIX Regime", regime)
    c4.metric("Entry", "Allowed" if can_enter else "Avoid")

    st.markdown("---")
    vix_scale = min(1.5, max(0.75, vix / 18.0))
    dte = 40 if vix > 25 else 50

    k1 = int(spx * (1 - 0.035 * vix_scale) / 5) * 5
    k2 = int(spx * (1 - 0.065 * vix_scale) / 5) * 5
    k3 = int(spx * (1 - 0.12 * vix_scale) / 5) * 5
    cvs = int(spx * (1 + 0.015) / 5) * 5
    cvl = cvs + 30
    exp = dt.date.today() + dt.timedelta(days=dte)

    st.subheader("Suggested Structure")
    st.markdown(
        f"- Expiry: {exp.isoformat()} (DTE {dte})\n"
        f"- Put BWB: Buy {k1} / Sell 2x {k2} / Buy {k3}\n"
        f"- Call Vertical: Sell {cvs} / Buy {cvl}"
    )

    if can_enter:
        st.success("Entry window is open under VIX filter")
    else:
        st.warning("No entry: wait for VIX to return to 13-28 band")

st.caption(f"Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
