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
from lib.metrics import compute_metrics
from lib.rrg import (
    RRG_BASKET,
    RRG_BENCHMARK,
    SECTOR_MAP,
    SECTOR_SPDR,
    compute_idle_signal,
    compute_rrg,
    detect_signals,
    get_rrg_ranking,
    resample_weekly,
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


@st.cache_data(ttl=3600)
def run_holdem_backtest(start: str, end: str) -> pd.Series:
    data = fetch_multiple(["SPY", "TQQQ", "SQQQ"], period="5y")
    if "SPY" not in data or "TQQQ" not in data:
        return pd.Series(dtype=float)

    spy = data["SPY"]["Close"]
    tqqq = data["TQQQ"]["Close"]
    sqqq = data["SQQQ"]["Close"] if "SQQQ" in data else None
    ma100 = spy.rolling(100).mean()

    idx = spy.index.intersection(tqqq.index)
    spy = spy.reindex(idx)
    tqqq = tqqq.reindex(idx)
    ma100 = ma100.reindex(idx)
    if sqqq is not None:
        sqqq = sqqq.reindex(idx)

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
    qqq = fetch_prices("QQQ", period="5y")
    tqqq = fetch_prices("TQQQ", period="5y")
    if qqq.empty or tqqq.empty:
        return pd.Series(dtype=float)

    idx = qqq.index.intersection(tqqq.index)
    qqq_c = qqq["Close"].reindex(idx)
    tqqq_c = tqqq["Close"].reindex(idx)
    rsi2 = compute_rsi2(qqq_c)
    sma200 = compute_sma(qqq_c, 200)

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

        weekly = resample_weekly(raw_data)
        if RRG_BENCHMARK not in weekly:
            st.info("RRG benchmark series unavailable")
            return

        tail_len = max(tail_weeks, 1)
        rrg_df = compute_rrg(weekly, RRG_BENCHMARK, tail_length=tail_len)
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
page = st.sidebar.radio("Navigation", ["Overview", "Hold'em", "TMT", "GEX Filter", "SPX BWB", "Report"])

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
        rrg_weekly = resample_weekly(rrg_raw)
        rrg_df = compute_rrg(rrg_weekly, RRG_BENCHMARK, tail_length=2)
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

    st.markdown("---")
    st.subheader("Backtest Snapshot")
    today = dt.date.today()
    start = str(today - dt.timedelta(days=365))
    end = str(today)

    b1, b2 = st.columns(2)
    with b1:
        h_curve = run_holdem_backtest(start, end)
        if not h_curve.empty:
            st.line_chart(h_curve, use_container_width=True)
            m = compute_metrics(h_curve)
            k1, k2, k3 = st.columns(3)
            k1.metric("Ann. Return", f"{m['annualized_return']*100:.1f}%")
            k2.metric("Max DD", f"{m['max_drawdown']*100:.1f}%")
            k3.metric("Sharpe", f"{m['sharpe']:.2f}")

    with b2:
        t_curve = run_tmt_backtest(start, end)
        if not t_curve.empty:
            st.line_chart(t_curve, use_container_width=True)
            m = compute_metrics(t_curve)
            k1, k2, k3 = st.columns(3)
            k1.metric("Ann. Return", f"{m['annualized_return']*100:.1f}%")
            k2.metric("Max DD", f"{m['max_drawdown']*100:.1f}%")
            k3.metric("Sharpe", f"{m['sharpe']:.2f}")

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
            tree_dot = f"""
digraph HoldemTree {{
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Arial", fontsize=10];
    edge [color="#64748b", fontname="Arial", fontsize=9];

    start [label="Start", fillcolor="#e2e8f0"];
    bull [label="SPY > MA100?"];
    overbought [label="RSI(TQQQ/UPRO) > 79?"];
    uvxy [label="Hold UVXY (Phase1)"];
    cooldown [label="Hold SGOV (Phase2)"];
    gate [label="SPY < MA50 for 2d?"];
    tqqq [label="Hold TQQQ"];
    bear [label="Bear branch"];
    sqqq [label="Hold SQQQ"];
    tlt [label="Hold TLT"];

    start -> bull;
    bull -> overbought [label="Yes"];
    bull -> bear [label="No"];
    overbought -> uvxy [label="Yes"];
    overbought -> gate [label="No"];
    gate -> cooldown [label="Yes"];
    gate -> tqqq [label="No"];
    bear -> sqqq [label="SQQQ RSI > TLT RSI"];
    bear -> tlt [label="otherwise"];
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
                st.markdown(
                    "- RSI(2) < 15 and QQQ > 200SMA -> BUY TQQQ\n"
                    "- RSI(2) > 80 -> EXIT TQQQ and rotate to Idle\n"
                    "- Otherwise -> Idle rotation (RRG breadth + breakout)"
                )
                st.caption(f"Idle hold: {tmt['dm_hold']} | {tmt['idle_reason']}")

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

elif page == "Report":
    st.title("📄 Strategy Report")
    st.caption("Backtest-only quarterly comparison (no private transaction data)")

    today = dt.date.today()
    current_year = today.year
    current_q = (today.month - 1) // 3 + 1

    def quarter_range(y: int, q: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        sm, sd = starts[q]
        em, ed = ends[q]
        return pd.Timestamp(y, sm, sd), pd.Timestamp(y, em, ed)

    c1, c2 = st.columns(2)
    with c1:
        sel_year = st.selectbox("Year", options=list(range(current_year, current_year - 3, -1)), index=0)
    with c2:
        sel_q = st.selectbox("Quarter", options=[1, 2, 3, 4], index=current_q - 1)

    q_start, q_end = quarter_range(sel_year, sel_q)
    st.caption(f"Period: {q_start.date()} -> {q_end.date()}")

    if st.button("Generate", type="primary"):
        h_curve = run_holdem_backtest(str(q_start.date()), str(q_end.date()))
        t_curve = run_tmt_backtest(str(q_start.date()), str(q_end.date()))
        spy = fetch_prices("SPY", period="5y")
        b_curve = pd.Series(dtype=float)
        if not spy.empty:
            s = spy["Close"]
            s = s[(s.index >= q_start) & (s.index <= q_end)]
            if len(s) > 1:
                b_curve = s / s.iloc[0] * 100000

        fig = go.Figure()
        if not h_curve.empty:
            fig.add_trace(go.Scatter(x=h_curve.index, y=h_curve.values, name="Holdem", line=dict(width=2)))
        if not t_curve.empty:
            fig.add_trace(go.Scatter(x=t_curve.index, y=t_curve.values, name="TMT", line=dict(width=2)))
        if not b_curve.empty:
            fig.add_trace(go.Scatter(x=b_curve.index, y=b_curve.values, name="SPY", line=dict(width=2, dash="dash")))
        fig.update_layout(height=430, margin=dict(t=20, b=30), yaxis_title="NAV")
        st.plotly_chart(fig, use_container_width=True)

        rows = []
        for name, curve in [("Holdem", h_curve), ("TMT", t_curve), ("SPY", b_curve)]:
            if curve.empty or len(curve) < 2:
                continue
            m = compute_metrics(curve)
            rows.append(
                {
                    "Strategy": name,
                    "Return": f"{m['total_return']*100:.2f}%",
                    "Ann": f"{m['annualized_return']*100:.2f}%",
                    "Vol": f"{m['annualized_volatility']*100:.2f}%",
                    "MaxDD": f"{m['max_drawdown']*100:.2f}%",
                    "Sharpe": f"{m['sharpe']:.2f}",
                    "Calmar": f"{m['calmar']:.2f}",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption(f"Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
