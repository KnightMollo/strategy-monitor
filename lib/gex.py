"""GEX (Gamma Exposure) filter engine.

Computes dealer gamma exposure from yfinance options chains and applies
a multi-factor screen for directional setups.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

GEX_TICKERS = [
    "SPY", "AAOI", "AAPL", "AEHR", "ALAB", "AMAT", "AMD", "ASTS",
    "AVGO", "AXTI", "CIEN", "CLS", "COHR", "CRDO", "CRWD", "DOCN",
    "GLW", "HOOD", "IREN", "LITE", "MRVL", "MSFT", "MU", "NBIS",
    "NFLX", "NVDA", "NVTS", "OKLO", "OUST", "PANW", "PENG", "PLTR",
    "RDDT", "RKLB", "SOFI", "SPCX", "TSEM", "TSLA",
]

TICKER_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "AVGO": "XLK", "AMAT": "XLK", "MU": "XLK", "MRVL": "XLK",
    "CRWD": "XLK", "PANW": "XLK", "CLS": "XLK", "GLW": "XLK",
    "LITE": "XLK", "CIEN": "XLK", "COHR": "XLK", "CRDO": "XLK",
    "DOCN": "XLK", "TSEM": "XLK", "AXTI": "XLK", "AEHR": "XLK",
    "NVTS": "XLK", "AAOI": "XLK", "NBIS": "XLK", "PENG": "XLK",
    "ALAB": "XLK",
    "TSLA": "XLY", "NFLX": "XLC", "HOOD": "XLF", "SOFI": "XLF",
    "PLTR": "XLK", "RDDT": "XLC", "RKLB": "XLI", "ASTS": "XLK",
    "IREN": "XLK", "OUST": "XLK", "OKLO": "XLU",
    "SPCX": "XLK",
    "SPY": "SPY",
}


def _bs_gamma(spot: float, strike: float, t: float, rate: float, sigma: float) -> float:
    """Black-Scholes gamma."""
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return float(norm.pdf(d1) / (spot * sigma * np.sqrt(t)))


def compute_gex(ticker: str, spot: Optional[float] = None, max_exps: int = 4) -> Optional[Dict]:
    """Compute GEX profile for a single ticker using yfinance options."""
    import yfinance as yf

    tk = yf.Ticker(ticker)

    if spot is None:
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

    hist_long = tk.history(period="6mo")
    ma50 = float(hist_long["Close"].rolling(50).mean().iloc[-1]) if len(hist_long) >= 50 else None

    try:
        exps = tk.options
    except Exception:
        return None
    if not exps:
        return None

    today = datetime.now()
    near_exps = [e for e in exps if datetime.strptime(e, "%Y-%m-%d") <= today + timedelta(days=45)]
    if not near_exps:
        near_exps = exps[:max_exps]
    else:
        near_exps = near_exps[:max_exps]

    rate = 0.05
    gex_by_strike = {}
    call_oi_by_strike = {}
    put_oi_by_strike = {}

    for exp in near_exps:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        t = max((exp_date - today).days / 365.0, 1 / 365)

        for _, row in chain.calls.iterrows():
            strike = float(row["strike"])
            oi_raw = row.get("openInterest", 0)
            oi = int(oi_raw) if (oi_raw is not None and not np.isnan(oi_raw)) else 0
            iv_raw = row.get("impliedVolatility", 0.3)
            iv = float(iv_raw) if (iv_raw is not None and not np.isnan(iv_raw) and iv_raw > 0) else 0.3
            if oi == 0 or abs(strike - spot) / spot > 0.30:
                continue

            gamma = _bs_gamma(spot, strike, t, rate, iv)
            gex = oi * gamma * 100 * spot
            gex_by_strike[strike] = gex_by_strike.get(strike, 0) + gex
            call_oi_by_strike[strike] = call_oi_by_strike.get(strike, 0) + oi

        for _, row in chain.puts.iterrows():
            strike = float(row["strike"])
            oi_raw = row.get("openInterest", 0)
            oi = int(oi_raw) if (oi_raw is not None and not np.isnan(oi_raw)) else 0
            iv_raw = row.get("impliedVolatility", 0.3)
            iv = float(iv_raw) if (iv_raw is not None and not np.isnan(iv_raw) and iv_raw > 0) else 0.3
            if oi == 0 or abs(strike - spot) / spot > 0.30:
                continue

            gamma = _bs_gamma(spot, strike, t, rate, iv)
            gex = -oi * gamma * 100 * spot
            gex_by_strike[strike] = gex_by_strike.get(strike, 0) + gex
            put_oi_by_strike[strike] = put_oi_by_strike.get(strike, 0) + oi

    if not gex_by_strike:
        return None

    above_spot = {k: v for k, v in gex_by_strike.items() if k > spot}
    max_gex_strike = max(above_spot, key=above_spot.get) if above_spot else max(gex_by_strike, key=gex_by_strike.get)

    net_gex = sum(gex_by_strike.values())

    puts_below = {k: v for k, v in put_oi_by_strike.items() if k <= spot}
    put_wall = max(puts_below, key=puts_below.get) if puts_below else None

    calls_above = {k: v for k, v in call_oi_by_strike.items() if k >= spot}
    call_wall = max(calls_above, key=calls_above.get) if calls_above else None

    near_money_oi = sum(
        call_oi_by_strike.get(k, 0) + put_oi_by_strike.get(k, 0)
        for k in set(list(call_oi_by_strike.keys()) + list(put_oi_by_strike.keys()))
        if abs(k - spot) / spot <= 0.10
    )

    upside_pct = (max_gex_strike / spot - 1) * 100 if max_gex_strike > spot else 0
    downside_pct = (1 - put_wall / spot) * 100 if put_wall and put_wall < spot else None

    if put_wall and max_gex_strike > spot and put_wall < spot:
        reward = max_gex_strike - spot
        risk = spot - put_wall
        rr = reward / risk if risk > 0 else 0
    else:
        rr = 0

    return {
        "ticker": ticker,
        "spot": round(spot, 2),
        "max_gex": round(max_gex_strike, 2),
        "upside_pct": round(upside_pct, 2),
        "put_wall": round(put_wall, 2) if put_wall else None,
        "call_wall": round(call_wall, 2) if call_wall else None,
        "downside_pct": round(downside_pct, 2) if downside_pct else None,
        "rr": round(rr, 2),
        "net_gex": net_gex,
        "net_gex_sign": "POSITIVE" if net_gex > 0 else "NEGATIVE",
        "near_money_oi": near_money_oi,
        "ma50": round(ma50, 2) if ma50 else None,
        "above_ma50": spot > ma50 if ma50 else None,
        "sector": TICKER_SECTOR.get(ticker, None),
    }


def apply_filters(gex_result: Dict, rrg_quadrants: Dict = None) -> Dict:
    """Apply 8 filters to a GEX result."""
    filters = {}

    filters["max_gex_above"] = {
        "pass": gex_result["max_gex"] > gex_result["spot"],
        "label": "MaxGEX > Spot",
        "detail": f"MaxGEX ${gex_result['max_gex']} vs Spot ${gex_result['spot']}",
    }

    filters["rr"] = {
        "pass": gex_result["rr"] >= 2.0,
        "label": "R/R >= 2.0",
        "detail": f"R/R = {gex_result['rr']:.2f}",
    }

    oi_threshold = 5000
    filters["oi_depth"] = {
        "pass": gex_result["near_money_oi"] >= oi_threshold,
        "label": f"OI depth >= {oi_threshold:,}",
        "detail": f"Near-money OI = {gex_result['near_money_oi']:,}",
    }

    filters["net_gex"] = {
        "pass": gex_result["net_gex"] > 0,
        "label": "Net GEX positive",
        "detail": f"Net GEX = {gex_result['net_gex']:,.0f} ({gex_result['net_gex_sign']})",
    }

    filters["momentum"] = {
        "pass": gex_result.get("above_ma50") is True,
        "label": "Above 50MA",
        "detail": f"Spot ${gex_result['spot']} vs MA50 ${gex_result.get('ma50', '?')}",
    }

    sector = gex_result.get("sector")
    rrg_pass = False
    rrg_detail = "No RRG data"
    if rrg_quadrants and sector and sector in rrg_quadrants:
        q = rrg_quadrants[sector]["quadrant"]
        rrg_pass = q in ("Leading", "Improving")
        rrg_detail = f"{sector} -> {q}"
    elif sector == "SPY":
        rrg_pass = True
        rrg_detail = "Benchmark (always pass)"

    filters["rrg_sector"] = {
        "pass": rrg_pass,
        "label": "RRG Leading/Improving",
        "detail": rrg_detail,
    }

    filters["put_wall"] = {
        "pass": gex_result["put_wall"] is not None and gex_result["put_wall"] < gex_result["spot"],
        "label": "Put Wall below entry",
        "detail": f"Put Wall = ${gex_result['put_wall']}" if gex_result["put_wall"] else "No Put Wall",
    }

    filters["min_upside"] = {
        "pass": gex_result["upside_pct"] >= 2.0,
        "label": "Upside >= 2%",
        "detail": f"Upside = {gex_result['upside_pct']:.1f}%",
    }

    passed = sum(1 for f in filters.values() if f["pass"])
    total = len(filters)
    status = "CONFIRMED" if passed == total else "BLOCKED"

    return {
        "status": status,
        "passed": passed,
        "total": total,
        "filters": filters,
    }
