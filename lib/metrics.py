"""Performance metrics calculations (no personal data)."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(curve: pd.Series, rf: float = 0.04) -> Dict[str, float]:
    """Compute performance metrics from an equity curve Series (indexed by date)."""
    if len(curve) < 2:
        return {"total_return": 0, "annualized_return": 0, "annualized_volatility": 0,
                "max_drawdown": 0, "sharpe": 0, "calmar": 0}

    vals = curve.values
    rets = np.diff(vals) / vals[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) == 0:
        return {"total_return": 0, "annualized_return": 0, "annualized_volatility": 0,
                "max_drawdown": 0, "sharpe": 0, "calmar": 0}

    tr = vals[-1] / vals[0] - 1
    days = max((curve.index[-1] - curve.index[0]).days, 1)
    ar = (1 + tr) ** (365 / days) - 1

    mu = np.mean(rets)
    std = np.std(rets, ddof=1) if len(rets) > 1 else 0
    avol = std * math.sqrt(TRADING_DAYS)

    peak = np.maximum.accumulate(vals)
    dd = vals / peak - 1
    mdd = float(np.min(dd))

    drf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    sharpe = (mu - drf) / std * math.sqrt(TRADING_DAYS) if std > 0 else 0
    calmar = ar / abs(mdd) if mdd < 0 else 0

    return {
        "total_return": float(tr),
        "annualized_return": float(ar),
        "annualized_volatility": float(avol),
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
    }
