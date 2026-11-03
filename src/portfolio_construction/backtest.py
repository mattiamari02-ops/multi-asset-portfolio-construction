"""Historical back-test of fixed-weight portfolios (rebalanced quarterly)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import drawdown, hist_var_cvar

Q = 4


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """r_p,t = sum_i w_i r_i,t. If an asset is not yet reported in a quarter,
    the remaining weights are rescaled to 100% for that quarter."""
    w = weights[weights > 0]
    r = returns[w.index]
    avail = r.notna()
    w_eff = avail.mul(w, axis=1)
    return (r.fillna(0) * w_eff).sum(axis=1) / w_eff.sum(axis=1)


def performance(r: pd.Series, rf: pd.Series, inflation: pd.Series) -> dict[str, float]:
    r = r.dropna()
    mean_ann = r.mean() * Q
    sd = r.std(ddof=1) * np.sqrt(Q)
    var, cvar = hist_var_cvar(r)
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod()])
    return {
        "mean_ann": mean_ann,
        "cagr": float(np.exp(np.log1p(r).sum() * Q / len(r)) - 1),
        "sd_ann": sd,
        "sharpe": (mean_ann - rf.loc[r.index].mean() * Q) / sd,
        "worst_q": r.min(),
        "var95_q": var,
        "cvar95_q": cvar,
        "max_drawdown": float(drawdown(wealth).min()),
        "skew": r.skew(),
        "inflation_beta": float(np.polyfit(inflation.loc[r.index], r, 1)[0]),
        "pct_negative_q": float((r < 0).mean()),
    }


def cumulative(r: pd.Series, quarters: list[str]) -> float:
    """Compounded return over a set of quarters (stress-episode analysis)."""
    return float(np.prod(1 + r.loc[quarters]) - 1)
