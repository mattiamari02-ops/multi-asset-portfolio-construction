"""Risk/return statistics of asset classes and portfolios (quarterly data, annualised)."""
from __future__ import annotations

import numpy as np
import pandas as pd

Q = 4  # quarters per year


def drawdown(levels: pd.Series) -> pd.Series:
    """Distance from the running peak within the window."""
    lv = levels.dropna()
    return lv / lv.cummax() - 1


def hist_var_cvar(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    r = r.dropna()
    var = float(np.percentile(r, alpha * 100, method="linear"))
    return var, float(r[r <= var].mean())


def asset_statistics(returns: pd.DataFrame, inflation: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """One column per asset: mean, CAGR, sigma, Sharpe, downside, tail and inflation metrics.

    `returns` must already be restricted to the estimation window.
    """
    rf_ann = rf.mean() * Q
    out = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 4:
            continue
        mean_ann = r.mean() * Q
        sd_ann = r.std(ddof=1) * np.sqrt(Q)
        var, cvar = hist_var_cvar(r)
        wealth = (1 + r).cumprod()
        infl = inflation.loc[r.index]
        out[col] = {
            "mean_ann": mean_ann,
            "cagr": float(np.exp(np.log1p(r).sum() * Q / len(r)) - 1),
            "sd_ann": sd_ann,
            "sharpe": (mean_ann - rf_ann) / sd_ann if col != "cash" else np.nan,
            "downside_dev_ann": float(np.sqrt((np.minimum(r, 0) ** 2).sum() / len(r)) * np.sqrt(Q)),
            "worst_q": r.min(), "best_q": r.max(),
            "var95_q": var, "cvar95_q": cvar,
            "max_drawdown": float(min(0.0, drawdown(pd.concat([pd.Series([1.0]), wealth])).min())),
            "skew": r.skew(), "excess_kurt": r.kurt(),
            "inflation_beta": float(np.polyfit(infl, r, 1)[0]),
            "corr_inflation": float(np.corrcoef(r, infl)[0, 1]),
            "ar1": float(np.corrcoef(r.values[1:], r.values[:-1])[0, 1]),
            "n_quarters": len(r),
        }
    return pd.DataFrame(out)


def portfolio_moments(w: pd.Series, mu_ann: pd.Series, cov_ann: pd.DataFrame) -> tuple[float, float]:
    w = w.reindex(mu_ann.index).fillna(0.0)
    return float(w @ mu_ann), float(np.sqrt(w @ cov_ann.loc[w.index, w.index] @ w))


def risk_contributions(w: pd.Series, cov_ann: pd.DataFrame) -> pd.Series:
    """Euler decomposition: share of portfolio variance from each asset (sums to 1)."""
    w = w.reindex(cov_ann.index).fillna(0.0)
    marginal = cov_ann @ w
    return w * marginal / float(w @ marginal)
