"""Mandate constraints, compliance checks and constrained mean-variance optimisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Mandate:
    """Investment policy of the endowment fund."""
    min_return: float = 0.07        # nominal, p.a., on both 5Y and 10Y estimates
    max_sd: float = 0.12            # p.a., on both 5Y and 10Y estimates
    min_cash: float = 0.08
    min_liquid: float = 0.50
    min_traditional: float = 0.50
    min_alternatives: int = 3       # number of alternative classes held ...
    min_alt_weight: float = 0.05    # ... with at least this weight
    max_cvar95_q: float = -0.10     # internal limit: quarterly CVaR 95% of back-test
    min_inflation_beta: float = -1.0  # internal limit (10Y)


def compliance_report(
    w: pd.Series, universe: pd.DataFrame, mandate: Mandate,
    mu5: pd.Series, cov5: pd.DataFrame, mu10: pd.Series, cov10: pd.DataFrame,
) -> pd.DataFrame:
    """Every mandate rule with the portfolio's value and a PASS/FAIL flag."""
    w = w.reindex(universe.index).fillna(0.0)
    r5, s5 = float(w @ mu5), float(np.sqrt(w @ cov5 @ w))
    r10, s10 = float(w @ mu10), float(np.sqrt(w @ cov10 @ w))
    n_alt = int(((w >= mandate.min_alt_weight - 1e-9) & (universe["alternative"] == 1)).sum())
    rows = [
        ("Weights sum to 100%", w.sum(), 1.0, abs(w.sum() - 1) < 1e-6),
        ("No short selling", w.min(), 0.0, w.min() >= -1e-9),
        ("Within max weights", (w - universe["max_weight"]).max(), 0.0, (w <= universe["max_weight"] + 1e-9).all()),
        ("Return >= 7% (5Y)", r5, mandate.min_return, r5 >= mandate.min_return),
        ("Return >= 7% (10Y)", r10, mandate.min_return, r10 >= mandate.min_return),
        ("Volatility <= 12% (5Y)", s5, mandate.max_sd, s5 <= mandate.max_sd),
        ("Volatility <= 12% (10Y)", s10, mandate.max_sd, s10 <= mandate.max_sd),
        ("Cash >= 8%", w["cash"], mandate.min_cash, w["cash"] >= mandate.min_cash - 1e-9),
        ("Liquid assets >= 50%", float(w @ universe["liquid"]), mandate.min_liquid,
         float(w @ universe["liquid"]) >= mandate.min_liquid - 1e-9),
        ("Traditional assets >= 50%", float(w @ universe["traditional"]), mandate.min_traditional,
         float(w @ universe["traditional"]) >= mandate.min_traditional - 1e-9),
        (">= 3 alternatives at >= 5%", n_alt, mandate.min_alternatives, n_alt >= mandate.min_alternatives),
    ]
    df = pd.DataFrame(rows, columns=["rule", "portfolio", "limit", "pass"])
    df["status"] = np.where(df["pass"], "PASS", "FAIL")
    return df.drop(columns="pass")


# ---------------------------------------------------------------------------
# Optimisation (convex part of the mandate; the "count of alternatives" rule is
# combinatorial and is checked afterwards)
# ---------------------------------------------------------------------------
def _linear_constraints(universe: pd.DataFrame, mandate: Mandate) -> list[dict]:
    liq, trad = universe["liquid"].values, universe["traditional"].values
    i_cash = list(universe.index).index("cash")
    return [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": lambda w: w[i_cash] - mandate.min_cash},
        {"type": "ineq", "fun": lambda w: w @ liq - mandate.min_liquid},
        {"type": "ineq", "fun": lambda w: w @ trad - mandate.min_traditional},
    ]


def _solve(objective, universe, mandate, extra=(), x0=None):
    n = len(universe)
    bounds = [(0.0, m) for m in universe["max_weight"].values]
    x0 = np.full(n, 1 / n) if x0 is None else x0
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=_linear_constraints(universe, mandate) + list(extra),
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(res.message)
    w = np.where(res.x < 1e-6, 0.0, res.x)
    return pd.Series(w / w.sum(), index=universe.index)


def min_variance(universe, mandate, cov, target_return=None, mu=None):
    c = cov.values
    extra = []
    if target_return is not None:
        extra.append({"type": "eq", "fun": lambda w: w @ mu.values - target_return})
    return _solve(lambda w: w @ c @ w, universe, mandate, extra)


def max_return(universe, mandate, mu, max_sd=None, cov=None):
    extra = [] if max_sd is None else [{"type": "ineq", "fun": lambda w: max_sd**2 - w @ cov.values @ w}]
    return _solve(lambda w: -(w @ mu.values), universe, mandate, extra)


def max_sharpe(universe, mandate, mu, cov, rf):
    c, m = cov.values, mu.values
    return _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ c @ w), universe, mandate)


def efficient_frontier(universe, mandate, mu, cov, n_points: int = 15) -> pd.DataFrame:
    """Minimum-variance portfolios for target returns between min-var and max-return."""
    w_min = min_variance(universe, mandate, cov)
    w_max = max_return(universe, mandate, mu)
    targets = np.linspace(float(w_min @ mu), float(w_max @ mu), n_points)
    rows = []
    for t in targets:
        w = min_variance(universe, mandate, cov, t, mu)
        rows.append({**w.to_dict(), "exp_return": float(w @ mu), "sd": float(np.sqrt(w @ cov @ w))})
    return pd.DataFrame(rows)
