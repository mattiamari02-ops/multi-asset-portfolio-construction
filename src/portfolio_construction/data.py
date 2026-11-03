"""Data preparation: total-return indices from yields, quarterly returns, desmoothing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Investable universe: key -> (label, traditional, liquid, alternative class, max weight)
UNIVERSE = pd.DataFrame(
    [
        ("cash", "Cash", 1, 1, "Traditional", 0.20),
        ("bond", "Bond (US 10Y govt)", 1, 1, "Traditional", 0.40),
        ("sp500", "Equity - S&P 500", 1, 1, "Traditional", 0.40),
        ("asia50", "Equity - Asia 50", 1, 1, "Traditional", 0.20),
        ("direct_property", "Direct Property", 0, 0, "Alt. A", 0.15),
        ("listed_property", "Listed Property", 0, 1, "Alt. B", 0.15),
        ("hedge_funds", "Hedge Funds", 0, 0, "Alt. C", 0.15),
        ("commodity", "Commodity", 0, 1, "Alt. D", 0.15),
        ("private_equity", "Private Equity", 0, 0, "Alt. E", 0.15),
        ("vaipx", "ETF - VAIPX (TIPS)", 0, 1, "Alt. F (ETF)", 0.15),
        ("gdx", "ETF - GDX (Gold miners)", 0, 1, "Alt. F (ETF)", 0.15),
        ("slv", "ETF - SLV (Silver)", 0, 1, "Alt. F (ETF)", 0.15),
        ("iwm", "ETF - IWM (Russell 2000)", 0, 1, "Alt. F (ETF)", 0.15),
        ("emgf", "ETF - EMGF (EM factors)", 0, 1, "Alt. F (ETF)", 0.15),
        ("agthx", "ETF - AGTHX (US growth)", 0, 1, "Alt. F (ETF)", 0.15),
        ("btc", "Crypto - BTC", 0, 1, "Alt. G (Crypto)", 0.05),
    ],
    columns=["key", "label", "traditional", "liquid", "class", "max_weight"],
).set_index("key")
UNIVERSE["alternative"] = 1 - UNIVERSE["traditional"]

SMOOTHED = ["direct_property", "hedge_funds"]  # appraisal / reported-NAV based


def load_levels(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "quarterly_levels.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("quarter")


def cash_index(tbill_yield_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Roll 3-month T-bills: I_t = I_{t-1} * (1 + y_{t-1} / 400)."""
    y = tbill_yield_pct
    first = y.first_valid_index()
    out = pd.Series(np.nan, index=y.index)
    pos = y.index.get_loc(first)
    out.iloc[pos] = base
    for i in range(pos + 1, len(y)):
        out.iloc[i] = out.iloc[i - 1] * (1 + y.iloc[i - 1] / 400)
    return out


def par_bond_duration_convexity(y_pct: float, years: int = 10, freq: int = 2) -> tuple[float, float]:
    """Modified duration and convexity of a par bond with semi-annual coupons (closed form)."""
    y = y_pct / 100
    n = years * freq
    mod_dur = (1 / y) * (1 - (1 + y / freq) ** (-n))
    convexity = (2 / y**2) * (1 - (1 + y / freq) ** (-n)) - (2 * years) / (y * (1 + y / freq) ** (n + 1))
    return mod_dur, convexity


def bond_index_from_yields(yield_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Quarterly total return of a constant-maturity 10Y Treasury from its yield.

    r_t = carry (y_{t-1}/4) - D_mod * dy + 0.5 * Convexity * dy^2
    (used because the provided bond index had three implausible quarters).
    """
    y = yield_pct
    first = y.first_valid_index()
    out = pd.Series(np.nan, index=y.index)
    pos = y.index.get_loc(first)
    out.iloc[pos] = base
    for i in range(pos + 1, len(y)):
        y0, y1 = y.iloc[i - 1], y.iloc[i]
        d, c = par_bond_duration_convexity(y0)
        dy = (y1 - y0) / 100
        out.iloc[i] = out.iloc[i - 1] * (1 + y0 / 400 - d * dy + 0.5 * c * dy**2)
    return out


def build_levels(raw: pd.DataFrame) -> pd.DataFrame:
    """Total-return levels for the whole universe (+ CPI)."""
    lv = raw.copy()
    lv["cash"] = cash_index(raw["tbill_yield_pct"])
    lv["bond"] = bond_index_from_yields(raw["ust10y_yield_pct"])
    return lv


def quarterly_returns(levels: pd.DataFrame) -> pd.DataFrame:
    cols = list(UNIVERSE.index) + ["eth", "sol", "cpi", "bond_index_original"]
    r = levels[cols].astype(float).pct_change(fill_method=None)
    return r.rename(columns={"cpi": "inflation"})


def ar1(x: pd.Series) -> float:
    """First-order autocorrelation (Pearson correlation of r_t with r_{t-1})."""
    x = x.dropna()
    return float(np.corrcoef(x.values[1:], x.values[:-1])[0, 1])


def geltner_desmooth(r: pd.Series, phi: float) -> pd.Series:
    """Geltner (1993): r*_t = (r_t - phi * r_{t-1}) / (1 - phi)."""
    return (r - phi * r.shift(1)) / (1 - phi)


def window(df: pd.DataFrame | pd.Series, start: str, end: str):
    """Slice by quarter labels, inclusive (index is 'Q1 2016' style)."""
    idx = list(df.index)
    return df.iloc[idx.index(start): idx.index(end) + 1]
