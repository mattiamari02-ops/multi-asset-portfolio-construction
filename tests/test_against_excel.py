"""Regression tests: the Python implementation must reproduce the original Excel model."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_analysis  # noqa: E402
from portfolio_construction.data import build_levels, load_levels  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return run_analysis.main()


def test_total_return_indices_from_yields():
    lv = build_levels(load_levels())
    assert lv.loc["Q4 2025", "cash"] == pytest.approx(123.874139, abs=1e-5)
    assert lv.loc["Q4 2025", "bond"] == pytest.approx(144.048317, abs=1e-5)


def test_asset_statistics(res):
    s10 = res["stats"]["10Y"]
    assert s10.loc["mean_ann", "sp500"] == pytest.approx(0.128885, abs=1e-6)
    assert s10.loc["sd_ann", "asia50"] == pytest.approx(0.201291, abs=1e-6)
    assert s10.loc["sharpe", "gdx"] == pytest.approx(0.744217, abs=1e-6)
    assert s10.loc["inflation_beta", "commodity"] == pytest.approx(8.8825, abs=1e-4)
    assert res["phis"]["5Y"]["direct_property"] == pytest.approx(0.579849, abs=1e-6)


def test_selected_portfolio(res):
    p = res["port_table"].loc["Selected"]
    assert p["E[R] 10Y"] == pytest.approx(0.1028, abs=1e-4)
    assert p["sd 10Y"] == pytest.approx(0.0788, abs=1e-4)
    assert p["sd 5Y"] == pytest.approx(0.0677, abs=1e-4)
    assert p["sd 10Y desmoothed"] == pytest.approx(0.0816, abs=1e-4)
    assert (res["compliance"]["status"] == "PASS").all()


def test_backtest_and_stress(res):
    assert res["perf"]["10Y"].loc["cvar95_q", "Selected"] == pytest.approx(-0.0571, abs=1e-4)
    assert res["perf"]["10Y"].loc["max_drawdown", "60/40"] == pytest.approx(-0.1610, abs=1e-4)
    assert res["stress"].loc["Inflation shock & Fed hikes (2022)", "60/40"] == pytest.approx(-0.1240, abs=1e-4)


def test_frontier_minimum_variance_point(res):
    ef = res["frontier"]
    assert ef["sd"].iloc[0] == pytest.approx(0.0272, abs=2e-4)
    assert ef["exp_return"].iloc[0] == pytest.approx(0.0355, abs=2e-4)
    assert (ef["sd"].diff().dropna() > 0).all()
