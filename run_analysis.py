"""Reproduce the full asset-allocation study.

    python run_analysis.py

Writes tables to outputs/ and charts to outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_construction.backtest import cumulative, performance, portfolio_returns  # noqa: E402
from portfolio_construction.data import (SMOOTHED, UNIVERSE, ar1, build_levels,  # noqa: E402
                                         geltner_desmooth, load_levels, quarterly_returns, window)
from portfolio_construction.optimize import (Mandate, compliance_report, efficient_frontier,  # noqa: E402
                                             max_sharpe)
from portfolio_construction.plotting import CRITICAL, INK_2, SERIES, apply_style, pct  # noqa: E402
from portfolio_construction.stats import asset_statistics, portfolio_moments, risk_contributions  # noqa: E402

OUT, FIG = ROOT / "outputs", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
apply_style()
pd.options.display.float_format = "{:,.4f}".format

WIN = {"5Y": ("Q1 2021", "Q4 2025"), "10Y": ("Q1 2016", "Q4 2025")}
ASSETS = list(UNIVERSE.index)

# Portfolios tested (weights in %)
PORTFOLIOS = {
    "Selected": {"cash": 10, "bond": 5, "sp500": 35, "asia50": 5, "direct_property": 10,
                 "hedge_funds": 10, "commodity": 10, "private_equity": 10, "gdx": 5},
    "Contingency": {"cash": 10, "bond": 10, "sp500": 25, "asia50": 10, "direct_property": 10,
                    "hedge_funds": 10, "commodity": 10, "private_equity": 10, "gdx": 5},
    "60/40": {"sp500": 60, "bond": 40},
    "S&P 500": {"sp500": 100},
}
STRESS = {
    "US-China trade war (2018-19)": [f"Q{q} {y}" for y in (2018, 2019) for q in range(1, 5)],
    "Fed tightening (Q4 2018)": ["Q4 2018"],
    "COVID-19 crash (Q1 2020)": ["Q1 2020"],
    "Inflation shock & Fed hikes (2022)": [f"Q{q} 2022" for q in range(1, 5)],
    "Middle East / oil shock (Q1 2026)*": ["Q1 2026"],
}


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> dict:
    res: dict = {}
    levels = build_levels(load_levels())
    rets = quarterly_returns(levels)

    # ---------------------------------------------------------------- 1. stats
    section("1. Asset-class statistics (annualised, quarterly data)")
    stats, mu, cov, phis = {}, {}, {}, {}
    for name, (a, b) in WIN.items():
        w = window(rets, a, b)
        stats[name] = asset_statistics(w[ASSETS], w["inflation"], w["cash"])
        mu[name] = stats[name].loc["mean_ann", ASSETS]
        cov[name] = w[ASSETS].cov() * 4
        phis[name] = {k: ar1(w[k]) for k in SMOOTHED}
        stats[name].T.to_csv(OUT / f"asset_stats_{name.lower()}.csv")
        print(f"\n{name} window {a} - {b}")
        print(stats[name].loc[["mean_ann", "sd_ann", "sharpe", "cvar95_q", "max_drawdown",
                               "inflation_beta"]].T.to_string())

    # Geltner desmoothing of appraisal-based series (robustness for risk)
    section("2. Desmoothing appraisal-based returns (Geltner)")
    w10 = window(rets, *WIN["10Y"]).copy()
    for k in SMOOTHED:
        full_ds = geltner_desmooth(rets[k], phis["10Y"][k])
        w10[k] = window(full_ds, *WIN["10Y"])
        print(f"{k:<16} phi(10Y) = {phis['10Y'][k]:.3f}  sd reported {stats['10Y'].loc['sd_ann', k]:.2%}"
              f" -> desmoothed {w10[k].std() * 2:.2%}")
    cov["10Y adj."] = w10[ASSETS].cov() * 4
    res.update(stats=stats, phis=phis)

    # ---------------------------------------------------------------- 3. portfolios
    section("3. Candidate portfolios")
    W = pd.DataFrame({k: pd.Series(v, dtype=float) / 100 for k, v in PORTFOLIOS.items()}).reindex(ASSETS).fillna(0)
    rows = {}
    for p in W:
        r5, s5 = portfolio_moments(W[p], mu["5Y"], cov["5Y"])
        r10, s10 = portfolio_moments(W[p], mu["10Y"], cov["10Y"])
        _, s10a = portfolio_moments(W[p], mu["10Y"], cov["10Y adj."])
        rf5, rf10 = mu["5Y"]["cash"], mu["10Y"]["cash"]
        rows[p] = {"E[R] 5Y": r5, "sd 5Y": s5, "Sharpe 5Y": (r5 - rf5) / s5,
                   "E[R] 10Y": r10, "sd 10Y": s10, "Sharpe 10Y": (r10 - rf10) / s10,
                   "sd 10Y desmoothed": s10a,
                   "Inflation beta 10Y": float(W[p] @ stats["10Y"].loc["inflation_beta", ASSETS])}
    port_table = pd.DataFrame(rows).T
    print(W.loc[(W > 0).any(axis=1)].to_string())
    print("\n" + port_table.to_string())
    port_table.to_csv(OUT / "portfolio_ex_ante.csv")
    res.update(weights=W, port_table=port_table)

    # ---------------------------------------------------------------- 4. backtest
    section("4. Back-test (quarterly rebalancing, Q1 2016 - Q4 2025) and stress episodes")
    bt = pd.DataFrame({p: portfolio_returns(rets[ASSETS], W[p]) for p in W})
    perf = {}
    for name, (a, b) in WIN.items():
        wbt = window(bt, a, b)
        perf[name] = pd.DataFrame({p: performance(wbt[p], window(rets["cash"], a, b),
                                                  window(rets["inflation"], a, b)) for p in W})
        print(f"\n{name}\n" + perf[name].to_string())
        perf[name].to_csv(OUT / f"backtest_{name.lower()}.csv")
    stress = pd.DataFrame({ep: {p: cumulative(bt[p], q) for p in W} for ep, q in STRESS.items()}).T
    print("\nStress episodes (cumulative return)\n" + stress.to_string())
    stress.to_csv(OUT / "stress_episodes.csv")
    res.update(perf=perf, stress=stress, bt=bt)

    # ---------------------------------------------------------------- 5. compliance
    section("5. Mandate compliance - selected portfolio")
    mandate = Mandate()
    rep = compliance_report(W["Selected"], UNIVERSE, mandate, mu["5Y"], cov["5Y"], mu["10Y"], cov["10Y"])
    extra = pd.DataFrame([
        ["Volatility <= 12% (10Y, desmoothed)", port_table.loc["Selected", "sd 10Y desmoothed"], mandate.max_sd],
        ["Back-test CVaR 95% >= -10% (10Y)", perf["10Y"].loc["cvar95_q", "Selected"], mandate.max_cvar95_q],
        ["Inflation beta >= -1 (10Y)", port_table.loc["Selected", "Inflation beta 10Y"], mandate.min_inflation_beta],
    ], columns=["rule", "portfolio", "limit"])
    extra["status"] = np.where([extra.portfolio[0] <= extra.limit[0], extra.portfolio[1] >= extra.limit[1],
                                extra.portfolio[2] >= extra.limit[2]], "PASS", "FAIL")
    rep = pd.concat([rep, extra], ignore_index=True)
    print(rep.to_string(index=False))
    rep.to_csv(OUT / "compliance_selected.csv", index=False)
    rc = pd.DataFrame({"weight": W["Selected"],
                       "return contrib. 10Y": W["Selected"] * mu["10Y"],
                       "risk contrib. 10Y": risk_contributions(W["Selected"], cov["10Y"])})
    rc = rc[rc.weight > 0]
    print("\n" + rc.to_string())
    rc.to_csv(OUT / "risk_contributions_selected.csv")
    res.update(compliance=rep, risk_contrib=rc)

    # ---------------------------------------------------------------- 6. frontier
    section("6. Constrained efficient frontier (10Y) and max-Sharpe portfolio")
    ef = efficient_frontier(UNIVERSE, mandate, mu["10Y"], cov["10Y"], n_points=15)
    ms = max_sharpe(UNIVERSE, mandate, mu["10Y"], cov["10Y"], mu["10Y"]["cash"])
    ms_r, ms_s = portfolio_moments(ms, mu["10Y"], cov["10Y"])
    print(ef[["exp_return", "sd"]].T.to_string())
    print(f"\nMax-Sharpe (convex constraints): E[R] {ms_r:.2%}, sd {ms_s:.2%}, "
          f"Sharpe {(ms_r - mu['10Y']['cash']) / ms_s:.2f}")
    print(ms[ms > 0].round(4).to_string())
    ef.to_csv(OUT / "efficient_frontier_10y.csv", index=False)
    ms.to_csv(OUT / "max_sharpe_weights.csv", header=["weight"])
    res.update(frontier=ef, max_sharpe=ms)

    # ---------------------------------------------------------------- charts
    labels = UNIVERSE["label"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(ef["sd"], ef["exp_return"], color=SERIES[0], label="Constrained efficient frontier")
    for k in ASSETS:
        if k == "btc":
            continue
        ax.scatter(stats["10Y"].loc["sd_ann", k], mu["10Y"][k], s=18, color=INK_2, zorder=3)
        ax.annotate(labels[k].replace("Equity - ", "").replace("ETF - ", ""),
                    (stats["10Y"].loc["sd_ann", k], mu["10Y"][k]),
                    xytext=(4, -10) if k in ("hedge_funds", "commodity") else (4, 2),
                    textcoords="offset points", fontsize=7.5, color=INK_2)
    for p, c in [("Selected", SERIES[1]), ("Contingency", SERIES[2]), ("60/40", SERIES[6])]:
        ax.scatter(port_table.loc[p, "sd 10Y"], port_table.loc[p, "E[R] 10Y"], s=70, color=c, zorder=4,
                   edgecolor="white", linewidth=1.5, label=p)
    ax.scatter(ms_s, ms_r, s=70, marker="D", color=SERIES[3], edgecolor="white", zorder=4, label="Max Sharpe (extension)")
    ax.axhline(0.07, color=CRITICAL, lw=1, ls="--"); ax.axvline(0.12, color=CRITICAL, lw=1, ls="--")
    ax.text(0.121, 0.005, "risk limit 12%", color=CRITICAL, fontsize=8)
    ax.text(0.005, 0.073, "return target 7%", color=CRITICAL, fontsize=8)
    pct(ax, "x"); pct(ax, "y"); ax.set_xlabel("Volatility p.a. (10Y)"); ax.set_ylabel("Expected return p.a. (10Y mean)")
    ax.set_xlim(0, 0.38); ax.set_ylim(0, 0.31); ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Asset classes, mandate limits and the constrained frontier (BTC point omitted)")
    fig.savefig(FIG / "efficient_frontier.png"); plt.close(fig)

    wealth = (1 + window(bt, *WIN["10Y"])).cumprod() * 100
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for p, c in zip(W, [SERIES[1], SERIES[2], SERIES[6], SERIES[0]]):
        axes[0].plot(range(len(wealth)), wealth[p], color=c, label=p)
        dd = wealth[p] / wealth[p].cummax() - 1
        axes[1].plot(range(len(wealth)), dd, color=c)
    axes[0].set_ylabel("Q4 2015 = 100"); axes[0].legend(ncol=4, loc="upper left")
    axes[0].set_title("Back-test 2016-2025: the selected mix earns ~10% p.a. with ~40% smaller drawdowns than equities")
    pct(axes[1]); axes[1].set_ylabel("Drawdown")
    ticks = list(range(3, len(wealth), 8))
    axes[1].set_xticks(ticks, [wealth.index[i] for i in ticks])
    fig.savefig(FIG / "backtest_wealth_drawdown.png"); plt.close(fig)

    corr = window(rets, *WIN["10Y"])[ASSETS].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [labels[k].replace("Equity - ", "").replace("ETF - ", "").replace("Crypto - ", "") for k in ASSETS]
    ax.set_xticks(range(len(ASSETS)), short, rotation=90, fontsize=8); ax.set_yticks(range(len(ASSETS)), short, fontsize=8)
    ax.grid(False)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "#333")
    fig.colorbar(im, ax=ax, shrink=0.75); ax.set_title("Correlation of quarterly returns, 2016-2025")
    fig.savefig(FIG / "correlation_10y.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(rc)); wdt = 0.4
    ax.bar(x - wdt / 2, rc["weight"], wdt, color=SERIES[0], label="Capital weight")
    ax.bar(x + wdt / 2, rc["risk contrib. 10Y"], wdt, color=SERIES[1], label="Share of risk (10Y)")
    ax.set_xticks(x, [labels[k].replace("Equity - ", "").replace("ETF - ", "") for k in rc.index], rotation=20, fontsize=8.5)
    pct(ax); ax.legend(); ax.set_title("Selected portfolio: S&P 500 is 35% of capital but ~51% of risk")
    fig.savefig(FIG / "risk_contributions.png"); plt.close(fig)

    print(f"\nTables written to {OUT.relative_to(ROOT)}/, charts to {FIG.relative_to(ROOT)}/")
    return res


if __name__ == "__main__":
    main()
