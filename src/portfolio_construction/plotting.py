"""Consistent, minimal chart style (thin marks, recessive grid, fixed colour order)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CRITICAL = "#d03b3b"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def pct(ax, axis: str = "y", decimals: int = 0) -> None:
    from matplotlib.ticker import PercentFormatter
    fmt = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
