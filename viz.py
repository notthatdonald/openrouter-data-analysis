"""Shared matplotlib styling for the notebooks.

Colours are a validated categorical palette (first three slots pass colour-blind
separation checks for every pair); text never uses series colours.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"

BLUE = "#2a78d6"    # series 1
ORANGE = "#eb6834"  # series 2
AQUA = "#1baf7a"    # series 3 (low contrast: always direct-label it)
NEUTRAL = "#c9c8c2" # de-emphasised context marks


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    })


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional muted subtitle."""
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=TEXT,
                 pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2, va="bottom")


def hbar(ax, labels, values, fmt="{:,.0f}", color=BLUE) -> None:
    """Horizontal bars, largest at top, value at each tip."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    bars = ax.barh(labels, values, height=0.62, color=color)
    ax.grid(axis="y", visible=False)
    vmax = max(values) if values else 0
    for b, v in zip(bars, values):
        ax.text(b.get_width() + vmax * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", fontsize=9, color=TEXT_2)
    ax.set_xlim(0, vmax * 1.12 if vmax else 1)
    ax.spines["left"].set_visible(False)
