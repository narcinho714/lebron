"""Figures. Palette validated with the dataviz validator (blue/orange pass CVD + normal-vision gates);
text always wears ink colours, never series colours, and every multi-series chart has a legend."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def _style(ax, xlabel=None, ylabel=None, title=None, subtitle=None):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.figure.text(0.01, 0.985, title, color=INK, fontsize=12, fontweight="bold", va="top")
    if subtitle:
        ax.figure.text(0.01, 0.925, subtitle, color=INK2, fontsize=9, va="top")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def adoption(ad: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _style(ax, ylabel="Leaders who fouled at first chance", title="Fouling up 3 has gone mainstream",
           subtitle="Share of 'up 3, under 24s, opponent has the ball' situations where the leader fouled")
    ax.fill_between(ad["season"], ad["lo"] * 100, ad["hi"] * 100, color=BLUE, alpha=0.12, lw=0)
    ax.plot(ad["season"], ad["rate"] * 100, color=BLUE, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("Season starting (95% Wilson band)", color=INK2, fontsize=9)
    _save(fig, path)


def forest(rows: pd.DataFrame, label_col: str, path: Path, title: str, subtitle: str,
           est="rd", lo="rd_lo", hi="rd_hi", sublabel_col: str | None = None):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8, 0.62 * n + 1.9))
    _style(ax, xlabel="Change in leader's win probability if they foul (percentage points)",
           title=title, subtitle=subtitle)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ys = np.arange(n)[::-1]
    for y, (_, r) in zip(ys, rows.iterrows()):
        ax.plot([r[lo] * 100, r[hi] * 100], [y, y], color=BLUE, lw=2.5, solid_capstyle="round")
        ax.plot(r[est] * 100, y, "o", color=BLUE, ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(r[hi] * 100 + 0.35, y, f"{r[est] * 100:+.1f}", color=INK, va="center", fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys)
    labels = [str(v) for v in rows[label_col]]
    if sublabel_col:
        labels = [f"{a}\n{b}" for a, b in zip(labels, rows[sublabel_col])]
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    _save(fig, path)


def mechanism(mech: pd.DataFrame, path: Path):
    order = ["No foul", "Foul: no free throws (not in bonus)", "Foul: opponent shot free throws"]
    m = mech.set_index("group").loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    _style(ax, xlabel="Share of games", title="How the game was settled in regulation",
           subtitle="Leader up 3 with under 24s left; by what the leader did")
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(order))[::-1]
    parts = [("win_reg_pct", "Leader wins in regulation", BLUE), ("tie_to_ot_pct", "Tied, goes to OT", ORANGE),
             ("lose_reg_pct", "Leader loses in regulation", INK2)]
    for y, name in zip(ys, order):
        left = 0.0
        for col, _, color in parts:
            w = m.loc[name, col] * 100
            ax.barh(y, w, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
            if w >= 6:
                ax.text(left + w / 2, y, f"{w:.0f}%", color="white", ha="center", va="center", fontsize=9,
                        fontweight="bold")
            left += w
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}\n(n={int(m.loc[n, 'n'])})" for n in order], color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    ax.legend(handles, [p[1] for p in parts], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, labelcolor=INK2, fontsize=9)
    _save(fig, path)


def by_time(tab: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    _style(ax, xlabel="Seconds left when the situation began", ylabel="Leader wins the game",
           title="Win rate by time left: foul vs no foul",
           subtitle="Raw rates, 95% Wilson intervals (fouling is rare early, so those bars are wide)")
    x = np.arange(len(tab))
    for off, lab, col, color in ((-0.1, "foul", "win_foul", BLUE), (0.1, "nofoul", "win_nofoul", ORANGE)):
        y = tab[col] * 100
        ax.errorbar(x + off, y, yerr=[y - tab[col + "_lo"] * 100, tab[col + "_hi"] * 100 - y], fmt="o",
                    color=color, ms=7, lw=2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={a}/{b})" for g, a, b in zip(tab["group"], tab["n_foul"], tab["n_nofoul"])],
                       color=INK2, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend([plt.Line2D([], [], marker="o", color=BLUE, lw=0), plt.Line2D([], [], marker="o", color=ORANGE, lw=0)],
              ["Leader fouled", "Leader did not foul"], frameon=False, labelcolor=INK2, fontsize=9,
              loc="lower left")
    _save(fig, path)


def breakeven(c: dict, path: Path, observed_ft: float | None = None, league_ft: float = 0.76):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax, xlabel="Free-throw % of the player fouled", ylabel="Fouling advantage (pp of win probability)",
           title="How good a shooter must they be before fouling stops helping?",
           subtitle="Modelled from the free throws made on real intentional fouls; shaded = 95% bootstrap band")
    f = c["f"] * 100
    ax.fill_between(f, c["lo"] * 100, c["hi"] * 100, color=BLUE, alpha=0.14, lw=0)
    ax.plot(f, c["advantage"] * 100, color=BLUE, lw=2.2)
    ax.axhline(0, color=INK2, lw=1)
    top = ax.get_ylim()[1]
    marks = [(league_ft * 100, "league average 76%", "left")]
    if observed_ft:
        marks.append((observed_ft * 100, f"shot on fouled trips {observed_ft:.0%}", "right"))
    for x, lab, side in marks:
        ax.axvline(x, color=GRAY, lw=1, ls=(0, (4, 3)))
        ax.text(x + (0.5 if side == "left" else -0.5), top * 0.97, lab, color=INK2, fontsize=8.5,
                va="top", ha=side)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _save(fig, path)
