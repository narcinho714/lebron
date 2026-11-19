"""A data-anchored break-even model: how good a free-throw shooter must the opponent be before
fouling up 3 stops paying off?

Idea. A free throw is close to a weighted coin flip, so once the leader fouls, the number of free
throws made (0, 1 or 2) is as-good-as random given the shooter's skill f. We can therefore estimate,
from real fouls, the leader's win rate after k made free throws, w_k, and rebuild the win probability
of fouling for ANY shooter skill:

        W_foul(f) = sum_k  Binomial(k; 2, f) * w_k

The alternative, not fouling, is the covariate-adjusted win rate of NO_FOUL states re-weighted to
look like the foul states (the 'effect on the treated' baseline). The curve  W_foul(f) - baseline
is the advantage of fouling; its zero crossing, if any, is the break-even free-throw percentage.

Only penalty fouls that produced exactly two free throws are used for w_k (clean 2-shot trips).
Caveat: w_k come from a few hundred trips, so the bands are wide - read the shape, not the decimals.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit

from .analysis import RHS_FULL

F_GRID = np.round(np.arange(0.40, 1.0001, 0.01), 2)


def binom2(f: np.ndarray | float) -> np.ndarray:
    """P(0, 1, 2 made of two) for shooter skill f. Shape (len(f), 3)."""
    f = np.atleast_1d(np.asarray(f, float))
    return np.stack([(1 - f) ** 2, 2 * f * (1 - f), f ** 2], axis=1)


def trip_table(df: pd.DataFrame) -> pd.DataFrame:
    """Leader win rate after 0 / 1 / 2 made free throws, from real two-shot intentional fouls."""
    trips = df[(df["foul"] == 1) & (df["ft_att"] == 2)]
    out = trips.groupby("ft_made").agg(n=("leader_win", "size"), win=("leader_win", "mean"),
                                       tie=("reg_outcome", lambda s: (s == "tie_to_ot").mean()),
                                       lose_reg=("reg_outcome", lambda s: (s == "lose_reg").mean()))
    return out.reindex([0, 1, 2]).reset_index()


def w_foul(f, w: np.ndarray) -> np.ndarray:
    return binom2(f) @ np.asarray(w, float)


def _estimate(df, X, idx):
    """w_k and the no-foul baseline for one (re)sample of row indices."""
    sub = df.iloc[idx]
    y = sub["leader_win"].to_numpy()
    foul = sub["foul"].to_numpy() == 1
    trips = foul & (sub["ft_att"].to_numpy() == 2)
    made = sub["ft_made"].to_numpy()
    pooled = y[trips].mean()
    w = np.array([y[trips & (made == k)].mean() if (trips & (made == k)).sum() >= 3 else pooled
                  for k in (0, 1, 2)])
    Xs = X[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sm.GLM(y[~foul], Xs[~foul], family=sm.families.Binomial()).fit(maxiter=200).params
    baseline = float(expit(Xs[trips] @ b).mean())          # no-foul win prob for the foul trips
    return w, baseline


def curve(df: pd.DataFrame, f_grid: np.ndarray = F_GRID, n_boot: int = 500, seed: int = 0) -> dict:
    X = np.asarray(patsy.dmatrix(RHS_FULL, df))
    w, base = _estimate(df, X, np.arange(len(df)))
    adv = w_foul(f_grid, w) - base
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        wb, bb = _estimate(df, X, rng.integers(0, len(df), len(df)))
        boots.append(w_foul(f_grid, wb) - bb)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"f": f_grid, "advantage": adv, "lo": lo, "hi": hi, "w": w, "baseline": base,
            "break_even": break_even(f_grid, adv)}


def break_even(f_grid: np.ndarray, adv: np.ndarray) -> float | None:
    """First free-throw % at which fouling stops helping (advantage crosses zero), else None."""
    s = np.sign(adv)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    if not len(idx):
        return None
    i = idx[0]
    return float(f_grid[i] - adv[i] * (f_grid[i + 1] - f_grid[i]) / (adv[i + 1] - adv[i]))
