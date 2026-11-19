import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from foulup3 import analysis as A
from foulup3 import config, model
from foulup3.features import team_priors


def test_wilson_known_value():
    lo, hi = A.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_newcombe_brackets_the_difference_and_is_null_when_equal():
    d, lo, hi = A.newcombe_rd(90, 100, 80, 100)
    assert lo < d < hi and d == pytest.approx(0.10)
    d0, lo0, hi0 = A.newcombe_rd(50, 100, 50, 100)
    assert lo0 < 0 < hi0


def _simulate(n=6000, true_effect_logit=0.35, seed=1):
    """Confounded world: fouling is likelier late, and lateness alone raises the win rate."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(1, 24, n)
    p_foul = expit(1.2 - 0.12 * t)
    z = (rng.uniform(size=n) < p_foul).astype(int)
    base = 2.6 - 0.09 * t
    p1, p0 = expit(base + true_effect_logit), expit(base)
    y = (rng.uniform(size=n) < np.where(z == 1, p1, p0)).astype(int)
    df = pd.DataFrame({"t_entry": t, "foul": z, "leader_win": y})
    df["t_bin"] = pd.cut(df["t_entry"], A.TIME_BINS, labels=A.TIME_LABELS, include_lowest=True)
    return df, float((p1 - p0).mean())


def test_adjustment_removes_confounding_bias():
    df, truth = _simulate()
    naive = df[df.foul == 1].leader_win.mean() - df[df.foul == 0].leader_win.mean()
    adj = A.gcomp(df, A.RHS_TIME, n_boot=30)
    assert naive - truth > 0.01                      # the naive estimate is visibly biased upward
    assert abs(adj["rd"] - truth) < 0.012            # the adjusted one lands on the truth
    assert abs(A.ipw(df, A.RHS_TIME, n_boot=30)["rd"] - truth) < 0.02
    assert adj["rd_lo"] < adj["rd"] < adj["rd_hi"]


def _tg(rows):
    cols = ["game_id", "season", "season_type", "team_id", "pts_for", "pts_against",
            "ft_att", "ft_made", "tp_att", "tp_made", "to_q123"]
    return pd.DataFrame(rows, columns=cols)


def test_priors_never_use_the_game_itself_or_the_future():
    g = lambda gid, ftm: ("00223%05d" % gid, 2023, "RS", 7, 100, 90, 20, ftm, 30, 10, 3)
    base = team_priors(_tg([g(1, 10), g(2, 20), g(3, 5)])).set_index("game_id")
    assert base.loc["0022300001", "ft_pct_prior"] == pytest.approx(config.FT_LEAGUE_PRIOR)  # no history
    # Changing game 3 must not move game 2's prior; changing game 1 must.
    later = team_priors(_tg([g(1, 10), g(2, 20), g(3, 0)])).set_index("game_id")
    earlier = team_priors(_tg([g(1, 0), g(2, 20), g(3, 5)])).set_index("game_id")
    assert later.loc["0022300002", "ft_pct_prior"] == base.loc["0022300002", "ft_pct_prior"]
    assert earlier.loc["0022300002", "ft_pct_prior"] != base.loc["0022300002", "ft_pct_prior"]


def test_binomial_weights_sum_to_one():
    assert model.binom2(np.array([0.3, 0.76, 1.0])).sum(axis=1) == pytest.approx(1.0)


def test_win_probability_is_a_weighted_average_of_trip_outcomes():
    w = np.array([1.0, 0.94, 0.91])
    assert model.w_foul(0.0, w)[0] == pytest.approx(1.0)       # shooter never makes one
    assert model.w_foul(1.0, w)[0] == pytest.approx(0.91)      # shooter always makes both
    assert model.w_foul(0.5, w)[0] == pytest.approx(0.25 * 1 + 0.5 * 0.94 + 0.25 * 0.91)


def test_break_even_finds_the_crossing_or_none():
    f = np.array([0.5, 0.6, 0.7, 0.8])
    assert model.break_even(f, np.array([0.04, 0.02, -0.02, -0.04])) == pytest.approx(0.65)
    assert model.break_even(f, np.array([0.04, 0.03, 0.02, 0.01])) is None
