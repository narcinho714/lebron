from pathlib import Path

import pandas as pd
import pytest

from foulup3 import config
from foulup3.config import Definition
from foulup3.states import find_states
from helpers import Game


def run(g: Game, defn: Definition = config.H_STRONG, **final):
    return find_states(g.events(), g.games(**final), defn)


def test_opponent_shoots_without_foul_is_no_foul():
    g = Game().fg("H", 40, three=True)               # home goes up 3 at 40s; away inbounds
    g.fg("A", 15, made=False, three=True)            # away shoots a three at 15s
    st, diag = run(g)
    assert len(st) == 1 and diag["possession_conflicts"] == 0
    s = st.iloc[0]
    assert (s.leader, s.kind, s.opp_shot) == ("H", "NO_FOUL", "3PA")
    assert s.t_entry == 24.0 and s.t_res == 15.0     # entry capped at the window edge
    assert s.leader_win == 1


def test_leader_foul_is_foul_with_free_throws():
    g = Game().fg("H", 20, three=True)               # up 3 at 20s (entry inside the window)
    g.foul("H", 18).ft("A", 18, 1, 2, made=True).ft("A", 18, 2, 2, made=False)
    st, _ = run(g)
    s = st.iloc[0]
    assert (s.kind, s.t_entry, s.t_res, s.ft_att, s.ft_made) == ("FOUL", 20.0, 18.0, 2, 1)


def test_free_throw_lookahead_skips_team_rebound_and_violation():
    g = Game().fg("H", 20, three=True)
    g.foul("H", 18)
    g._add(7, 18, "A")                                # violation between foul and free throws
    g.ft("A", 18, 1, 2, made=False).reb("A", 18).ft("A", 18, 2, 2, made=True)
    st, _ = run(g)
    assert (st.iloc[0].ft_att, st.iloc[0].ft_made) == (2, 1)


def test_shot_before_the_window_creates_no_state():
    g = Game().fg("H", 40, three=True)
    g.fg("A", 30, made=False).reb("H", 29)           # opp shoots at 30s: outside the <24s window
    st, _ = run(g)
    assert st.empty


def test_window_is_strictly_less_than_24_seconds():
    g = Game().fg("H", 40, three=True).fg("A", 24, made=False)
    assert run(g)[0].empty
    g = Game().fg("H", 40, three=True).fg("A", 23.5, made=False)
    assert len(run(g)[0]) == 1


def test_only_leader_fouls_count_as_treatment():
    g = Game().fg("H", 40, three=True)
    g.foul("A", 20, act=26)                            # opponent's offensive foul with the ball
    st, _ = run(g)
    assert st.iloc[0].kind == "NO_FOUL" and st.iloc[0].opp_shot == "OFF_FOUL"


def test_shooting_foul_is_not_intentional():
    g = Game().fg("H", 40, three=True).fg("A", 18, made=False, three=True)
    g.foul("H", 18, act=2)                             # shooting foul, not a deliberate foul
    st, _ = run(g)
    assert st.iloc[0].kind == "NO_FOUL" and not st.iloc[0].fouled_later


def test_fouled_later_flag():
    g = Game().fg("H", 40, three=True).fg("A", 18, made=False, three=True)
    g.reb("A", 17).foul("H", 10)                      # leader fouls on a later possession
    st, _ = run(g)
    assert st.iloc[0].kind == "NO_FOUL" and st.iloc[0].fouled_later


def test_leader_possession_is_not_a_state():
    g = Game().fg("H", 40, three=True).fg("A", 35, made=False).reb("H", 34)
    g.fg("H", 15, made=False)                          # leader has the ball: no decision
    st, diag = run(g)
    assert st.empty and diag["possession_conflicts"] == 0


def test_missed_shot_hands_ball_to_the_rebounder():
    g = Game().fg("H", 40, three=True).fg("A", 30, made=False).reb("A", 29)  # opp offensive board
    g.fg("A", 15, made=False)                          # opp still has the ball inside the window
    st, _ = run(g)
    assert len(st) == 1 and st.iloc[0].kind == "NO_FOUL"


def test_exactly_three_required():
    g = Game().fg("H", 40, three=False).fg("A", 15, made=False)   # up 2, not 3
    assert run(g)[0].empty


def test_trailing_team_up_three_roles_flip():
    g = Game(home=100, away=100).fg("A", 20, three=True)           # away leads by 3
    g.foul("A", 18).ft("H", 18, 1, 2).ft("H", 18, 2, 2)
    st, _ = run(g)
    s = st.iloc[0]
    assert (s.leader, s.kind, s.ft_made, s.reg_margin) == ("A", "FOUL", 2, 1)


def test_regulation_outcome_labels():
    g = Game().fg("H", 20, three=True).foul("H", 18)
    g.ft("A", 18, 1, 2).ft("A", 18, 2, 2).fg("H", 10, made=False).reb("A", 9).fg("A", 3, three=True)
    s = run(g, final_home=110, final_away=112)[0].iloc[0]       # tied then lost in OT
    assert s.reg_outcome == "lose_reg" and s.leader_win == 0


GOLDEN = config.PROCESSED / "events_late.parquet"


@pytest.mark.skipif(not GOLDEN.exists(), reason="run `make ingest` first")
def test_golden_game_pacers_celtics_2024_ecf_game1():
    """Pacers up 3, Celtics ball inside 24s, no foul -> Brown ties it (real data)."""
    ev = pd.read_parquet(GOLDEN)
    ev = ev[ev.game_id == "0042300301"]
    if ev.empty:
        pytest.skip("2023-24 playoffs not ingested")
    games = pd.read_parquet(config.PROCESSED / "games.parquet")
    st, _ = find_states(ev, games)
    s = st.iloc[0]
    assert (s.leader, s.kind, s.leader_win, s.reg_outcome) == ("A", "NO_FOUL", 0, "tie_to_ot")
    st_all, _ = find_states(ev, games, Definition(first_state_only=False))
    last = st_all.iloc[-1]                              # the famous possession: Brown's tying three
    assert (last.opp_shot, last.opp_made) == ("3PA", True) and last.t_res <= 8
