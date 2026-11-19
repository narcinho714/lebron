"""Scan play-by-play and extract one 'decision state' per game for the H_strong hypothesis.

A STATE is the first moment, with strictly less than `defn.max_seconds` left in the period, when one
team (the LEADER) is ahead by exactly `defn.up_by` and the other team (the OPPONENT) has the ball.

The state is RESOLVED by the first thing that happens next:
    FOUL     the leader commits a non-shooting defensive foul before the opponent shoots
    NO_FOUL  the opponent shoots, turns the ball over, or commits an offensive foul

Possession is not recorded in the play-by-play, so it is inferred event by event (see `_next_poss`).
That inference is the riskiest piece of the project: it is unit-tested and sanity-checked in
`diagnostics()` on the real data.

Design note: the state is anchored to when the *situation* begins (`t_entry`), not to when the foul
or shot happens. Conditioning on the resolution time would be conditioning on a consequence of the
decision being studied. `res_delay` records how long after entry the resolution came.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import H_STRONG, Definition


def other(side: str) -> str:
    return "A" if side == "H" else "H"


def _next_poss(poss, msg, act, side, is_kicked, ft_special, ft_n, ft_of, is_miss):
    """Possession after one event. None = unknown / loose ball."""
    if not side and msg not in (10, 12):
        return poss
    if msg == 1:                                   # made field goal -> other team inbounds
        return other(side)
    if msg == 2:                                   # missed field goal -> wait for the rebound
        return None
    if msg == 4:                                   # rebound (player or team)
        return side
    if msg == 5:                                   # turnover (incl. steals)
        return other(side)
    if msg == 6:                                   # foul
        if act in config.TECH_FOUL_ACTIONS:
            return poss
        return other(side)                         # the fouled team gets the ball / free throws
    if msg == 3:                                   # free throw
        if ft_special or ft_of == 0 or ft_n != ft_of:
            return poss
        return None if is_miss else other(side)    # last FT: made -> other inbounds, miss -> rebound
    if msg == 7 and is_kicked:                     # kicked-ball violation
        return other(side)
    if msg in (10, 12):                            # jump ball / start of period
        return None
    return poss


def scan_game(a: dict, lo: int, hi: int, defn: Definition) -> tuple[list[dict], int]:
    """Scan events [lo, hi) of one game (arrays in dict `a`). Returns (states, possession_conflicts)."""
    states: list[dict] = []
    conflicts = 0
    periods_ok = (lambda p: p >= 4) if defn.include_ot else (lambda p: p == 4)

    cur_period = None
    poss = None
    prev_margin = int(a["home_score"][lo] - a["away_score"][lo])
    foul = {"H": 0, "A": 0}
    tos = {"H": 0, "A": 0}
    cond_since = None          # clock when the current (leader up N, opp has ball) episode began
    snap = None                # counters at episode start
    resolved_ep = False
    end_margin_p4 = prev_margin
    cur = None                 # the state being tracked for post-resolution facts

    for i in range(lo, hi):
        p = int(a["period"][i])
        if p != cur_period:
            cur_period, poss, cond_since, resolved_ep = p, None, None, False
            foul = {"H": 0, "A": 0}
        clk = a["clock_s"][i]
        msg = int(a["msg"][i]); act = int(a["act"][i]); sd = a["side"][i]

        # ---- 1. classify this event against the PRE-event situation ----
        if (cond_since is not None and not resolved_ep and periods_ok(p) and clk < defn.max_seconds
                and (not defn.first_state_only or not states)):
            leader = "H" if prev_margin == defn.up_by else "A"
            opp = other(leader)
            kind = None
            if msg == 6 and sd == leader and act in defn.foul_actions:
                kind = "FOUL"
            elif sd == opp and (msg in (1, 2, 5) or (msg == 6 and act in config.OFFENSIVE_FOUL_ACTIONS)):
                kind = "NO_FOUL"
            elif sd == leader and msg in (1, 2):
                conflicts += 1                      # leader shooting while we think opp has ball
            if kind:
                rec = {
                    "game_id": a["game_id"][lo], "season": int(a["season"][lo]),
                    "season_type": a["season_type"][lo], "leader": leader, "kind": kind,
                    "t_entry": min(defn.max_seconds, float(cond_since)), "t_res": float(clk),
                    "period": p, "q4_leader_fouls": snap[leader], "q4_opp_fouls": snap[opp],
                    "to_q4_leader": snap["to_" + leader], "to_q4_opp": snap["to_" + opp],
                    "ft_att": 0, "ft_made": 0, "opp_shot": "", "opp_made": False,
                    "fouled_later": False, "res_idx": i,
                }
                if kind == "FOUL":
                    j = i + 1
                    while j < hi:
                        mj = a["msg"][j]
                        if mj in (7, 8, 9):         # violation / sub / timeout between foul and FTs
                            j += 1
                            continue
                        if mj == 4 and rec["ft_att"] > 0:   # team rebound after a missed non-final FT
                            j += 1
                            continue
                        if mj == 3 and a["side"][j] == opp and not a["ft_special"][j]:
                            rec["ft_att"] += 1
                            rec["ft_made"] += int(not a["is_miss"][j])
                            if a["ft_of"][j] and a["ft_n"][j] == a["ft_of"][j]:
                                break               # final free throw of the trip
                            j += 1
                            continue
                        break
                else:
                    if msg in (1, 2):
                        rec["opp_shot"] = "3PA" if a["is_3pt"][i] else "2PA"
                        rec["opp_made"] = msg == 1
                    elif msg == 5:
                        rec["opp_shot"] = "TO"
                    else:
                        rec["opp_shot"] = "OFF_FOUL"
                states.append(rec)
                cur = rec
                resolved_ep = True
        elif (cur is not None and cur["kind"] == "NO_FOUL" and sd == cur["leader"] and msg == 6
              and act in defn.foul_actions and clk < defn.max_seconds and i > cur["res_idx"]
              and p == cur["period"]):
            cur["fouled_later"] = True

        # ---- 2. apply the event ----
        poss = _next_poss(poss, msg, act, sd, a["is_kicked"][i], a["ft_special"][i],
                          int(a["ft_n"][i]), int(a["ft_of"][i]), a["is_miss"][i])
        margin = int(a["home_score"][i] - a["away_score"][i])
        if sd and msg == 6 and act in config.TEAM_FOUL_ACTIONS:
            foul[sd] += 1
        if sd and msg == 9 and act != 7:
            tos[sd] += 1
        if p == 4:
            end_margin_p4 = margin

        # ---- 3. update the episode flag ----
        cond = poss is not None and (
            (margin == defn.up_by and poss == "A") or (margin == -defn.up_by and poss == "H"))
        if cond and cond_since is None:
            cond_since = clk
            snap = {"H": foul["H"], "A": foul["A"], "to_H": tos["H"], "to_A": tos["A"]}
        elif not cond:
            cond_since, resolved_ep = None, False
        prev_margin = margin

    for s in states:
        sign = 1 if s["leader"] == "H" else -1
        s["reg_margin"] = sign * end_margin_p4
        s.pop("res_idx", None)
    return states, conflicts


def find_states(events: pd.DataFrame, games: pd.DataFrame, defn: Definition = H_STRONG):
    """All decision states across `events`, joined to game outcomes. Returns (states, diagnostics)."""
    ev = events.reset_index(drop=True)
    a = {c: ev[c].to_numpy() for c in
         ["game_id", "season", "season_type", "period", "clock_s", "msg", "act", "side", "is_miss",
          "is_3pt", "ft_special", "ft_n", "ft_of", "is_kicked", "home_score", "away_score"]}
    gid = a["game_id"]
    starts = np.r_[0, np.flatnonzero(gid[1:] != gid[:-1]) + 1]
    ends = np.r_[starts[1:], len(gid)]
    out, conflicts = [], 0
    for lo, hi in zip(starts, ends):
        s, c = scan_game(a, int(lo), int(hi), defn)
        out.extend(s)
        conflicts += c
    states = pd.DataFrame(out)
    diag = {"games_scanned": int(len(starts)), "states": len(states), "possession_conflicts": conflicts}
    if states.empty:
        return states, diag
    g = games.set_index("game_id")[["final_home", "final_away", "home_team_id", "away_team_id"]]
    states = states.join(g, on="game_id")
    home_won = states["final_home"] > states["final_away"]
    states["leader_win"] = np.where(states["leader"] == "H", home_won, ~home_won).astype(int)
    states["foul"] = (states["kind"] == "FOUL").astype(int)
    states["res_delay"] = states["t_entry"] - states["t_res"]
    states["reg_outcome"] = np.select(
        [states["reg_margin"] > 0, states["reg_margin"] == 0], ["win_reg", "tie_to_ot"], "lose_reg")
    return states, diag
