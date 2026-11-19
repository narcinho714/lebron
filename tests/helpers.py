"""Tiny synthetic-game builder so each rule of the state scanner can be tested in isolation."""
from __future__ import annotations

import pandas as pd

COLS = ["game_id", "season", "season_type", "row", "period", "clock_s", "msg", "act", "side",
        "is_miss", "is_3pt", "ft_special", "ft_n", "ft_of", "is_kicked", "home_score", "away_score"]


class Game:
    """Fourth-quarter event stream. Scores are tracked for you (post-event, like the real data)."""

    def __init__(self, home: int = 100, away: int = 100, gid: str = "0022300001"):
        self.gid, self.h, self.a, self.rows = gid, home, away, []
        self._add(12, 720.0, "")          # start of 4th period carries the pre-period score

    def _add(self, msg, clock, side, act=0, miss=False, three=False, ft_n=0, ft_of=0, special=False):
        self.rows.append(dict(
            game_id=self.gid, season=2023, season_type="RS", row=len(self.rows), period=4,
            clock_s=float(clock), msg=msg, act=act, side=side, is_miss=miss, is_3pt=three,
            ft_special=special, ft_n=ft_n, ft_of=ft_of, is_kicked=False,
            home_score=self.h, away_score=self.a))
        return self

    def _score(self, side, pts):
        if side == "H":
            self.h += pts
        else:
            self.a += pts

    def fg(self, side, clock, made=True, three=False):
        pts = 3 if three else 2
        if made:
            self._score(side, pts)
        return self._add(1 if made else 2, clock, side, miss=not made, three=three)

    def reb(self, side, clock):
        return self._add(4, clock, side)

    def to(self, side, clock):
        return self._add(5, clock, side)

    def foul(self, side, clock, act=1):
        return self._add(6, clock, side, act=act)

    def ft(self, side, clock, n, of, made=True):
        if made:
            self._score(side, 1)
        return self._add(3, clock, side, miss=not made, ft_n=n, ft_of=of)

    def timeout(self, side, clock):
        return self._add(9, clock, side, act=1)

    def sub(self, side, clock):
        return self._add(8, clock, side)

    def events(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)[COLS]

    def games(self, final_home: int | None = None, final_away: int | None = None) -> pd.DataFrame:
        return pd.DataFrame([{
            "game_id": self.gid, "final_home": self.h if final_home is None else final_home,
            "final_away": self.a if final_away is None else final_away,
            "home_team_id": 1.0, "away_team_id": 2.0}])
