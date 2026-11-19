"""Pre-game covariates for each decision state.

The rule that matters: every team-level number (free-throw %, 3-point %, strength) is computed from
games BEFORE the one being analysed. Using the full-season value would quietly leak the future
(including the game itself) into the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def team_priors(team_games: pd.DataFrame) -> pd.DataFrame:
    """Expanding, shrunk, strictly-prior team stats per team-game.

    Order within a season is regular season first, then playoffs, then by game id (ids are issued
    in schedule order, so this is chronological to within same-day games).
    """
    tg = team_games.copy()
    tg["po"] = (tg["season_type"] == "PO").astype(int)
    tg = tg.sort_values(["team_id", "season", "po", "game_id"]).reset_index(drop=True)
    grp = tg.groupby(["team_id", "season"], sort=False)

    for c in ("ft_att", "ft_made", "tp_att", "tp_made"):
        tg[c + "_prior"] = grp[c].cumsum() - tg[c]            # sum over strictly earlier games
    tg["n_prior"] = grp.cumcount()
    tg["net"] = tg["pts_for"] - tg["pts_against"]
    tg["net_prior"] = grp["net"].cumsum() - tg["net"]

    tg["ft_pct_prior"] = ((tg["ft_made_prior"] + config.FT_PRIOR_ATT * config.FT_LEAGUE_PRIOR)
                          / (tg["ft_att_prior"] + config.FT_PRIOR_ATT))
    tg["tp_pct_prior"] = ((tg["tp_made_prior"] + config.TP_PRIOR_ATT * config.TP_LEAGUE_PRIOR)
                          / (tg["tp_att_prior"] + config.TP_PRIOR_ATT))
    tg["net_pg_prior"] = tg["net_prior"] / (tg["n_prior"] + config.NET_PRIOR_GAMES)
    keep = ["game_id", "team_id", "ft_pct_prior", "tp_pct_prior", "net_pg_prior", "n_prior", "to_q123"]
    return tg[keep]


def build_features(states: pd.DataFrame, games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach covariates to the state table. Adds leader_/opp_ prefixed columns."""
    if states.empty:
        return states
    pri = team_priors(team_games)
    df = states.copy()
    df["leader_team_id"] = np.where(df["leader"] == "H", df["home_team_id"], df["away_team_id"])
    df["opp_team_id"] = np.where(df["leader"] == "H", df["away_team_id"], df["home_team_id"])

    for who in ("leader", "opp"):
        p = pri.rename(columns={
            "team_id": f"{who}_team_id",
            "ft_pct_prior": f"{who}_ft_pct_prior", "tp_pct_prior": f"{who}_tp_pct_prior",
            "net_pg_prior": f"{who}_net_pg_prior", "n_prior": f"{who}_n_prior",
            "to_q123": f"{who}_to_q123"})
        df = df.merge(p, on=["game_id", f"{who}_team_id"], how="left")

    df["to_used_leader"] = df["leader_to_q123"] + df["to_q4_leader"]
    df["to_used_opp"] = df["opp_to_q123"] + df["to_q4_opp"]
    df["home_leader"] = (df["leader"] == "H").astype(int)
    df["season_c"] = (df["season"] - 2010) / 10.0
    df["post_2018"] = (df["season"] >= 2018).astype(int)
    df["bonus_leader"] = (df["q4_leader_fouls"] >= 5).astype(int)
    return df
