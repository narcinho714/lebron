"""Raw stats.nba.com play-by-play (shufinskiy/nba_data archives) -> tidy parquet tables.

Outputs (data/processed/):
    events_late.parquet  period >= 4 events, with the fields the state scanner needs
    games.parquet        one row per game: team ids, final score
    team_games.parquet   one row per team-game: shooting / timeout counts and points for/against
    ingest_quality.csv   per-season data-quality diagnostics

Things learned from the raw files (keep these in mind when touching this module):
  * GAME_ID loses its leading zeros when parsed as a number -> read as str, zfill(10).
  * SCORE is "visitor - home"; it is only populated on scoring events -> forward-fill per game.
  * PERSON1TYPE: 2 home team, 3 visitor team, 4 home player, 5 visitor player. For team events
    (e.g. team rebounds, timeouts) PLAYER1_ID is the team id.
  * Row order is chronological; EVENTNUM is NOT monotone. Never sort on it.
"""
from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

USECOLS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
    "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
    "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_TEAM_ID",
]
EVENT_COLS = [
    "game_id", "season", "season_type", "row", "period", "clock_s", "msg", "act", "side",
    "is_miss", "is_3pt", "ft_special", "ft_n", "ft_of", "is_kicked", "home_score", "away_score",
]
FILE_RE = re.compile(r"^nbastats(?P<po>_po)?_(?P<season>\d{4})\.tar\.xz$")


def list_season_files(source: Path, seasons: set[int] | None = None):
    """(season, 'RS'|'PO', path) for every stats.nba.com play-by-play archive in `source`."""
    out = []
    for path in sorted(source.glob("nbastats*_*.tar.xz")):
        m = FILE_RE.match(path.name)  # skips nbastatsv3_* and friends
        if not m:
            continue
        season = int(m["season"])
        if seasons and season not in seasons:
            continue
        out.append((season, "PO" if m["po"] else "RS", path))
    return out


def read_season_file(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:xz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
        return pd.read_csv(tf.extractfile(member), usecols=USECOLS, dtype={"GAME_ID": str})


def parse_clock(s: pd.Series) -> pd.Series:
    """'M:SS' (or 'M:SS.t') -> seconds remaining in the period."""
    parts = s.astype("string").str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def normalise(raw: pd.DataFrame, season: int, season_type: str) -> pd.DataFrame:
    df = raw.copy()
    df["game_id"] = df["GAME_ID"].str.zfill(10)
    df["row"] = np.arange(len(df))
    df["season"] = season
    df["season_type"] = season_type
    df["period"] = df["PERIOD"].fillna(0).astype(int)
    df["clock_s"] = parse_clock(df["PCTIMESTRING"])
    df["msg"] = df["EVENTMSGTYPE"].fillna(0).astype(int)
    df["act"] = df["EVENTMSGACTIONTYPE"].fillna(0).astype(int)

    # Which side (H/A) acted. PERSON1TYPE first; fall back to which description column is filled.
    p = df["PERSON1TYPE"]
    has_h = df["HOMEDESCRIPTION"].notna()
    has_v = df["VISITORDESCRIPTION"].notna()
    df["side"] = np.select(
        [p.isin([2, 4]), p.isin([3, 5]), has_h & ~has_v, has_v & ~has_h],
        ["H", "A", "H", "A"],
        default="",
    )

    # Both descriptions matter: a blocked shot has the shot in one column and BLOCK in the other.
    desc = (
        df["HOMEDESCRIPTION"].fillna("") + " " + df["VISITORDESCRIPTION"].fillna("") + " "
        + df["NEUTRALDESCRIPTION"].fillna("")
    ).str.upper()
    df["is_miss"] = desc.str.contains("MISS", regex=False)
    df["is_3pt"] = desc.str.contains("3PT", regex=False)
    df["is_kicked"] = desc.str.contains("KICKED", regex=False)
    df["ft_special"] = desc.str.contains("TECHNICAL|FLAGRANT|CLEAR PATH", regex=True)
    ft = desc.str.extract(r"(\d)\s+OF\s+(\d)")
    df["ft_n"] = pd.to_numeric(ft[0], errors="coerce").fillna(0).astype(int)
    df["ft_of"] = pd.to_numeric(ft[1], errors="coerce").fillna(0).astype(int)

    # Scores: SCORE is "visitor - home", only on scoring plays. Forward-fill within the game.
    sc = df["SCORE"].str.extract(r"(\d+)\s*-\s*(\d+)")
    df["away_score"] = pd.to_numeric(sc[0])
    df["home_score"] = pd.to_numeric(sc[1])
    filled = df.groupby("game_id", sort=False)[["away_score", "home_score"]].ffill().fillna(0)
    df["away_score"] = filled["away_score"].astype(int)
    df["home_score"] = filled["home_score"].astype(int)

    # Team id of the acting side: team events carry it in PLAYER1_ID, player events in the team col.
    df["tid"] = np.where(p.isin([2, 3]), df["PLAYER1_ID"], df["PLAYER1_TEAM_ID"])
    return df


def summarise_games(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-game and per-team-game tables from the full (all periods) normalised events."""
    sided = df[df["side"].isin(["H", "A"])]
    tid_h = sided[sided["side"] == "H"].dropna(subset=["tid"]).groupby("game_id")["tid"].first()
    tid_a = sided[sided["side"] == "A"].dropna(subset=["tid"]).groupby("game_id")["tid"].first()

    last = df.groupby("game_id", sort=False).tail(1).set_index("game_id")
    games = pd.DataFrame({
        "season": last["season"], "season_type": last["season_type"],
        "home_team_id": tid_h, "away_team_id": tid_a,
        "final_home": last["home_score"], "final_away": last["away_score"],
        "max_period": df.groupby("game_id")["period"].max(),
    }).reset_index()

    flags = df.assign(
        ft_att=(df["msg"] == 3) & ~df["ft_special"],
        tp_att=df["msg"].isin([1, 2]) & df["is_3pt"],
        to_q123=(df["msg"] == 9) & (df["act"] != 7) & (df["period"] <= 3),
    )
    flags["ft_made"] = flags["ft_att"] & ~flags["is_miss"]
    flags["tp_made"] = flags["tp_att"] & (flags["msg"] == 1)
    cols = ["ft_att", "ft_made", "tp_att", "tp_made", "to_q123"]
    side_stats = (
        flags[flags["side"].isin(["H", "A"])].groupby(["game_id", "side"])[cols].sum().reset_index()
    )

    rows = []
    for side, tcol, pf, pa in (("H", "home_team_id", "final_home", "final_away"),
                               ("A", "away_team_id", "final_away", "final_home")):
        t = games[["game_id", "season", "season_type", tcol, pf, pa]].rename(
            columns={tcol: "team_id", pf: "pts_for", pa: "pts_against"})
        t["side"] = side
        rows.append(t.merge(side_stats[side_stats["side"] == side].drop(columns="side"),
                            on="game_id", how="left"))
    team_games = pd.concat(rows, ignore_index=True)
    team_games[cols] = team_games[cols].fillna(0).astype(int)
    return games, team_games


def quality_row(df: pd.DataFrame, games: pd.DataFrame, season: int, season_type: str) -> dict:
    fouls = df[df["msg"] == 6]
    q4 = df[df["period"] == 4]
    return {
        "season": season, "season_type": season_type,
        "games": len(games),
        "games_with_team_ids": int(games[["home_team_id", "away_team_id"]].notna().all(axis=1).sum()),
        "games_with_q4": int(q4["game_id"].nunique()),
        "foul_side_known": round(float((fouls["side"] != "").mean()), 4) if len(fouls) else np.nan,
        "clock_parsed": round(float(df["clock_s"].notna().mean()), 4),
        "has_scores": round(float((df["home_score"] + df["away_score"] > 0).mean()), 4),
    }


def ingest_all(source: Path = config.RAW_SOURCE, out_dir: Path = config.PROCESSED,
               seasons: set[int] | None = None, verbose: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_season_files(source, seasons)
    if not files:
        raise FileNotFoundError(f"no nbastats*_*.tar.xz archives under {source} (run `make data`)")
    events, games, team_games, quality = [], [], [], []
    for season, stype, path in files:
        df = normalise(read_season_file(path), season, stype)
        g, tg = summarise_games(df)
        events.append(df.loc[df["period"] >= 4, EVENT_COLS])
        games.append(g)
        team_games.append(tg)
        quality.append(quality_row(df, g, season, stype))
        if verbose:
            print(f"{season} {stype}: {len(g):>5} games, {len(df):>8} events", flush=True)
    pd.concat(events, ignore_index=True).to_parquet(out_dir / "events_late.parquet", index=False)
    pd.concat(games, ignore_index=True).to_parquet(out_dir / "games.parquet", index=False)
    pd.concat(team_games, ignore_index=True).to_parquet(out_dir / "team_games.parquet", index=False)
    pd.DataFrame(quality).to_csv(out_dir / "ingest_quality.csv", index=False)
