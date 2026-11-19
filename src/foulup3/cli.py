"""Command line entry point:  python -m foulup3.cli {ingest,states,analyze}"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config


def cmd_ingest(args) -> None:
    from .ingest import ingest_all
    seasons = set(args.seasons) if args.seasons else None
    ingest_all(config.RAW_SOURCE, config.PROCESSED, seasons=seasons)


def cmd_states(args) -> None:
    from .features import build_features
    from .states import find_states
    events = pd.read_parquet(config.PROCESSED / "events_late.parquet")
    games = pd.read_parquet(config.PROCESSED / "games.parquet")
    team_games = pd.read_parquet(config.PROCESSED / "team_games.parquet")
    states, diag = find_states(events, games, config.H_STRONG)
    feats = build_features(states, games, team_games)
    feats.to_parquet(config.PROCESSED / "states.parquet", index=False)
    (config.PROCESSED / "states_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))
    print(feats["kind"].value_counts().to_string())


def cmd_analyze(args) -> None:
    from .report import run_report
    run_report()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="foulup3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="raw archives -> parquet tables")
    p.add_argument("--seasons", type=int, nargs="*", help="e.g. 2022 2023 (default: all)")
    p.set_defaults(fn=cmd_ingest)
    sub.add_parser("states", help="extract decision states + features").set_defaults(fn=cmd_states)
    sub.add_parser("analyze", help="estimates, figures, reports/results.md").set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
