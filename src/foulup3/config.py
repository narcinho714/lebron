"""Paths, priors and the one definition everything else hangs off: what counts as a 'state'."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_SOURCE = DATA / "external" / "nba_data" / "datasets"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

# Shrinkage priors for pre-game team features (constants, so they cannot leak the future).
FT_LEAGUE_PRIOR = 0.77
TP_LEAGUE_PRIOR = 0.355
FT_PRIOR_ATT = 150.0       # pseudo free-throw attempts behind the prior
TP_PRIOR_ATT = 300.0       # pseudo three-point attempts behind the prior
NET_PRIOR_GAMES = 10.0     # pseudo games of zero point differential

# stats.nba.com EVENTMSGACTIONTYPE codes for EVENTMSGTYPE 6 (fouls), verified against the data.
# Non-shooting defensive fouls that can be an intentional "foul up 3":
#   1 personal, 5 inbound, 6 away-from-play, 28 personal take, 31 transition take
INTENTIONAL_FOUL_ACTIONS = frozenset({1, 5, 6, 28, 31})
# Fouls that count toward the team-foul total (bonus) for the period.
TEAM_FOUL_ACTIONS = frozenset({1, 2, 3, 5, 6, 9, 14, 15, 28, 31})
OFFENSIVE_FOUL_ACTIONS = frozenset({4, 26})
TECH_FOUL_ACTIONS = frozenset({10, 11, 12, 13, 16, 17, 18, 25, 30, 32, 33})


@dataclass(frozen=True)
class Definition:
    """What a decision state is. H_STRONG is the hypothesis under test:

    'Foul every single time when up 3 with under 24 seconds left.'
    A state is the first moment, with strictly less than `max_seconds` on the clock, at which one
    team leads by exactly `up_by` and the other team has the ball.
    """

    up_by: int = 3
    max_seconds: float = 24.0
    include_ot: bool = False
    first_state_only: bool = True
    foul_actions: frozenset = INTENTIONAL_FOUL_ACTIONS


H_STRONG = Definition()
