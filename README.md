# foul-up-3

**Should the leading team foul when up 3 with under 24 seconds left?**
A data-driven myth-buster using 29 seasons of NBA play-by-play (1996-97 to 2024-25).

## Hypothesis tested (H_strong)

Leader is up exactly 3, strictly under 24 seconds remain, the opponent has the ball.
Claim: the leader should foul **every single time**.

## Headline result: NOT PROVEN

| | |
|---|---|
| Situations | 3,918 (487 fouled, 3,431 not) |
| Naive effect on win prob. | +2.9 pp |
| Fully adjusted effect | **+2.2 pp**, 95% CI -0.5 to +5.1 |
| Verdict | Point estimate favours fouling; interval includes zero |

Fouling teams are stronger and foul with less time left, which inflates the naive gap.
The adjustment ladder (time, timeouts, team fouls, opponent shooting, quality, era) removes that,
using both g-computation and inverse-propensity weighting. A break-even model asks what free-throw %
the fouled player needs for fouling to pay off. Full write-up: [`reports/results.md`](reports/results.md).

![ladder](reports/figures/ladder.png)

## Reproduce

```bash
pip install -e . pytest
git clone --depth 1 https://github.com/shufinskiy/nba_data data/external/nba_data   # GIT_LFS_SKIP_SMUDGE=1
make ingest states analyze   # ~2 min for ingest
make test
```

## Structure

- `src/foulup3/config.py` - paths, priors, the `Definition` of H_strong
- `ingest.py` - raw play-by-play to tidy parquet
- `states.py` - infers possession event by event, finds the decision state, classifies FOUL / NO_FOUL
- `features.py` - leak-free pre-game priors (shrunk FT%, 3P%, net rating from earlier games only)
- `analysis.py` - adjustment ladder, balance, subgroups
- `model.py` - break-even free-throw model
- `plots.py`, `report.py` - figures and generated `results.md`
- `tests/` - synthetic-game unit tests plus a golden real-game check

## Limitations

- Observational: unmeasured confounding (who is fouled, coach intent) can remain.
- Only 487 fouling cases; the interval is wide (about 3.2x more data needed to confirm +2.2 pp).
- Fouled players shot 70% vs a 76% team prior, so team FT% is a rough proxy.
- Possession is inferred from events, not observed (1 conflict in 36,668 games).
