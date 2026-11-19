PY ?= python3
EXT = data/external/nba_data

.PHONY: data ingest states simple analyze report test all

# 1. fetch the public play-by-play archive (large: ~2 GB, shallow clone)
data:
	GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/shufinskiy/nba_data $(EXT)

# 2. raw archives -> tidy parquet tables
ingest:
	PYTHONPATH=src $(PY) -m foulup3.cli ingest

# 3. play-by-play -> one row per "up 3, under 24s" decision state
states:
	PYTHONPATH=src $(PY) -m foulup3.cli states

# 4a. THE SIMPLE TEST: raw foul vs defend win %, two eras
simple:
	PYTHONPATH=src $(PY) -m foulup3.simple

# 4b. (optional, deeper) estimates (naive, regression-adjusted, IPW) + figures + reports/results.md
analyze:
	PYTHONPATH=src $(PY) -m foulup3.cli analyze

test:
	PYTHONPATH=src $(PY) -m pytest -q

all: ingest states simple analyze
