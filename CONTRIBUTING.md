# Contributing

Thanks for your interest! This is a research project — correctness and *honesty*
of results matter more than features.

## Ground rules

1. **No look-ahead, no leakage.** Any PR that touches features, splits, or the
   backtester must keep the no-look-ahead tests passing and add new ones for new
   behavior. See [`docs/warnings.md`](docs/warnings.md).
2. **Out-of-sample or it didn't happen.** Report metrics on a time-ordered holdout
   or walk-forward, always next to a naive baseline.
3. **Small, focused commits** with conventional-commit prefixes
   (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `build:`, `ci:`).

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                          # run the suite
ruff check backend tests scripts
```

## Adding things

- **A model:** register a builder in `backend/app/models/registry.py`. Document
  *when* it should be used.
- **A feature:** add a function returning a date-indexed frame and wire it into
  `backend/app/features/pipeline.py`. Add it to `docs/data_dictionary.md`.
- **A data source:** implement a fetcher returning the documented OHLCV contract;
  nothing else should need to change.

## What not to do

- Don't commit data files, `.env`, or model artifacts (they're git-ignored).
- Don't present a backtest as a guaranteed strategy. This is research.
