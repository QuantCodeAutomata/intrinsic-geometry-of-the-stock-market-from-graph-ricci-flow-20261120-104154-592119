# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A replication of "Intrinsic Geometry of the Stock Market from Graph Ricci
Flow" (arXiv:2510.15942): the ARCUS algorithm (Ollivier-Ricci curvature,
normalised Ricci flow, curvature-gap surgery) applied to stock correlation
graphs, with two experiments — the NASDAQ-100 hierarchical decomposition
(`exp/exp1.py`) and the Mantegna (1999) ten-stock benchmark (`exp/exp2.py`).

## Layout

- `src/` — flat library modules: `graph_builder.py`, `curvature.py`,
  `ricci_flow.py`, `arcus.py`, `baselines.py`, `metrics.py`, `config.py`.
- `exp/` — experiment scripts, run as `python -m exp.exp1` / `exp.exp2`.
- `data/` — `data_loader.py` loads OHLCV bars through the QuantX Dataloader
  REST API; downloaded bars are cached under `data/cache/` (ignored by git).
- `tests/` — pytest tests for `src/` (synthetic fixtures) and for the
  experiments (a cached market-data slice, marked `market_data`).
- `results/` — `RESULTS.md` plus every table, metric file and plot the
  experiments produce; rerunning an experiment overwrites its files here.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE constants, each with a paper reference or "not specified in the
  paper". Do not hardcode parameters elsewhere.
- Market data is loaded only through `data/data_loader.py`; no other module
  calls an external data API.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` (see `.env.example`); they must never appear in code, logs or
  results.
- Tests: `python -m pytest -m "not market_data"` runs without network access;
  the full suite additionally needs the API keys above.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"
python -m exp.exp1            # reduced NASDAQ-100 universe (see RESULTS.md)
python -m exp.exp1 --full      # full 101-ticker paper universe
python -m exp.exp2             # Mantegna ten-stock benchmark
./reproduce.sh
```

After a rerun, the files in `results/` are overwritten; the numbers and
verdicts in `results/RESULTS.md` must then be checked against the new files
and updated by hand.
