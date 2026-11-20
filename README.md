# Intrinsic Geometry of the Stock Market from Graph Ricci Flow

Replication of selected experiments from **"Intrinsic Geometry of the Stock Market
from Graph Ricci Flow"** (arXiv:2510.15942).

This repository implements the ARCUS (Adaptive Ricci Curvature Unsupervised
Surgery) algorithm: it builds a complete weighted correlation graph of a stock
universe, computes Ollivier-Ricci curvature (ORC) on every edge, runs a
normalised discrete Ricci flow, and performs curvature-gap surgery to obtain a
parameter-free hierarchical community decomposition.

Two experiments are reproduced:

1. **`exp1`** — ARCUS hierarchical decomposition of the NASDAQ-100 correlation
   graph (the paper's main NASDAQ-100 experiment).
2. **`exp2`** — Validation against the Mantegna (1999) / Tumminello et al.
   (2008) ten-stock benchmark, comparing ARCUS against MST average-linkage and
   PCA sign-clustering baselines using ARI, NMI and modularity Q.

## Repository layout

```
src/     ARCUS algorithm: graph construction, Ollivier-Ricci curvature, Ricci
         flow + surgery, hierarchical decomposition, baselines, metrics,
         config.py (all parameters, one place, with their paper source)
exp/     exp1.py, exp2.py — runnable experiment scripts
data/    data_loader.py — OHLCV loading through the QuantX Dataloader REST API
tests/   pytest unit tests for src/ and data-dependent tests for exp/
results/ RESULTS.md and the metrics/plots produced by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Both experiments need market data (NASDAQ-100 bars for `exp1`, the ten legacy
tickers of the Mantegna benchmark for `exp2`) loaded through
`data/data_loader.py`, which calls the QuantX Dataloader REST API.

## Running the tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access needed
python -m pytest                        # full suite, needs the API keys above
```

## Running the experiments

```bash
python -m exp.exp1        # NASDAQ-100 decomposition (reduced universe, see RESULTS.md)
python -m exp.exp1 --full # full paper universe (101 tickers)
python -m exp.exp2        # Mantegna ten-stock benchmark
```

Every run overwrites the tables, metrics and plots in `results/`; the
narrative, comparison numbers and verdicts in `results/RESULTS.md` are updated
by hand after inspecting the new files.

## Reproduce everything

```bash
./reproduce.sh
```

See `results/RESULTS.md` for the comparison against the paper's reported
numbers, the deviations from the paper, and the claims not reproduced.
