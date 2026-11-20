"""Shared fixtures for the ARCUS Ricci-flow replication test suite."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src import config


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader / Massive API call"
    )


@pytest.fixture(scope="session")
def synthetic_returns() -> pd.DataFrame:
    """Small synthetic daily-return panel with a clear block/community structure.

    Three groups of 5 correlated tickers each, used to test that the ARCUS
    pipeline (correlation -> distance -> ORC -> Ricci flow -> surgery) recovers
    a planted community structure on data with a known answer.
    """
    rng = np.random.default_rng(0)
    n_days = 500
    tickers = [f"G{g}_{i}" for g in range(3) for i in range(5)]
    factors = rng.normal(size=(n_days, 3)) * 0.01
    returns = {}
    for g in range(3):
        for i in range(5):
            idio = rng.normal(size=n_days) * 0.003
            returns[f"G{g}_{i}"] = factors[:, g] + idio
    return pd.DataFrame(returns, columns=tickers)


@pytest.fixture(scope="session")
def mantegna_slice_returns(request):
    """A short real-market slice of the Mantegna benchmark tickers.

    Loads about one year of daily bars for a few of the ten benchmark tickers
    through the shared ``data.data_loader.load_ohlcv`` path and caches the
    result between test runs with pytest's cache API. Skipped when
    DATALOADER_API_KEY / MASSIVE_API_KEY are not set.
    """
    if not (os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY")):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set: skipping market-data test")

    cache_key = "arcus/mantegna_slice_returns"
    cached = request.config.cache.get(cache_key, None)
    if cached is not None:
        return pd.DataFrame(cached["data"], index=pd.to_datetime(cached["index"]))

    from data.data_loader import load_ohlcv, log_returns

    tickers = config.MANTEGNA_TICKERS[:4]  # a few instruments of the paper's universe
    bars, meta = load_ohlcv(
        tickers, "2003-09-10", "2004-03-31",  # proportionally shorter window
        identity_date=config.MANTEGNA_IDENTITY_DATE,
    )
    if meta["failed"]:
        pytest.skip(f"Could not load Mantegna slice tickers: {meta['failed']}")
    returns = log_returns(bars)
    request.config.cache.set(cache_key, {
        "data": returns.values.tolist(),
        "index": returns.index.astype(str).tolist(),
        "columns": list(returns.columns),
    })
    returns.columns = list(returns.columns)
    return returns
