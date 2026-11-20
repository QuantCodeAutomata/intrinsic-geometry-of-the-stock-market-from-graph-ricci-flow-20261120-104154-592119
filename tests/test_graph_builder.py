"""Unit tests for src.graph_builder (correlation -> distance -> complete graph)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graph_builder import (
    build_complete_graph,
    correlation_matrix,
    correlation_to_distance,
    graph_from_correlation,
    mean_edge_weight,
)


def test_correlation_matrix_diagonal_is_one(synthetic_returns):
    corr = correlation_matrix(synthetic_returns)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert corr.shape == (15, 15)


def test_correlation_matrix_rejects_single_row():
    with pytest.raises(ValueError):
        correlation_matrix(pd.DataFrame({"A": [0.01], "B": [0.02]}))


def test_correlation_to_distance_range_and_metric_properties():
    corr = pd.DataFrame(
        [[1.0, 0.5, -1.0], [0.5, 1.0, -0.5], [-1.0, -0.5, 1.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    dist = correlation_to_distance(corr)
    assert np.allclose(np.diag(dist.values), 0.0)
    # Perfect correlation -> distance 0; perfect anti-correlation -> distance 2.
    assert np.isclose(dist.loc["A", "C"], 2.0)
    assert dist.values.min() >= 0.0
    assert dist.values.max() <= 2.0 + 1e-9
    # Symmetry, required for a proper metric (Mantegna 1999).
    assert np.allclose(dist.values, dist.values.T)


def test_build_complete_graph_edge_count():
    tickers = ["A", "B", "C", "D"]
    dist = pd.DataFrame(np.ones((4, 4)) - np.eye(4), index=tickers, columns=tickers)
    G = build_complete_graph(dist)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 4 * 3 // 2
    assert G["A"]["B"]["weight"] == 1.0


def test_graph_from_correlation_and_mean_edge_weight(synthetic_returns):
    corr = correlation_matrix(synthetic_returns)
    G = graph_from_correlation(corr)
    assert G.number_of_nodes() == 15
    w = mean_edge_weight(G)
    assert w > 0.0
