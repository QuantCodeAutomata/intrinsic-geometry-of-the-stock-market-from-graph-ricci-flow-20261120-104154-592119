"""Unit tests for src.curvature (Ollivier-Ricci curvature, Appendix B)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.curvature import (
    is_orc_spherical,
    min_curvature,
    neighborhood_measures,
    ollivier_ricci_curvature,
)
from src.graph_builder import build_complete_graph, correlation_matrix, correlation_to_distance


def test_neighborhood_measures_sum_to_one_and_respect_alpha():
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + abs(u - v) * 0.1
    nodes, W, M = neighborhood_measures(G, alpha=0.5)
    assert np.allclose(M.sum(axis=1), 1.0)
    assert np.allclose(np.diag(M), 0.5)


def test_complete_graph_equal_weights_matches_closed_form_curvature():
    """On K_n with all equal weights and idleness alpha, the shortest-path
    metric between distinct nodes is the discrete (0/1) metric, so the
    1-Wasserstein distance between any two neighbourhood measures equals
    their total-variation distance. For n=6, alpha=0.5: m_i(i)=0.5,
    m_i(j)=0.1 (j != i); the TV distance between m_i and m_j is 0.4, so
    kappa = 1 - W1/d(i,j) = 1 - 0.4 = 0.6 (closed-form sanity check for the
    weight-proportional measure of Appendix B on a fully symmetric graph).
    """
    G = nx.complete_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    curvature = ollivier_ricci_curvature(G, alpha=0.5)
    for kappa in curvature.values():
        assert kappa == pytest.approx(0.6, abs=1e-6)


def test_is_orc_spherical_true_and_false():
    assert is_orc_spherical({(0, 1): 0.1, (1, 2): 0.2})
    assert not is_orc_spherical({(0, 1): 0.1, (1, 2): -0.01})


def test_min_curvature():
    assert min_curvature({(0, 1): 0.3, (1, 2): 0.1, (2, 3): 0.5}) == pytest.approx(0.1)


def test_synthetic_correlation_graph_is_orc_spherical(synthetic_returns):
    """A correlation graph built from clearly positively-correlated blocks should
    be ORC-spherical before any flow, matching the paper's starting condition
    (research objective: 'Verify the graph is ORC-spherical before any flow')."""
    corr = correlation_matrix(synthetic_returns)
    dist = correlation_to_distance(corr)
    G = build_complete_graph(dist)
    curvature = ollivier_ricci_curvature(G, alpha=0.5)
    assert is_orc_spherical(curvature)
    assert min_curvature(curvature) > 0.0
