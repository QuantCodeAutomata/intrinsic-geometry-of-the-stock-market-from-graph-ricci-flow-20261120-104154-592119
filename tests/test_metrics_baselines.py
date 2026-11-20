"""Unit tests for src.metrics and src.baselines."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.baselines import mst_average_linkage, pca_sign_clusters
from src.metrics import clustering_agreement, modularity


def test_clustering_agreement_identical_labelings_is_perfect():
    labels = ["a", "a", "b", "b", "c", "c"]
    agreement = clustering_agreement(labels, labels)
    assert agreement["ARI"] == pytest.approx(1.0)
    assert agreement["NMI"] == pytest.approx(1.0)


def test_clustering_agreement_independent_labelings_is_low():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = rng.integers(0, 3, size=300)
    agreement = clustering_agreement(y_true, y_pred)
    assert agreement["ARI"] < 0.2


def test_modularity_high_for_well_separated_cliques():
    G = nx.Graph()
    for block in range(2):
        nodes = [f"{block}_{i}" for i in range(5)]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j], weight=1.0)
    G.add_edge("0_0", "1_0", weight=0.01)
    communities = [{f"0_{i}" for i in range(5)}, {f"1_{i}" for i in range(5)}]
    q = modularity(G, communities)
    assert q > 0.3


def test_mst_average_linkage_recovers_two_well_separated_blocks():
    tickers = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
    n = len(tickers)
    dist = np.ones((n, n)) * 1.9
    np.fill_diagonal(dist, 0.0)
    for i in range(4):
        for j in range(4):
            if i != j:
                dist[i, j] = 0.1
                dist[4 + i, 4 + j] = 0.1
    distance = pd.DataFrame(dist, index=tickers, columns=tickers)
    labels = mst_average_linkage(distance, k=2)
    group_a = {labels[f"A{i}"] for i in range(4)}
    group_b = {labels[f"B{i}"] for i in range(4)}
    assert len(group_a) == 1
    assert len(group_b) == 1
    assert group_a != group_b


def test_pca_sign_clusters_skips_market_mode_and_returns_v2_v3():
    rng = np.random.default_rng(2)
    tickers = [f"T{i}" for i in range(6)]
    base = rng.normal(size=(400, 6))
    base[:, 3:] *= -1  # anti-correlate the second half with the first
    returns = pd.DataFrame(base, columns=tickers)
    corr = returns.corr()
    result = pca_sign_clusters(corr)
    assert set(result.keys()) <= {"v2", "v3"}
    for entry in result.values():
        assert set(entry["labels"].keys()) == set(tickers)
        assert set(entry["labels"].values()) <= {0, 1}
