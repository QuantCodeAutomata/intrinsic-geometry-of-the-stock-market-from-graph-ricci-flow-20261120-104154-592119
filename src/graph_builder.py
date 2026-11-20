"""Complete weighted correlation graphs (paper Sec. 2, "Stock market as a metric graph").

Daily log-returns are turned into a Pearson correlation matrix C_ij, which is
mapped to a metric distance d_ij = sqrt(2 * (1 - C_ij)) (the standard
Mantegna 1999 correlation-distance metric, used throughout the paper). The
resulting complete graph on all stocks carries d_ij as its edge weight.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily log-returns, one column per ticker."""
    if returns.shape[0] < 2:
        raise ValueError("Need at least 2 return observations to correlate")
    corr = returns.corr(method="pearson")
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-8), "diagonal must be 1"
    return corr


def correlation_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Map correlations to edge weights w_ij = sqrt(2 * (1 - C_ij)) (paper Sec. 2).

    This is a proper metric (Mantegna 1999): w_ij in [0, 2], w_ii = 0,
    symmetric, and satisfies the triangle inequality.
    """
    values = 2.0 * (1.0 - corr.values)
    values = np.clip(values, 0.0, None)  # guard against floating-point noise < 0
    dist = pd.DataFrame(np.sqrt(values), index=corr.index, columns=corr.columns)
    np.fill_diagonal(dist.values, 0.0)
    return dist


def build_complete_graph(distance: pd.DataFrame) -> nx.Graph:
    """Complete weighted graph on all tickers, edge weight = distance d_ij."""
    tickers = list(distance.index)
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            G.add_edge(a, b, weight=float(distance.loc[a, b]))
    assert G.number_of_edges() == len(tickers) * (len(tickers) - 1) // 2
    return G


def graph_from_correlation(corr: pd.DataFrame) -> nx.Graph:
    """Full pipeline: correlation matrix -> distance matrix -> complete graph."""
    distance = correlation_to_distance(corr)
    return build_complete_graph(distance)


def mean_edge_weight(G: nx.Graph) -> float:
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    return float(weights.mean())
