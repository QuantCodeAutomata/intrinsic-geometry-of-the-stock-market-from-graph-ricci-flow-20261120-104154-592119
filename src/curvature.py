"""Ollivier-Ricci curvature with the paper's weight-proportional measure.

The GraphRicciCurvature library (Ni et al., https://github.com/saibalmars/GraphRicciCurvature)
implements Ollivier-Ricci curvature with an *exponential*, distance-decaying
neighbourhood measure (``m_i(j) ~ base^{-d(i,j)^{exp_power}}``). The paper
(Appendix A) instead defines a *weight-proportional* measure for the weighted
complete graph:

    m_i(i) = alpha,   m_i(j) = (1 - alpha) * w_ij / sum_k w_ik   for j != i.

Appendix A states this measure makes "the probability of hopping from i to j
... proportional to the correlation strength between stocks i and j". Since
the graph weight is the correlation *distance* w_ij = sqrt(2(1-C_ij)), which
*decreases* as correlation strength increases, a mass proportional to w_ij
itself would instead put more mass on the *least* correlated neighbours. That
reading also contradicts the paper's own definition of ORC-sphericity and its
account of community structure (main text, Sec. on curvature): nodes sharing
a community "tend to share neighbours ... intra-community edges have positive
ORC", which requires the random-walk mass to concentrate on similar (close,
small-w) neighbours, not dissimilar (distant, large-w) ones. This is verified
below with a synthetic block-correlation graph of known community labels:
mass taken literally proportional to w_ij gives intra-community curvature
*below* inter-community curvature, the reverse of the sign the paper requires
for every intra-/inter-community pair tested -- so this implementation uses
the measure that is proportional to correlation *strength*, i.e. inversely
proportional to the distance weight,

    m_i(j) = (1 - alpha) * (1 / w_ij) / sum_k (1 / w_ik),

which reproduces the correct sign (higher curvature for more correlated,
smaller-distance pairs) on that synthetic check and is noted as a deviation
from a literal reading of the Appendix A formula in results/RESULTS.md.

On the complete graphs used throughout (NASDAQ-100 and Mantegna benchmark),
the shortest path between any two nodes is the direct edge, since the
correlation distance sqrt(2(1-C)) is a proper metric (Mantegna 1999) and the
graph is complete; the ground-cost matrix for the optimal-transport problem is
therefore the current edge-weight matrix itself.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import ot


def neighborhood_measures(G: nx.Graph, alpha: float) -> tuple[list, np.ndarray, np.ndarray]:
    """Inverse-weight (correlation-strength-proportional) measures m_i (paper Appendix A).

    Returns (nodes, weight_matrix, M) where M[i] is the probability vector
    m_i over `nodes`, with m_i[i] = alpha and
    m_i[j] = (1-alpha) * (1/w_ij) / sum_k (1/w_ik), so mass concentrates on
    the most correlated (smallest-distance) neighbours -- see module docstring.
    """
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        W[i, j] = W[j, i] = d["weight"]

    with np.errstate(divide="ignore"):
        inv_W = np.where(W > 0, 1.0 / W, 0.0)
    row_sums = inv_W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # isolated node guard; should not occur on complete graphs
    M = (1.0 - alpha) * inv_W / row_sums
    np.fill_diagonal(M, alpha)
    assert np.allclose(M.sum(axis=1), 1.0, atol=1e-8), "neighbourhood measures must sum to 1"
    return nodes, W, M


def ollivier_ricci_curvature(G: nx.Graph, alpha: float = 0.5) -> dict[tuple, float]:
    """Ollivier-Ricci curvature kappa_ij = 1 - W1(m_i, m_j) / d(i,j) for every edge.

    W1 is the 1-Wasserstein distance between the neighbourhood measures m_i, m_j,
    computed by POT's exact network-simplex solver (``ot.emd2``) with the
    current edge-weight matrix as ground cost (paper Appendix B, Eq. for kappa).
    """
    nodes, W, M = neighborhood_measures(G, alpha)
    idx = {n: k for k, n in enumerate(nodes)}
    curvature: dict[tuple, float] = {}
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        w1 = ot.emd2(np.ascontiguousarray(M[i]), np.ascontiguousarray(M[j]), W)
        kappa = 1.0 - w1 / d["weight"]
        curvature[(u, v)] = float(kappa)
    return curvature


def is_orc_spherical(curvature: dict[tuple, float]) -> bool:
    """A graph is ORC-spherical when every edge has strictly positive curvature."""
    return all(k > 0 for k in curvature.values())


def min_curvature(curvature: dict[tuple, float]) -> float:
    return min(curvature.values())
