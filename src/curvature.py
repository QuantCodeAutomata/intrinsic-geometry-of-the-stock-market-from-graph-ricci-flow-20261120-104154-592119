"""Ollivier-Ricci curvature with the paper's weight-proportional measure.

The GraphRicciCurvature library (Ni et al.; Context7 /saibalmars/graphriccicurvature)
implements Ollivier-Ricci curvature with an *exponential*, distance-decaying
neighbourhood measure (``m_i(j) ~ base^{-d(i,j)^{exp_power}}``). The paper's
measure (implementation_notes) is instead *weight-proportional*:

    m_i(i) = alpha,   m_i(j) = (1 - alpha) * w_ij / sum_k w_ik   for j != i,

i.e. mass grows with edge weight rather than shrinking with distance. Because
this measure differs from the library's built-in family, the measure and the
curvature formula are implemented here from the paper's definition (Appendix B),
while the optimal-transport subproblem itself -- computing the 1-Wasserstein
(earth mover's) distance between two neighbourhood measures -- is solved with
the POT (Python Optimal Transport) library's network-simplex solver
``ot.emd2``, rather than re-implementing a transportation-problem solver.

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
    """Weight-proportional measures m_i for every node (paper Appendix B).

    Returns (nodes, weight_matrix, M) where M[i] is the probability vector
    m_i over `nodes`, with m_i[i] = alpha and m_i[j] = (1-alpha) * w_ij / sum_k w_ik.
    """
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        W[i, j] = W[j, i] = d["weight"]

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # isolated node guard; should not occur on complete graphs
    M = (1.0 - alpha) * W / row_sums
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
