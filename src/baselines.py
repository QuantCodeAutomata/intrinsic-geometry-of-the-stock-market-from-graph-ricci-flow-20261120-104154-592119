"""Baseline community-detection methods for comparison against ARCUS (Appendix F.2).

MST average-linkage requires the number of clusters k to be specified in
advance; PCA sign clustering on the correlation matrix's second and third
eigenvectors is parameter-free but, as the paper notes, cannot separate
financial from technology on the Mantegna benchmark.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def mst_average_linkage(distance: pd.DataFrame, k: int) -> dict[str, int]:
    """Minimum-spanning-tree-based average-linkage clustering into k clusters.

    Builds the MST of the distance graph (networkx, Kruskal's algorithm),
    then cuts the corresponding average-linkage hierarchical dendrogram built
    from the full distance matrix into k clusters (scipy hierarchical
    clustering), following the standard Mantegna/Tumminello MST-partition
    comparison protocol.
    """
    tickers = list(distance.index)
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return dict(zip(tickers, labels.tolist()))


def pca_sign_clusters(corr: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Sign-based clustering on the eigenvectors of the correlation matrix.

    Returns one label dict per non-leading eigenvector v2, v3, ... with
    cluster = sign(v_component); the sign of the market-mode eigenvector v1
    is not informative and is skipped (paper Appendix F.2).
    """
    tickers = list(corr.index)
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]  # descending eigenvalues: v1 = market mode
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    results = {}
    for k in range(1, min(3, len(tickers))):
        vec = eigvecs[:, k]
        labels = (vec >= 0).astype(int)
        results[f"v{k + 1}"] = {
            "eigenvalue": float(eigvals[k]),
            "labels": dict(zip(tickers, labels.tolist())),
        }
    return results
