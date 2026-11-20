"""Clustering-agreement metrics used to validate ARCUS partitions (paper Appendix F.2).

ARI and NMI are standard clustering-agreement scores; modularity Q is the
usual Newman-Girvan community-quality score. All three are taken from
scikit-learn / NetworkX rather than re-implemented.
"""

from __future__ import annotations

import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_agreement(labels_true: list, labels_pred: list) -> dict[str, float]:
    """Adjusted Rand Index and Normalised Mutual Information between two labelings."""
    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))
    return {"ARI": ari, "NMI": nmi}


def modularity(G: nx.Graph, communities: list[set]) -> float:
    """Newman-Girvan modularity Q of a partition of G into `communities`."""
    return float(nx.algorithms.community.modularity(G, communities, weight="weight"))
