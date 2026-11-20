"""Normalised discrete Ricci flow and curvature-gap-based surgery (paper Appendix B).

One flow iteration updates every edge weight by its curvature,

    w_ij^(l+1) = (1 - kappa_ij^(l)) * w_ij^(l),

then rescales all weights so the mean edge weight is restored to its value
before the iteration (methodology Step 3). Iterating this flow on an
ORC-spherical graph is expected to develop a discrete neckpinch: the
curvature distribution polarises into a high population (intra-community,
kappa+) and a near-zero population (surgery candidates, kappa-), separated by
an empty gap. The curvature-gap ratio

    G = [mu(kappa+) - mu(kappa-)] / sigma_pooled

(paper Appendix B) measures how sharp that separation is; the two populations
are found with 1-D k-means (scikit-learn ``KMeans``, k=2) on the curvature
values, which is the standard tool for splitting a sample into two clusters
and needs no manually tuned threshold.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from .curvature import ollivier_ricci_curvature


def gap_ratio(curvature: dict[tuple, float]) -> tuple[float, dict[tuple, float], set[tuple]]:
    """Curvature-gap ratio G and the inferred surgery set (paper Appendix B).

    Returns (G, labels_by_edge, surgery_edges) where surgery_edges is the set
    of edges assigned to the lower (near-zero) curvature population.
    """
    edges = list(curvature.keys())
    values = np.array([curvature[e] for e in edges]).reshape(-1, 1)
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0, {}, set()

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(values)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()
    low_label = int(np.argmin(centers))
    high_label = 1 - low_label

    low_vals = values[labels == low_label].flatten()
    high_vals = values[labels == high_label].flatten()
    if len(low_vals) == 0 or len(high_vals) == 0:
        return 0.0, {}, set()

    mu_low, mu_high = low_vals.mean(), high_vals.mean()
    n_low, n_high = len(low_vals), len(high_vals)
    pooled_var = (n_low * low_vals.var() + n_high * high_vals.var()) / (n_low + n_high)
    sigma_pooled = np.sqrt(pooled_var) if pooled_var > 0 else 1e-12
    G = float((mu_high - mu_low) / sigma_pooled)

    surgery_edges = {edges[k] for k in range(len(edges)) if labels[k] == low_label}
    return G, {e: curvature[e] for e in edges}, surgery_edges


def flow_step(G: nx.Graph, alpha: float, target_mean: float) -> tuple[nx.Graph, dict[tuple, float]]:
    """One normalised Ricci-flow iteration; returns the updated graph and curvature."""
    curvature = ollivier_ricci_curvature(G, alpha=alpha)
    H = G.copy()
    for (u, v), kappa in curvature.items():
        H[u][v]["weight"] = max((1.0 - kappa) * G[u][v]["weight"], 1e-10)
    current_mean = np.mean([d["weight"] for _, _, d in H.edges(data=True)])
    scale = target_mean / current_mean
    for u, v in H.edges():
        H[u][v]["weight"] *= scale
    return H, curvature


def run_flow_to_convergence(
    G: nx.Graph,
    alpha: float,
    g_min: float,
    stability_window: int,
    max_iter: int,
    no_split_g: float,
) -> dict:
    """Iterate the normalised Ricci flow until the curvature-gap criterion fires.

    Stops when G >= g_min and the surgery edge set is identical for
    `stability_window` consecutive iterations, or after `max_iter` iterations.
    Returns a dict with the flow history and the final (graph, surgery_edges).
    """
    target_mean = np.mean([d["weight"] for _, _, d in G.edges(data=True)])
    history = []
    stable_count = 0
    last_surgery_set = None
    current = G
    final_surgery: set[tuple] = set()
    final_curvature: dict[tuple, float] = {}

    for iteration in range(1, max_iter + 1):
        current, curvature = flow_step(current, alpha, target_mean)
        G_ratio, _, surgery_edges = gap_ratio(curvature)
        history.append({"iteration": iteration, "G": G_ratio, "n_surgery_edges": len(surgery_edges)})
        final_surgery, final_curvature = surgery_edges, curvature

        if surgery_edges and surgery_edges == last_surgery_set:
            stable_count += 1
        else:
            stable_count = 1
        last_surgery_set = surgery_edges

        if G_ratio >= g_min and stable_count >= stability_window:
            return {
                "graph": current,
                "curvature": final_curvature,
                "surgery_edges": final_surgery,
                "G": G_ratio,
                "iterations": iteration,
                "history": history,
                "converged": True,
            }

    final_G = history[-1]["G"] if history else 0.0
    return {
        "graph": current,
        "curvature": final_curvature,
        "surgery_edges": final_surgery,
        "G": final_G,
        "iterations": max_iter,
        "history": history,
        "converged": final_G >= no_split_g,
    }


def run_flow_fixed_iterations(G: nx.Graph, alpha: float, n_iter: int) -> dict:
    """Run the normalised Ricci flow for exactly `n_iter` iterations (no early stop).

    Used for the Mantegna benchmark (Appendix F.2), where the paper reports
    surgery at a fixed iteration count per level (20, then 10 on the 7-stock
    subgraph) rather than the curvature-gap stability criterion used for the
    full NASDAQ-100 decomposition (Appendix B, Step 3).
    """
    target_mean = np.mean([d["weight"] for _, _, d in G.edges(data=True)])
    history = []
    current = G
    for iteration in range(1, n_iter + 1):
        current, curvature = flow_step(current, alpha, target_mean)
        G_ratio, _, surgery_edges = gap_ratio(curvature)
        history.append({"iteration": iteration, "G": G_ratio, "n_surgery_edges": len(surgery_edges)})
    return {
        "graph": current,
        "curvature": curvature,
        "surgery_edges": surgery_edges,
        "G": G_ratio,
        "iterations": n_iter,
        "history": history,
    }


def apply_surgery(G: nx.Graph, surgery_edges: set[tuple]) -> nx.Graph:
    """Remove the surgery edges from a copy of G and return the result."""
    H = G.copy()
    H.remove_edges_from(surgery_edges)
    return H


def connected_components_at_least(G: nx.Graph, min_size: int) -> list[set]:
    """Connected components of G with at least `min_size` nodes."""
    return [c for c in nx.connected_components(G) if len(c) >= min_size]
