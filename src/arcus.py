"""ARCUS: Adaptive Ricci Curvature Unsupervised Surgery (paper Appendix B).

Recursive hierarchical community decomposition of a stock correlation graph:
on each connected component with at least ``n_min`` nodes, run the normalised
Ricci flow to a curvature-gap singularity, cut the near-zero-curvature
("surgery") edges, and recurse on every resulting component that is still
large enough. No cluster count or distance threshold is specified by the
user; the method is parameter-free in that sense (paper Sec. "ARCUS").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .curvature import is_orc_spherical, min_curvature, ollivier_ricci_curvature
from .graph_builder import build_complete_graph, correlation_to_distance
from .ricci_flow import apply_surgery, connected_components_at_least, run_flow_to_convergence


@dataclass
class ArcusNode:
    """One node of the ARCUS decomposition tree."""

    level: int
    path: str
    tickers: list
    orc_spherical_initial: bool
    min_initial_curvature: float
    is_leaf: bool
    G: float | None = None
    iterations: int | None = None
    history: list = field(default_factory=list)
    surgery_edges: list = field(default_factory=list)
    children: list = field(default_factory=list)  # list[ArcusNode]

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "path": self.path,
            "n_tickers": len(self.tickers),
            "tickers": self.tickers,
            "orc_spherical_initial": self.orc_spherical_initial,
            "min_initial_curvature": self.min_initial_curvature,
            "is_leaf": self.is_leaf,
            "G": self.G,
            "iterations": self.iterations,
            "n_surgery_edges": len(self.surgery_edges),
            "children": [c.to_dict() for c in self.children],
        }


def leaves(node: ArcusNode) -> list[ArcusNode]:
    if node.is_leaf:
        return [node]
    out = []
    for c in node.children:
        out.extend(leaves(c))
    return out


def decompose(
    corr: pd.DataFrame,
    tickers: list,
    *,
    alpha: float,
    g_min: float,
    stability_window: int,
    n_min: int,
    max_iter: int,
    no_split_g: float,
    level: int = 1,
    path: str = "root",
) -> ArcusNode:
    """Recursively decompose the sub-correlation-matrix of `tickers` (Appendix B Steps 1-5)."""
    sub_corr = corr.loc[tickers, tickers]
    distance = correlation_to_distance(sub_corr)
    G_graph = build_complete_graph(distance)

    curvature0 = ollivier_ricci_curvature(G_graph, alpha=alpha)
    spherical = is_orc_spherical(curvature0)
    min_kappa0 = min_curvature(curvature0)

    if len(tickers) < n_min:
        # Below N_min: terminate without attempting Ricci flow (Appendix B, Step 5).
        return ArcusNode(level, path, tickers, spherical, min_kappa0, is_leaf=True)

    result = run_flow_to_convergence(
        G_graph, alpha=alpha, g_min=g_min, stability_window=stability_window,
        max_iter=max_iter, no_split_g=no_split_g,
    )
    node = ArcusNode(
        level=level, path=path, tickers=tickers,
        orc_spherical_initial=spherical, min_initial_curvature=min_kappa0,
        is_leaf=False, G=result["G"], iterations=result["iterations"],
        history=result["history"], surgery_edges=list(result["surgery_edges"]),
    )

    if not result["surgery_edges"]:
        node.is_leaf = True
        return node

    cut_graph = apply_surgery(G_graph, result["surgery_edges"])
    components = connected_components_at_least(cut_graph, min_size=1)

    if len(components) < 2:
        node.is_leaf = True  # surgery produced no disconnection: terminate here
        return node

    big_components = [c for c in components if len(c) >= n_min]
    if len(big_components) < 1:
        node.is_leaf = True
        return node

    for k, component in enumerate(sorted(components, key=len, reverse=True)):
        child_tickers = [t for t in tickers if t in component]
        if len(child_tickers) >= n_min:
            child = decompose(
                corr, child_tickers, alpha=alpha, g_min=g_min,
                stability_window=stability_window, n_min=n_min, max_iter=max_iter,
                no_split_g=no_split_g, level=level + 1, path=f"{path}.{k}",
            )
        else:
            sub = corr.loc[child_tickers, child_tickers]
            sub_dist = correlation_to_distance(sub)
            small_graph = build_complete_graph(sub_dist)
            small_curv = ollivier_ricci_curvature(small_graph, alpha=alpha) if len(child_tickers) > 1 else {}
            child = ArcusNode(
                level=level + 1, path=f"{path}.{k}", tickers=child_tickers,
                orc_spherical_initial=is_orc_spherical(small_curv) if small_curv else True,
                min_initial_curvature=min_curvature(small_curv) if small_curv else float("nan"),
                is_leaf=True,
            )
        node.children.append(child)

    return node
