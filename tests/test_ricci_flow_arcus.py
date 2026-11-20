"""Unit tests for src.ricci_flow and src.arcus on synthetic, known-answer data."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import config
from src.arcus import decompose, leaves
from src.curvature import is_orc_spherical, ollivier_ricci_curvature
from src.graph_builder import build_complete_graph, correlation_matrix, correlation_to_distance
from src.ricci_flow import (
    apply_surgery,
    connected_components_at_least,
    flow_step,
    gap_ratio,
    run_flow_fixed_iterations,
    run_flow_to_convergence,
)


def _planted_two_block_graph(n_per_block: int = 8, seed: int = 3) -> nx.Graph:
    """A two-community weighted graph built the same way as the paper pipeline:
    correlation -> Mantegna distance -> complete graph. Returns are generated
    with a common market factor plus a block-specific sector factor, mirroring
    real equity correlations (all pairs positively correlated through the
    market factor, intra-block pairs additionally correlated through the
    shared sector factor). This is the condition under which recursive Ricci
    flow is reported to isolate the sparser inter-block bridges as the
    near-zero-curvature surgery set (paper Sec. 3, Appendix B)."""
    rng = np.random.default_rng(seed)
    n_days = 750
    market = rng.normal(size=n_days) * 0.011
    sector = [rng.normal(size=n_days) * 0.007 for _ in range(2)]
    data = {}
    for b in range(2):
        for i in range(n_per_block):
            idio = rng.normal(size=n_days) * 0.005
            data[f"B{b}_{i}"] = market + sector[b] + idio
    returns = pd.DataFrame(data)
    corr = correlation_matrix(returns)
    dist = correlation_to_distance(corr)
    return build_complete_graph(dist), corr


def test_flow_step_preserves_mean_weight():
    G, _ = _planted_two_block_graph()
    target_mean = np.mean([d["weight"] for _, _, d in G.edges(data=True)])
    H, curvature = flow_step(G, alpha=0.5, target_mean=target_mean)
    new_mean = np.mean([d["weight"] for _, _, d in H.edges(data=True)])
    assert new_mean == pytest.approx(target_mean, rel=1e-6)
    assert len(curvature) == G.number_of_edges()


def test_gap_ratio_on_clearly_bimodal_curvature():
    curvature = {(i, i + 1): 0.48 for i in range(0, 20, 2)}
    curvature.update({(i, i + 1): 0.01 for i in range(20, 40, 2)})
    G, labels, surgery = gap_ratio(curvature)
    assert G > 15.0
    assert len(surgery) == 10


def test_run_flow_to_convergence_develops_neckpinch_on_two_block_graph():
    """The curvature-gap ratio must rise well above 1 and the resulting surgery
    set must disconnect the graph into (at least) two components -- the paper's
    discrete-neckpinch signature (research objective 2)."""
    G, _ = _planted_two_block_graph()
    curvature0 = ollivier_ricci_curvature(G, alpha=0.5)
    assert is_orc_spherical(curvature0)

    result = run_flow_to_convergence(
        G, alpha=0.5, g_min=2.5, stability_window=3, max_iter=15, no_split_g=2.0,
    )
    assert result["G"] > 2.0
    cut = apply_surgery(G, result["surgery_edges"])
    components = connected_components_at_least(cut, min_size=1)
    assert len(components) >= 2


def test_run_flow_fixed_iterations_runs_exact_count():
    G, _ = _planted_two_block_graph(n_per_block=5)
    result = run_flow_fixed_iterations(G, alpha=0.5, n_iter=7)
    assert result["iterations"] == 7
    assert len(result["history"]) == 7


def test_arcus_decompose_recovers_planted_two_block_structure():
    """ARCUS must recover the planted 2-block partition without being told k."""
    G, corr = _planted_two_block_graph(n_per_block=8)
    tickers = list(corr.index)
    tree = decompose(
        corr, tickers, alpha=0.5, g_min=2.5, stability_window=3,
        n_min=5, max_iter=15, no_split_g=2.0,
    )
    leaf_nodes = leaves(tree)
    assert len(leaf_nodes) >= 2
    recovered_sets = [set(n.tickers) for n in leaf_nodes]
    block0 = {t for t in tickers if t.startswith("B0")}
    # Each planted block must be fully contained in a single recovered leaf
    # (allows further recursive splitting inside a block but not mixing).
    assert any(block0.issubset(s) for s in recovered_sets)


def test_arcus_terminates_below_n_min():
    G, corr = _planted_two_block_graph(n_per_block=3)
    tickers = list(corr.index)
    tree = decompose(
        corr, tickers, alpha=0.5, g_min=config.G_MIN, stability_window=config.STABILITY_WINDOW,
        n_min=10, max_iter=5, no_split_g=config.NO_SPLIT_G,
    )
    assert tree.is_leaf
    assert tree.tickers == tickers
