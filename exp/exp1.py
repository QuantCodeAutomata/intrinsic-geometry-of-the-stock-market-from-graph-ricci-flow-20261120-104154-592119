"""Experiment 1: ARCUS hierarchical decomposition of the NASDAQ-100 correlation graph.

Reproduces the paper's main result (Appendix B/D): builds the complete
weighted correlation graph of the NASDAQ-100 constituents, verifies it is
ORC-spherical, and recursively applies the normalised Ricci flow and
curvature-gap surgery (ARCUS) to obtain a parameter-free hierarchical
community decomposition, down to the terminal leaves (N_min = 10 nodes).

Run: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import fetch_nasdaq100_tickers, load_ohlcv, log_returns
from src import config
from src.arcus import decompose, leaves
from src.curvature import is_orc_spherical, min_curvature, ollivier_ricci_curvature
from src.graph_builder import build_complete_graph, correlation_matrix, correlation_to_distance, mean_edge_weight

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def flatten_tree(node, rows: list) -> None:
    rows.append({
        "level": node.level,
        "path": node.path,
        "n_tickers": len(node.tickers),
        "is_leaf": node.is_leaf,
        "G": node.G,
        "iterations": node.iterations,
        "orc_spherical_initial": node.orc_spherical_initial,
        "min_initial_curvature": node.min_initial_curvature,
        "tickers": ",".join(sorted(node.tickers)),
    })
    for child in node.children:
        flatten_tree(child, rows)


def run(use_cache: bool = True) -> dict:
    print("Fetching NASDAQ-100 constituents from Wikipedia (paper Sec. 2, data_requirements)")
    tickers = fetch_nasdaq100_tickers()
    print(f"Universe size: {len(tickers)} (paper: {config.NASDAQ100_UNIVERSE_SIZE_PAPER})")

    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    bars, meta = load_ohlcv(
        tickers,
        config.NASDAQ100_START_DATE,
        config.NASDAQ100_END_DATE,
        identity_date=config.NASDAQ100_IDENTITY_DATE,
        use_cache=use_cache,
    )
    loaded_tickers = sorted(bars["ticker"].unique())
    print(f"Loaded {len(loaded_tickers)}/{len(tickers)} tickers")
    if meta["failed"]:
        print(f"Failed to load (excluded): {meta['failed']}")
    if meta["non_massive_source"]:
        print(f"Non-Massive data sources: {meta['non_massive_source']}")

    returns = log_returns(bars)
    returns = returns[loaded_tickers]
    print(f"Return matrix: {returns.shape[0]} trading days x {returns.shape[1]} tickers, "
          f"{returns.index.min().date()}..{returns.index.max().date()}")

    corr = correlation_matrix(returns)
    distance = correlation_to_distance(corr)
    root_graph = build_complete_graph(distance)
    mean_w = mean_edge_weight(root_graph)
    print(f"Mean edge weight: {mean_w:.4f} (paper: ~1.11)")

    curvature0 = ollivier_ricci_curvature(root_graph, alpha=config.ALPHA)
    spherical0 = is_orc_spherical(curvature0)
    min_kappa0 = min_curvature(curvature0)
    print(f"Root ORC-spherical: {spherical0}, min kappa = {min_kappa0:.4f}")
    if not spherical0:
        print("WARNING: root graph is not ORC-spherical; paper requires this before any flow")

    print(f"Running recursive ARCUS decomposition (alpha={config.ALPHA}, g_min={config.G_MIN}, "
          f"n_min={config.N_MIN}, max_iter={config.MAX_ITER})")
    tree = decompose(
        corr, loaded_tickers,
        alpha=config.ALPHA, g_min=config.G_MIN, stability_window=config.STABILITY_WINDOW,
        n_min=config.N_MIN, max_iter=config.MAX_ITER, no_split_g=config.NO_SPLIT_G,
    )

    rows: list = []
    flatten_tree(tree, rows)
    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "exp1_decomposition_table.csv", index=False)

    with open(RESULTS_DIR / "exp1_tree.json", "w") as f:
        json.dump(tree.to_dict(), f, indent=2, default=str)

    leaf_nodes = leaves(tree)
    leaf_rows = [{"path": n.path, "level": n.level, "n_tickers": len(n.tickers),
                  "tickers": sorted(n.tickers)} for n in leaf_nodes]
    with open(RESULTS_DIR / "exp1_clusters.json", "w") as f:
        json.dump(leaf_rows, f, indent=2)

    # Curvature-distribution plot: root initial (spherical) vs. after flow (polarised).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(list(curvature0.values()), bins=40, color="steelblue")
    axes[0].set_title("Root graph, iteration 0\n(ORC-spherical)")
    axes[0].set_xlabel("Ollivier-Ricci curvature kappa")
    axes[0].set_ylabel("Edge count")
    axes[0].axvline(0, color="black", linewidth=0.8)

    internal_nodes = [r for r in rows if not r["is_leaf"]]
    non_root = [r for r in internal_nodes if r["path"] == "root"]
    final_history = tree.history[-1] if tree.history else None
    if final_history is not None:
        axes[1].bar(["G at final iteration"], [final_history["G"]], color="indianred")
        axes[1].set_title(f"Level-1 curvature-gap ratio G\n(final: {final_history['G']:.1f}, paper: 25.03)")
        axes[1].set_ylabel("G")
    fig.suptitle("NASDAQ-100 Ricci-flow neckpinch: curvature polarisation at Level 1")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_curvature_distributions.png", dpi=140)
    plt.close(fig)

    summary = {
        "universe_size_loaded": len(loaded_tickers),
        "universe_size_paper": config.NASDAQ100_UNIVERSE_SIZE_PAPER,
        "sample_period": [str(returns.index.min().date()), str(returns.index.max().date())],
        "n_trading_days": int(returns.shape[0]),
        "mean_edge_weight": mean_w,
        "mean_edge_weight_paper": 1.11,
        "root_orc_spherical": bool(spherical0),
        "min_initial_kappa": float(min_kappa0),
        "n_leaves": len(leaf_nodes),
        "n_internal_nodes": int(sum(1 for r in rows if not r["is_leaf"])),
        "level1_G": tree.G,
        "level1_G_paper": 25.03,
        "level1_split_sizes": sorted([len(c.tickers) for c in tree.children]) if tree.children else [],
        "level1_split_sizes_paper": [35, 66],
        "non_massive_source": meta["non_massive_source"],
        "failed_tickers": meta["failed"],
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print("Saved results/exp1_tree.json, exp1_clusters.json, "
          "exp1_decomposition_table.csv, exp1_curvature_distributions.png, exp1_summary.json")
    return summary


if __name__ == "__main__":
    run()
