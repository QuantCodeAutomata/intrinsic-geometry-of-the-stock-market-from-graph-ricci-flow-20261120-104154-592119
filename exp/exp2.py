"""Experiment 2: validation against the Mantegna (1999) ten-stock benchmark.

Reproduces Appendix F of the paper: builds the complete weighted correlation
graph of ten U.S. equities (AIG, IBM, BAC, AXP, MER, TXN, MOT, SLB, OXY, RD),
confirms it is ORC-spherical, runs the normalised Ricci flow for a fixed
number of iterations at each of two hierarchical levels, and compares the
recovered partition against the three-sector ground truth (energy, financial,
technology) using ARI, NMI and modularity Q. MST average-linkage (k=3) and
PCA sign clustering are run as baselines (Appendix F.3, Table 3).

Run: python -m exp.exp2
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.data_loader import load_ohlcv, log_returns
from src import config
from src.baselines import mst_average_linkage, pca_sign_clusters
from src.curvature import is_orc_spherical, min_curvature, ollivier_ricci_curvature
from src.graph_builder import build_complete_graph, correlation_matrix, correlation_to_distance
from src.metrics import clustering_agreement, modularity
from src.ricci_flow import apply_surgery, connected_components_at_least, run_flow_fixed_iterations

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def ground_truth_labels(tickers: list[str]) -> list[str]:
    return [config.MANTEGNA_GROUND_TRUTH[t] for t in tickers]


def run(use_cache: bool = True) -> dict:
    tickers = config.MANTEGNA_TICKERS
    print(f"Loading {len(tickers)} Mantegna benchmark tickers "
          f"{config.MANTEGNA_START_DATE}..{config.MANTEGNA_END_DATE}")

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
        config.MANTEGNA_START_DATE,
        config.MANTEGNA_END_DATE,
        identity_date=config.MANTEGNA_IDENTITY_DATE,
        use_cache=use_cache,
    )
    loaded_tickers = sorted(bars["ticker"].unique())
    print(f"Loaded tickers ({len(loaded_tickers)}): {loaded_tickers}")
    print(f"Non-massive sources: {meta['non_massive_source']}")
    if meta["failed"]:
        raise RuntimeError(f"Tickers failed to load: {meta['failed']}")
    missing = set(tickers) - set(loaded_tickers)
    if missing:
        raise RuntimeError(f"Missing Mantegna benchmark tickers after load: {missing}")

    returns = log_returns(bars)
    returns = returns[tickers]  # fixed column order matching config.MANTEGNA_TICKERS
    print(f"Return matrix: {returns.shape[0]} trading days x {returns.shape[1]} tickers, "
          f"{returns.index.min().date()}..{returns.index.max().date()}")

    corr = correlation_matrix(returns)
    distance = correlation_to_distance(corr)
    graph = build_complete_graph(distance)

    curvature0 = ollivier_ricci_curvature(graph, alpha=config.ALPHA)
    spherical0 = is_orc_spherical(curvature0)
    min_kappa0 = min_curvature(curvature0)
    print(f"Initial ORC-spherical: {spherical0}, min kappa = {min_kappa0:.4f} "
          f"(paper: {config.MANTEGNA_MIN_KAPPA_PAPER})")

    # Level 1: fixed-iteration Ricci flow on the full 10-stock graph (Appendix F.2).
    level1 = run_flow_fixed_iterations(graph, alpha=config.ALPHA, n_iter=config.MANTEGNA_LEVEL1_ITER)
    cut1 = apply_surgery(graph, level1["surgery_edges"])
    components1 = connected_components_at_least(cut1, min_size=1)
    print(f"Level 1 (iter={config.MANTEGNA_LEVEL1_ITER}): G={level1['G']:.2f} "
          f"(paper {config.MANTEGNA_G_LEVEL1_PAPER}), components={[sorted(c) for c in components1]}")

    energy_component = min(components1, key=len)
    rest_component = [c for c in components1 if c is not energy_component]
    rest_tickers = sorted(set().union(*rest_component)) if rest_component else []

    # Level 2: fresh Ricci flow on the 7-stock remainder subgraph (Appendix F.2).
    sub_corr = corr.loc[rest_tickers, rest_tickers]
    sub_distance = correlation_to_distance(sub_corr)
    sub_graph = build_complete_graph(sub_distance)
    level2 = run_flow_fixed_iterations(sub_graph, alpha=config.ALPHA, n_iter=config.MANTEGNA_LEVEL2_ITER)
    cut2 = apply_surgery(sub_graph, level2["surgery_edges"])
    components2 = connected_components_at_least(cut2, min_size=1)
    print(f"Level 2 (iter={config.MANTEGNA_LEVEL2_ITER}): G={level2['G']:.2f} "
          f"(paper {config.MANTEGNA_G_LEVEL2_PAPER}), components={[sorted(c) for c in components2]}")

    # Assemble the recovered 3-way partition: energy from Level 1, the two
    # Level-2 communities for the remaining seven stocks.
    predicted = {}
    for t in energy_component:
        predicted[t] = "energy_cluster"
    for k, comp in enumerate(components2):
        for t in comp:
            predicted[t] = f"level2_cluster_{k}"

    ordered_tickers = tickers
    y_true = ground_truth_labels(ordered_tickers)
    y_pred = [predicted.get(t, "unassigned") for t in ordered_tickers]
    agreement = clustering_agreement(y_true, y_pred)

    label_to_idx = {lab: i for i, lab in enumerate(sorted(set(y_pred)))}
    communities = {}
    for t, lab in zip(ordered_tickers, y_pred):
        communities.setdefault(label_to_idx[lab], set()).add(t)
    q = modularity(graph, list(communities.values()))
    print(f"ARI={agreement['ARI']:.3f} NMI={agreement['NMI']:.3f} Q={q:.3f} "
          f"(paper: ARI=1.000 NMI=1.000 Q=0.018)")

    # Baselines (Appendix F.3, Table 3).
    mst_labels = mst_average_linkage(distance, k=3)
    mst_pred = [mst_labels[t] for t in ordered_tickers]
    mst_agreement = clustering_agreement(y_true, mst_pred)

    pca = pca_sign_clusters(corr)
    pca_v2_labels = [pca["v2"]["labels"][t] for t in ordered_tickers]
    pca_v2_ari = clustering_agreement(y_true, pca_v2_labels)["ARI"]
    pca_v2v3_labels = [
        f"{pca['v2']['labels'][t]}_{pca['v3']['labels'][t]}" for t in ordered_tickers
    ]
    pca_v2v3_ari = clustering_agreement(y_true, pca_v2v3_labels)["ARI"]

    comparison = pd.DataFrame([
        {"method": "Ricci flow (ARCUS, Levels 1+2)", "k": 3, "ARI": agreement["ARI"],
         "NMI": agreement["NMI"], "note": f"G={level1['G']:.1f}, {level2['G']:.1f}; no parameters"},
        {"method": "MST average-linkage", "k": 3, "ARI": mst_agreement["ARI"],
         "NMI": mst_agreement["NMI"], "note": "Requires specifying k=3"},
        {"method": "PCA v2 sign", "k": 2, "ARI": pca_v2_ari, "NMI": float("nan"),
         "note": f"lambda2={pca['v2']['eigenvalue']:.3f}"},
        {"method": "PCA v2+v3 sign", "k": 3, "ARI": pca_v2v3_ari, "NMI": float("nan"),
         "note": f"lambda3={pca['v3']['eigenvalue']:.3f}"},
    ])

    RESULTS_DIR.mkdir(exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "exp2_method_comparison.csv", index=False)

    summary = {
        "data_source": meta["non_massive_source"] or "massive",
        "sample_period": [str(returns.index.min().date()), str(returns.index.max().date())],
        "n_trading_days": int(returns.shape[0]),
        "initial_orc_spherical": bool(spherical0),
        "min_initial_kappa": float(min_kappa0),
        "min_initial_kappa_paper": config.MANTEGNA_MIN_KAPPA_PAPER,
        "level1": {"iterations": config.MANTEGNA_LEVEL1_ITER, "G": level1["G"],
                   "G_paper": config.MANTEGNA_G_LEVEL1_PAPER,
                   "components": [sorted(c) for c in components1]},
        "level2": {"iterations": config.MANTEGNA_LEVEL2_ITER, "G": level2["G"],
                   "G_paper": config.MANTEGNA_G_LEVEL2_PAPER,
                   "components": [sorted(c) for c in components2]},
        "ARI": agreement["ARI"], "NMI": agreement["NMI"], "Q": q,
        "ARI_paper": 1.000, "NMI_paper": 1.000, "Q_paper": 0.018,
        "predicted_labels": predicted,
        "ground_truth": dict(zip(ordered_tickers, y_true)),
        "mst_ARI": mst_agreement["ARI"], "pca_v2_ARI": pca_v2_ari, "pca_v2v3_ARI": pca_v2v3_ari,
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved results/exp2_summary.json and results/exp2_method_comparison.csv")
    return summary


if __name__ == "__main__":
    run()
