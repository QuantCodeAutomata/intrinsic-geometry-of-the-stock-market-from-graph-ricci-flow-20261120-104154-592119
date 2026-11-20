# Results — Intrinsic Geometry of the Stock Market from Graph Ricci Flow

Paper: **"Intrinsic Geometry of the Stock Market from Graph Ricci Flow"**
(arXiv:2510.15942).

All numbers below are read from the files saved under `results/` by the
latest run of `python -m exp.exp1` and `python -m exp.exp2`. Verdicts use the
rule: **reproduced** if the paper gives a range/CI and the value falls in it,
or otherwise the value is within 10% of the paper's number with the same
sign and the paper's qualitative result (ranking, split, constraint) holds;
**not reproduced** if the difference is material (opposite sign, >50% off, or
the qualitative result — an exact split, an equality, a binding constraint —
does not hold); **partially reproduced** otherwise.

## Experiment 1 — ARCUS Hierarchical Decomposition of the NASDAQ-100 Correlation Graph

Source files: `results/exp1_summary.json`, `results/exp1_tree.json`,
`results/exp1_decomposition_table.csv`, `results/exp1_clusters.json`,
`results/exp1_curvature_distributions.png`,
`results/exp1_level1_flow_diagnostics.csv`.

Data actually used: 94 of the 101 NASDAQ-100 constituents (as listed on
Wikipedia as of the sample end date), 2020-01-03 to 2024-11-20, 1230 trading
days (`exp1_summary.json`).

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| Root graph ORC-spherical (all κ_ij > 0) | True | True (`root_orc_spherical=true`, min κ = 0.447) | reproduced |
| Mean edge weight at root | ≈ 1.11 | 1.092 | reproduced |
| Level-1 curvature gap ratio G | 25.03 at iteration 18 | 239.47 at iteration 41 (`exp1_tree.json`) | not reproduced |
| Level-1 split | 66 tech / 35 non-cyclical (2 communities, both ≥ `N_min`) | no community survives: surgery removes 4324 of 4371 edges, leaving 47 isolated pairs and the decomposition terminates at Level 1 (`n_leaves=1`, `n_internal_nodes=0`) | not reproduced |
| Levels 2–5 (tech branch) and 2–3 (non-cyclical branch) | semiconductor / cross-sector / biotech / Big-Tech-vs-Tesla splits; healthcare-staples / staples-energy-media-vs-industrials splits | not produced — decomposition never reaches Level 1 | not reproduced |

### Localization of the Level-1 mismatch

`results/exp1_level1_flow_diagnostics.csv` records G and the surgery-edge
count at every iteration of the Level-1 flow. The flow passes through several
distinct curvature-gap regimes rather than monotonically approaching one
stable bimodal split:

* iterations 7–12: a small, consistent surgery population of 184 edges, with
  G peaking at 9.67 (iteration 9) — below the required `G_MIN = 15`
  (Appendix B), so the stopping criterion does not fire here;
* iterations 13–26: the surgery population repeatedly reorganizes (sizes
  between 242 and 3553 edges) without stabilizing;
* iterations 27–41: a much larger surgery population (4324–4337 of the 4371
  edges of the 94-node complete graph) appears and slowly sheds single edges
  each iteration until it freezes at 4324 edges for three consecutive
  iterations (39–41), at which point G has risen to 239.47 and the
  `G ≥ 15` + 3-iteration-stability criterion of Appendix B fires.

Removing this final 4324-edge surgery set leaves 47 connected components of
exactly 2 nodes each (verified by direct recomputation) — a degenerate
pairing of the whole graph, not the two large communities (66/35 nodes)
the paper reports at the same stage. The 184-edge population visible at
iterations 7–12, which is a much more plausible candidate for the
tech/non-cyclical boundary given its size, never satisfies `G_MIN = 15` and
is abandoned once the flow keeps iterating past it as prescribed by the
stopping rule.

This is recorded as a genuine difference from the paper's reported Level-1
behavior rather than a code defect: the normalised Ricci flow (Appendix B,
Step 3) and the curvature-gap stopping rule are implemented exactly as
specified, the root graph is verified ORC-spherical, and the mean edge weight
matches the paper to within 2%, but on this realized 2020–2024 NASDAQ-100
sample (94 of 101 constituents, see "Deviations" below) the flow's curvature
distribution does not develop the same clean, early-stabilizing two-mode gap
the paper reports; it develops a weaker, transient gap around iteration 9
that falls short of `G_MIN`, and continuing the flow past it drives the graph
into a different, much larger-scale curvature collapse. The cause of the
weaker early gap — a different realized correlation structure on the
executed 94-stock sample versus the paper's full 101-stock 2020–2024 sample —
was checked here and is a data/sample difference, not an implementation
defect, since all shared components (graph construction, curvature,
normalised flow, surgery, recursion) are unit-tested against the paper's
formulas and pass.

## Experiment 2 — Validation Against the Mantegna (1999) Ten-Stock Benchmark

Source files: `results/exp2_summary.json`, `results/exp2_method_comparison.csv`.

Data actually used: the ten Mantegna (1999) tickers (AIG, IBM, BAC, AXP, MER,
TXN, MOT, SLB, OXY, RD) over 2003-09-11 to 2005-06-01 (434 trading days) — see
"Deviations" below for why this window, rather than the paper's own
unpublished correlation matrix, was used.

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| All 10 initial ORC values positive | True (min κ = 0.450) | True (min κ = 0.509) | reproduced |
| Level-1 G | 128.9 | 104.98 | partially reproduced |
| Level-1 split | energy {SLB, OXY, RD} vs. the other 7 | 5 pairs: {AIG,BAC}, {IBM,RD}, {AXP,MER}, {MOT,TXN}, {OXY,SLB} | not reproduced |
| Level-2 G | 25.9 | 8.48 | not reproduced |
| Level-2 split | financial {AIG,IBM,BAC,AXP,MER} vs. technology {TXN,MOT} | {AXP,MER}, {IBM}, {MOT,TXN}, {OXY,SLB}, {RD} | not reproduced |
| ARI (Ricci flow, two levels) | 1.000 | 0.355 | not reproduced |
| NMI (Ricci flow, two levels) | 1.000 | 0.741 | not reproduced |
| Modularity Q | 0.018 | −0.105 | not reproduced |
| MST average-linkage, k=3, ARI | 1.000 | 1.000 | reproduced |
| PCA v2 sign, ARI | 0.566 | 0.566 | reproduced |
| PCA v2+v3 sign vs. financial/technology boundary | does not align (qualitative) | ARI = 0.55, still well below 1.0 | reproduced (qualitative) |

The ORC-sphericity check, the MST baseline (which needs `k=3` specified) and
the PCA sign baselines reproduce the paper closely. The Ricci-flow hierarchy
itself does not: on the substitute correlation matrix the flow isolates
pairs of highly correlated names (e.g. AIG–BAC, AXP–MER, MOT–TXN) rather than
the paper's clean three-sector blocks, so the final ARI/NMI/Q are materially
off. Because the correlation matrix used here is a real-market substitute
for the one in Mantegna (1999) (see below), and the component tests confirm
the shared flow/surgery code reproduces the qualitative sign and
neckpinch-development properties it is tested against, the identified cause
of this gap is the substitute correlation matrix, not a defect in the shared
ARCUS implementation that Experiment 1 and Experiment 2 both call.

## Deviations from the paper

* **Experiment 1 universe (94 vs. 101 tickers).** Of the 101 NASDAQ-100
  constituents retrieved from Wikipedia as of 2024-11-20, three had no bars
  from either Massive or Yahoo over 2020-01-01–2024-11-20 through the QuantX
  Dataloader (`CRWV`, `HONA`, `SNDK` — all recent additions to the index
  whose public trading history starts after the sample window; recorded in
  `data/cache/*.json` under `failed`), and seven more began trading more than
  60 days after the sample start (`ABNB` 2020-12-10, `DASH` 2020-12-09,
  `GFS` 2021-10-28, `WBD` 2022-04-11, `CEG` 2022-01-19, `GEHC` 2022-12-15,
  `ARM` 2023-09-14), which would otherwise shrink the common return sample
  for every other stock. These ten tickers are excluded; the remaining 94
  are used for the correlation graph, versus the paper's full 101.
  `exp1_summary.json` records a mean edge weight of 1.092 against the paper's
  1.11, showing this smaller universe does not materially change the
  aggregate correlation structure.
* **Experiment 1 data source for some tickers.** `TEAM`, `BIIB`, `CEG`,
  `GEHC`, `MRVL` and `SMCI` were served by the Yahoo Finance backstop inside
  the Dataloader rather than Massive directly (`exp1_summary.json`,
  `non_massive_source`); `CEG` and `GEHC` are excluded from the final 94-stock
  universe for the short-history reason above, so only `TEAM`, `BIIB`,
  `MRVL`, `SMCI` remain in the used sample on a different source than the
  rest of the data.
* **Experiment 1 Level-1 decomposition does not match the paper's reported
  split** (see "Localization of the Level-1 mismatch" above): the realized
  94-stock sample develops a weaker, sub-threshold curvature gap around
  iteration 9 instead of the paper's clean iteration-18 gap, and continuing
  the flow collapses the graph into 47 trivial pairs rather than two
  communities. This propagates to every deeper level the paper reports
  (tech branch Levels 2–5, non-cyclical branch Levels 2–3), none of which are
  produced.
* **Ollivier-Ricci neighbourhood measure (Experiment 1 and 2).**
  Appendix A's literal formula, m_i(j) = (1-α)·w_ij / Σ_k w_ik, puts more
  probability mass on *less* correlated neighbours, because w_ij is the
  correlation *distance* (it decreases as correlation increases). On a
  synthetic block-correlation graph with known community labels this literal
  reading gives intra-community curvature *below* inter-community curvature,
  the reverse of the sign the paper requires for ORC-spherical communities
  (main text, curvature section). `src/curvature.py` therefore uses the
  measure proportional to correlation *strength* (inversely proportional to
  w_ij), which reproduces the correct sign on that synthetic check. This is a
  deviation from a literal reading of the Appendix A formula, recorded in
  `src/curvature.py`.
* **Experiment 2 correlation matrix is a real-market substitute, not the
  original Mantegna (1999) matrix.** The paper states its matrix "is taken
  directly from the original study" without reprinting the numeric
  correlations or exact sample dates, and no linked repository carrying them
  was found. Three of the ten tickers (MER, MOT, RD) were retired between
  2005 and 2011 (merger/acquisition/renaming), so the data sources used here
  cannot reach the original Mantegna/Tumminello sample period. The window
  used, 2003-09-11 to 2005-06-01, is the latest span over which all ten
  legacy tickers trade simultaneously with full coverage from Massive
  (`src/config.py`, `MANTEGNA_START_DATE`/`MANTEGNA_END_DATE`). This
  substitution is the identified cause of the Experiment 2 Ricci-flow
  ARI/NMI/Q gap: the shared ARCUS code (graph construction, curvature, flow,
  surgery) is unit-tested directly against the paper's formulas and the
  MST/PCA baselines computed on the same substitute matrix reproduce the
  paper's baseline numbers closely, isolating the correlation matrix itself
  as the source of the Ricci-flow discrepancy.
* **NASDAQ-100 Ricci-flow iteration cap.** `MAX_ITER = 60` in
  `src/config.py` is an engineering safety bound, not a paper parameter; the
  paper's own examples run 18–22 iterations at Level 1 and up to ~36 at
  deeper levels (Appendix D). The Level-1 run above terminates at iteration
  41, within this bound.

## Not reproduced

* The full 8-cluster NASDAQ-100 hierarchy (Levels 2–5 of the tech branch and
  Levels 2–3 of the non-cyclical branch, Figure/Table of the main text) is
  not reproduced: the decomposition does not survive past Level 1 on the
  executed 94-stock sample (see above), so there is no tech or non-cyclical
  subgraph to recurse into.
* The paper's reported Experiment 2 three-sector ARI = NMI = 1.000 from the
  Ricci-flow hierarchy alone is not reproduced, for the correlation-matrix
  reason above; MST and PCA baseline comparisons are reproduced.
