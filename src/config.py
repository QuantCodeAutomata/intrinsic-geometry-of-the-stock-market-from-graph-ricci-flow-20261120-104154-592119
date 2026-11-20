"""Parameters for the ARCUS (Adaptive Ricci Curvature Unsupervised Surgery) replication.

All experiment parameters live here as module-level constants. Each constant
names the paper section/equation it comes from, or states that the paper does
not specify it.
"""

# --- Shared ARCUS algorithm parameters (Appendix B) -------------------------------
ALPHA = 0.5  # Ollivier-Ricci idleness parameter alpha, used throughout (Appendix B, Input)
G_MIN = 15.0  # Minimum curvature gap ratio to stop the flow, G_min (Appendix B / Fig. 4 caption)
STABILITY_WINDOW = 3  # Number of consecutive iterations the surgery set must be stable (Appendix B)
N_MIN = 10  # Minimum subgraph size to keep recursing, N_min (Appendix B, Input; Methodology)
SURGERY_TAU = 0.25  # Surgery threshold kappa*, any value in the empty gap [0.05, 0.45] (Appendix B Eq. 8, Appendix C)
MAX_ITER = 60  # Safety cap on Ricci-flow iterations per subgraph; paper's L_max=20 is a floor,
# deeper levels need more (e.g. Level 4, N=40, needs ~36 iterations per Appendix D Table 1)
NO_SPLIT_G = 2.0  # G below which recursion terminates with no disconnection (Appendix B, Step 5)
RICCI_METHOD = "OTD"  # Optimal Transport Distance method of GraphRicciCurvature (Appendix B)

# --- Experiment 1: ARCUS decomposition of the NASDAQ-100 --------------------------
NASDAQ100_START_DATE = "2020-01-01"  # Sample start (paper_pipeline data_requirements)
NASDAQ100_END_DATE = "2024-11-20"  # Sample end (paper_pipeline data_requirements; paper text, Sec. "NASDAQ-100 complete graph")
NASDAQ100_UNIVERSE_SIZE_PAPER = 101  # Full NASDAQ-100 index constituents used in the paper (Sec. 2)
NASDAQ100_UNIVERSE_SIZE_RUN = 101  # Executed universe size; equal to the paper's, no narrowing needed
NASDAQ100_IDENTITY_DATE = "2024-11-20"  # Point-in-time index-membership date (= sample end; paper gives no other date)

# --- Experiment 2: Mantegna (1999) / Tumminello et al. (2008) ten-stock benchmark --
MANTEGNA_TICKERS = ["AIG", "IBM", "BAC", "AXP", "MER", "TXN", "MOT", "SLB", "OXY", "RD"]
# Ground-truth three-sector partition (Appendix F.1)
MANTEGNA_GROUND_TRUTH = {
    "AIG": "financial", "IBM": "financial", "BAC": "financial", "AXP": "financial", "MER": "financial",
    "TXN": "technology", "MOT": "technology",
    "SLB": "energy", "OXY": "energy", "RD": "energy",
}
MANTEGNA_LEVEL1_ITER = 20  # Iterations before Level-1 surgery (energy vs. rest), Appendix F.2
MANTEGNA_LEVEL2_ITER = 10  # Iterations before Level-2 surgery on the 7-stock subgraph (Appendix F.2)
MANTEGNA_G_LEVEL1_PAPER = 128.9  # Expected G at Level 1 (Appendix F.2)
MANTEGNA_G_LEVEL2_PAPER = 25.9  # Expected G at Level 2 (Appendix F.2)
MANTEGNA_MIN_KAPPA_PAPER = 0.450  # Expected minimum initial ORC (Appendix F.1)
# Sample window for the real-market substitute correlation matrix: the paper states the
# matrix "is taken directly from the original study" (Appendix F.1) without reprinting
# its numeric values or sample dates, and no linked repository carries them (see
# Deviations in RESULTS.md). MER (Merrill Lynch), MOT (Motorola) and RD (Royal Dutch
# Petroleum) were retired tickers (acquired/merged/split) by 2005-2011 and have no
# market-data coverage before September 2003 in the data sources used here; the window
# below is the latest span for which all ten legacy tickers trade simultaneously with
# full coverage from a single data source (Massive, no Yahoo backstop needed).
MANTEGNA_START_DATE = "2003-09-10"
MANTEGNA_END_DATE = "2004-12-31"
MANTEGNA_IDENTITY_DATE = "2004-06-01"  # Mid-sample identity date for legacy/renamed tickers
