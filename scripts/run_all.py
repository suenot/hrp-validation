"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + records.csv
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below; wall-clock timing is printed to
stdout only and never written into results. Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hrp_experiments import __version__
from hrp_experiments import analysis as A
from hrp_experiments.simulate import mv_rcond_sweep, run_batch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED_MAIN = 101       # gaussian batch
SEED_STUDENT_T = 707  # student-t robustness batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_main = 200 if args.quick else 4000
    n_t = 60 if args.quick else 800
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    print(f"[1/4] gaussian batch (n={n_main}, seed={SEED_MAIN}) ...", flush=True)
    recs = run_batch(n_main, family="random", dist="gaussian", seed=SEED_MAIN,
                     progress_every=max(1, n_main // 8))
    t1 = time.perf_counter()
    print(f"      {t1 - t0:.1f}s", flush=True)

    print(f"[2/4] student-t robustness batch (n={n_t}, seed={SEED_STUDENT_T}) ...", flush=True)
    recs_t = run_batch(n_t, family="random", dist="student_t", seed=SEED_STUDENT_T,
                       progress_every=max(1, n_t // 4))
    t2 = time.perf_counter()
    print(f"      {t2 - t1:.1f}s", flush=True)

    print(f"[3/4] sample-MV pinv rcond sweep at T/N=1 (replays seed {SEED_MAIN}) ...",
          flush=True)
    rcond_sweep = mv_rcond_sweep(n_main, seed=SEED_MAIN)
    t2b = time.perf_counter()
    print(f"      {t2b - t2:.1f}s", flush=True)

    print("[4/4] summaries ...", flush=True)
    df = A.to_frame(recs + recs_t)
    df.to_csv(RESULTS / "records.csv", index=False)
    summary = A.summarize(df)
    summary["mv_pinv_rcond_sweep"] = rcond_sweep
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "n_gaussian": n_main,
            "n_student_t": n_t,
            "seeds": {"gaussian": SEED_MAIN, "student_t": SEED_STUDENT_T},
            "quick": bool(args.quick),
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        **summary,
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS / 'results.json'} and records.csv "
          f"({time.perf_counter() - t0:.1f}s total).")

    # ---- headline numbers to stdout ------------------------------------- #
    g = df[df["cfg_dist"] == "gaussian"]
    fams = ("one_factor", "hierarchical", "unstructured", "equicorr")
    methods = ("ew", "iv", "mv", "mv_lo", "mv_lw", "hrp_single")

    print("\n--- HEADLINE NUMBERS ---")
    print(f"experiments: {len(df)} ({len(g)} gaussian + {len(df) - len(g)} student-t); "
          f"oracle = exact min-variance on the true covariance (risk ratio >= 1)")

    for band, mask in (("T/N = 0.5", g["cfg_tn_ratio"] == 0.5),
                       ("T/N = 10", g["cfg_tn_ratio"] == 10.0)):
        print(f"\nMedian realized risk / oracle at {band}, by structure family:")
        print(f"  {'method':12}" + "".join(f"{f:>14}" for f in fams))
        for m in methods:
            vals = [g[mask & (g['cfg_family'] == f)][f"risk_ratio_{m}"].median() for f in fams]
            print(f"  {m:12}" + "".join(f"{v:14.3f}" for v in vals))

    print("\nHRP(single) win rate (realized risk lower) at T/N <= 1, by family:")
    low = g[g["cfg_tn_ratio"] <= 1.0]
    print(f"  {'opponent':12}" + "".join(f"{f:>14}" for f in fams))
    for opp in ("ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo"):
        vals = [(low[low['cfg_family'] == f]["risk_ratio_hrp_single"]
                 < low[low['cfg_family'] == f][f"risk_ratio_{opp}"]).mean() for f in fams]
        print(f"  {opp:12}" + "".join(f"{v:14.2f}" for v in vals))

    disp_rows = summary["hierarchical_dispersion_effect"]
    if disp_rows:
        print("\nHierarchical family, T/N <= 2: HRP edge vs cross-block tightness dispersion:")
        for row in disp_rows:
            print(f"  {row['block_disp_tercile']:10} (disp~{row['mean_block_disp']:.2f}, "
                  f"n={row['n']:4d})  win_vs_iv {row['win_vs_iv']:.2f}  "
                  f"win_vs_mv_lw {row['win_vs_mv_lw']:.2f}  "
                  f"med_log10_vs_iv {row['median_log10_vs_iv']:+.3f}")

    print("\nLinkage sensitivity (median risk ratio, all T/N):")
    for row in summary["linkage_sensitivity"]:
        print(f"  {row['family']:14} single {row['median_hrp_single']:.3f}  "
              f"average {row['median_hrp_average']:.3f}  ward {row['median_hrp_ward']:.3f}  "
              f"(best: {row['best_linkage'].replace('hrp_', '')})")

    print("\nDistance convention (single linkage on condensed d vs distance-of-distances):")
    for row in summary["distance_convention"]:
        print(f"  {row['family']:14} median d {row['median_hrp_single']:.3f}  "
              f"d-tilde {row['median_hrp_dod']:.3f}  "
              f"identical risk {row['frac_identical_risk']:.2f}  "
              f"win d-tilde {row['win_dod_vs_single']:.2f}")

    sw = summary["mv_pinv_rcond_sweep"]
    print(f"\nSample-MV pinv rcond sweep at T/N={sw['tn_ratio']:g} (n={sw['n']}):")
    for k, v in sw["median_risk_ratio_mv_by_rcond"].items():
        print(f"  rcond {k:>6}: median risk ratio {v:.2f}")

    print("\nSharpe captured (median sr / oracle tangency), low T/N <= 1, all families pooled:")
    for col, name in (("sr_ratio_ew", "equal weight (mu-free)"),
                      ("sr_ratio_mv_lw", "LW min-var (mu-free)"),
                      ("sr_ratio_hrp_single", "HRP single (mu-free)"),
                      ("sr_ratio_mv_mu", "Markowitz w/ estimated mu"),
                      ("sr_ratio_mv_mu_lw", "Markowitz w/ est. mu + LW")):
        s = low[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {name:28} {s.median():+.3f}")
    print(f"  frac. negative true Sharpe for Markowitz w/ estimated mu: "
          f"{(low['sr_true_mv_mu'] < 0).mean():.2f}")

    print("\nStudent-t robustness (HRP vs LW min-var win rate, by family):")
    for row in summary["student_t_robustness"]:
        if row["dist"] == "student_t":
            print(f"  {row['family']:14} win_hrp_vs_mv_lw {row['win_hrp_single_vs_mv_lw']:.2f} "
                  f"(n={row['n']})")


if __name__ == "__main__":
    main()
