"""Assert that every quantitative claim in paper/main.tex matches results/.

    python scripts/check_paper_numbers.py

Each check formats a value from results/results.json (or, for pooled text
claims, recomputes it from results/records.csv) exactly the way the paper
quotes it and asserts the resulting token appears in main.tex (and, where the
paper states an inequality or a range, asserts the underlying inequality
holds). Exits non-zero on the first group of failures, so it can gate a
release.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = re.sub(r"\s+", " ", (ROOT / "paper" / "main.tex").read_text())  # normalize wraps
R = json.loads((ROOT / "results" / "results.json").read_text())
DF = pd.read_csv(ROOT / "results" / "records.csv")
G = DF[DF["cfg_dist"] == "gaussian"]
LOW = G[G["cfg_tn_ratio"] <= 1.0]
HIGH = G[G["cfg_tn_ratio"] >= 5.0]
T1 = G[G["cfg_tn_ratio"] == 1.0]

failures: list[str] = []
n_checks = 0


def check(label: str, token: str, cond: bool = True) -> None:
    """Assert ``token`` appears in main.tex (whitespace-normalized) and ``cond``."""
    global n_checks
    n_checks += 1
    ok_tex = token in TEX
    if ok_tex and cond:
        print(f"  PASS  {label:62} {token!r}")
    else:
        why = [] if ok_tex else [f"token {token!r} not in main.tex"]
        if not cond:
            why.append("condition failed")
        failures.append(f"{label}: {'; '.join(why)}")
        print(f"  FAIL  {label:62} {token!r}  <-- {'; '.join(why)}")


def f2(v) -> str:
    return f"{v:.2f}"


def grid(family: str, tn: float, method: str) -> float:
    for r in R["headline_grid"]:
        if r["family"] == family and r["tn_ratio"] == tn and r["method"] == method:
            return r["median_risk_ratio"]
    raise KeyError((family, tn, method))


def winrate(family: str, tn: float, opp: str) -> float:
    for r in R["hrp_win_rates"]:
        if r["family"] == family and r["tn_ratio"] == tn and r["opponent"] == opp:
            return r["hrp_win_rate"]
    raise KeyError((family, tn, opp))


def medlog(family: str, tn: float, opp: str) -> float:
    for r in R["hrp_win_rates"]:
        if r["family"] == family and r["tn_ratio"] == tn and r["opponent"] == opp:
            return r["median_log10_risk_ratio"]
    raise KeyError((family, tn, opp))


def scorecard(band: str, family: str, opp: str) -> float:
    for r in R["hrp_scorecard"]:
        if r["band"].startswith(band) and r["family"] == family:
            return r[f"win_vs_{opp}"]
    raise KeyError((band, family, opp))


FAMS = ("one_factor", "hierarchical", "unstructured", "equicorr")

# ---------------------------------------------------------------- counts ----
print("[design counts]")
check("total experiments", "4{,}800", R["n_experiments"] == 4800)
check("gaussian batch", "4{,}000", R["n_gaussian"] == 4000)
check("student-t batch", "800", R["n_student_t"] == 800)
nf = R["n_by_family"]
check("one-factor count", "1{,}007", nf["one_factor"] == 1007)
check("hierarchical count", "1{,}018", nf["hierarchical"] == 1018)
check("unstructured count", "1{,}011", nf["unstructured"] == 1011)
check("equicorr count", "964", nf["equicorr"] == 964)
cells = [r["n"] for r in R["headline_grid"] if r["method"] == "ew"]
check("per-cell n range", "$179$--$222$", min(cells) == 179 and max(cells) == 222)
check("seeds quoted", "$101$", R["meta"]["seeds"]["gaussian"] == 101)
check("seeds quoted (t)", "$707$", R["meta"]["seeds"]["student_t"] == 707)

# ------------------------------------------------ Table 1: headline grid ----
print("[table 1: headline medians, T/N=1 and T/N=10]")
T1_METHODS = ("ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo", "hrp_single")
for m in T1_METHODS:
    for fam in FAMS:
        for tn in (1.0, 10.0):
            v = grid(fam, tn, m)
            check(f"grid {m} {fam} tn={tn:g}", f2(v))

# ------------------------------------------------ Table 2: scorecard --------
print("[table 2: HRP win-rate scorecard]")
for band in ("low", "high"):
    for fam in FAMS:
        for opp in ("ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo"):
            v = scorecard(band, fam, opp)
            check(f"scorecard {band} {fam} vs {opp}", f2(v))

# ------------------------------------------------ Table 3: Sharpe track -----
print("[table 3: Sharpe track]")
for r in R["sharpe_summary"]:
    band, fam = r["band"], r["family"]
    for col, name in (
        ("median_sr_ratio_ew", "EW"),
        ("median_sr_ratio_mv_lw", "LW"),
        ("median_sr_ratio_hrp_single", "HRP"),
        ("median_sr_ratio_mv_mu", "MV-mu"),
        ("median_sr_ratio_mv_mu_lw", "MV-mu-LW"),
        ("frac_negative_sr_mv_mu", "neg"),
        ("median_sr_ratio_oracle_minvar", "oracleMV"),
    ):
        check(f"sharpe {band[:4]} {fam} {name}", f2(r[col]))

# ------------------------------------------------ Table 4: Student-t --------
print("[table 4: Student-t robustness]")
for r in R["student_t_robustness"]:
    dist, fam = r["dist"], r["family"]
    for col, name in (
        ("median_risk_ratio_ew", "EW"),
        ("median_risk_ratio_mv", "MV"),
        ("median_risk_ratio_mv_lw", "LW"),
        ("median_risk_ratio_hrp_single", "HRP"),
        ("win_hrp_single_vs_mv_lw", "win"),
    ):
        check(f"t-robust {dist} {fam} {name}", f2(r[col]))

# ------------------------------------------------ data-starved text ---------
print("[data-starved regime]")
check("MV pooled median at T/N=1", "16.8",
      f"{T1['risk_ratio_mv'].median():.1f}" == "16.8")
check("HRP pooled median at T/N=1", "2.3",
      f"{T1['risk_ratio_hrp_single'].median():.1f}" == "2.3")
check("LW pooled median at T/N=1", "1.8",
      f"{T1['risk_ratio_mv_lw'].median():.1f}" == "1.8")
check("EW pooled median at T/N=1", "3.8",
      f"{T1['risk_ratio_ew'].median():.1f}" == "3.8")
check("IV pooled median at T/N=1", "2.2",
      f"{T1['risk_ratio_iv'].median():.1f}" == "2.2")
check("MV >10x share at T/N=1", "65\\%",
      round((T1["risk_ratio_mv"] > 10).mean() * 100) == 65)
check("MV >100x share at T/N=1", "9\\%",
      round((T1["risk_ratio_mv"] > 100).mean() * 100) == 9)
check("HRP win vs MV pooled low", "87\\%",
      round((LOW["risk_ratio_hrp_single"] < LOW["risk_ratio_mv"]).mean() * 100) == 87)
check("unstructured tn0.5 win vs MV is exactly 1",
      "$1.00$ of them ($201$ of $201$)",
      winrate("unstructured", 0.5, "mv") == 1.0
      and int([r["n"] for r in R["hrp_win_rates"]
               if r["family"] == "unstructured" and r["tn_ratio"] == 0.5
               and r["opponent"] == "mv"][0]) == 201)
check("all low-T/N records rank-deficient", "$T-1<N$",
      bool((LOW["rank_deficient"] == 1).all()))
check("HRP >10x share low band", "8.8\\%",
      f"{(LOW['risk_ratio_hrp_single'] > 10).mean() * 100:.1f}" == "8.8")
check("EW >10x share low band", "19.9\\%",
      f"{(LOW['risk_ratio_ew'] > 10).mean() * 100:.1f}" == "19.9")
check("IV >10x share low band", "8.6\\%",
      f"{(LOW['risk_ratio_iv'] > 10).mean() * 100:.1f}" == "8.6")
check("MV >10x share low band", "42.6\\%",
      f"{(LOW['risk_ratio_mv'] > 10).mean() * 100:.1f}" == "42.6")
mv05 = [grid(f, 0.5, "mv") for f in FAMS]
check("MV medians at tn0.5 range", "$4.1$--$10.4\\times$",
      f"{min(mv05):.1f}" == "4.1" and f"{max(mv05):.1f}" == "10.4")
mv1 = [grid(f, 1.0, "mv") for f in FAMS]
check("MV medians at tn1 range", "$16.2$--$20.0\\times$",
      f"{min(mv1):.1f}" == "16.2" and f"{max(mv1):.1f}" == "20.0")
mvlo1 = [grid(f, 1.0, "mv_lo") for f in FAMS]
check("MV-LO medians at tn1 range", "$2.8$--$3.8\\times$",
      f"{min(mvlo1):.1f}" == "2.8" and f"{max(mvlo1):.1f}" == "3.8")
wlo = [scorecard("low", f, "mv_lo") for f in FAMS]
check("HRP vs MV-LO low-band per-family range", "$85$--$95\\%$",
      round(min(wlo) * 100) == 85 and round(max(wlo) * 100) == 95)

# ------------------------------------------------ data-rich text ------------
print("[data-rich regime]")
t10 = G[G["cfg_tn_ratio"] == 10.0]
check("MV+LW pooled medians at tn10", "1.10",
      f"{t10['risk_ratio_mv'].median():.2f}" == "1.10"
      and f"{t10['risk_ratio_mv_lw'].median():.2f}" == "1.10")
hrp10 = sorted(grid(f, 10.0, "hrp_single") for f in FAMS)
check("HRP plateau range at tn10", "$1.40$--$3.20\\times$",
      f2(hrp10[0]) == "1.40" and f2(hrp10[-1]) == "3.20")

# ------------------------------------------------ shrinkage comparison ------
print("[HRP vs Ledoit-Wolf]")
check("HRP win vs LW pooled low", "$0.34$",
      f2((LOW["risk_ratio_hrp_single"] < LOW["risk_ratio_mv_lw"]).mean()) == "0.34")
check("HRP win vs LW pooled high", "$0.05$",
      f2((HIGH["risk_ratio_hrp_single"] < HIGH["risk_ratio_mv_lw"]).mean()) == "0.05")
check("unstructured tn0.5 win vs LW", "$0.54$",
      f2(winrate("unstructured", 0.5, "mv_lw")) == "0.54")
check("unstructured tn1 win vs LW", "$0.57$",
      f2(winrate("unstructured", 1.0, "mv_lw")) == "0.57")
check("unstructured tn1 paired log-ratio", "$-0.02$ dex",
      f"{medlog('unstructured', 1.0, 'mv_lw'):.2f}" == "-0.02")
check("unstructured low band win vs LW (abstract 0.55)", "0.55",
      f2(scorecard("low", "unstructured", "mv_lw")) == "0.55")
struct = [f for f in FAMS if f != "unstructured"]
cond_lw_all = all(
    grid(f, tn, "mv_lw") < grid(f, tn, "hrp_single")
    for f in struct for tn in (0.5, 1.0, 2.0, 5.0, 10.0))
check("LW median beats HRP at every T/N in structured families",
      "at \\emph{every} $\\TN$ in the one-factor, hierarchical, and equicorrelation families",
      cond_lw_all)
cond_lw_cells = all(
    winrate(f, tn, "mv_lw") < 0.5 for f in struct for tn in (0.5, 1.0, 2.0, 5.0, 10.0))
check("HRP win vs LW <0.5 in every structured cell",
      "below $0.5$ in every cell", cond_lw_cells)
lwwins = [1 - winrate(f, tn, "mv_lw") for f in struct for tn in (0.5, 1.0, 2.0, 5.0, 10.0)]
check("LW per-cell win range vs HRP", "$0.63$--$0.99$",
      f2(min(lwwins)) == "0.63" and f2(max(lwwins)) == "0.99")
check("IV beats HRP in unstructured tn1 (medians)", "$1.54\\times$",
      grid("unstructured", 1.0, "iv") < grid("unstructured", 1.0, "hrp_single")
      and f2(grid("unstructured", 1.0, "iv")) == "1.54")
check("HRP median unstructured tn1", "$1.67\\times$",
      f2(grid("unstructured", 1.0, "hrp_single")) == "1.67")
check("HRP win vs IV unstructured low band", "$0.16$",
      f2(scorecard("low", "unstructured", "iv")) == "0.16")
unstr_small = [winrate("unstructured", tn, "iv") for tn in (0.5, 1.0, 2.0)]
check("HRP vs IV unstructured small-T cells", "$0.13$--$0.29$",
      f2(min(unstr_small)) == "0.13" and f2(max(unstr_small)) == "0.29")

# ------------------------------------------------ IVP core / dispersion -----
print("[HRP vs its IVP core]")
disp = R["hierarchical_dispersion_effect"]
check("dispersion tercile wins vs IV",
      "$0.35$ (mean $d=0.13$), $0.40$ (mean $d=0.38$), $0.56$ (mean $d=0.66$)",
      [f2(t["win_vs_iv"]) for t in disp] == ["0.35", "0.40", "0.56"]
      and [f2(t["mean_block_disp"]) for t in disp] == ["0.13", "0.38", "0.66"])
check("dispersion terciles vs LW", "$0.28/0.19/0.19$",
      [f2(t["win_vs_mv_lw"]) for t in disp] == ["0.28", "0.19", "0.19"])
check("top-tercile median paired advantage", "$-0.002$ dex",
      f"{disp[-1]['median_log10_vs_iv']:.3f}" == "-0.002")
hier_logs = [abs(medlog("hierarchical", tn, "iv")) for tn in (0.5, 1.0, 2.0, 5.0, 10.0)]
check("hierarchical paired log-ratios vs IV tiny", "$\\pm0.006$ dex",
      max(hier_logs) <= 0.006)

# ------------------------------------------------ linkage -------------------
print("[linkage sensitivity]")
LK = {r["family"]: r for r in R["linkage_sensitivity"]}
check("linkage medians one-factor", "$2.95/2.93/3.00$",
      [f2(LK["one_factor"][f"median_hrp_{k}"]) for k in ("single", "average", "ward")]
      == ["2.95", "2.93", "3.00"])
check("linkage medians hierarchical", "$3.04/3.06/3.04$",
      [f2(LK["hierarchical"][f"median_hrp_{k}"]) for k in ("single", "average", "ward")]
      == ["3.04", "3.06", "3.04"])
check("linkage medians unstructured", "$1.57/1.53/1.54$",
      [f2(LK["unstructured"][f"median_hrp_{k}"]) for k in ("single", "average", "ward")]
      == ["1.57", "1.53", "1.54"])
check("linkage medians equicorr", "$2.42/2.42/2.39$",
      [f2(LK["equicorr"][f"median_hrp_{k}"]) for k in ("single", "average", "ward")]
      == ["2.42", "2.42", "2.39"])
spreads = [LK[f]["median_spread"] for f in FAMS]
check("linkage spread range", "$0.02$--$0.06$",
      f2(min(spreads)) == "0.02" and f2(max(spreads)) == "0.06")

# ------------------------------------------------ concentration/turnover ----
print("[concentration and turnover]")
CT = R["concentration_turnover"]
to1 = next(r for r in CT["median_turnover_by_tn"] if r["tn_ratio"] == 1.0)
check("MV turnover at tn1", "$4.43$", f2(to1["median_turnover_mv"]) == "4.43")
check("LW turnover at tn1", "$0.52$", f2(to1["median_turnover_mv_lw"]) == "0.52")
check("HRP turnover at tn1", "$0.22$", f2(to1["median_turnover_hrp_single"]) == "0.22")
check("IV turnover at tn1", "$0.12$", f2(to1["median_turnover_iv"]) == "0.12")
hrp_to = [r["median_turnover_hrp_single"] for r in CT["median_turnover_by_tn"]]
check("HRP turnover range", "$0.11$--$0.30$",
      f2(min(hrp_to)) == "0.11" and f2(max(hrp_to)) == "0.30")
lw_to = [r["median_turnover_mv_lw"] for r in CT["median_turnover_by_tn"]]
check("HRP below LW turnover throughout", "below Ledoit--Wolf throughout",
      all(h < l for h, l in zip(hrp_to, lw_to)))
check("MV gross short", "$0.57$", f2(CT["median_short_gross_mv"]) == "0.57")
check("LW gross short", "$0.25$", f2(CT["median_short_gross_mv_lw"]) == "0.25")
hhi_all = [v for row in CT["median_hhi_by_family"] for k, v in row.items()
           if k.startswith("median_hhi")]
check("HHI range", "$0.02$--$0.08$",
      f2(min(hhi_all)) == "0.02" and f2(max(hhi_all)) == "0.08")
lw_delta = G.groupby("cfg_tn_ratio")["lw_shrinkage"].median()
check("LW intensity at tn0.5", "$0.42$", f2(lw_delta[0.5]) == "0.42")
check("LW intensity at tn10", "$0.04$", f2(lw_delta[10.0]) == "0.04")

# ------------------------------------------------ noisy-mu ------------------
print("[noisy-mu Markowitz]")
low_mu = [r for r in R["sharpe_summary"] if r["band"].startswith("low")]
mu_meds = [r["median_sr_ratio_mv_mu"] for r in low_mu]
check("noisy-mu capture by family", "$8$--$11\\%$",
      round(min(mu_meds) * 100) == 8 and round(max(mu_meds) * 100) == 11)
check("noisy-mu pooled capture low", "$10\\%$",
      round(LOW["sr_ratio_mv_mu"].median() * 100) == 10)
check("noisy-mu negative share low", "$28\\%$",
      round((LOW["sr_true_mv_mu"] < 0).mean() * 100) == 28)
mu_lw_meds = [r["median_sr_ratio_mv_mu_lw"] for r in low_mu]
check("noisy-mu+LW capture", "$19$--$22\\%$",
      round(min(mu_lw_meds) * 100) == 19 and round(max(mu_lw_meds) * 100) == 22)
high_mu = [r for r in R["sharpe_summary"] if r["band"].startswith("high")]
hm = [r["median_sr_ratio_mv_mu"] for r in high_mu]
check("noisy-mu capture high band", "$48$--$71\\%$",
      round(min(hm) * 100) == 48 and round(max(hm) * 100) == 71)
hneg = [r["frac_negative_sr_mv_mu"] for r in high_mu]
check("noisy-mu negative share high", "$2$--$6\\%$",
      round(min(hneg) * 100) == 2 and round(max(hneg) * 100) == 6)
or_struct = [r["median_sr_ratio_oracle_minvar"] for r in R["sharpe_summary"]
             if r["family"] != "unstructured"]
check("oracle minvar capture structured", "$0.20$--$0.26$",
      f2(min(or_struct)) == "0.20" and f2(max(or_struct)) == "0.26")
or_unstr = [r["median_sr_ratio_oracle_minvar"] for r in R["sharpe_summary"]
            if r["family"] == "unstructured"]
check("oracle minvar capture unstructured", "$0.84$--$0.85$",
      f2(min(or_unstr)) == "0.84" and f2(max(or_unstr)) == "0.85")
mufree = [r[k] for r in low_mu
          for k in ("median_sr_ratio_ew", "median_sr_ratio_mv_lw",
                    "median_sr_ratio_hrp_single")]
check("mu-free capture range low band", "$26$--$70\\%$",
      round(min(mufree) * 100) == 26 and round(max(mufree) * 100) == 70)

# ------------------------------------------------ student-t text ------------
print("[student-t text claims]")
ST = {(r["dist"], r["family"]): r for r in R["student_t_robustness"]}
tw = [ST[("student_t", f)]["win_hrp_single_vs_mv_lw"] for f in FAMS]
gw = [ST[("gaussian", f)]["win_hrp_single_vs_mv_lw"] for f in FAMS]
check("t-batch HRP vs LW win range", "$0.13$--$0.38$",
      f2(min(tw)) == "0.13" and f2(max(tw)) == "0.38")
check("gaussian HRP vs LW win range", "$0.14$--$0.32$",
      f2(min(gw)) == "0.14" and f2(max(gw)) == "0.32")
mv_shift = [ST[("student_t", f)]["median_risk_ratio_mv"]
            - ST[("gaussian", f)]["median_risk_ratio_mv"] for f in FAMS]
check("MV worsens under t by +0.04..+0.25", "$+0.04$ to $+0.25$",
      f2(min(mv_shift)) == "0.04" and f2(max(mv_shift)) == "0.25"
      and min(mv_shift) > 0)
lw_shift = [ST[("student_t", f)]["median_risk_ratio_mv_lw"]
            - ST[("gaussian", f)]["median_risk_ratio_mv_lw"] for f in FAMS]
check("LW worsens under t by +0.02..+0.11", "$+0.02$ to $+0.11$",
      f2(min(lw_shift)) == "0.02" and f2(max(lw_shift)) == "0.11"
      and min(lw_shift) > 0)
tn_counts = sorted(DF[DF["cfg_dist"] == "student_t"]["cfg_family"].value_counts())
check("t-cells n range", "$193$--$205$",
      tn_counts[0] == 193 and tn_counts[-1] == 205)

# ---------------------------------------- adversarial-review additions ------
print("[review fixes: long-only tie, sample-MV capture, N gradient, d-tilde, rcond]")
# methods / record-count bookkeeping (hrp_dod added to the risk track)
check("ten risk-track methods", "ten methods",
      len([c for c in DF.columns if c.startswith("risk_ratio_")]) == 10)
check("record field count (setup)", "$4{,}800\\times86$", DF.shape[1] == 86)
check("record field count (reproducibility)", "$86$ recorded fields",
      DF.shape[1] == 86)
check("table-1 caption per-cell range", "$180$--$212$",
      min(r["n"] for r in R["headline_grid"]
          if r["method"] == "ew" and r["tn_ratio"] in (1.0, 10.0)) == 180
      and max(r["n"] for r in R["headline_grid"]
              if r["method"] == "ew" and r["tn_ratio"] in (1.0, 10.0)) == 212)
# like-for-like long-only comparison at T/N=1 (abstract)
check("LW-LO pooled median at tn1", "$2.32\\times$",
      f2(T1["risk_ratio_mv_lw_lo"].median()) == "2.32")
check("HRP pooled median at tn1 (2dp)", "$2.29\\times$",
      f2(T1["risk_ratio_hrp_single"].median()) == "2.29")
# HRP vs IV in the data-rich band (abstract claim re-scope)
iv_high = [scorecard("high", f, "iv") for f in FAMS]
check("HRP vs IV high-band win range", "$0.46$--$0.63$",
      f2(min(iv_high)) == "0.46" and f2(max(iv_high)) == "0.63"
      and sum(v > 0.5 for v in iv_high) == 3)
# mu-free sample MV Sharpe capture (Section 6.6 / contribution 5)
mv_cap = LOW["sr_ratio_mv"].replace([np.inf, -np.inf], np.nan).dropna()
check("sample-MV Sharpe capture pooled low", "$18\\%$",
      round(mv_cap.median() * 100) == 18)
mv_cap_fam = [LOW[LOW["cfg_family"] == f]["sr_ratio_mv"]
              .replace([np.inf, -np.inf], np.nan).dropna().median() for f in FAMS]
check("sample-MV capture per-family range", "$0.15$--$0.25$",
      f2(min(mv_cap_fam)) == "0.15" and f2(max(mv_cap_fam)) == "0.25")
# unstructured low-band exception: N gradient (footnote)
U = LOW[LOW["cfg_family"] == "unstructured"]
wins_by_n = {}
for n, gn in U.groupby("cfg_n_assets"):
    a, b = gn["risk_ratio_hrp_single"], gn["risk_ratio_mv_lw"]
    ok = a.notna() & b.notna()
    wins_by_n[int(n)] = float((a[ok] < b[ok]).mean())
check("unstructured low-band win vs LW by N",
      "$0.34$ at $N=10$, $0.59$ at $N=30$, $0.60$ at $N=50$, and $0.69$ at $N=100$",
      [f2(wins_by_n[n]) for n in (10, 30, 50, 100)] == ["0.34", "0.59", "0.60", "0.69"])
# distance-matrix convention: condensed d vs distance-of-distances d-tilde
DC = {r["family"]: r for r in R["distance_convention"]}
check("d vs d-tilde per-family medians",
      "$2.95\\to2.94$ one-factor, $3.04\\to3.05$ hierarchical, "
      "$1.57\\to1.58$ unstructured, $2.42\\to2.44$ equicorrelation",
      [f2(DC[f]["median_hrp_single"]) for f in FAMS] == ["2.95", "3.04", "1.57", "2.42"]
      and [f2(DC[f]["median_hrp_dod"]) for f in FAMS] == ["2.94", "3.05", "1.58", "2.44"])
check("d-tilde max median shift", "at most $0.02$",
      max(abs(DC[f]["median_hrp_dod"] - DC[f]["median_hrp_single"]) for f in FAMS) <= 0.02)
check("d-tilde paired median log-ratio", "exactly zero",
      all(DC[f]["median_log10_dod_vs_single"] == 0.0 for f in FAMS))
dwin = [DC[f]["win_dod_vs_single"] for f in FAMS]
check("d-tilde win-rate range", "$0.44$--$0.48$",
      f2(min(dwin)) == "0.44" and f2(max(dwin)) == "0.48")
ident = [DC[f]["frac_identical_risk"] for f in FAMS]
check("identical-risk share range", "$4$--$9\\%$",
      round(min(ident) * 100) == 4 and round(max(ident) * 100) == 9)
# pinv rcond sweep behind the 16.8x headline
SW = R["mv_pinv_rcond_sweep"]["median_risk_ratio_mv_by_rcond"]
check("rcond sweep medians", "$16.82/16.82/15.61/3.98$",
      [f2(SW[k]) for k in ("1e-15", "1e-10", "1e-06", "0.001")]
      == ["16.82", "16.82", "15.61", "3.98"])
check("rcond default consistent with stored records", "$\\texttt{rcond}=10^{-15}$",
      abs(SW["1e-15"] - T1["risk_ratio_mv"].median()) < 1e-9)

# ------------------------------------------------ discussion ----------------
print("[discussion claims]")
crossings_ok = all(
    winrate(f, 1.0, "mv") > 0.5 and winrate(f, 2.0, "mv") < 0.5 for f in struct
) and winrate("unstructured", 2.0, "mv") > 0.5 and winrate("unstructured", 5.0, "mv") < 0.5
check("break-even crossing location",
      "between $\\TN=1$ and $2$", crossings_ok)
check("specification gap percent", "$220\\%$",
      round((grid("one_factor", 10.0, "hrp_single") - 1) * 100 / 10) * 10 == 220)
check("specification gap percent low", "$40\\%$",
      round((grid("unstructured", 10.0, "hrp_single") - 1) * 100 / 10) * 10 == 40)

# ---------------------------------------------------------------- result ----
print(f"\n{n_checks} checks; {len(failures)} failures")
if failures:
    sys.exit(1)
print("OK: every quoted number matches results/")
