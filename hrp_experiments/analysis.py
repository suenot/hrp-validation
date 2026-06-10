"""Turn a batch of experiment records into the paper's quantitative results.

Five questions, all answered with measured numbers and ALL stratified by
correlation-structure family (no aggregate-only claims -- HRP's advantage is
structure-dependent by hypothesis):

1. **Headline grid.** Median realized-risk ratio vs the oracle minimum-variance
   floor, by method x T/N ratio x structure family.
2. **Where HRP wins / loses.** Pairwise win rates of HRP (single linkage)
   against each competitor, per family x T/N cell, plus median paired log-ratio.
3. **Linkage sensitivity.** single vs average vs ward; plus the distance
   convention (condensed ``d`` vs distance-of-distances ``d_tilde``).
4. **Concentration & turnover.** Herfindahl of gross weights; estimation-noise
   turnover between two independent windows from the same truth.
5. **Sharpe track.** True Sharpe (vs the oracle tangency ceiling) of the mu-free
   methods and the noisy-mu Markowitz directions, recorded separately.

Everything returned is JSON-able. Gaussian and Student-t records are kept
separate via the ``cfg_dist`` column; the robustness summary compares them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import RISK_METHODS

#: competitors HRP is scored against in the win-rate analysis
OPPONENTS: tuple[str, ...] = ("ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo")
HRP_MAIN = "hrp_single"
LINKAGES: tuple[str, ...] = ("hrp_single", "hrp_average", "hrp_ward")


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _gauss(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["cfg_dist"] == "gaussian"]


def _q(s: pd.Series, q: float) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.quantile(q)) if len(s) else float("nan")


# --------------------------------------------------------------------------- #
# 1. headline grid: realized-risk ratio vs oracle
# --------------------------------------------------------------------------- #
def headline_grid(df: pd.DataFrame, methods: tuple[str, ...] = RISK_METHODS) -> list[dict]:
    """Median (+IQR) of ``w' Sigma w / oracle`` per family x T/N x method."""
    rows = []
    for (fam, ratio), g in _gauss(df).groupby(["cfg_family", "cfg_tn_ratio"]):
        for m in methods:
            col = g[f"risk_ratio_{m}"]
            rows.append({
                "family": fam, "tn_ratio": float(ratio), "method": m,
                "median_risk_ratio": _q(col, 0.5),
                "q25": _q(col, 0.25), "q75": _q(col, 0.75),
                "n": int(col.notna().sum()),
            })
    return rows


# --------------------------------------------------------------------------- #
# 2. where HRP wins / loses
# --------------------------------------------------------------------------- #
def hrp_win_rates(df: pd.DataFrame, hrp: str = HRP_MAIN) -> list[dict]:
    """P(HRP realized risk < opponent's) and the median paired log10 risk ratio
    (negative => HRP typically less risky), per family x T/N x opponent."""
    rows = []
    for (fam, ratio), g in _gauss(df).groupby(["cfg_family", "cfg_tn_ratio"]):
        for opp in OPPONENTS:
            a, b = g[f"risk_ratio_{hrp}"], g[f"risk_ratio_{opp}"]
            ok = a.notna() & b.notna()
            if ok.sum() == 0:
                continue
            ratio_log = np.log10(a[ok] / b[ok])
            rows.append({
                "family": fam, "tn_ratio": float(ratio), "opponent": opp,
                "hrp_win_rate": float((a[ok] < b[ok]).mean()),
                "median_log10_risk_ratio": float(ratio_log.median()),
                "n": int(ok.sum()),
            })
    return rows


def hrp_scorecard(df: pd.DataFrame) -> list[dict]:
    """Coarse summary: HRP win rate vs each opponent per family, pooled over the
    low-sample regime (T/N <= 1) and the data-rich regime (T/N >= 5)."""
    g = _gauss(df)
    bands = {"low_T/N (<=1)": g["cfg_tn_ratio"] <= 1.0, "high_T/N (>=5)": g["cfg_tn_ratio"] >= 5.0}
    rows = []
    for band, mask in bands.items():
        sub = g[mask]
        for fam, gf in sub.groupby("cfg_family"):
            row: dict = {"band": band, "family": fam, "n": int(len(gf))}
            for opp in OPPONENTS:
                a, b = gf[f"risk_ratio_{HRP_MAIN}"], gf[f"risk_ratio_{opp}"]
                ok = a.notna() & b.notna()
                row[f"win_vs_{opp}"] = float((a[ok] < b[ok]).mean())
            rows.append(row)
    return rows


def hierarchical_dispersion_effect(df: pd.DataFrame) -> list[dict]:
    """Within the hierarchical family at low T/N (<= 2): does HRP's edge over
    the structure-blind allocators grow with cross-block tightness dispersion
    (``cfg_block_disp``)? Terciles of block_disp; win rates vs IV and LW."""
    g = _gauss(df)
    h = g[(g["cfg_family"] == "hierarchical") & (g["cfg_tn_ratio"] <= 2.0)].copy()
    if len(h) < 30:
        return []
    h["disp_bin"] = pd.qcut(h["cfg_block_disp"], 3,
                            labels=["low_disp", "mid_disp", "high_disp"])
    rows = []
    for b, gb in h.groupby("disp_bin", observed=True):
        a = gb[f"risk_ratio_{HRP_MAIN}"]
        iv, lw = gb["risk_ratio_iv"], gb["risk_ratio_mv_lw"]
        ok_iv = a.notna() & iv.notna()
        ok_lw = a.notna() & lw.notna()
        rows.append({
            "block_disp_tercile": str(b),
            "mean_block_disp": float(gb["cfg_block_disp"].mean()),
            "n": int(len(gb)),
            "win_vs_iv": float((a[ok_iv] < iv[ok_iv]).mean()),
            "win_vs_mv_lw": float((a[ok_lw] < lw[ok_lw]).mean()),
            "median_log10_vs_iv": float(np.log10(a[ok_iv] / iv[ok_iv]).median()),
        })
    return rows


# --------------------------------------------------------------------------- #
# 3. linkage sensitivity
# --------------------------------------------------------------------------- #
def linkage_sensitivity(df: pd.DataFrame) -> list[dict]:
    """Median risk ratio per linkage per family (+ spread between best/worst).

    ``median_hrp_dod`` (single linkage under the distance-of-distances
    convention; see ``distance_convention``) is reported alongside for
    reference but kept out of the canonical three-linkage best/spread.
    """
    rows = []
    for fam, g in _gauss(df).groupby("cfg_family"):
        med = {lk: _q(g[f"risk_ratio_{lk}"], 0.5) for lk in LINKAGES}
        rows.append({
            "family": fam,
            **{f"median_{lk}": v for lk, v in med.items()},
            "median_hrp_dod": _q(g["risk_ratio_hrp_dod"], 0.5),
            "best_linkage": min(med, key=med.get),  # type: ignore[arg-type]
            "median_spread": float(max(med.values()) - min(med.values())),
            "n": int(len(g)),
        })
    return rows


def distance_convention(df: pd.DataFrame) -> list[dict]:
    """Single-linkage HRP under the two distance conventions, per family (plus
    a pooled row): condensed correlation distance ``d`` (our main
    implementation, ``hrp_single``) vs the Euclidean distance-of-distances
    ``d_tilde`` of Lopez de Prado's (2016) code listing (``hrp_dod``).

    ``frac_identical_risk`` is the fraction of experiments whose realized risk
    is bit-identical under the two conventions (the leaf orders coincided);
    the win rate and paired median quantify the (im)materiality of the rest.
    """
    g = _gauss(df)
    groups = [("pooled", g), *list(g.groupby("cfg_family"))]
    rows = []
    for fam, gf in groups:
        a, b = gf["risk_ratio_hrp_single"], gf["risk_ratio_hrp_dod"]
        ok = a.notna() & b.notna()
        rows.append({
            "family": str(fam),
            "n": int(ok.sum()),
            "median_hrp_single": _q(a, 0.5),
            "median_hrp_dod": _q(b, 0.5),
            "median_log10_dod_vs_single": float(np.log10(b[ok] / a[ok]).median()),
            "frac_identical_risk": float((a[ok] == b[ok]).mean()),
            "win_dod_vs_single": float((b[ok] < a[ok]).mean()),
        })
    return rows


# --------------------------------------------------------------------------- #
# 4. concentration & turnover diagnostics
# --------------------------------------------------------------------------- #
def concentration_turnover(df: pd.DataFrame) -> dict:
    g = _gauss(df)
    hhi = []
    for fam, gf in g.groupby("cfg_family"):
        row: dict = {"family": fam}
        for m in RISK_METHODS:
            row[f"median_hhi_{m}"] = _q(gf[f"hhi_{m}"], 0.5)
        hhi.append(row)
    to = []
    for ratio, gr in g.groupby("cfg_tn_ratio"):
        row = {"tn_ratio": float(ratio)}
        for m in RISK_METHODS:
            row[f"median_turnover_{m}"] = _q(gr[f"turnover_{m}"], 0.5)
        to.append(row)
    return {
        "median_hhi_by_family": hhi,
        "median_turnover_by_tn": to,
        "median_short_gross_mv": _q(g["short_gross_mv"], 0.5),
        "median_short_gross_mv_lw": _q(g["short_gross_mv_lw"], 0.5),
    }


# --------------------------------------------------------------------------- #
# 5. Sharpe track (mu-free vs noisy-mu, vs the oracle tangency ceiling)
# --------------------------------------------------------------------------- #
def sharpe_summary(df: pd.DataFrame) -> list[dict]:
    """Median Sharpe ratio captured (sr / oracle tangency sr) per family x T/N
    band, for mu-free methods AND the noisy-mu Markowitz directions."""
    g = _gauss(df)
    methods = [*[f"sr_ratio_{m}" for m in ("ew", "mv_lw", HRP_MAIN)],
               "sr_ratio_mv_mu", "sr_ratio_mv_mu_lw"]
    bands = {"low_T/N (<=1)": g["cfg_tn_ratio"] <= 1.0, "high_T/N (>=5)": g["cfg_tn_ratio"] >= 5.0}
    rows = []
    for band, mask in bands.items():
        for fam, gf in g[mask].groupby("cfg_family"):
            row: dict = {"band": band, "family": fam, "n": int(len(gf))}
            for col in methods:
                row[f"median_{col}"] = _q(gf[col], 0.5)
            row["frac_negative_sr_mv_mu"] = float((gf["sr_true_mv_mu"] < 0).mean())
            row["median_sr_ratio_oracle_minvar"] = _q(
                gf["sr_true_oracle_minvar"] / gf["sr_oracle_tangency"], 0.5)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 6. Student-t robustness
# --------------------------------------------------------------------------- #
def student_t_robustness(df: pd.DataFrame) -> list[dict]:
    """Median risk ratios for gaussian vs student_t, per family, for the key
    methods -- a qualitative-stability check, not a separate headline."""
    rows = []
    for dist, gd in df.groupby("cfg_dist"):
        for fam, gf in gd.groupby("cfg_family"):
            row: dict = {"dist": dist, "family": fam, "n": int(len(gf))}
            for m in ("ew", "mv", "mv_lw", HRP_MAIN):
                row[f"median_risk_ratio_{m}"] = _q(gf[f"risk_ratio_{m}"], 0.5)
            a, b = gf[f"risk_ratio_{HRP_MAIN}"], gf["risk_ratio_mv_lw"]
            ok = a.notna() & b.notna()
            row[f"win_{HRP_MAIN}_vs_mv_lw"] = float((a[ok] < b[ok]).mean())
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# top-level summary
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> dict:
    g = _gauss(df)
    return {
        "n_experiments": int(len(df)),
        "n_gaussian": int(len(g)),
        "n_student_t": int((df["cfg_dist"] == "student_t").sum()),
        "n_by_family": {str(k): int(v) for k, v in g["cfg_family"].value_counts().items()},
        "headline_grid": headline_grid(df),
        "hrp_win_rates": hrp_win_rates(df),
        "hrp_scorecard": hrp_scorecard(df),
        "hierarchical_dispersion_effect": hierarchical_dispersion_effect(df),
        "linkage_sensitivity": linkage_sensitivity(df),
        "distance_convention": distance_convention(df),
        "concentration_turnover": concentration_turnover(df),
        "sharpe_summary": sharpe_summary(df),
        "student_t_robustness": student_t_robustness(df),
    }
