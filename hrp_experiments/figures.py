"""Generate the paper's figures (vector PDF) from the saved results.

    python -m hrp_experiments.figures      # writes paper/figures/*.pdf

Reads results/records.csv; the setup panels are recomputed deterministically
from fixed canonical configs + seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from .allocators import corr_from_cov, hrp_linkage, quasi_diag_order
from .model import FAMILIES, canonical_configs, true_covariance
from .simulate import RISK_METHODS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

METHOD_STYLE = {
    "ew":         ("#9aa6b2", "o", "equal weight (1/N)"),
    "iv":         ("#e0a458", "v", "inverse variance"),
    "mv":         ("#c0392b", "s", "min-var (sample, shorts)"),
    "mv_lo":      ("#e74c3c", "D", "min-var (sample, LO clip)"),
    "mv_lw":      ("#2e8b57", "P", "min-var (Ledoit-Wolf)"),
    "mv_lw_lo":   ("#27ae60", "X", "min-var (LW, LO clip)"),
    "hrp_single": ("#1f3b73", "^", "HRP (single)"),
    "hrp_average":("#4a69bd", "<", "HRP (average)"),
    "hrp_ward":   ("#7f9bd1", ">", "HRP (ward)"),
}
FAMILY_LABEL = {
    "one_factor": "one-factor", "hierarchical": "hierarchical",
    "unstructured": "unstructured", "equicorr": "equicorrelation",
}
SETUP_SEED = 42  # fixed seed for the illustrative setup figure only


def _gauss(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["cfg_dist"] == "gaussian"]


# --------------------------------------------------------------------------- #
# Fig 1: setup -- the four true structures + HRP's view of the hierarchical one
# --------------------------------------------------------------------------- #
def fig_setup(path: Path) -> None:
    cfgs = canonical_configs()
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])

    corrs: dict[str, np.ndarray] = {}
    for j, fam in enumerate(FAMILIES):
        rng = np.random.default_rng(SETUP_SEED + j)
        sigma, _, info = true_covariance(cfgs[fam], rng)
        corrs[fam] = corr_from_cov(sigma)
        shown = corrs[fam]
        if fam == "hierarchical":
            # the generator randomly permutes assets; panel (b) shows the truth
            # in its natural block order (panel (f) shows the permuted view)
            natural = np.lexsort((info["sub"], info["block"]))
            shown = shown[np.ix_(natural, natural)]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(shown, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(f"({'abcd'[j]}) {FAMILY_LABEL[fam]}")
        ax.set_xticks([]); ax.set_yticks([])

    c_h = corrs["hierarchical"]   # asset order randomly permuted by the generator
    link = hrp_linkage(c_h, "single")
    order = quasi_diag_order(link)

    ax = fig.add_subplot(gs[1, 0:2])
    sch.dendrogram(link, ax=ax, no_labels=True, color_threshold=0.0,
                   above_threshold_color="#1f3b73")
    ax.set_title("(e) single-linkage dendrogram on $d_{ij}=\\sqrt{(1-\\rho_{ij})/2}$")
    ax.set_ylabel("merge distance")
    ax.tick_params(axis="x", length=0)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(c_h, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    ax.set_title("(f) hierarchical, shuffled order")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(c_h[np.ix_(order, order)], vmin=-1, vmax=1, cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_title("(g) quasi-diagonalized (leaf order)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: realized-risk ratio vs T/N, per structure family
# --------------------------------------------------------------------------- #
def fig_risk_vs_tn(path: Path, df: pd.DataFrame) -> None:
    g = _gauss(df)
    show = ["ew", "iv", "mv", "mv_lo", "mv_lw", "hrp_single"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        gf = g[g["cfg_family"] == fam]
        for m in show:
            color, marker, label = METHOD_STYLE[m]
            med = gf.groupby("cfg_tn_ratio")[f"risk_ratio_{m}"].median()
            ax.plot(med.index, med.values, marker=marker, ms=4, lw=1.4,
                    color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(1.0, color="k", lw=0.7, ls=":")
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel("$T/N$ (estimation window / assets)")
    axes[0].set_ylabel("median realized risk / oracle\n($w^\\top\\Sigma w$, true $\\Sigma$)")
    axes[-1].legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: HRP win-rate heatmaps, opponent x (family x T/N)
# --------------------------------------------------------------------------- #
def fig_winrate(path: Path, df: pd.DataFrame) -> None:
    g = _gauss(df)
    opponents = ["ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo"]
    ratios = sorted(g["cfg_tn_ratio"].unique())
    fig, axes = plt.subplots(1, len(opponents), figsize=(13.5, 2.9), sharey=True)
    for ax, opp in zip(axes, opponents):
        mat = np.full((len(FAMILIES), len(ratios)), np.nan)
        for i, fam in enumerate(FAMILIES):
            for j, r in enumerate(ratios):
                sub = g[(g["cfg_family"] == fam) & (g["cfg_tn_ratio"] == r)]
                if len(sub):
                    a, b = sub["risk_ratio_hrp_single"], sub[f"risk_ratio_{opp}"]
                    ok = a.notna() & b.notna()
                    if ok.sum():
                        mat[i, j] = (a[ok] < b[ok]).mean()
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdBu", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(mat[i, j] - 0.5) > 0.3 else "black")
        ax.set_title(f"vs {METHOD_STYLE[opp][2]}", fontsize=7.5)
        ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios], fontsize=7)
        ax.set_xlabel("$T/N$", fontsize=8)
    axes[0].set_yticks(range(len(FAMILIES)), [FAMILY_LABEL[f] for f in FAMILIES], fontsize=7.5)
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cbar.set_label("P(HRP realized risk < opponent)", fontsize=7)
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: linkage sensitivity + concentration + estimation-noise turnover
# --------------------------------------------------------------------------- #
def fig_sensitivity(path: Path, df: pd.DataFrame) -> None:
    g = _gauss(df)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.1))

    # (a) linkage sensitivity per family
    ax = axes[0]
    linkages = ["hrp_single", "hrp_average", "hrp_ward"]
    x = np.arange(len(FAMILIES)); w = 0.26
    for k, lk in enumerate(linkages):
        med = [g[g["cfg_family"] == fam][f"risk_ratio_{lk}"].median() for fam in FAMILIES]
        ax.bar(x + (k - 1) * w, med, w, color=METHOD_STYLE[lk][0],
               label=lk.replace("hrp_", ""))
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xticks(x, [FAMILY_LABEL[f] for f in FAMILIES], fontsize=7, rotation=12)
    ax.set_ylabel("median realized risk / oracle")
    ax.set_title("(a) HRP linkage sensitivity")
    ax.legend(fontsize=7, frameon=False)

    # (b) weight concentration (HHI of gross weights) by method
    ax = axes[1]
    show = ["ew", "iv", "mv", "mv_lw", "hrp_single"]
    x = np.arange(len(show)); w = 0.19
    for k, fam in enumerate(FAMILIES):
        gf = g[g["cfg_family"] == fam]
        med = [gf[f"hhi_{m}"].median() for m in show]
        ax.bar(x + (k - 1.5) * w, med, w, label=FAMILY_LABEL[fam],
               color=plt.cm.viridis(k / 3.2))
    short = {"ew": "EW", "iv": "IV", "mv": "MV (sample)",
             "mv_lw": "MV (LW)", "hrp_single": "HRP"}
    ax.set_xticks(x, [short[m] for m in show], fontsize=7)
    ax.set_ylabel("median HHI of $|w|/\\Vert w\\Vert_1$")
    ax.set_title("(b) weight concentration")
    ax.legend(fontsize=6.5, frameon=False)

    # (c) estimation-noise turnover vs T/N
    ax = axes[2]
    for m in ["ew", "iv", "mv", "mv_lw", "hrp_single"]:
        color, marker, label = METHOD_STYLE[m]
        med = g.groupby("cfg_tn_ratio")[f"turnover_{m}"].median()
        ax.plot(med.index, med.values, marker=marker, ms=4, lw=1.4, color=color,
                label=label)
    ax.set_xscale("log")
    ax.set_xlabel("$T/N$")
    ax.set_ylabel("median turnover between\nindependent windows")
    ax.set_title("(c) estimation-noise turnover")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "records.csv")
    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_risk_vs_tn(FIGDIR / "fig_risk_vs_tn.pdf", df)
    fig_winrate(FIGDIR / "fig_winrate.pdf", df)
    fig_sensitivity(FIGDIR / "fig_sensitivity.pdf", df)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
