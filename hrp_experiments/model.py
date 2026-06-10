"""Ground-truth covariance generator for the HRP-vs-Markowitz comparison.

The whole experimental design rests on one device: the TRUE covariance matrix
``Sigma`` is known by construction, so for any weight vector ``w`` the realized
portfolio risk ``w' Sigma w`` is computable *exactly* -- no backtest noise, no
data snooping. Estimation error (the entire argument for HRP in Lopez de Prado
2016) is then isolated by handing every allocator a finite sample of ``T``
observations drawn from that truth and scoring the resulting weights on the
truth itself.

Four correlation families, spanning the cases where HRP's clustering prior is
right, partially right, and wrong:

* ``one_factor``     -- single common factor with heterogeneous loadings plus
                        idiosyncratic noise (CAPM-like; weak cluster structure).
* ``hierarchical``   -- nested global/block/sub-block factor model: the exact
                        world HRP's dendrogram is built for. Asset order is
                        randomly permuted so no method can exploit input order.
* ``unstructured``   -- random correlation from a normalized Wishart draw
                        (``C = cov2corr(X'X)``, ``X`` Gaussian with ``dof``
                        rows); no cluster structure for HRP to find.
* ``equicorr``       -- constant correlation ``rho`` (PSD for ``rho > -1/(N-1)``;
                        we sample positive values only).

All families use heterogeneous per-asset volatilities (log-uniform) and a true
mean vector ``mu_i = s_i * sigma_i`` with per-asset Sharpe ``s_i`` drawn around
a sampled level, so a true-Sharpe evaluation is available alongside the pure
minimum-risk track.

Honesty contract (a sibling project was burned by hidden constants): every
scalar that influences the generated truth is either (a) a field of
``CovConfig`` -- drawn in ``sample_config`` from the explicit ranges in
``SAMPLING_RANGES`` and stored in every output record -- or (b) an asset-level
draw made by the generator from the experiment RNG, whose realized summary
(mean off-diagonal correlation, vol spread) is recorded per experiment. There
are no other knobs.

Limitation, stated up front: returns are i.i.d. Gaussian (one Student-t
robustness batch aside) -- no fat joint tails beyond multivariate-t, no
autocorrelation, no regime switching, and the true covariance is constant over
the estimation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Family = Literal["one_factor", "hierarchical", "unstructured", "equicorr"]
FAMILIES: tuple[Family, ...] = ("one_factor", "hierarchical", "unstructured", "equicorr")

# --------------------------------------------------------------------------- #
# every sampled range lives here, in one visible place
# --------------------------------------------------------------------------- #
SAMPLING_RANGES: dict[str, tuple] = {
    "n_assets_choices": (10, 30, 50, 100),
    "tn_ratio_choices": (0.5, 1.0, 2.0, 5.0, 10.0),   # T = round(ratio * N), the key axis
    "vol_lo": (0.005, 0.02),          # lowest per-asset vol (per period)
    "vol_ratio": (1.5, 6.0),          # sigma_hi / sigma_lo (log-uniform spread)
    "sharpe_mean": (0.0, 0.10),       # mean per-asset per-period Sharpe
    "sharpe_disp": (0.0, 0.05),       # cross-sectional sd of per-asset Sharpe
    "t_df": (4.0, 10.0),              # Student-t dof (robustness batch only)
    # one_factor
    "factor_r2": (0.10, 0.70),        # mean variance share of the common factor
    "factor_r2_disp": (0.0, 0.15),    # per-asset spread of that share
    # hierarchical
    "n_blocks_choices": (2, 3, 4, 5),
    "n_sub_choices": (1, 2, 3),       # sub-clusters per block
    "global_share": (0.05, 0.30),     # variance share of the global factor
    "block_share": (0.10, 0.40),      # ... of the block factor
    "sub_share": (0.00, 0.30),        # ... of the sub-block factor
    "max_total_share": 0.90,          # cap so idiosyncratic variance >= 0.10
    "loading_jitter": (0.0, 0.30),    # per-asset multiplicative jitter on shares
    "block_disp": (0.0, 0.8),         # per-BLOCK share dispersion: 0 => all blocks
                                      # equally tight (IV near-optimal); large =>
                                      # blocks of very different tightness (the
                                      # case cluster-level allocation is for)
    # unstructured
    "wishart_dof_ratio": (1.2, 5.0),  # dof = round(ratio * N); lower => wilder corr
    # equicorr
    "equi_rho": (0.05, 0.70),
}


@dataclass(frozen=True)
class CovConfig:
    """Everything that defines one ground-truth (Sigma, mu) draw.

    Unused family parameters stay at their NaN/0 defaults and are still written
    to the records CSV, so the provenance of every experiment is auditable.
    """

    family: Family = "hierarchical"
    n_assets: int = 30
    t_obs: int = 60                    # estimation-window length T
    tn_ratio: float = 2.0              # the recorded T/N ratio (t_obs = round(ratio*N))
    vol_lo: float = 0.01
    vol_ratio: float = 3.0
    dist: Literal["gaussian", "student_t"] = "gaussian"
    t_df: float = float("nan")         # Student-t dof (NaN for gaussian)
    sharpe_mean: float = 0.05
    sharpe_disp: float = 0.02
    # one_factor
    factor_r2: float = float("nan")
    factor_r2_disp: float = float("nan")
    # hierarchical
    n_blocks: int = 0
    n_sub: int = 0
    global_share: float = float("nan")
    block_share: float = float("nan")
    sub_share: float = float("nan")
    loading_jitter: float = float("nan")
    block_disp: float = float("nan")
    # unstructured
    wishart_dof_ratio: float = float("nan")
    # equicorr
    equi_rho: float = float("nan")
    label: str = "custom"


# --------------------------------------------------------------------------- #
# correlation builders (all PSD by construction)
# --------------------------------------------------------------------------- #
def _one_factor_corr(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """C = beta beta' + diag(1 - beta^2) with beta_i = sqrt(r2_i)."""
    n = cfg.n_assets
    lo = max(0.01, cfg.factor_r2 - cfg.factor_r2_disp)
    hi = min(0.90, cfg.factor_r2 + cfg.factor_r2_disp)
    r2 = rng.uniform(lo, hi, size=n)
    beta = np.sqrt(r2)
    corr = np.outer(beta, beta)
    np.fill_diagonal(corr, 1.0)
    return corr, {"beta": beta, "block": None, "sub": None}


def _hierarchical_corr(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Nested factor model: global + block + sub-block factors + idiosyncratic.

    Variance shares (global_share, block_share, sub_share) are jittered per
    asset, capped so idiosyncratic variance stays positive; loadings are the
    square roots, which keeps ``C = A A' + diag(idio)`` PSD by construction.
    Expected correlations: ~global_share across blocks, ~global+block within a
    block, ~global+block+sub within a sub-block. Assets are randomly permuted.

    ``block_disp`` additionally scales each block's (and sub-block's) share by a
    per-cluster multiplier ``~U(1-block_disp, 1+block_disp)``: at 0 all clusters
    are equally tight (inverse-variance is then near-optimal by symmetry); at
    large values clusters differ strongly in tightness, which is exactly the
    asymmetry cluster-level allocators are meant to exploit. It is a sampled,
    recorded config field -- not a hidden constant.
    """
    n, n_blocks = cfg.n_assets, cfg.n_blocks
    sizes = [len(a) for a in np.array_split(np.arange(n), n_blocks)]
    block = np.repeat(np.arange(n_blocks), sizes)

    sub = np.empty(n, dtype=int)
    sid = 0
    for b in range(n_blocks):
        idx = np.where(block == b)[0]
        k = min(cfg.n_sub, max(1, len(idx) // 2))   # >=2 assets per sub-cluster
        for part in np.array_split(idx, k):
            sub[part] = sid
            sid += 1
    n_sub_total = sid

    def jittered(share: float) -> np.ndarray:
        return share * (1.0 + cfg.loading_jitter * rng.uniform(-1.0, 1.0, size=n))

    disp = 0.0 if not np.isfinite(cfg.block_disp) else cfg.block_disp
    block_mult = rng.uniform(1.0 - disp, 1.0 + disp, size=n_blocks)
    sub_mult = rng.uniform(1.0 - disp, 1.0 + disp, size=n_sub_total)
    g = jittered(cfg.global_share)
    b_ = jittered(cfg.block_share) * block_mult[block]
    s = jittered(cfg.sub_share) * sub_mult[sub]
    total = g + b_ + s
    cap = SAMPLING_RANGES["max_total_share"]
    scale = np.where(total > cap, cap / total, 1.0)
    g, b_, s = g * scale, b_ * scale, s * scale

    a_mat = np.zeros((n, 1 + n_blocks + n_sub_total))
    a_mat[:, 0] = np.sqrt(g)
    a_mat[np.arange(n), 1 + block] = np.sqrt(b_)
    a_mat[np.arange(n), 1 + n_blocks + sub] = np.sqrt(s)
    idio = 1.0 - (g + b_ + s)
    corr = a_mat @ a_mat.T + np.diag(idio)

    perm = rng.permutation(n)   # destroy any informative input ordering
    corr = corr[np.ix_(perm, perm)]
    return corr, {"beta": None, "block": block[perm], "sub": sub[perm]}


def _unstructured_corr(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Random correlation: normalize a Wishart draw ``X'X`` with
    ``dof = max(N+2, round(wishart_dof_ratio * N))`` rows (a.s. full rank)."""
    n = cfg.n_assets
    dof = max(n + 2, int(round(cfg.wishart_dof_ratio * n)))
    x = rng.standard_normal((dof, n))
    s = x.T @ x
    d = np.sqrt(np.diag(s))
    corr = s / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr, {"beta": None, "block": None, "sub": None}


def _equicorr_corr(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n, rho = cfg.n_assets, cfg.equi_rho
    if not (-1.0 / (n - 1) < rho <= 1.0):
        raise ValueError(f"equicorrelation rho={rho} not PSD-valid for N={n}")
    corr = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    return corr, {"beta": None, "block": None, "sub": None}


_BUILDERS = {
    "one_factor": _one_factor_corr,
    "hierarchical": _hierarchical_corr,
    "unstructured": _unstructured_corr,
    "equicorr": _equicorr_corr,
}


def true_correlation(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Population correlation matrix + structure info (block/sub labels, betas)."""
    return _BUILDERS[cfg.family](cfg, rng)


def true_covariance(cfg: CovConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(Sigma, mu, info)`` -- the ground truth for one experiment.

    Volatilities are log-uniform on ``[vol_lo, vol_lo * vol_ratio]``; the true
    mean is ``mu_i = s_i * sigma_i`` with ``s_i ~ N(sharpe_mean, sharpe_disp)``.
    ``info`` records the realized structure for diagnostics and tests.
    """
    corr, info = true_correlation(cfg, rng)
    n = cfg.n_assets
    vols = cfg.vol_lo * cfg.vol_ratio ** rng.uniform(0.0, 1.0, size=n)
    sigma = corr * np.outer(vols, vols)
    s = rng.normal(cfg.sharpe_mean, cfg.sharpe_disp, size=n)
    mu = s * vols
    off = corr[~np.eye(n, dtype=bool)]
    info.update({
        "vols": vols,
        "true_mean_corr": float(off.mean()),
        "true_min_eig": float(np.linalg.eigvalsh(corr).min()),
        "vol_spread": float(vols.max() / vols.min()),
    })
    return sigma, mu, info


# --------------------------------------------------------------------------- #
# return sampling
# --------------------------------------------------------------------------- #
def sample_returns(
    sigma: np.ndarray,
    mu: np.ndarray,
    t_obs: int,
    rng: np.random.Generator,
    *,
    dist: str = "gaussian",
    t_df: float = float("nan"),
) -> np.ndarray:
    """Draw ``(T, N)`` i.i.d. returns with mean ``mu`` and covariance ``sigma``.

    ``student_t`` draws a multivariate t (common chi-square mixing across the
    cross-section) rescaled by ``sqrt((df-2)/df)`` so its covariance is exactly
    ``sigma`` -- same truth, fatter joint tails.
    """
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma + 1e-14 * np.trace(sigma) / n * np.eye(n))
    z = rng.standard_normal((t_obs, n)) @ chol.T
    if dist == "student_t":
        if not (t_df > 2.0):
            raise ValueError("student_t requires t_df > 2")
        w = rng.chisquare(t_df, size=t_obs)
        z = z * np.sqrt((t_df - 2.0) / w)[:, None]
    elif dist != "gaussian":
        raise ValueError(f"unknown dist {dist!r}")
    return mu[None, :] + z


# --------------------------------------------------------------------------- #
# config sampler for Monte-Carlo batches
# --------------------------------------------------------------------------- #
def sample_config(
    rng: np.random.Generator,
    *,
    family: Family | Literal["random"] = "random",
    dist: Literal["gaussian", "student_t"] = "gaussian",
) -> CovConfig:
    """Draw one experiment config; every sampled value lands in a CovConfig field."""
    r = SAMPLING_RANGES
    fam: Family = rng.choice(FAMILIES) if family == "random" else family
    n = int(rng.choice(r["n_assets_choices"]))
    ratio = float(rng.choice(r["tn_ratio_choices"]))
    t_obs = max(5, int(round(ratio * n)))

    common = dict(
        family=fam,
        n_assets=n,
        t_obs=t_obs,
        tn_ratio=ratio,
        vol_lo=float(rng.uniform(*r["vol_lo"])),
        vol_ratio=float(rng.uniform(*r["vol_ratio"])),
        dist=dist,
        t_df=float(rng.uniform(*r["t_df"])) if dist == "student_t" else float("nan"),
        sharpe_mean=float(rng.uniform(*r["sharpe_mean"])),
        sharpe_disp=float(rng.uniform(*r["sharpe_disp"])),
        label=fam,
    )
    if fam == "one_factor":
        return CovConfig(**common,
                         factor_r2=float(rng.uniform(*r["factor_r2"])),
                         factor_r2_disp=float(rng.uniform(*r["factor_r2_disp"])))
    if fam == "hierarchical":
        n_blocks = int(rng.choice([b for b in r["n_blocks_choices"] if b <= n // 4]))
        return CovConfig(**common,
                         n_blocks=n_blocks,
                         n_sub=int(rng.choice(r["n_sub_choices"])),
                         global_share=float(rng.uniform(*r["global_share"])),
                         block_share=float(rng.uniform(*r["block_share"])),
                         sub_share=float(rng.uniform(*r["sub_share"])),
                         loading_jitter=float(rng.uniform(*r["loading_jitter"])),
                         block_disp=float(rng.uniform(*r["block_disp"])))
    if fam == "unstructured":
        return CovConfig(**common,
                         wishart_dof_ratio=float(rng.uniform(*r["wishart_dof_ratio"])))
    return CovConfig(**common, equi_rho=float(rng.uniform(*r["equi_rho"])))


def canonical_configs() -> dict[str, CovConfig]:
    """Fixed illustrative configs for the setup figure and sanity tests."""
    return {
        "one_factor": CovConfig(
            family="one_factor", n_assets=30, t_obs=60, tn_ratio=2.0,
            factor_r2=0.40, factor_r2_disp=0.10, label="one_factor"),
        "hierarchical": CovConfig(
            family="hierarchical", n_assets=30, t_obs=60, tn_ratio=2.0,
            n_blocks=3, n_sub=2, global_share=0.15, block_share=0.30,
            sub_share=0.20, loading_jitter=0.10, block_disp=0.0,
            label="hierarchical"),
        "unstructured": CovConfig(
            family="unstructured", n_assets=30, t_obs=60, tn_ratio=2.0,
            wishart_dof_ratio=1.5, label="unstructured"),
        "equicorr": CovConfig(
            family="equicorr", n_assets=30, t_obs=60, tn_ratio=2.0,
            equi_rho=0.40, label="equicorr"),
    }
