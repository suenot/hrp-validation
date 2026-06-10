"""Portfolio allocators under comparison.

All allocators map an *estimated* covariance (or raw returns) to full-investment
weights ``sum(w) = 1``; they are scored on the TRUE covariance elsewhere.

* ``equal_weight``        -- 1/N (DeMiguel, Garlappi & Uppal 2009 benchmark).
* ``inverse_variance``    -- w_i proportional to 1/sigma_i^2 (diagonal-only; the
                             degenerate HRP with no clustering information).
* ``min_variance``        -- closed-form sample minimum variance
                             ``w = S^+ 1 / (1' S^+ 1)``, shorts allowed. With
                             ``T <= N`` the sample covariance is singular; we use
                             the Moore-Penrose pseudoinverse (minimum-norm
                             solution) and flag the record as rank-deficient.
* ``clip_long_only``      -- the practitioner heuristic: clip negative weights
                             to zero and renormalize (NOT the long-only QP; this
                             is deliberate -- it is what naive implementations do).
* ``ledoit_wolf_cov``     -- Ledoit & Wolf (2004) shrinkage toward the scaled
                             identity ``mu I`` via scikit-learn (analytic
                             optimal intensity); fed to ``min_variance``.
* ``hrp_weights``         -- Hierarchical Risk Parity per Lopez de Prado (2016)
                             with the linkage computed on the CONDENSED
                             correlation distance ``d`` (the convention used by
                             many library implementations); see the
                             step-by-step docstring. Linkage is ``single`` in
                             the original; ``average`` / ``ward`` are exposed
                             as sensitivity variants.
* ``hrp_weights_dod``     -- the same pipeline with the linkage computed on the
                             Euclidean distance-of-distances ``d_tilde`` (the
                             square matrix Lopez de Prado's 2016 code listing
                             passes to ``scipy.cluster.hierarchy.linkage``);
                             stored as the faithfulness robustness variant.
* ``tangency_direction``  -- mean-variance direction ``S^+ mu_hat`` for the
                             noisy-mu Markowitz track (Sharpe is scale-invariant,
                             so the direction is evaluated unnormalized; a wrong
                             sign honestly shows up as negative true Sharpe).
"""

from __future__ import annotations

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import ledoit_wolf as _sk_ledoit_wolf

# --------------------------------------------------------------------------- #
# simple allocators
# --------------------------------------------------------------------------- #
def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_variance(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.clip(np.diag(cov), 1e-18, None)
    return ivp / ivp.sum()


def min_variance(cov: np.ndarray) -> np.ndarray:
    """Unconstrained minimum-variance: ``w = Sigma^+ 1 / (1' Sigma^+ 1)``.

    Uses the (Hermitian) pseudoinverse at NumPy's default cutoff
    (``rcond = 1e-15`` relative to the largest singular value) so rank-deficient
    sample covariances (T <= N) yield the minimum-norm solution instead of an
    error. Raises ``ValueError`` in the measure-zero event ``1' Sigma^+ 1 ~ 0``
    (never observed in the experiments) rather than silently substituting
    another portfolio.
    """
    n = cov.shape[0]
    ones = np.ones(n)
    w = np.linalg.pinv(cov, hermitian=True) @ ones
    denom = float(w.sum())
    if abs(denom) < 1e-12:
        raise ValueError(
            f"min_variance is undefined: |1' Sigma^+ 1| = {abs(denom):.3e} < 1e-12")
    return w / denom


def clip_long_only(w: np.ndarray) -> np.ndarray:
    """Clip shorts to zero and renormalize (crude long-only projection).

    Since ``sum(w) = 1`` at least one weight is positive, so the denominator is
    strictly positive.
    """
    wc = np.clip(w, 0.0, None)
    return wc / wc.sum()


def ledoit_wolf_cov(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance ``(1-d) S_mle + d (tr(S_mle)/N) I``.

    Returns ``(shrunk_cov, shrinkage_intensity d)``. scikit-learn computes the
    analytic optimal intensity; ``S_mle`` is the MLE (ddof=0) covariance of the
    demeaned returns -- the constant scaling vs ddof=1 does not affect
    min-variance weights.
    """
    cov, delta = _sk_ledoit_wolf(returns, assume_centered=False)
    return cov, float(delta)


def tangency_direction(cov: np.ndarray, mu_hat: np.ndarray) -> np.ndarray:
    """Markowitz mean-variance direction ``Sigma^+ mu_hat`` (unnormalized)."""
    return np.linalg.pinv(cov, hermitian=True) @ mu_hat


# --------------------------------------------------------------------------- #
# Hierarchical Risk Parity -- Lopez de Prado (2016), both distance conventions
# --------------------------------------------------------------------------- #
def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    return np.clip(corr, -1.0, 1.0)


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Step 1 -- distance ``d_ij = sqrt((1 - rho_ij) / 2)`` in [0, 1]."""
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def hrp_linkage(corr: np.ndarray, method: str = "single") -> np.ndarray:
    """Step 2 -- agglomerative clustering on the CONDENSED correlation distance.

    The linkage is computed directly on the pairwise distances ``d_ij`` (passed
    to scipy in condensed form). This is the convention used by many library
    implementations of HRP, but it is NOT the one in Lopez de Prado's (2016)
    code listing, which passes the *square* matrix of ``d`` to
    ``scipy.cluster.hierarchy.linkage``; scipy then treats its rows as
    observation vectors and clusters on the Euclidean distance-of-distances
    ``d_tilde_ij = ||d_.i - d_.j||_2`` (see ``hrp_linkage_dod``). Lopez de
    Prado (2016) prescribes ``single`` linkage; ``average`` and ``ward`` are
    sensitivity variants (note ward is formally defined for Euclidean distances
    only -- scipy still computes it, and we report it as a variant, not as HRP).
    """
    dist = correlation_distance(corr)
    return sch.linkage(squareform(dist, checks=False), method=method)


def hrp_linkage_dod(corr: np.ndarray, method: str = "single") -> np.ndarray:
    """Step 2' -- clustering on the Euclidean distance-of-distances ``d_tilde``.

    Reproduces the linkage of Lopez de Prado's (2016) code listing, which hands
    the square correlation-distance matrix ``d`` to scipy's ``linkage``: scipy
    treats each row ``d_.i`` as an observation vector and clusters on
    ``d_tilde_ij = ||d_.i - d_.j||_2``. For 3 assets the resulting tree always
    coincides with the condensed-``d`` convention; from 4+ assets the two can
    (and sometimes do) differ.
    """
    dist = correlation_distance(corr)
    return sch.linkage(pdist(dist, metric="euclidean"), method=method)


def quasi_diag_order(link: np.ndarray) -> np.ndarray:
    """Step 3 -- quasi-diagonalization: the dendrogram leaf order, which places
    mutually correlated assets adjacently (equivalent to getQuasiDiag in the
    original paper; scipy's ``leaves_list`` returns exactly this order)."""
    return np.asarray(sch.leaves_list(link), dtype=int)


def cluster_variance(cov: np.ndarray, idx: np.ndarray | list[int]) -> float:
    """Variance of a cluster under its inverse-variance-weighted sub-portfolio
    (getClusterVar in the original): ``w = ivp / sum(ivp)``, ``var = w' V w``."""
    sub = cov[np.ix_(idx, idx)]
    w = inverse_variance(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Step 4 -- top-down recursive bisection (getRecBipart in the original).

    The *ordered* asset list is split into contiguous halves (first
    ``len(cluster) // 2`` items vs the rest -- the 2016 paper bisects the list,
    not the dendrogram); each half gets capital inversely proportional to its
    cluster variance: ``alpha = 1 - varL / (varL + varR)`` for the left half.
    Weights are positive and sum to 1 by construction.
    """
    n = cov.shape[0]
    weights = np.ones(n)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        nxt: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_l = cluster_variance(cov, left)
            var_r = cluster_variance(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return weights / weights.sum()


def hrp_weights(cov: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """Full HRP (condensed-``d`` linkage convention): correlation -> distance ->
    linkage -> leaf order -> bisection."""
    corr = corr_from_cov(cov)
    link = hrp_linkage(corr, method=linkage_method)
    order = quasi_diag_order(link)
    return recursive_bisection(cov, order)


def hrp_weights_dod(cov: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """Full HRP with the distance-of-distances (``d_tilde``) linkage convention
    of Lopez de Prado's (2016) code listing; bisection identical to
    ``hrp_weights``."""
    corr = corr_from_cov(cov)
    link = hrp_linkage_dod(corr, method=linkage_method)
    order = quasi_diag_order(link)
    return recursive_bisection(cov, order)


# --------------------------------------------------------------------------- #
# evaluation helpers (all on the TRUE covariance)
# --------------------------------------------------------------------------- #
def true_variance(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(w @ sigma @ w)


def true_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    v = true_variance(w, sigma)
    return float(w @ mu / np.sqrt(v)) if v > 0 else float("nan")


def herfindahl(w: np.ndarray) -> float:
    """Concentration of *gross* exposure: HHI of ``|w| / sum|w|`` (1/N for equal
    weight, 1 for a single position; well-defined for short-selling portfolios)."""
    a = np.abs(w)
    a = a / a.sum()
    return float(np.sum(a**2))


def turnover(w1: np.ndarray, w2: np.ndarray) -> float:
    """Half L1 distance between two weight vectors (one-way turnover)."""
    return float(0.5 * np.abs(w1 - w2).sum())


def short_gross(w: np.ndarray) -> float:
    return float(-w[w < 0].sum())
