"""Sanity tests for the HRP-vs-Markowitz experiment harness.

Run: python -m pytest -q   (from the project root)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch

from hrp_experiments import (
    FAMILIES,
    CovConfig,
    canonical_configs,
    clip_long_only,
    corr_from_cov,
    correlation_distance,
    equal_weight,
    hrp_linkage_dod,
    hrp_weights,
    hrp_weights_dod,
    inverse_variance,
    ledoit_wolf_cov,
    min_variance,
    run_experiment,
    sample_config,
    sample_returns,
    true_covariance,
)
from hrp_experiments.simulate import RISK_METHODS


def _random_pd_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4 * n, n))
    return (x.T @ x) / (4 * n) + 0.05 * np.eye(n)


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def test_generator_psd_all_families():
    """Every family yields a symmetric PSD covariance with the configured vols."""
    for i, fam in enumerate(FAMILIES):
        rng = np.random.default_rng(100 + i)
        cfg = sample_config(rng, family=fam)
        sigma, mu, info = true_covariance(cfg, rng)
        assert sigma.shape == (cfg.n_assets, cfg.n_assets)
        assert np.allclose(sigma, sigma.T)
        assert np.linalg.eigvalsh(sigma).min() > -1e-12
        vols = np.sqrt(np.diag(sigma))
        assert vols.min() >= cfg.vol_lo * 0.999
        assert vols.max() <= cfg.vol_lo * cfg.vol_ratio * 1.001
        assert mu.shape == (cfg.n_assets,)


def test_hierarchical_structure_is_nested():
    """Mean correlation: within sub-block > within block (across subs) > across
    blocks -- the intended nested hierarchy."""
    cfg = canonical_configs()["hierarchical"]
    rng = np.random.default_rng(7)
    sigma, _, info = true_covariance(cfg, rng)
    d = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(d, d)
    block, sub = info["block"], info["sub"]
    n = cfg.n_assets
    iu = np.triu_indices(n, k=1)
    same_block = block[iu[0]] == block[iu[1]]
    same_sub = sub[iu[0]] == sub[iu[1]]
    c = corr[iu]
    within_sub = c[same_sub].mean()
    within_block = c[same_block & ~same_sub].mean()
    across = c[~same_block].mean()
    assert within_sub > within_block + 0.05
    assert within_block > across + 0.05
    # and the levels match the configured variance shares
    assert across == pytest.approx(cfg.global_share, abs=0.05)
    assert within_sub == pytest.approx(
        cfg.global_share + cfg.block_share + cfg.sub_share, abs=0.07)


def test_one_factor_and_equicorr_structure():
    """One-factor off-diagonals equal beta_i*beta_j; equicorr off-diagonals == rho."""
    rng = np.random.default_rng(11)
    cfg = canonical_configs()["one_factor"]
    sigma, _, info = true_covariance(cfg, rng)
    d = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(d, d)
    beta = info["beta"]
    expected = np.outer(beta, beta)
    off = ~np.eye(cfg.n_assets, dtype=bool)
    assert np.allclose(corr[off], expected[off], atol=1e-10)

    cfg_e = canonical_configs()["equicorr"]
    sigma_e, _, _ = true_covariance(cfg_e, np.random.default_rng(12))
    d = np.sqrt(np.diag(sigma_e))
    corr_e = sigma_e / np.outer(d, d)
    off = ~np.eye(cfg_e.n_assets, dtype=bool)
    assert np.allclose(corr_e[off], cfg_e.equi_rho, atol=1e-10)


def test_sample_returns_match_truth():
    """Empirical covariance of a long Gaussian AND Student-t sample ~ Sigma."""
    cfg = CovConfig(family="equicorr", n_assets=5, t_obs=60, equi_rho=0.4,
                    vol_lo=0.01, vol_ratio=2.0)
    rng = np.random.default_rng(3)
    sigma, mu, _ = true_covariance(cfg, rng)
    for dist, df in (("gaussian", float("nan")), ("student_t", 6.0)):
        r = sample_returns(sigma, mu, 200_000, rng, dist=dist, t_df=df)
        emp = np.cov(r, rowvar=False)
        assert np.allclose(emp, sigma, rtol=0.12, atol=1e-6), dist
        assert np.allclose(r.mean(axis=0), mu, atol=5 * sigma.max() ** 0.5 / 400)


# --------------------------------------------------------------------------- #
# allocators
# --------------------------------------------------------------------------- #
def test_min_variance_matches_closed_form():
    """w = Sigma^{-1} 1 / (1' Sigma^{-1} 1), and it is the variance minimizer."""
    cov = _random_pd_cov(12, seed=5)
    w = min_variance(cov)
    ones = np.ones(12)
    expected = np.linalg.solve(cov, ones)
    expected /= expected.sum()
    assert np.allclose(w, expected, atol=1e-10)
    assert w.sum() == pytest.approx(1.0)
    for other in (equal_weight(12), inverse_variance(cov)):
        assert w @ cov @ w <= other @ cov @ other + 1e-15


def test_clip_long_only_properties():
    cov = _random_pd_cov(10, seed=6)
    w = clip_long_only(min_variance(cov))
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_ledoit_wolf_between_sample_and_target():
    """Shrunk covariance == (1-d)*S_mle + d*(tr(S_mle)/N)*I with d in [0, 1]."""
    rng = np.random.default_rng(8)
    r = rng.standard_normal((40, 15)) * 0.02
    lw, delta = ledoit_wolf_cov(r)
    assert 0.0 <= delta <= 1.0
    s_mle = np.cov(r, rowvar=False, ddof=0)
    target = np.trace(s_mle) / 15 * np.eye(15)
    assert np.allclose(lw, (1 - delta) * s_mle + delta * target, atol=1e-12)


def test_hrp_weights_positive_sum_one_all_linkages():
    for seed in (1, 2, 3):
        cov = _random_pd_cov(20, seed=seed)
        for method in ("single", "average", "ward"):
            w = hrp_weights(cov, method)
            assert w.shape == (20,)
            assert (w > 0).all()
            assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_hrp_matches_hand_computed_three_asset_case():
    """3 assets: 1&2 strongly correlated (rho=0.9), asset 3 weak (rho=0.1).

    Single linkage merges {0,1} first, leaf order = [2, 0, 1]; bisection splits
    [2] vs [0,1]. By hand (IVP cluster variances):
      var({2})   = 0.0225
      var({0,1}) = 0.01376   (ivp weights (0.8, 0.2))
      alpha_left = 1 - 0.0225/(0.0225+0.01376)  -> w2 = 0.37948...
      inner split: alpha = 1 - 0.01/(0.01+0.04) = 0.8 -> w0, w1 = 0.8/0.2 of rest
    """
    s1, s2, s3 = 0.1, 0.2, 0.15
    rho12, rho_w = 0.9, 0.1
    cov = np.array([
        [s1 * s1, rho12 * s1 * s2, rho_w * s1 * s3],
        [rho12 * s1 * s2, s2 * s2, rho_w * s2 * s3],
        [rho_w * s1 * s3, rho_w * s2 * s3, s3 * s3],
    ])
    w = hrp_weights(cov, "single")
    var_pair = 0.8**2 * 0.01 + 2 * 0.8 * 0.2 * 0.018 + 0.2**2 * 0.04  # 0.01376
    w2 = var_pair / (0.0225 + var_pair)
    expected = np.array([(1 - w2) * 0.8, (1 - w2) * 0.2, w2])
    assert np.allclose(w, expected, atol=1e-12)


def test_hrp_dod_reproduces_original_square_matrix_linkage():
    """``hrp_linkage_dod`` must equal what Lopez de Prado's (2016) code listing
    computes: ``sch.linkage`` fed the SQUARE correlation-distance matrix, which
    scipy converts to Euclidean distances between its rows (the
    distance-of-distances d-tilde)."""
    for seed in (0, 1, 2):
        corr = corr_from_cov(_random_pd_cov(8, seed=seed))
        dist = correlation_distance(corr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scipy warns on square-matrix input
            link_original = sch.linkage(dist, method="single")
        assert np.allclose(hrp_linkage_dod(corr, "single"), link_original)


def test_hrp_distance_conventions_differ_beyond_three_assets():
    """Condensed-d and distance-of-distances linkage are genuinely different
    conventions: with 3 assets the trees always coincide, but from >= 5 assets
    they can produce different leaf orders and therefore different weights."""
    # 3 assets: identical for many random covariances
    for seed in range(20):
        cov3 = _random_pd_cov(3, seed=seed)
        assert np.allclose(hrp_weights(cov3, "single"), hrp_weights_dod(cov3, "single"))
    # 5 assets, fixed covariance: the conventions disagree
    cov5 = _random_pd_cov(5, seed=1)
    w_d = hrp_weights(cov5, "single")
    w_dod = hrp_weights_dod(cov5, "single")
    for w in (w_d, w_dod):
        assert (w > 0).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(w_d - w_dod).max() > 1e-3   # materially different weights


def test_min_variance_raises_when_undefined():
    """``1' Sigma^+ 1 = 0`` (e.g. the centering projector) must raise, not be
    silently replaced by another portfolio."""
    n = 4
    cov = np.eye(n) - np.ones((n, n)) / n   # PSD, null space spanned by 1
    with pytest.raises(ValueError, match="min_variance is undefined"):
        min_variance(cov)


# --------------------------------------------------------------------------- #
# experiment pipeline
# --------------------------------------------------------------------------- #
def test_oracle_is_a_floor_for_every_method():
    """risk_ratio_m = w'Sigma w / oracle >= 1 for every method and family."""
    for i, fam in enumerate(FAMILIES):
        rng = np.random.default_rng(50 + i)
        cfg = sample_config(rng, family=fam)
        rec = run_experiment(cfg, rng)
        for m in RISK_METHODS:
            assert rec[f"risk_ratio_{m}"] >= 1.0 - 1e-9, (fam, m)
            assert np.isfinite(rec[f"risk_ratio_{m}"])


def _records_equal(a: dict, b: dict) -> bool:
    """Dict equality that treats NaN == NaN (unused config fields are NaN)."""
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb):
            continue
        if va != vb:
            return False
    return True


def test_run_experiment_is_deterministic():
    cfg = canonical_configs()["hierarchical"]
    a = run_experiment(cfg, np.random.default_rng(123))
    b = run_experiment(cfg, np.random.default_rng(123))
    assert _records_equal(a, b)


def test_run_batch_is_deterministic():
    """Same seed => bit-identical records, including config sampling."""
    from hrp_experiments import run_batch

    a = run_batch(6, seed=2024)
    b = run_batch(6, seed=2024)
    assert all(_records_equal(x, y) for x, y in zip(a, b))


def test_sample_config_records_t_over_n():
    """T = round(ratio * N) (>= 5) and every family parameter lands in the config."""
    rng = np.random.default_rng(99)
    for _ in range(50):
        cfg = sample_config(rng)
        assert cfg.t_obs == max(5, int(round(cfg.tn_ratio * cfg.n_assets)))
        assert cfg.family in FAMILIES
        if cfg.family == "hierarchical":
            assert cfg.n_blocks >= 2 and np.isfinite(cfg.global_share)
            assert 0.0 <= cfg.block_disp <= 0.8
        if cfg.family == "equicorr":
            assert 0.0 < cfg.equi_rho <= 0.7
        if cfg.family == "one_factor":
            assert np.isfinite(cfg.factor_r2)
        if cfg.family == "unstructured":
            assert np.isfinite(cfg.wishart_dof_ratio)


def test_record_has_expected_keys_and_finite_metrics():
    rng = np.random.default_rng(0)
    cfg = sample_config(rng, family="hierarchical")
    rec = run_experiment(cfg, rng)
    for m in RISK_METHODS:
        for prefix in ("risk_ratio", "sr_true", "hhi", "turnover"):
            key = f"{prefix}_{m}"
            assert key in rec and np.isfinite(rec[key]), key
    for key in ("sr_true_mv_mu", "sr_true_mv_mu_lw", "sr_oracle_tangency",
                "oracle_var", "cfg_family", "cfg_tn_ratio", "lw_shrinkage",
                "true_mean_corr", "rank_deficient"):
        assert key in rec
