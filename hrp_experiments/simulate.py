"""One experiment = one ground-truth (Sigma, mu) + one finite sample, end to end.

Per experiment:

1. draw the true ``(Sigma, mu)`` from the configured family;
2. draw TWO independent estimation windows of ``T`` observations (the second
   exists only to measure estimation-noise turnover: how much each method's
   weights move when the data, but not the truth, changes);
3. estimate (sample covariance + Ledoit-Wolf), allocate with every method;
4. score every weight vector on the TRUTH: realized variance ``w' Sigma w``
   relative to the oracle minimum-variance floor, true Sharpe ``w'mu /
   sqrt(w'Sigma w)``, concentration (Herfindahl of gross weights), turnover;
5. record the noisy-mu Markowitz track (``Sigma_hat^+ mu_hat`` direction)
   separately from the mu-free risk-only track.

The oracle is the unconstrained minimum-variance portfolio on the true Sigma --
the exact lower bound on realized variance over all full-investment weight
vectors, so every ``risk_ratio_* >= 1`` by construction.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .allocators import (
    clip_long_only,
    equal_weight,
    herfindahl,
    hrp_weights,
    hrp_weights_dod,
    inverse_variance,
    ledoit_wolf_cov,
    min_variance,
    short_gross,
    tangency_direction,
    true_sharpe,
    true_variance,
    turnover,
)
from .model import CovConfig, sample_config, sample_returns, true_covariance

#: mu-free methods scored on the risk track (and, with the true mu, on Sharpe);
#: hrp_dod is single-linkage HRP under the distance-of-distances convention of
#: the original 2016 code listing (see allocators.hrp_weights_dod)
RISK_METHODS: tuple[str, ...] = (
    "ew", "iv", "mv", "mv_lo", "mv_lw", "mv_lw_lo",
    "hrp_single", "hrp_average", "hrp_ward", "hrp_dod",
)
#: methods that use the estimated mean (recorded separately, Sharpe track only)
MU_METHODS: tuple[str, ...] = ("mv_mu", "mv_mu_lw")


def allocate_all(returns: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Run every mu-free allocator on one estimation window.

    Returns ``(weights_by_method, extras)`` where extras carries the estimated
    covariances (for the noisy-mu track) and the Ledoit-Wolf intensity.
    """
    n = returns.shape[1]
    sample_cov = np.cov(returns, rowvar=False, ddof=1)
    lw_cov, lw_delta = ledoit_wolf_cov(returns)

    w_mv = min_variance(sample_cov)
    w_mv_lw = min_variance(lw_cov)
    weights = {
        "ew": equal_weight(n),
        "iv": inverse_variance(sample_cov),
        "mv": w_mv,
        "mv_lo": clip_long_only(w_mv),
        "mv_lw": w_mv_lw,
        "mv_lw_lo": clip_long_only(w_mv_lw),
        "hrp_single": hrp_weights(sample_cov, "single"),
        "hrp_average": hrp_weights(sample_cov, "average"),
        "hrp_ward": hrp_weights(sample_cov, "ward"),
        "hrp_dod": hrp_weights_dod(sample_cov, "single"),
    }
    extras = {"sample_cov": sample_cov, "lw_cov": lw_cov, "lw_shrinkage": lw_delta}
    return weights, extras


def run_experiment(cfg: CovConfig, rng: np.random.Generator) -> dict:
    """Simulate one experiment and return a flat, JSON/CSV-able record."""
    sigma, mu, info = true_covariance(cfg, rng)
    kw = {"dist": cfg.dist, "t_df": cfg.t_df}
    r1 = sample_returns(sigma, mu, cfg.t_obs, rng, **kw)
    r2 = sample_returns(sigma, mu, cfg.t_obs, rng, **kw)   # turnover window only

    w1, extras = allocate_all(r1)
    w2, _ = allocate_all(r2)

    # oracle floor: exact minimum variance on the TRUE covariance
    w_star = min_variance(sigma)
    var_star = true_variance(w_star, sigma)
    sr_tan = float(np.sqrt(max(mu @ np.linalg.solve(sigma, mu), 0.0)))  # oracle tangency Sharpe

    record: dict = {
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        "true_mean_corr": info["true_mean_corr"],
        "true_min_eig": info["true_min_eig"],
        "vol_spread": info["vol_spread"],
        "lw_shrinkage": extras["lw_shrinkage"],
        "rank_deficient": int(cfg.t_obs - 1 < cfg.n_assets),
        "oracle_var": var_star,
        "sr_oracle_tangency": sr_tan,
        "sr_true_oracle_minvar": true_sharpe(w_star, mu, sigma),
    }

    for m in RISK_METHODS:
        w = w1[m]
        v = true_variance(w, sigma)
        sr = true_sharpe(w, mu, sigma)
        record[f"risk_ratio_{m}"] = v / var_star
        record[f"sr_true_{m}"] = sr
        record[f"sr_ratio_{m}"] = sr / sr_tan if sr_tan > 1e-10 else float("nan")
        record[f"hhi_{m}"] = herfindahl(w)
        record[f"turnover_{m}"] = turnover(w, w2[m])
        if m in ("mv", "mv_lw"):
            record[f"short_gross_{m}"] = short_gross(w)

    # noisy-mu Markowitz track: direction Sigma_hat^+ mu_hat, scale-free Sharpe
    mu_hat = r1.mean(axis=0)
    for name, cov_est in (("mv_mu", extras["sample_cov"]), ("mv_mu_lw", extras["lw_cov"])):
        w_dir = tangency_direction(cov_est, mu_hat)
        v = true_variance(w_dir, sigma)
        sr = float(w_dir @ mu / np.sqrt(v)) if v > 1e-30 else float("nan")
        record[f"sr_true_{name}"] = sr
        record[f"sr_ratio_{name}"] = sr / sr_tan if sr_tan > 1e-10 else float("nan")
    return record


def run_batch(
    n_experiments: int,
    *,
    family: str = "random",
    dist: str = "gaussian",
    seed: int = 0,
    progress_every: int = 0,
) -> list[dict]:
    """Run ``n_experiments`` with independently-spawned child RNGs (reproducible:
    a single SeedSequence drives config sampling, truth and returns)."""
    ss = np.random.SeedSequence(seed)
    records = []
    for i, child in enumerate(ss.spawn(n_experiments)):
        rng = np.random.default_rng(child)
        cfg = sample_config(rng, family=family, dist=dist)  # type: ignore[arg-type]
        records.append(run_experiment(cfg, rng))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_experiments}", flush=True)
    return records


def mv_rcond_sweep(
    n_experiments: int,
    *,
    seed: int,
    tn_ratio: float = 1.0,
    rconds: tuple[float, ...] = (1e-15, 1e-10, 1e-6, 1e-3),
    family: str = "random",
    dist: str = "gaussian",
) -> dict:
    """Counterfactual sweep of the pseudoinverse cutoff behind sample MV.

    ``min_variance`` uses ``np.linalg.pinv`` at NumPy's default relative cutoff
    (``rcond = 1e-15``); this replays the exact config/truth/window draws of
    ``run_batch(n_experiments, seed=seed)`` (same SeedSequence spawning -- each
    experiment has its own child RNG, so skipping the ones at other T/N ratios
    changes nothing) and recomputes ONLY the sample-MV risk ratio at the
    requested cutoffs, for the experiments at ``tn_ratio``. Stored in
    results.json so the paper's rcond-robustness sentence is a checked number.
    """
    ss = np.random.SeedSequence(seed)
    ratios: dict[str, list[float]] = {f"{rc:g}": [] for rc in rconds}
    for child in ss.spawn(n_experiments):
        rng = np.random.default_rng(child)
        cfg = sample_config(rng, family=family, dist=dist)  # type: ignore[arg-type]
        if cfg.tn_ratio != tn_ratio:
            continue
        sigma, mu, _ = true_covariance(cfg, rng)
        r1 = sample_returns(sigma, mu, cfg.t_obs, rng, dist=cfg.dist, t_df=cfg.t_df)
        sample_cov = np.cov(r1, rowvar=False, ddof=1)
        var_star = true_variance(min_variance(sigma), sigma)
        ones = np.ones(cfg.n_assets)
        for rc in rconds:
            w = np.linalg.pinv(sample_cov, rcond=rc, hermitian=True) @ ones
            w = w / w.sum()
            ratios[f"{rc:g}"].append(true_variance(w, sigma) / var_star)
    return {
        "tn_ratio": tn_ratio,
        "n": len(next(iter(ratios.values()))),
        "median_risk_ratio_mv_by_rcond": {
            k: float(np.median(v)) for k, v in ratios.items()},
    }
