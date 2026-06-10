"""Controlled comparison of HRP vs Markowitz-family allocators under known covariance."""

from .allocators import (
    clip_long_only,
    cluster_variance,
    corr_from_cov,
    correlation_distance,
    equal_weight,
    herfindahl,
    hrp_linkage,
    hrp_linkage_dod,
    hrp_weights,
    hrp_weights_dod,
    inverse_variance,
    ledoit_wolf_cov,
    min_variance,
    quasi_diag_order,
    recursive_bisection,
    tangency_direction,
    true_sharpe,
    true_variance,
    turnover,
)
from .model import (
    FAMILIES,
    SAMPLING_RANGES,
    CovConfig,
    canonical_configs,
    sample_config,
    sample_returns,
    true_correlation,
    true_covariance,
)
from .simulate import MU_METHODS, RISK_METHODS, allocate_all, run_batch, run_experiment

__all__ = [
    "CovConfig", "FAMILIES", "SAMPLING_RANGES",
    "canonical_configs", "sample_config", "sample_returns",
    "true_correlation", "true_covariance",
    "equal_weight", "inverse_variance", "min_variance", "clip_long_only",
    "ledoit_wolf_cov", "tangency_direction",
    "corr_from_cov", "correlation_distance", "hrp_linkage", "hrp_linkage_dod",
    "quasi_diag_order",
    "cluster_variance", "recursive_bisection", "hrp_weights", "hrp_weights_dod",
    "true_variance", "true_sharpe", "herfindahl", "turnover",
    "RISK_METHODS", "MU_METHODS", "allocate_all", "run_experiment", "run_batch",
]
__version__ = "0.1.0"
