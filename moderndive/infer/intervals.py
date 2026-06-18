"""Confidence intervals from a simulated (bootstrap) distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .core import Distribution


def get_confidence_interval(
    distribution: Distribution,
    level: float = 0.95,
    type: str = "percentile",
    *,
    point_estimate: float | None = None,
) -> pl.DataFrame:
    """Return a one-row frame with ``lower_ci`` and ``upper_ci``.

    - ``type="percentile"``: the ``(1-level)/2`` and ``1-(1-level)/2`` quantiles
      of the bootstrap distribution.
    - ``type="se"``: ``point_estimate ± z* · SE`` where SE is the SD of the
      bootstrap distribution (requires ``point_estimate``).
    """
    stats = distribution.stats
    if type == "percentile":
        alpha = 1.0 - level
        lower = float(np.quantile(stats, alpha / 2))
        upper = float(np.quantile(stats, 1 - alpha / 2))
    elif type in ("se", "standard error"):
        if point_estimate is None:
            raise ValueError("type='se' requires `point_estimate`")
        from scipy.stats import norm

        z = float(norm.ppf(1 - (1 - level) / 2))
        se = float(np.std(stats, ddof=1))
        lower = float(point_estimate) - z * se
        upper = float(point_estimate) + z * se
    elif type in ("bias-corrected", "bias_corrected", "bc"):
        if point_estimate is None:
            raise ValueError("type='bias-corrected' requires `point_estimate`")
        from scipy.stats import norm

        alpha = 1.0 - level
        # Bias-correction factor from the fraction of replicates below the estimate.
        prop_less = float(np.mean(stats < float(point_estimate)))
        prop_less = min(max(prop_less, 1e-9), 1 - 1e-9)
        z0 = float(norm.ppf(prop_less))
        lo_q = float(norm.cdf(2 * z0 + norm.ppf(alpha / 2)))
        hi_q = float(norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2)))
        lower = float(np.quantile(stats, lo_q))
        upper = float(np.quantile(stats, hi_q))
    else:
        raise ValueError("type must be 'percentile', 'se', or 'bias-corrected'")
    return pl.DataFrame({"lower_ci": [lower], "upper_ci": [upper]})


def get_fit_confidence_interval(fit, level: float = 0.95) -> pl.DataFrame:
    """Per-term percentile confidence intervals from a bootstrap fit distribution."""
    alpha = 1.0 - level
    return (
        fit.data.group_by("term")
        .agg(
            lower_ci=pl.col("estimate").quantile(alpha / 2),
            upper_ci=pl.col("estimate").quantile(1 - alpha / 2),
        )
        .sort("term")
    )
