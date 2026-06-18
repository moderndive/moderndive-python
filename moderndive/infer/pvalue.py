"""p-values from a simulated null distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .core import Distribution

# Direction aliases accepted by infer.
_RIGHT = {"right", "greater"}
_LEFT = {"left", "less"}
_BOTH = {"two-sided", "two_sided", "both", "two sided"}


def _as_float(obs_stat) -> float:
    return float(obs_stat)


def get_p_value(
    distribution: Distribution,
    obs_stat,
    direction: str,
) -> pl.DataFrame:
    """Return a one-row frame with the simulation-based ``p_value``.

    ``direction`` is one of ``right``/``greater``, ``left``/``less``, or
    ``two-sided``. The two-sided p-value uses infer's convention: twice the
    smaller one-sided tail proportion, capped at 1.
    """
    stats = distribution.stats
    obs = _as_float(obs_stat)
    direction = direction.lower()

    if direction in _RIGHT:
        p = float(np.mean(stats >= obs))
    elif direction in _LEFT:
        p = float(np.mean(stats <= obs))
    elif direction in _BOTH:
        p_right = float(np.mean(stats >= obs))
        p_left = float(np.mean(stats <= obs))
        p = min(1.0, 2.0 * min(p_right, p_left))
    else:
        raise ValueError("direction must be 'right'/'greater', 'left'/'less', or 'two-sided'")
    return pl.DataFrame({"p_value": [p]})


def get_fit_p_value(fit, obs_stat, direction: str = "two-sided") -> pl.DataFrame:
    """Per-term simulation p-values: compare a null fit distribution to observed.

    ``obs_stat`` is the observed :class:`FitResult` (one estimate per term).
    """
    observed = {row["term"]: row["estimate"] for row in obs_stat.data.iter_rows(named=True)}
    direction = direction.lower()
    out_terms, out_p = [], []
    for term, sub in fit.data.group_by("term"):
        term = term[0] if isinstance(term, tuple) else term
        null_vals = sub["estimate"].to_numpy()
        obs = float(observed[term])
        if direction in _RIGHT:
            p = float(np.mean(null_vals >= obs))
        elif direction in _LEFT:
            p = float(np.mean(null_vals <= obs))
        else:
            center = float(np.mean(null_vals))
            p = float(np.mean(np.abs(null_vals - center) >= abs(obs - center)))
        out_terms.append(term)
        out_p.append(p)
    return pl.DataFrame({"term": out_terms, "p_value": out_p}).sort("term")
