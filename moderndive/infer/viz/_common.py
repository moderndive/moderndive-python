"""Engine-agnostic helpers shared by the plotly and plotnine viz backends.

No plotly/plotnine imports live here — only data preparation and the geometry of
p-value / confidence-interval shading, so both backends compute identical regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..core import Distribution

# Palette shared across backends.
_OBS_COLOR = "#d62728"  # observed statistic / p-value shading (red)
_SHADE_COLOR = "#1f77b4"  # confidence-interval shading (blue)
_HIST_FILL = "#7f7f7f"  # histogram bars (gray)
_INF = float("inf")

_ENGINES = ("plotly", "plotnine")
_METHODS = ("simulation", "theoretical", "both")
_RIGHT = {"right", "greater"}
_LEFT = {"left", "less"}
_BOTH = {"two-sided", "two_sided", "both", "two sided"}


def resolve_engine(engine: str) -> str:
    """Validate the plotting engine, returning it unchanged."""
    if engine not in _ENGINES:
        raise ValueError(f"engine must be one of {_ENGINES}, got {engine!r}")
    return engine


def resolve_method(method: str) -> str:
    """Validate the visualization method, returning it unchanged."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    return method


def stat_label(stat: str | None) -> str:
    """Axis label for the statistic (falls back to ``"stat"``)."""
    return "stat" if stat is None else stat


def dist_title(null) -> str:
    """Plot title reflecting whether the distribution is a null or bootstrap one."""
    return (
        "Simulation-Based Null Distribution"
        if null is not None
        else "Simulation-Based Bootstrap Distribution"
    )


def stat_values(distribution: Distribution) -> np.ndarray:
    """The simulated statistics as a numpy array."""
    return distribution.data["stat"].to_numpy()


def ci_endpoints(endpoints) -> tuple[float, float]:
    """Resolve ``(lower, upper)`` from a CI DataFrame or a 2-tuple."""
    if isinstance(endpoints, pl.DataFrame):
        return float(endpoints["lower_ci"][0]), float(endpoints["upper_ci"][0])
    lower, upper = (float(x) for x in endpoints)
    return lower, upper


def pvalue_regions(obs_stat, direction: str) -> tuple[list[tuple[float, bool]], list[tuple]]:
    """Geometry for p-value shading, computed once for both backends.

    Returns ``(vlines, rects)`` where each vline is ``(x, dashed)`` and each rect
    is ``(xmin, xmax)`` (possibly infinite). Two-sided mirrors the observed
    statistic about 0, the center of a null distribution.
    """
    obs = float(obs_stat)
    d = direction.lower()
    vlines: list[tuple[float, bool]] = [(obs, False)]
    if d in _RIGHT:
        rects = [(obs, _INF)]
    elif d in _LEFT:
        rects = [(-_INF, obs)]
    elif d in _BOTH:
        mirror = -obs
        lo, hi = sorted((obs, mirror))
        vlines.append((mirror, True))
        rects = [(-_INF, lo), (hi, _INF)]
    else:
        raise ValueError("direction must be right/greater, left/less, or two-sided")
    return vlines, rects


def normal_overlay(values: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """``(x, density)`` for a normal curve matching the mean/SD of ``values``.

    Used by ``method="theoretical"`` / ``"both"`` as the theory-based normal
    approximation overlaid on (or replacing) the simulation histogram.
    """
    from scipy import stats

    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    x = np.linspace(float(np.min(values)), float(np.max(values)), n)
    return x, stats.norm(mu, sigma).pdf(x)
