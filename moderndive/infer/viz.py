"""plotnine-based visualization of simulated distributions (infer's visualize()).

``visualize()`` returns a plotnine ``ggplot`` (a histogram of the ``stat``
column), so it composes with ``+`` exactly like ggplot layers — preserving the
grammar-of-graphics muscle memory from Chapter 2:

    visualize(null_distribution, bins=25) + shade_p_value(obs_diff, direction="right")
    visualize(bootstrap_means) + shade_confidence_interval(endpoints=percentile_ci)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from plotnine import (
    aes,
    annotate,
    facet_wrap,
    geom_histogram,
    geom_vline,
    ggplot,
    labs,
    theme_light,
)

if TYPE_CHECKING:
    from .core import Distribution

_OBS_COLOR = "#d62728"
_SHADE_COLOR = "#1f77b4"


def _stat_label(stat: str | None) -> str:
    return "stat" if stat is None else stat


def visualize(distribution: Distribution, bins: int = 20, **kwargs):
    """Histogram of the simulated statistics (a plotnine ggplot)."""
    pdf = distribution.data.select("stat").to_pandas()
    title = (
        "Simulation-Based Null Distribution"
        if distribution.null is not None
        else "Simulation-Based Bootstrap Distribution"
    )
    return (
        ggplot(pdf, aes(x="stat"))
        + geom_histogram(bins=bins, color="white", fill="#7f7f7f")
        + labs(x=_stat_label(distribution.stat), y="count", title=title)
        + theme_light()
    )


_INF = float("inf")


def visualize_fit(fit, bins: int = 20):
    """Faceted histogram of a regression fit distribution, one panel per term."""
    pdf = fit.data.select("term", "estimate").to_pandas()
    title = (
        "Simulation-Based Null Distribution"
        if fit.null is not None
        else "Simulation-Based Bootstrap Distribution"
    )
    return (
        ggplot(pdf, aes(x="estimate"))
        + geom_histogram(bins=bins, color="white", fill="#7f7f7f")
        + facet_wrap("term", scales="free")
        + labs(x="estimate", y="count", title=title)
        + theme_light()
    )


def _full_height_rect(xmin: float, xmax: float, fill: str):
    """A translucent rectangle spanning the full panel height between xmin and xmax."""
    return annotate("rect", xmin=xmin, xmax=xmax, ymin=-_INF, ymax=_INF, alpha=0.3, fill=fill)


def shade_p_value(obs_stat, direction: str) -> list:
    """Shade the p-value tail(s) and mark the observed statistic.

    Returns a list of plotnine layers (add it to a ``visualize()`` plot with ``+``).
    ``direction`` ∈ {right/greater, left/less, two-sided}.
    """
    obs = float(obs_stat)
    direction = direction.lower()

    layers = [geom_vline(xintercept=obs, color=_OBS_COLOR, size=1.0)]
    if direction in {"right", "greater"}:
        layers.append(_full_height_rect(obs, _INF, _OBS_COLOR))
    elif direction in {"left", "less"}:
        layers.append(_full_height_rect(-_INF, obs, _OBS_COLOR))
    else:  # two-sided: mirror about 0 (null statistics center on 0)
        mirror = -obs
        lo, hi = sorted((obs, mirror))
        layers.append(geom_vline(xintercept=mirror, color=_OBS_COLOR, size=1.0, linetype="dashed"))
        layers.append(_full_height_rect(-_INF, lo, _OBS_COLOR))
        layers.append(_full_height_rect(hi, _INF, _OBS_COLOR))
    return layers


def shade_confidence_interval(endpoints, color: str = _SHADE_COLOR) -> list:
    """Shade the confidence interval between its endpoints.

    Returns a list of plotnine layers. ``endpoints`` may be a polars DataFrame
    with ``lower_ci``/``upper_ci`` columns (the output of
    :func:`get_confidence_interval`) or a 2-tuple ``(lower, upper)``.
    """
    if isinstance(endpoints, pl.DataFrame):
        lower = float(endpoints["lower_ci"][0])
        upper = float(endpoints["upper_ci"][0])
    else:
        lower, upper = (float(x) for x in endpoints)

    return [
        _full_height_rect(lower, upper, color),
        geom_vline(xintercept=lower, color=color, size=1.0),
        geom_vline(xintercept=upper, color=color, size=1.0),
    ]
