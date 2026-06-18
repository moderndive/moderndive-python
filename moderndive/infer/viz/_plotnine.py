"""plotnine backend for infer visualization (engine="plotnine")."""

from __future__ import annotations

import pandas as pd
from plotnine import (
    aes,
    after_stat,
    annotate,
    facet_wrap,
    geom_histogram,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    theme_light,
)

from . import _common as C


def _full_height_rect(xmin: float, xmax: float, fill: str):
    return annotate("rect", xmin=xmin, xmax=xmax, ymin=-C._INF, ymax=C._INF, alpha=0.3, fill=fill)


def density_curve_gg(x, density, title: str, xlab: str = "statistic"):
    """A standalone theoretical density curve."""
    pdf = pd.DataFrame({"x": x, "density": density})
    return (
        ggplot(pdf, aes(x="x", y="density"))
        + geom_line(color=C._OBS_COLOR, size=1.0)
        + labs(x=xlab, y="density", title=title)
        + theme_light()
    )


def visualize_gg(distribution, bins: int, method: str):
    """Histogram of simulated statistics, optionally overlaid with a normal curve."""
    values = C.stat_values(distribution)
    xlab = C.stat_label(distribution.stat)
    title = C.dist_title(distribution.null)

    if method == "theoretical":
        x, dens = C.normal_overlay(values)
        return density_curve_gg(x, dens, "Theoretical Distribution", xlab)

    pdf = pd.DataFrame({"stat": values})
    if method == "both":
        x, dens = C.normal_overlay(values)
        return (
            ggplot(pdf, aes(x="stat"))
            + geom_histogram(
                aes(y=after_stat("density")), bins=bins, color="white", fill=C._HIST_FILL
            )
            + geom_line(
                aes(x="stat", y="density"),
                data=pd.DataFrame({"stat": x, "density": dens}),
                color=C._OBS_COLOR,
                size=1.0,
            )
            + labs(x=xlab, y="density", title=title)
            + theme_light()
        )

    return (
        ggplot(pdf, aes(x="stat"))
        + geom_histogram(bins=bins, color="white", fill=C._HIST_FILL)
        + labs(x=xlab, y="count", title=title)
        + theme_light()
    )


def visualize_fit_gg(fit, bins: int):
    """Faceted histogram of a regression fit distribution, one panel per term."""
    pdf = fit.data.select("term", "estimate").to_pandas()
    return (
        ggplot(pdf, aes(x="estimate"))
        + geom_histogram(bins=bins, color="white", fill=C._HIST_FILL)
        + facet_wrap("term", scales="free")
        + labs(x="estimate", y="count", title=C.dist_title(fit.null))
        + theme_light()
    )


def shade_pvalue_layers(spec) -> list:
    """plotnine layers shading the p-value tail(s) and marking the observed stat."""
    vlines, rects = C.pvalue_regions(spec.obs_stat, spec.direction)
    layers: list = []
    for x, dashed in vlines:
        extra = {"linetype": "dashed"} if dashed else {}
        layers.append(geom_vline(xintercept=x, color=C._OBS_COLOR, size=1.0, **extra))
    for xmin, xmax in rects:
        layers.append(_full_height_rect(xmin, xmax, C._OBS_COLOR))
    return layers


def shade_ci_layers(spec) -> list:
    """plotnine layers shading the confidence interval between its endpoints."""
    color = spec.color or C._SHADE_COLOR
    return [
        _full_height_rect(spec.lower, spec.upper, color),
        geom_vline(xintercept=spec.lower, color=color, size=1.0),
        geom_vline(xintercept=spec.upper, color=color, size=1.0),
    ]
