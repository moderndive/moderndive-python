"""Dual-engine visualization for the infer grammar (plotly default, plotnine optional).

Every plotting function takes ``engine="plotly"|"plotnine"`` and defaults to
``"plotly"`` (the book's default engine); pass ``engine="plotnine"`` to get the
grammar-of-graphics objects instead.

``visualize()`` returns an :class:`InferPlot` wrapper that composes with shading
via ``+`` for *both* engines::

    visualize(null_dist, bins=25) + shade_p_value(obs_diff, direction="right")
    visualize(boot_means) + shade_confidence_interval(endpoints=percentile_ci)

``shade_p_value()`` / ``shade_confidence_interval()`` return an engine-neutral
:class:`ShadeSpec`; ``InferPlot.__add__`` turns it into plotnine layers or applies
plotly shapes as appropriate. A keyword form is also supported and is the most
ergonomic path for plotly::

    visualize(null_dist, shade_pvalue={"obs_stat": obs, "direction": "right"})
    visualize(boot_means, shade_ci=percentile_ci)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _common as C

__all__ = [
    "InferPlot",
    "ShadeSpec",
    "visualize",
    "visualize_fit",
    "visualize_theoretical",
    "shade_p_value",
    "shade_confidence_interval",
]


@dataclass(frozen=True)
class ShadeSpec:
    """Engine-neutral description of a shading layer (p-value tail or CI).

    For a single-panel plot the scalar fields are used. For a faceted
    regression-fit plot (:func:`visualize_fit`), ``per_term`` holds one value per
    term so each facet is shaded independently — a tuple of ``(term, payload)``
    pairs where ``payload`` is the observed statistic (p-value) or a
    ``(lower, upper)`` pair (confidence interval).
    """

    kind: str  # "p_value" | "confidence_interval"
    obs_stat: float | None = None
    direction: str | None = None
    lower: float | None = None
    upper: float | None = None
    color: str | None = None
    fill: str | None = None
    per_term: tuple | None = None


class InferPlot:
    """A plot plus its engine, composable with :class:`ShadeSpec` (and, for the
    plotnine engine, any plotnine layer) via ``+``.

    Access the wrapped figure with :attr:`figure`; for the plotnine engine, the
    raw ``ggplot`` is also available via :attr:`gg`.
    """

    def __init__(self, figure, engine: str, terms: list[str] | None = None):
        self.figure = figure
        self.engine = engine
        # Facet terms for a regression-fit plot (enables per-facet shading).
        self.terms = terms

    def __add__(self, other):
        if isinstance(other, ShadeSpec) and other.per_term is not None:
            return self._add_per_facet(other)
        if self.engine == "plotnine":
            from . import _plotnine as P

            if isinstance(other, ShadeSpec):
                layers = (
                    P.shade_pvalue_layers(other)
                    if other.kind == "p_value"
                    else P.shade_ci_layers(other)
                )
                return InferPlot(self.figure + layers, "plotnine", self.terms)
            # Any other plotnine object/layer/list.
            return InferPlot(self.figure + other, "plotnine", self.terms)

        from . import _plotly as PX

        if isinstance(other, ShadeSpec):
            return InferPlot(PX.apply_shade_px(self.figure, other), "plotly", self.terms)
        raise TypeError(
            "Only a ShadeSpec (from shade_p_value/shade_confidence_interval) can be "
            "added to a plotly InferPlot."
        )

    def _add_per_facet(self, spec: ShadeSpec):
        """Apply per-term shading to a faceted regression-fit plot (both engines)."""
        if self.terms is None:
            raise TypeError(
                "Per-term shading requires a faceted fit plot from visualize_fit(); "
                "pass a per-term obs_stat/endpoints (a FitResult or a term-keyed table)."
            )
        if self.engine == "plotnine":
            from . import _plotnine as P

            return InferPlot(
                P.apply_fit_shade_gg(self.figure, spec, self.terms), "plotnine", self.terms
            )
        from . import _plotly as PX

        return InferPlot(PX.apply_fit_shade_px(self.figure, spec, self.terms), "plotly", self.terms)

    @property
    def gg(self):
        """The underlying plotnine ``ggplot`` (only when ``engine='plotnine'``)."""
        if self.engine != "plotnine":
            raise AttributeError("`.gg` is only available when engine='plotnine'.")
        return self.figure

    def show(self, **kwargs):
        """Display the figure (delegates to the underlying figure)."""
        return self.figure.show(**kwargs)

    def save(self, path, **kwargs):
        """Save the figure. plotnine forwards to ``ggplot.save``; plotly writes an
        HTML file for ``.html`` paths and a static image otherwise (the latter
        needs the optional ``kaleido`` dependency — install ``moderndive[image]``).
        """
        if self.engine == "plotnine":
            return self.figure.save(path, **kwargs)
        path_str = str(path)
        if path_str.endswith(".html"):
            return self.figure.write_html(path_str)
        return self.figure.write_image(path_str)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Rich display for Jupyter/Quarto across both engines.

        Both plotnine ``ggplot`` and plotly ``Figure`` expose
        ``_repr_mimebundle_``, so we delegate to it and the wrapped figure renders
        exactly as a bare figure would. We fall back to ``text/html`` and finally
        to no output. This is required because ``_repr_html_`` alone returns
        ``None`` for a ``ggplot`` — without this method, plotnine-engine
        ``InferPlot``s render blank in notebooks and Quarto.
        """
        fig = self.figure
        if hasattr(fig, "_repr_mimebundle_"):
            return fig._repr_mimebundle_(include=include, exclude=exclude)
        if hasattr(fig, "_repr_html_"):
            return {"text/html": fig._repr_html_()}
        return None

    def _repr_html_(self):
        fig = self.figure
        return fig._repr_html_() if hasattr(fig, "_repr_html_") else None

    def __repr__(self) -> str:
        return repr(self.figure)


def _coerce_pvalue_spec(value) -> ShadeSpec:
    if isinstance(value, ShadeSpec):
        return value
    if isinstance(value, dict):
        return shade_p_value(**value)
    raise TypeError("shade_pvalue must be a ShadeSpec or a dict of shade_p_value() kwargs.")


def _coerce_ci_spec(value) -> ShadeSpec:
    if isinstance(value, ShadeSpec):
        return value
    return shade_confidence_interval(value)


def visualize(
    distribution,
    bins: int = 20,
    *,
    engine: str = "plotly",
    method: str = "simulation",
    dens_color: str | None = None,
    shade_pvalue=None,
    shade_ci=None,
    **kwargs,
) -> InferPlot:
    """Histogram of the simulated statistics, as an :class:`InferPlot`.

    ``method`` is ``"simulation"`` (histogram, default), ``"theoretical"`` (a
    normal-approximation density curve), or ``"both"`` (histogram in density units
    overlaid with the normal curve), mirroring R ``infer``'s ``visualize(method=)``.
    ``dens_color`` sets the theoretical-curve color (for ``"theoretical"``/``"both"``).
    Pass ``shade_pvalue=``/``shade_ci=`` to shade in one call, or compose with ``+``.
    """
    engine = C.resolve_engine(engine)
    method = C.resolve_method(method)
    if engine == "plotnine":
        from . import _plotnine as P

        fig = P.visualize_gg(distribution, bins, method, dens_color)
    else:
        from . import _plotly as PX

        fig = PX.visualize_px(distribution, bins, method, dens_color)

    plot = InferPlot(fig, engine)
    if shade_pvalue is not None:
        plot = plot + _coerce_pvalue_spec(shade_pvalue)
    if shade_ci is not None:
        plot = plot + _coerce_ci_spec(shade_ci)
    return plot


def visualize_fit(
    fit, bins: int = 20, *, engine: str = "plotly", shade_pvalue=None, shade_ci=None
) -> InferPlot:
    """Faceted histogram of a regression fit distribution, one panel per term.

    Pass ``shade_pvalue=``/``shade_ci=`` to shade each facet from per-term values
    (a ``FitResult`` of observed estimates, or a term-keyed CI/p-value table), or
    compose the same per-term :class:`ShadeSpec` with ``+``.
    """
    engine = C.resolve_engine(engine)
    terms = fit.data["term"].unique(maintain_order=True).to_list()
    if engine == "plotnine":
        from . import _plotnine as P

        fig = P.visualize_fit_gg(fit, bins)
    else:
        from . import _plotly as PX

        fig = PX.visualize_fit_px(fit, bins)

    plot = InferPlot(fig, engine, terms)
    if shade_pvalue is not None:
        plot = plot + _coerce_pvalue_spec(shade_pvalue)
    if shade_ci is not None:
        plot = plot + _coerce_ci_spec(shade_ci)
    return plot


def visualize_theoretical(theoretical, bins: int = 100, *, engine: str = "plotly") -> InferPlot:
    """Density curve for a :class:`~moderndive.infer.TheoreticalDistribution`."""
    engine = C.resolve_engine(engine)
    dist = theoretical._dist()
    lo, hi = dist.ppf(0.001), dist.ppf(0.999)
    x = np.linspace(lo, hi, 400)
    density = dist.pdf(x)
    title = f"Theoretical {theoretical.distribution} distribution"
    if engine == "plotnine":
        from . import _plotnine as P

        return InferPlot(P.density_curve_gg(x, density, title), engine)
    from . import _plotly as PX

    return InferPlot(PX.density_curve_px(x, density, title), engine)


def _per_term_obs(obs_stat) -> dict | None:
    """Extract a ``{term: observed}`` mapping for per-facet p-value shading.

    Accepts an observed ``FitResult`` (term/estimate), a term-keyed polars frame
    (``term`` + ``estimate`` or ``stat``), or a dict. Returns ``None`` for a scalar.
    """
    import polars as pl

    if isinstance(obs_stat, pl.DataFrame) and "term" in obs_stat.columns:
        valcol = "estimate" if "estimate" in obs_stat.columns else "stat"
        return {r["term"]: float(r[valcol]) for r in obs_stat.iter_rows(named=True)}
    data = getattr(obs_stat, "data", None)
    if isinstance(data, pl.DataFrame) and {"term", "estimate"} <= set(data.columns):
        return {r["term"]: float(r["estimate"]) for r in data.iter_rows(named=True)}
    if isinstance(obs_stat, dict):
        return {k: float(v) for k, v in obs_stat.items()}
    return None


def _per_term_ci(endpoints) -> dict | None:
    """Extract a ``{term: (lower, upper)}`` mapping for per-facet CI shading."""
    import polars as pl

    if isinstance(endpoints, pl.DataFrame) and "term" in endpoints.columns:
        return {
            r["term"]: (float(r["lower_ci"]), float(r["upper_ci"]))
            for r in endpoints.iter_rows(named=True)
        }
    return None


def shade_p_value(
    obs_stat, direction: str, *, color: str | None = None, fill: str | None = None
) -> ShadeSpec:
    """A p-value shading spec; add it to a ``visualize()`` plot with ``+``.

    ``direction`` ∈ {right/greater, left/less, two-sided}. For a faceted
    :func:`visualize_fit` plot, pass a per-term ``obs_stat`` — an observed
    ``FitResult``, a ``term``-keyed frame, or a dict — to shade each facet.
    """
    per = _per_term_obs(obs_stat)
    if per is not None:
        return ShadeSpec(
            kind="p_value",
            direction=direction,
            color=color,
            fill=fill,
            per_term=tuple(sorted(per.items())),
        )
    return ShadeSpec(
        kind="p_value", obs_stat=float(obs_stat), direction=direction, color=color, fill=fill
    )


def shade_confidence_interval(
    endpoints, color: str | None = None, fill: str | None = None
) -> ShadeSpec:
    """A confidence-interval shading spec; add it to a ``visualize()`` plot with ``+``.

    ``endpoints`` is a CI DataFrame (``lower_ci``/``upper_ci``) or a ``(lower, upper)``
    tuple. For a faceted :func:`visualize_fit` plot, pass a per-term CI table (with a
    ``term`` column) to shade each facet from its own interval.
    """
    per = _per_term_ci(endpoints)
    if per is not None:
        return ShadeSpec(
            kind="confidence_interval",
            color=color,
            fill=fill,
            per_term=tuple(sorted(per.items())),
        )
    lower, upper = C.ci_endpoints(endpoints)
    return ShadeSpec(kind="confidence_interval", lower=lower, upper=upper, color=color, fill=fill)
