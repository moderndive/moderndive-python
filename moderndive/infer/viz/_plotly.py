"""plotly backend for infer visualization (engine="plotly", the default)."""

from __future__ import annotations

import numpy as np

from . import _common as C


def _go():
    import plotly.graph_objects as go

    return go


def _layout(fig, title: str, xlab: str, ylab: str):
    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        bargap=0.02,
        showlegend=False,
    )
    return fig


def density_curve_px(x, density, title: str, xlab: str = "statistic", color: str | None = None):
    """A standalone theoretical density curve."""
    go = _go()
    fig = go.Figure(go.Scatter(x=x, y=density, mode="lines", line={"color": color or C._OBS_COLOR}))
    return _layout(fig, title, xlab, "density")


def visualize_px(distribution, bins: int, method: str, dens_color: str | None = None):
    """Histogram of simulated statistics, optionally overlaid with a normal curve."""
    go = _go()
    values = C.stat_values(distribution)
    xlab = C.stat_label(distribution.stat)
    title = C.dist_title(distribution.null)
    curve_color = dens_color or C._OBS_COLOR

    if method == "theoretical":
        x, dens = C.normal_overlay(values)
        return density_curve_px(x, dens, "Theoretical Distribution", xlab, curve_color)

    histnorm = "probability density" if method == "both" else None
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm=histnorm,
            marker={"color": C._HIST_FILL, "line": {"color": "white", "width": 1}},
        )
    )
    if method == "both":
        x, dens = C.normal_overlay(values)
        fig.add_scatter(x=x, y=dens, mode="lines", line={"color": curve_color})
    return _layout(fig, title, xlab, "density" if method == "both" else "count")


def visualize_fit_px(fit, bins: int):
    """Faceted histogram of a regression fit distribution, one subplot per term."""
    from plotly.subplots import make_subplots

    go = _go()
    pdf = fit.data.select("term", "estimate")
    terms = pdf["term"].unique(maintain_order=True).to_list()
    fig = make_subplots(rows=1, cols=len(terms), subplot_titles=terms)
    for i, term in enumerate(terms, start=1):
        est = pdf.filter(pdf["term"] == term)["estimate"].to_numpy()
        fig.add_trace(
            go.Histogram(
                x=est,
                nbinsx=bins,
                marker={"color": C._HIST_FILL, "line": {"color": "white", "width": 1}},
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        title=C.dist_title(fit.null), template="plotly_white", bargap=0.02, showlegend=False
    )
    return fig


def _data_range(fig) -> tuple[float, float]:
    """Finite x-range across the figure's traces, for clipping infinite shades."""
    xs = []
    for trace in fig.data:
        if getattr(trace, "x", None) is not None and len(trace.x):
            xs.append(np.asarray(trace.x, dtype=float))
    if not xs:
        return -1.0, 1.0
    allx = np.concatenate(xs)
    lo, hi = float(np.min(allx)), float(np.max(allx))
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def _clip(value: float, lo: float, hi: float) -> float:
    if value == C._INF:
        return hi
    if value == -C._INF:
        return lo
    return value


def apply_shade_px(fig, spec):
    """Return a copy of ``fig`` with p-value or confidence-interval shading added."""
    go = _go()
    out = go.Figure(fig)
    lo, hi = _data_range(out)

    if spec.kind == "p_value":
        lc = spec.color or C._OBS_COLOR
        fc = spec.fill or spec.color or C._OBS_COLOR
        vlines, rects = C.pvalue_regions(spec.obs_stat, spec.direction)
        for x, dashed in vlines:
            out.add_vline(
                x=x, line={"color": lc, "width": 2, "dash": "dash" if dashed else "solid"}
            )
        for xmin, xmax in rects:
            out.add_vrect(
                x0=_clip(xmin, lo, hi),
                x1=_clip(xmax, lo, hi),
                fillcolor=fc,
                opacity=0.3,
                line_width=0,
            )
    else:  # confidence_interval
        lc = spec.color or C._SHADE_COLOR
        fc = spec.fill or spec.color or C._SHADE_COLOR
        out.add_vrect(x0=spec.lower, x1=spec.upper, fillcolor=fc, opacity=0.3, line_width=0)
        out.add_vline(x=spec.lower, line={"color": lc, "width": 2})
        out.add_vline(x=spec.upper, line={"color": lc, "width": 2})
    return out


def _subplot_xrange(fig, col: int) -> tuple[float, float]:
    """Finite x-range of the histogram in subplot ``col`` (for clipping infinite shades)."""
    axis = "x" if col == 1 else f"x{col}"
    xs = []
    for trace in fig.data:
        tx = getattr(trace, "xaxis", None) or "x"
        if tx == axis and getattr(trace, "x", None) is not None and len(trace.x):
            xs.append(np.asarray(trace.x, dtype=float))
    if not xs:
        return -1.0, 1.0
    allx = np.concatenate(xs)
    lo, hi = float(np.min(allx)), float(np.max(allx))
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def apply_fit_shade_px(fig, spec, terms):
    """Per-facet shading for a faceted fit figure: shade each term's subplot via row/col."""
    go = _go()
    out = go.Figure(fig)
    term_to_col = {t: i + 1 for i, t in enumerate(terms)}
    per = dict(spec.per_term)

    if spec.kind == "p_value":
        lc = spec.color or C._OBS_COLOR
        fc = spec.fill or spec.color or C._OBS_COLOR
        for term, obs in per.items():
            col = term_to_col.get(term)
            if col is None:
                continue
            lo, hi = _subplot_xrange(out, col)
            vlines, rects = C.pvalue_regions(obs, spec.direction)
            for x, dashed in vlines:
                out.add_vline(
                    x=x,
                    line={"color": lc, "width": 2, "dash": "dash" if dashed else "solid"},
                    row=1,
                    col=col,
                )
            for xmin, xmax in rects:
                out.add_vrect(
                    x0=_clip(xmin, lo, hi),
                    x1=_clip(xmax, lo, hi),
                    fillcolor=fc,
                    opacity=0.3,
                    line_width=0,
                    row=1,
                    col=col,
                )
    else:  # confidence_interval
        lc = spec.color or C._SHADE_COLOR
        fc = spec.fill or spec.color or C._SHADE_COLOR
        for term, (lower, upper) in per.items():
            col = term_to_col.get(term)
            if col is None:
                continue
            out.add_vrect(
                x0=lower, x1=upper, fillcolor=fc, opacity=0.3, line_width=0, row=1, col=col
            )
            out.add_vline(x=lower, line={"color": lc, "width": 2}, row=1, col=col)
            out.add_vline(x=upper, line={"color": lc, "width": 2}, row=1, col=col)
    return out
