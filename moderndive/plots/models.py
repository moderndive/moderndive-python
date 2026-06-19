"""Regression-model plotting helpers ported from the R ``moderndive`` package.

- :func:`gg_parallel_slopes`  ~ ``moderndive::gg_parallel_slopes`` (scatter + parallel lines)
- :func:`geom_parallel_slopes` ~ ``moderndive::geom_parallel_slopes`` (plotnine layer)
- :func:`gg_categorical_model` ~ ``moderndive::geom_categorical_model`` (categorical predictor)

The non-plotly backend is plotnine. All functions accept polars or pandas frames.
``gg_*`` return a figure (plotly ``go.Figure`` or plotnine ``ggplot``);
``geom_parallel_slopes`` returns plotnine layers to add to a ``ggplot`` with ``+``.
"""

from __future__ import annotations

import polars as pl

# Qualitative palette (matches plotly's default) so plotnine/plotly look alike.
_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]
_FIT_COLOR = "#d62728"


def _to_pandas(data, columns):
    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    return df.select(columns).drop_nulls().to_pandas()


def _resolve_engine(engine: str) -> str:
    if engine not in ("plotly", "plotnine"):
        raise ValueError(f"engine must be 'plotly' or 'plotnine', got {engine!r}")
    return engine


def _parallel_slopes_fit(pdf, response: str, explanatory: str, by: str):
    """Fit ``response ~ explanatory + C(by)``; return (intercepts, slope, levels).

    ``intercepts`` maps each level of ``by`` to its line intercept (common slope,
    differing intercepts — the definition of a parallel-slopes model).
    """
    import statsmodels.formula.api as smf

    pdf = pdf.copy()
    pdf[by] = pdf[by].astype(str)
    model = smf.ols(f"{response} ~ {explanatory} + C({by})", data=pdf).fit()
    params = model.params
    slope = float(params[explanatory])
    base = float(params["Intercept"])
    levels = sorted(pdf[by].unique())
    intercepts = {}
    for lvl in levels:
        key = f"C({by})[T.{lvl}]"
        intercepts[lvl] = base + (float(params[key]) if key in params.index else 0.0)
    return intercepts, slope, levels


def gg_parallel_slopes(data, response: str, explanatory: str, by: str, *, engine: str = "plotly"):
    """Scatterplot with a parallel-slopes regression model overlaid.

    Fits ``response ~ explanatory + C(by)`` (one common slope, a separate intercept
    per level of ``by``) and draws one fitted line per group over the data.
    """
    engine = _resolve_engine(engine)
    pdf = _to_pandas(data, [response, explanatory, by])
    pdf[by] = pdf[by].astype(str)
    intercepts, slope, levels = _parallel_slopes_fit(pdf, response, explanatory, by)
    xmin, xmax = float(pdf[explanatory].min()), float(pdf[explanatory].max())

    if engine == "plotly":
        import plotly.express as px

        fig = px.scatter(pdf, x=explanatory, y=response, color=by)
        for i, lvl in enumerate(levels):
            b = intercepts[lvl]
            fig.add_scatter(
                x=[xmin, xmax],
                y=[b + slope * xmin, b + slope * xmax],
                mode="lines",
                line={"color": _PALETTE[i % len(_PALETTE)]},
                name=f"{lvl} fit",
                showlegend=False,
            )
        fig.update_layout(template="plotly_white")
        return fig

    from plotnine import aes, geom_point, ggplot, labs, theme_light

    return (
        ggplot(pdf, aes(x=explanatory, y=response, color=by))
        + geom_point()
        + geom_parallel_slopes(pdf, response, explanatory, by)
        + labs(x=explanatory, y=response, title="Parallel slopes model")
        + theme_light()
    )


def geom_parallel_slopes(data, response: str, explanatory: str, by: str, color: str | None = None):
    """plotnine layer(s) drawing the parallel-slopes fitted lines.

    Add to a ``ggplot`` with ``+`` (plotnine-only; for a plotly version call
    :func:`gg_parallel_slopes` with ``engine="plotly"``).
    """
    import pandas as pd
    from plotnine import aes, geom_line

    pdf = _to_pandas(data, [response, explanatory, by])
    pdf[by] = pdf[by].astype(str)
    intercepts, slope, levels = _parallel_slopes_fit(pdf, response, explanatory, by)
    xmin, xmax = float(pdf[explanatory].min()), float(pdf[explanatory].max())

    rows = []
    for lvl in levels:
        b = intercepts[lvl]
        rows.append({explanatory: xmin, response: b + slope * xmin, by: lvl})
        rows.append({explanatory: xmax, response: b + slope * xmax, by: lvl})
    line_df = pd.DataFrame(rows)

    if color is None:
        return [geom_line(aes(x=explanatory, y=response, color=by), data=line_df)]
    return [geom_line(aes(x=explanatory, y=response, group=by), data=line_df, color=color)]


def gg_categorical_model(data, response: str, explanatory: str, *, engine: str = "plotly"):
    """Regression with one categorical predictor (~ ``geom_categorical_model``).

    Fits ``response ~ C(explanatory)``; each category's fitted value is its group
    mean, drawn as a horizontal marker over the (jittered) data points.
    """
    engine = _resolve_engine(engine)
    pdf = _to_pandas(data, [response, explanatory])
    pdf[explanatory] = pdf[explanatory].astype(str)
    means = pdf.groupby(explanatory)[response].mean()
    levels = sorted(pdf[explanatory].unique())

    if engine == "plotly":
        import plotly.express as px

        fig = px.strip(pdf, x=explanatory, y=response, category_orders={explanatory: levels})
        fig.add_scatter(
            x=levels,
            y=[float(means[lvl]) for lvl in levels],
            mode="markers",
            marker={"symbol": "line-ew", "size": 28, "line": {"color": _FIT_COLOR, "width": 3}},
            name="fitted mean",
            showlegend=False,
        )
        fig.update_layout(template="plotly_white")
        return fig

    import pandas as pd
    from plotnine import aes, geom_point, ggplot, labs, theme_light

    mean_df = pd.DataFrame({explanatory: levels, response: [float(means[lvl]) for lvl in levels]})
    return (
        ggplot(pdf, aes(x=explanatory, y=response))
        + geom_point(position="jitter", alpha=0.5)
        + geom_point(data=mean_df, color=_FIT_COLOR, shape="_", size=8)
        + labs(x=explanatory, y=response, title="Categorical model")
        + theme_light()
    )


# R-parity alias: R's helper is named geom_categorical_model(); expose both names.
geom_categorical_model = gg_categorical_model
