"""Regression-model plotting helpers ported from the R ``moderndive`` package.

- :func:`gg_parallel_slopes`  ~ ``moderndive::gg_parallel_slopes`` (scatter + parallel lines)
- :func:`geom_parallel_slopes` ~ ``moderndive::geom_parallel_slopes`` (plotnine layer)
- :func:`gg_categorical_model` ~ ``moderndive::geom_categorical_model`` (categorical predictor)
- :func:`plot_3d_regression` ~ ``moderndive::plot_3d_regression`` (3D scatter + plane)

The non-plotly backend is plotnine. All functions accept polars or pandas frames.
``gg_*`` return a figure (plotly ``go.Figure`` or plotnine ``ggplot``);
``geom_parallel_slopes`` returns plotnine layers to add to a ``ggplot`` with ``+``.
"""

from __future__ import annotations

import re

import polars as pl

from .._messaging import helpful_error

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


def gg_parallel_slopes(
    data, response: str, explanatory: str, by: str, *, alpha: float = 1.0, engine: str = "plotly"
):
    """Scatterplot with a parallel-slopes regression model overlaid.

    Fits ``response ~ explanatory + C(by)`` (one common slope, a separate intercept
    per level of ``by``) and draws one fitted line per group over the data.
    ``alpha`` sets the point transparency (0–1), useful when points overlap.
    """
    engine = _resolve_engine(engine)
    pdf = _to_pandas(data, [response, explanatory, by])
    pdf[by] = pdf[by].astype(str)
    intercepts, slope, levels = _parallel_slopes_fit(pdf, response, explanatory, by)
    xmin, xmax = float(pdf[explanatory].min()), float(pdf[explanatory].max())

    if engine == "plotly":
        import plotly.express as px

        fig = px.scatter(pdf, x=explanatory, y=response, color=by, opacity=alpha)
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
        + geom_point(alpha=alpha)
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


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def plot_3d_regression(data, formula: str, n: int = 25):
    """Interactive 3D scatterplot with a fitted regression plane.

    Mirrors ``moderndive::plot_3d_regression``. Pass a formula ``z ~ x + y`` —
    one numeric outcome and exactly two numeric predictors — and get a plotly
    ``go.Figure`` with the data points and the fitted ``lm`` plane.

    In-formula transformations (e.g. ``log(z) ~ x + y``) are **not** supported,
    since the plane and the raw points would be on different scales; transform
    the columns of ``data`` first and pass plain names. ``n`` sets the plane's
    grid resolution per axis.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.formula.api as smf

    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    if not isinstance(n, int) or n < 2:
        raise ValueError(
            helpful_error(f"`n` must be an integer ≥ 2, got {n!r}.", "Try n=25 (the default).")
        )
    if "~" not in formula:
        raise ValueError(
            helpful_error(
                f"formula must look like 'z ~ x + y', got {formula!r}.",
                "Put the outcome on the left of ~ and exactly two predictors on the right.",
            )
        )
    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    predictors = [v.strip() for v in rhs.split("+") if v.strip()]
    if not _IDENTIFIER.match(lhs) or not all(_IDENTIFIER.match(p) for p in predictors):
        raise ValueError(
            helpful_error(
                "plot_3d_regression() does not support in-formula transformations "
                "(e.g. 'log(z) ~ x + y').",
                "Transform the columns of `data` first, then pass plain names like 'z ~ x + y'.",
            )
        )
    if len(predictors) != 2:
        raise ValueError(
            helpful_error(
                f"The right-hand side must name exactly two predictors (got {len(predictors)}).",
                "Use a formula like 'z ~ x + y'.",
            )
        )
    missing = [c for c in [lhs, *predictors] if c not in df.columns]
    if missing:
        raise ValueError(
            helpful_error(
                f"Column(s) not found in the data: {', '.join(missing)}.",
                f"Available columns: {', '.join(df.columns)}.",
            )
        )
    for var in [lhs, *predictors]:
        if not df.schema[var].is_numeric():
            raise ValueError(
                helpful_error(
                    f"`{var}` must be numeric — a regression plane needs three continuous "
                    "variables.",
                    "For a categorical predictor, see gg_categorical_model().",
                )
            )

    x_var, y_var = predictors
    pdf = df.select(lhs, x_var, y_var).drop_nulls().to_pandas()
    model = smf.ols(f"{lhs} ~ {x_var} + {y_var}", data=pdf).fit()

    x_seq = np.linspace(pdf[x_var].min(), pdf[x_var].max(), n)
    y_seq = np.linspace(pdf[y_var].min(), pdf[y_var].max(), n)
    grid_x, grid_y = np.meshgrid(x_seq, y_seq)
    grid = pd.DataFrame({x_var: grid_x.ravel(), y_var: grid_y.ravel()})
    z_pred = np.asarray(model.predict(grid)).reshape(grid_x.shape)

    fig = go.Figure(
        go.Scatter3d(
            x=pdf[x_var],
            y=pdf[y_var],
            z=pdf[lhs],
            mode="markers",
            marker={"size": 4},
            name="data",
        )
    )
    fig.add_surface(
        x=x_seq, y=y_seq, z=z_pred, opacity=0.6, showscale=False, name="regression plane"
    )
    fig.update_layout(
        template="plotly_white",
        scene={"xaxis_title": x_var, "yaxis_title": y_var, "zaxis_title": lhs},
    )
    return fig
