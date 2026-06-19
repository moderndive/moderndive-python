"""Plotting helpers (dual-engine: plotly by default, plotnine/seaborn optional).

- :func:`pairplot` — scatterplot matrix (~ GGally::ggpairs)
- :func:`gg_parallel_slopes` / :func:`geom_parallel_slopes` — parallel-slopes model
- :func:`gg_categorical_model` — regression with one categorical predictor

Every public plotting function takes ``engine="plotly"|"plotnine"`` defaulting to
``"plotly"`` (the book's default engine).
"""

from __future__ import annotations

import polars as pl

from .models import (
    geom_categorical_model,
    geom_parallel_slopes,
    gg_categorical_model,
    gg_parallel_slopes,
)

__all__ = [
    "pairplot",
    "gg_parallel_slopes",
    "geom_parallel_slopes",
    "gg_categorical_model",
    "geom_categorical_model",
]


def pairplot(
    data: pl.DataFrame,
    columns: list[str] | None = None,
    hue: str | None = None,
    *,
    engine: str = "plotly",
):
    """Scatterplot matrix of the numeric columns (the analog of ``GGally::ggpairs``).

    ``engine="plotly"`` (default) returns a plotly ``go.Figure`` from
    ``plotly.express.scatter_matrix``. ``engine="plotnine"`` (alias ``"seaborn"``)
    returns the matplotlib ``Figure`` from ``seaborn.pairplot`` — the non-plotly
    backend here is seaborn-backed, since plotnine has no first-class SPLOM.
    ``hue`` colors points by a categorical column.
    """
    if engine not in ("plotly", "plotnine", "seaborn"):
        raise ValueError("engine must be 'plotly', 'plotnine', or 'seaborn'")
    cols = columns or [
        c for c, dt in zip(data.columns, data.dtypes, strict=False) if dt.is_numeric()
    ]
    if engine == "plotly":
        import plotly.express as px

        keep = list(cols) + ([hue] if hue is not None and hue not in cols else [])
        fig = px.scatter_matrix(data.select(keep).to_pandas(), dimensions=cols, color=hue)
        fig.update_layout(template="plotly_white")
        fig.update_traces(diagonal_visible=False)
        return fig

    import seaborn as sns

    pdf = data.to_pandas()
    keep = list(cols) + ([hue] if hue is not None and hue not in cols else [])
    grid = sns.pairplot(pdf[keep], vars=cols, hue=hue, corner=False, diag_kind="hist")
    return grid.figure
