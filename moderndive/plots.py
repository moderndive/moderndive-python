"""Small plotting helpers that don't fit plotnine's grammar cleanly.

`pairplot` is a thin wrapper over seaborn's pairplot — the closest analog to R's
GGally `ggpairs` scatterplot matrix — accepting a polars DataFrame.
"""

from __future__ import annotations

import polars as pl

__all__ = ["pairplot"]


def pairplot(data: pl.DataFrame, columns: list[str] | None = None, hue: str | None = None):
    """Scatterplot matrix of the numeric columns (a seaborn PairGrid).

    The analog of R's ``GGally::ggpairs``: pairwise scatterplots off the diagonal
    and per-variable histograms on the diagonal. ``hue`` colors points by a
    categorical column. Returns the matplotlib ``Figure`` so it displays directly
    in notebooks and Quarto.
    """
    import seaborn as sns

    pdf = data.to_pandas()
    cols = columns or [c for c, dt in zip(data.columns, data.dtypes) if dt.is_numeric()]
    keep = list(cols)
    if hue is not None and hue not in keep:
        keep = keep + [hue]
    grid = sns.pairplot(pdf[keep], vars=cols, hue=hue, corner=False, diag_kind="hist")
    return grid.figure
