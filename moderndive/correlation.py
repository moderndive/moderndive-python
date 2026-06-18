"""Correlation and population-spread helpers mirroring the R ``moderndive`` package.

- :func:`get_correlation` ~ ``moderndive::get_correlation`` (tidy 1-row ``cor`` frame)
- :func:`pop_sd`          ~ ``moderndive::pop_sd`` (population standard deviation)
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["get_correlation", "pop_sd"]


def _parse_pair(formula: str | None, x: str | None, y: str | None) -> tuple[str, str]:
    """Resolve the (y, x) column pair from a ``"y ~ x"`` formula or x=/y= kwargs."""
    if formula is not None:
        if x is not None or y is not None:
            raise ValueError("Pass either formula or x=/y=, not both.")
        if "~" not in formula:
            raise ValueError(f"formula must look like 'y ~ x', got {formula!r}.")
        lhs, rhs = (part.strip() for part in formula.split("~", 1))
        if not lhs or not rhs:
            raise ValueError(f"formula must look like 'y ~ x', got {formula!r}.")
        return lhs, rhs
    if x is None or y is None:
        raise ValueError("Provide a formula 'y ~ x' or both x= and y=.")
    return y, x


def get_correlation(
    data,
    formula: str | None = None,
    *,
    x: str | None = None,
    y: str | None = None,
) -> pl.DataFrame:
    """Pearson correlation as a tidy 1-row frame with a ``cor`` column.

    Mirrors ``moderndive::get_correlation(data, y ~ x)``. Specify the variable
    pair either as a formula string (``"y ~ x"``) or via the ``x=`` and ``y=``
    keyword arguments. Rows with a null in either column are dropped.
    """
    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    y_col, x_col = _parse_pair(formula, x, y)
    for col in (y_col, x_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} is not in the data.")
    pair = df.select(x_col, y_col).drop_nulls()
    value = float(np.corrcoef(pair[x_col].to_numpy(), pair[y_col].to_numpy())[0, 1])
    return pl.DataFrame({"cor": [value]})


def pop_sd(x) -> float:
    """Population standard deviation (divides by ``n``, not ``n - 1``).

    Mirrors ``moderndive::pop_sd``. Accepts a polars Series, list, numpy array,
    or any sequence; nulls/NaNs are dropped before computing.
    """
    if isinstance(x, pl.Series):
        values = x.drop_nulls().to_numpy()
    else:
        values = np.asarray(list(x), dtype=float)
        values = values[~np.isnan(values)]
    return float(np.std(values, ddof=0))
