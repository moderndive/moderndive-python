"""Regression + summary helpers mirroring the R ``moderndive`` package.

Built on ``statsmodels`` (which, unlike scikit-learn, reports standard errors,
test statistics, p-values, and confidence intervals — the inferential output the
book teaches). Inputs may be polars or pandas frames; outputs are polars frames.

- :func:`get_regression_table`  ~ ``broom::tidy`` + CIs
- :func:`get_regression_points` ~ ``broom::augment`` (fitted values + residuals)
- :func:`tidy_summary`          ~ per-variable summary statistics
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

__all__ = ["get_regression_table", "get_regression_points", "tidy_summary"]


def _to_pandas(data):
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    return data


def _clean_term(term: str) -> str:
    """Turn patsy term labels into the tidy names moderndive uses.

    Examples:
        ``Intercept``                         -> ``intercept``
        ``income[T.Lower middle income]``     -> ``income: Lower middle income``
        ``life_exp:income[T.High income]``    -> ``life_exp:income: High income``
        ``C(income, levels=[...])[T.High income]`` -> ``income: High income``
    """
    term = term.replace("Intercept", "intercept")
    # Drop any C(col, ...) wrapper down to just the column name.
    term = re.sub(r"C\(\s*([A-Za-z_]\w*)[^)]*\)", r"\1", term)
    # Turn a categorical level marker [T.level] into ": level".
    term = re.sub(r"\[T\.(.*?)\]", r": \1", term)
    return term


def get_regression_table(model, digits: int = 3, conf_level: float = 0.95) -> pl.DataFrame:
    """Tidy regression table: term, estimate, std_error, statistic, p_value, lower/upper_ci.

    ``model`` is a fitted ``statsmodels`` results object (e.g. from
    ``statsmodels.formula.api.ols("y ~ x", data).fit()``).
    """
    conf = model.conf_int(alpha=1 - conf_level)
    terms = [_clean_term(t) for t in model.params.index]
    table = pl.DataFrame(
        {
            "term": terms,
            "estimate": model.params.to_numpy(),
            "std_error": model.bse.to_numpy(),
            "statistic": model.tvalues.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
            "lower_ci": np.asarray(conf)[:, 0],
            "upper_ci": np.asarray(conf)[:, 1],
        }
    )
    numeric = [c for c in table.columns if c != "term"]
    return table.with_columns(pl.col(numeric).round(digits))


def get_regression_points(model, digits: int = 3) -> pl.DataFrame:
    """Fitted values + residuals per observation (~ ``broom::augment``).

    Columns: ``ID``, the response, each explanatory term, ``<response>_hat``,
    ``residual``.
    """
    endog_name = model.model.endog_names
    exog_names = [n for n in model.model.exog_names if n != "Intercept"]
    frame = _to_pandas(model.model.data.frame)

    out = {
        "ID": np.arange(1, len(frame) + 1, dtype=np.int64),
        endog_name: np.asarray(frame[endog_name]),
    }
    for name in exog_names:
        if name in frame.columns:
            out[name] = np.asarray(frame[name])
    out[f"{endog_name}_hat"] = np.asarray(model.fittedvalues)
    out["residual"] = np.asarray(model.resid)

    df = pl.DataFrame(out)
    round_cols = [endog_name, f"{endog_name}_hat", "residual"]
    round_cols = [c for c in round_cols if df.schema[c].is_numeric()]
    return df.with_columns(pl.col(round_cols).round(digits))


def tidy_summary(data, columns: list[str] | None = None, digits: int = 3) -> pl.DataFrame:
    """Per-variable summary statistics for the selected columns.

    Mirrors the R ``moderndive::tidy_summary`` column layout:
    ``column, n, group, type, min, Q1, mean, median, Q3, max, sd``.
    Numeric columns get the five-number summary + mean/sd; non-numeric columns
    report ``n`` and ``type`` with the numeric fields left null.
    """
    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    columns = columns or df.columns

    rows = []
    for col in columns:
        series = df[col]
        is_num = series.dtype.is_numeric()
        n = int(series.len() - series.null_count())
        row = {
            "column": col,
            "n": n,
            "group": None,
            "type": "numeric" if is_num else "categorical",
            "min": None,
            "Q1": None,
            "mean": None,
            "median": None,
            "Q3": None,
            "max": None,
            "sd": None,
        }
        if is_num:
            s = series.drop_nulls()
            row.update(
                min=round(float(s.min()), digits),
                Q1=round(float(s.quantile(0.25)), digits),
                mean=round(float(s.mean()), digits),
                median=round(float(s.median()), digits),
                Q3=round(float(s.quantile(0.75)), digits),
                max=round(float(s.max()), digits),
                sd=round(float(s.std()), digits),
            )
        rows.append(row)

    schema = {
        "column": pl.Utf8,
        "n": pl.Int64,
        "group": pl.Utf8,
        "type": pl.Utf8,
        "min": pl.Float64,
        "Q1": pl.Float64,
        "mean": pl.Float64,
        "median": pl.Float64,
        "Q3": pl.Float64,
        "max": pl.Float64,
        "sd": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)
