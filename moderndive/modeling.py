"""Regression + summary helpers mirroring the R ``moderndive`` package.

Built on ``statsmodels`` (which, unlike scikit-learn, reports standard errors,
test statistics, p-values, and confidence intervals — the inferential output the
book teaches). Both ``ols()`` (linear) and ``glm()`` (e.g. logistic) fitted
models are supported. Inputs may be polars or pandas frames; outputs are polars.

- :func:`get_regression_table`  ~ ``broom::tidy`` + CIs
- :func:`get_regression_points` ~ ``broom::augment`` (fitted values + residuals)
- :func:`get_regression_summaries` ~ ``broom::glance``
- :func:`tidy_summary`          ~ per-variable summary statistics
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from ._messaging import helpful_error

__all__ = [
    "get_regression_table",
    "get_regression_points",
    "get_regression_summaries",
    "tidy_summary",
    "count_missing",
]


def _to_pandas(data):
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    return data


def _check_model(model) -> None:
    """Beginner-friendly guard: ``model`` must be a fitted statsmodels result."""
    if not (hasattr(model, "params") and hasattr(model, "model")):
        raise TypeError(
            helpful_error(
                f"Expected a fitted statsmodels model, got {type(model).__name__}.",
                'Fit one first, e.g. smf.ols("y ~ x", data).fit() or smf.glm(...).fit().',
            )
        )


def _is_glm(model) -> bool:
    """A fitted statsmodels GLM result (logistic, Poisson, …) exposes ``family``."""
    return hasattr(model, "family")


def _clean_term(term: str) -> str:
    """Turn patsy term labels into the tidy names moderndive uses.

    Examples:
        ``Intercept`` / ``const``             -> ``intercept``
        ``income[T.Lower middle income]``     -> ``income: Lower middle income``
        ``life_exp:income[T.High income]``    -> ``life_exp:income: High income``
        ``C(income, levels=[...])[T.High income]`` -> ``income: High income``
    """
    if term == "const":  # the constant column from sm.add_constant (array API)
        return "intercept"
    term = term.replace("Intercept", "intercept")
    # Drop any C(col, ...) wrapper down to just the column name.
    term = re.sub(r"C\(\s*([A-Za-z_]\w*)[^)]*\)", r"\1", term)
    # Turn a categorical level marker [T.level] into ": level".
    term = re.sub(r"\[T\.(.*?)\]", r": \1", term)
    return term


def _term_names(model) -> list[str]:
    """Coefficient names for the table. Works for formula/pandas (``params`` is a
    Series with an index) and array-API numpy fits (``params`` is a bare array,
    so fall back to ``exog_names``)."""
    index = getattr(model.params, "index", None)
    return list(index) if index is not None else list(model.model.exog_names)


def _has_formula_frame(model) -> bool:
    """Whether the model carries the original data + formula (the formula API).

    True for ``smf.ols("y ~ x", data).fit()``; False for an array-API fit like
    ``sm.OLS(y, X).fit()``, which exposes only the design matrix.
    """
    frame = getattr(getattr(model.model, "data", None), "frame", None)
    return frame is not None and getattr(model.model, "formula", None) is not None


def _clean_name(expr: str) -> str:
    """Sanitize a transformed term into a tidy column name.

    ``np.log(mpg)`` -> ``log_mpg``; ``np.sqrt(price)`` -> ``sqrt_price``.
    """
    expr = expr.replace("np.", "")
    return re.sub(r"[^0-9A-Za-z]+", "_", expr).strip("_").lower()


def _outcome_info(model) -> tuple[str, str, bool]:
    """Resolve the outcome's tidy column name (and whether the LHS is transformed).

    An untransformed LHS like ``mpg`` keeps its name; a transformed LHS like
    ``np.log(mpg)`` becomes ``log_mpg`` (with ``log_mpg_hat`` for fitted values).
    """
    endog = model.model.endog_names
    if endog in model.model.data.frame.columns:
        return endog, f"{endog}_hat", False
    name = _clean_name(endog)
    return name, f"{name}_hat", True


def _predictor_vars(model) -> list[str]:
    """Original predictor variable names from the RHS, in formula order.

    For ``y ~ poly(hp, 2) + wt`` this returns ``["hp", "wt"]`` — the original
    columns, never the patsy basis/transform columns.
    """
    frame_cols = list(model.model.data.frame.columns)
    rhs = model.model.formula.split("~", 1)[1]
    ordered: list[str] = []
    for token in re.findall(r"[A-Za-z_]\w*", rhs):
        if token in frame_cols and token not in ordered:
            ordered.append(token)
    return ordered


def get_regression_table(
    model,
    digits: int = 3,
    conf_level: float = 0.95,
    exponentiate: bool = False,
) -> pl.DataFrame:
    """Tidy regression table: term, estimate, std_error, statistic, p_value, lower/upper_ci.

    ``model`` is a fitted ``statsmodels`` results object — either OLS
    (``smf.ols("y ~ x", data).fit()``) or GLM (``smf.glm(...).fit()``).

    For GLMs with a log or logit link, pass ``exponentiate=True`` to report the
    coefficient estimate and its confidence interval as rate / odds ratios
    (``std_error``, ``statistic``, and ``p_value`` stay on the model's link scale,
    matching ``broom::tidy``).
    """
    _check_model(model)
    conf = np.asarray(model.conf_int(alpha=1 - conf_level), dtype=float)
    estimate = np.asarray(model.params, dtype=float)
    lower, upper = conf[:, 0], conf[:, 1]
    if exponentiate:
        estimate, lower, upper = np.exp(estimate), np.exp(lower), np.exp(upper)
    table = pl.DataFrame(
        {
            "term": [_clean_term(t) for t in _term_names(model)],
            "estimate": estimate,
            "std_error": np.asarray(model.bse, dtype=float),
            "statistic": np.asarray(model.tvalues, dtype=float),
            "p_value": np.asarray(model.pvalues, dtype=float),
            "lower_ci": lower,
            "upper_ci": upper,
        }
    )
    numeric = [c for c in table.columns if c != "term"]
    return table.with_columns(pl.col(numeric).round(digits))


def get_regression_points(model, digits: int = 3) -> pl.DataFrame:
    """Fitted values + residuals per observation (~ ``broom::augment``).

    Columns: ``ID``, the outcome, each original predictor, ``<outcome>_hat``,
    ``residual``. In-formula transformations are handled gracefully: a
    transformed outcome (``np.log(mpg)``) is shown on the model's scale under a
    sanitized name (``log_mpg`` / ``log_mpg_hat``), and transformed predictors
    (``poly()``, ``scale()``, ``I()``) are shown as their original columns rather
    than leaking basis matrices. For GLMs, fitted values and residuals are on the
    response scale (e.g. probabilities for logistic regression).
    """
    _check_model(model)
    name, name_hat, out = (
        _points_columns_formula(model)
        if _has_formula_frame(model)
        else _points_columns_array(model)
    )
    fitted = np.asarray(model.fittedvalues, dtype=float)
    if _is_glm(model):
        residual = np.asarray(model.resid_response, dtype=float)
    else:
        residual = np.asarray(out[name], dtype=float) - fitted
    out[name_hat] = fitted
    out["residual"] = residual

    df = pl.DataFrame(out).drop_nulls().with_row_index("ID", offset=1)
    df = df.with_columns(pl.col("ID").cast(pl.Int64))
    float_cols = [c for c, dt in df.schema.items() if dt.is_float()]
    return df.with_columns(pl.col(float_cols).round(digits))


def _points_columns_formula(model):
    """Outcome + original predictor columns for a formula-API model (transform-aware)."""
    frame = _to_pandas(model.model.data.frame)
    name, name_hat, transformed = _outcome_info(model)
    outcome_vals = (
        np.asarray(model.model.endog, dtype=float) if transformed else np.asarray(frame[name])
    )
    out = {name: outcome_vals}
    for predictor in _predictor_vars(model):
        out[predictor] = np.asarray(frame[predictor])
    return name, name_hat, out


def _points_columns_array(model):
    """Outcome + predictor columns for an array-API model (e.g. ``sm.OLS(y, X)``).

    There are no in-formula transforms here, so each non-constant design-matrix
    column is a predictor and the outcome is the endog array.
    """
    name = model.model.endog_names or "y"
    out = {name: np.asarray(model.model.endog, dtype=float)}
    exog = np.asarray(model.model.exog, dtype=float)
    for j, raw in enumerate(model.model.exog_names):
        if raw in ("const", "Intercept"):
            continue
        out[raw] = exog[:, j]
    return name, f"{name}_hat", out


def get_regression_summaries(model, digits: int = 3) -> pl.DataFrame:
    """Model-fit summaries as a tidy 1-row frame (~ ``broom::glance``).

    For an **OLS** model: ``r_squared``, ``adj_r_squared``, ``mse``, ``rmse``,
    ``sigma``, ``statistic`` (overall F), ``p_value``, ``df``, ``nobs``.

    For a **GLM** (no R² applies): ``mse``, ``rmse``, ``deviance``,
    ``null_deviance``, ``aic``, ``bic``, ``log_lik``, ``df_residual``,
    ``df_null``, ``nobs``. ``mse``/``rmse`` use response-scale residuals.

    ``mse`` is the mean squared residual using ``n`` in the denominator (so
    ``rmse = sqrt(mse)``); for OLS ``sigma`` is the residual standard error
    using ``n - p`` — matching the R package.
    """
    _check_model(model)
    nobs = int(model.nobs)

    if _is_glm(model):
        res = np.asarray(model.resid_response, dtype=float)
        mse = float(np.mean(res**2))
        table = pl.DataFrame(
            {
                "mse": [mse],
                "rmse": [float(np.sqrt(mse))],
                "deviance": [float(model.deviance)],
                "null_deviance": [float(model.null_deviance)],
                "aic": [float(model.aic)],
                "bic": [float(model.bic_llf)],
                "log_lik": [float(model.llf)],
                "df_residual": [int(model.df_resid)],
                "df_null": [nobs - 1],
                "nobs": [nobs],
            }
        )
    else:
        mse = float(model.ssr) / nobs
        table = pl.DataFrame(
            {
                "r_squared": [float(model.rsquared)],
                "adj_r_squared": [float(model.rsquared_adj)],
                "mse": [mse],
                "rmse": [float(np.sqrt(mse))],
                "sigma": [float(np.sqrt(model.mse_resid))],
                "statistic": [float(model.fvalue)],
                "p_value": [float(model.f_pvalue)],
                "df": [int(model.df_model)],
                "nobs": [nobs],
            }
        )
    float_cols = [c for c, dt in table.schema.items() if dt.is_float()]
    return table.with_columns(pl.col(float_cols).round(digits))


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


def count_missing(data, columns: list[str] | None = None) -> pl.DataFrame:
    """Count missing (``null``) values in each column.

    A beginner-friendly alternative to ``df.select(pl.all().is_null().sum())``:
    it returns a tidy two-column data frame with one row per column
    (``column``, ``n_missing``), sorted from most to fewest missing values so the
    columns needing attention surface first.

    Parameters
    ----------
    data:
        A polars (or pandas) data frame.
    columns:
        Optional list of column names to check; defaults to every column.
    """
    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
    columns = columns or df.columns
    return pl.DataFrame(
        {
            "column": columns,
            "n_missing": [int(df[col].null_count()) for col in columns],
        },
        schema={"column": pl.Utf8, "n_missing": pl.Int64},
    ).sort("n_missing", descending=True)
