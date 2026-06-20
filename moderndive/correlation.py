"""Correlation and population-spread helpers mirroring the R ``moderndive`` package.

- :func:`get_correlation` ~ ``moderndive::get_correlation`` (one or more predictors)
- :func:`pop_sd`          ~ ``moderndive::pop_sd`` (population standard deviation)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._messaging import helpful_error, inform

__all__ = ["get_correlation", "pop_sd"]


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Resolve ``"y ~ x1 + x2"`` into the outcome name and a list of predictors."""
    if "~" not in formula:
        raise ValueError(
            helpful_error(
                f"formula must look like 'y ~ x' (or 'y ~ x1 + x2'), got {formula!r}.",
                "Put the outcome on the left of ~ and one or more predictors on the right.",
            )
        )
    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    predictors = [v.strip() for v in rhs.split("+") if v.strip()]
    if not lhs or not predictors:
        raise ValueError(
            helpful_error(
                f"formula must name an outcome and at least one predictor, got {formula!r}.",
                "Example: 'mpg ~ wt' or 'mpg ~ wt + hp'.",
            )
        )
    return lhs, predictors


def get_correlation(
    data,
    formula: str | None = None,
    *,
    x: str | None = None,
    y: str | None = None,
    wide: bool = False,
    quiet: bool = False,
) -> pl.DataFrame:
    """Pearson correlation between an outcome and one or more predictors.

    Mirrors ``moderndive::get_correlation``. Give the variables either as a
    formula (``"y ~ x"`` or ``"y ~ x1 + x2 + x3"``) or, for a single predictor,
    via ``x=`` and ``y=``.

    With **one** predictor the result is a 1-row frame with a ``cor`` column.
    With **multiple** predictors the result is long by default — columns
    ``predictor`` and ``cor`` (one row each) — or pass ``wide=True`` for one
    column per predictor. Rows with a null in either column are dropped per pair.

    A short note points to a full pairwise correlation matrix when there are
    multiple predictors; silence it with ``quiet=True``.
    """
    df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)

    if formula is not None:
        if x is not None or y is not None:
            raise ValueError(
                helpful_error(
                    "Pass either a formula or x=/y=, not both.",
                    "Use a formula ('y ~ x') for one or more predictors, or x=/y= for one.",
                )
            )
        outcome, predictors = _parse_formula(formula)
    else:
        if x is None or y is None:
            raise ValueError(
                helpful_error(
                    "Provide a formula ('y ~ x') or both x= and y=.",
                    "For several predictors use a formula: 'y ~ x1 + x2'.",
                )
            )
        outcome, predictors = y, [x]

    missing = [c for c in [outcome, *predictors] if c not in df.columns]
    if missing:
        raise ValueError(
            helpful_error(
                f"Column(s) not found in the data: {', '.join(missing)}.",
                f"Available columns: {', '.join(df.columns)}.",
            )
        )

    cors: dict[str, float] = {}
    for predictor in predictors:
        pair = df.select(predictor, outcome).drop_nulls()
        cors[predictor] = float(
            np.corrcoef(pair[predictor].to_numpy(), pair[outcome].to_numpy())[0, 1]
        )

    if len(predictors) == 1:
        return pl.DataFrame({"cor": [cors[predictors[0]]]})

    if not quiet:
        inform(
            f"Computing correlations of `{outcome}` against {len(predictors)} predictors.",
            "For a full pairwise matrix (incl. predictor–predictor correlations), "
            "use `df.to_pandas().corr()`.",
            "Pass quiet=True to silence this message.",
        )

    if wide:
        return pl.DataFrame({predictor: [cors[predictor]] for predictor in predictors})
    return pl.DataFrame(
        {"predictor": predictors, "cor": [cors[predictor] for predictor in predictors]}
    )


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
