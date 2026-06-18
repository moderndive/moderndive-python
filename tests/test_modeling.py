"""Tests for regression + summary helpers (statsmodels-backed)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.formula.api as smf

from moderndive import get_regression_points, get_regression_table, tidy_summary


def _linear_frame():
    x = np.arange(20, dtype=float)
    # y = 3x + 5 with small deterministic wiggle so std errors are finite
    y = 3.0 * x + 5.0 + np.sin(x)
    return pl.DataFrame({"x": x, "y": y})


def test_regression_table_columns_and_estimates():
    df = _linear_frame()
    model = smf.ols("y ~ x", data=df.to_pandas()).fit()
    table = get_regression_table(model)
    assert table.columns == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "lower_ci",
        "upper_ci",
    ]
    assert table["term"].to_list() == ["intercept", "x"]
    slope = table.filter(pl.col("term") == "x")["estimate"][0]
    assert slope == pytest.approx(3.0, abs=0.05)


def test_regression_points_columns():
    df = _linear_frame()
    model = smf.ols("y ~ x", data=df.to_pandas()).fit()
    pts = get_regression_points(model)
    assert pts.columns == ["ID", "y", "x", "y_hat", "residual"]
    assert pts.height == df.height
    # residual = y - y_hat (each column independently rounded to 3 digits)
    resid = (pts["y"] - pts["y_hat"]).to_numpy()
    assert np.allclose(resid, pts["residual"].to_numpy(), atol=2e-3)


def test_tidy_summary_layout_and_values():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
    out = tidy_summary(df)
    assert out.columns == [
        "column",
        "n",
        "group",
        "type",
        "min",
        "Q1",
        "mean",
        "median",
        "Q3",
        "max",
        "sd",
    ]
    a_row = out.filter(pl.col("column") == "a")
    assert a_row["type"][0] == "numeric"
    assert a_row["mean"][0] == pytest.approx(2.5)
    assert a_row["min"][0] == pytest.approx(1.0)
    assert a_row["max"][0] == pytest.approx(4.0)
    b_row = out.filter(pl.col("column") == "b")
    assert b_row["type"][0] == "categorical"
    assert b_row["n"][0] == 4
    assert b_row["mean"][0] is None
