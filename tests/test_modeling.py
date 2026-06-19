"""Tests for regression + summary helpers (statsmodels-backed)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.formula.api as smf

from moderndive import (
    count_missing,
    get_regression_points,
    get_regression_table,
    tidy_summary,
)


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


def test_count_missing_counts_and_sorts():
    df = pl.DataFrame(
        {
            "a": [1, None, 3, None],  # 2 missing
            "b": [None, None, None, 4],  # 3 missing
            "c": [1, 2, 3, 4],  # 0 missing
        }
    )
    out = count_missing(df)
    assert out.columns == ["column", "n_missing"]
    # sorted most-missing first
    assert out["column"].to_list() == ["b", "a", "c"]
    assert out["n_missing"].to_list() == [3, 2, 0]


def test_count_missing_columns_subset_and_pandas_input():
    df = pl.DataFrame({"a": [1, None], "b": [None, None]})
    sub = count_missing(df, columns=["a"])
    assert sub["column"].to_list() == ["a"] and sub["n_missing"].to_list() == [1]
    # pandas input is accepted too
    out = count_missing(df.to_pandas())
    assert dict(zip(out["column"], out["n_missing"], strict=True)) == {"a": 1, "b": 2}
