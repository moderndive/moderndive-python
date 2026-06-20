"""Tests for the moderndive R argument-parity additions:

- get_correlation(method=, na_rm=)
- get_regression_points(newdata=, ID=)
- get_regression_table(default_categorical_levels=)
- gg_parallel_slopes(alpha=)
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
import statsmodels.formula.api as smf
from plotnine import ggplot
from scipy import stats

import moderndive as md
from moderndive import get_correlation, get_regression_points, get_regression_table


@pytest.fixture(scope="module")
def houses():
    return md.load_saratoga_houses()


# ============================ get_correlation ============================


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_get_correlation_methods_match_scipy(houses, method):
    x = houses["living_area"].to_numpy()
    y = houses["price"].to_numpy()
    got = get_correlation(houses, "price ~ living_area", method=method).item()
    if method == "pearson":
        expected = float(np.corrcoef(x, y)[0, 1])
    elif method == "spearman":
        expected = float(stats.spearmanr(x, y).statistic)
    else:
        expected = float(stats.kendalltau(x, y).statistic)
    assert got == pytest.approx(expected)


def test_get_correlation_method_applies_to_multi(houses):
    out = get_correlation(houses, "price ~ living_area + bedrooms", method="spearman", quiet=True)
    expected = float(
        stats.spearmanr(houses["bedrooms"].to_numpy(), houses["price"].to_numpy()).statistic
    )
    assert out.filter(pl.col("predictor") == "bedrooms")["cor"].item() == pytest.approx(expected)


def test_get_correlation_bad_method(houses):
    with pytest.raises(ValueError, match="method must be one of"):
        get_correlation(houses, "price ~ living_area", method="bogus")


def test_get_correlation_na_rm_false_yields_nan():
    df = pl.DataFrame({"y": [1.0, 2.0, None, 4.0], "x": [1.0, 2.0, 3.0, 4.0]})
    # default drops nulls and gives a real number
    assert math.isfinite(get_correlation(df, "y ~ x").item())
    # na_rm=False keeps the null → nan
    assert math.isnan(get_correlation(df, "y ~ x", na_rm=False).item())


# ============================ get_regression_points =====================


@pytest.fixture(scope="module")
def fit_train_test(houses):
    train = houses.head(800)
    test = houses.tail(200)
    model = smf.ols("price ~ living_area + bedrooms", data=train.to_pandas()).fit()
    return model, test


def test_points_newdata_with_outcome(fit_train_test):
    model, test = fit_train_test
    out = get_regression_points(model, newdata=test)
    assert out.columns == ["ID", "price", "living_area", "bedrooms", "price_hat", "residual"]
    assert out.height == test.height


def test_points_newdata_without_outcome(fit_train_test):
    model, test = fit_train_test
    out = get_regression_points(model, newdata=test.drop("price"))
    # no outcome → predictions only, no residual
    assert out.columns == ["ID", "living_area", "bedrooms", "price_hat"]


def test_points_newdata_with_id(fit_train_test):
    model, test = fit_train_test
    test_id = test.with_row_index("house_id")
    out = get_regression_points(model, newdata=test_id, ID="house_id")
    assert out.columns[0] == "house_id"


def test_points_newdata_missing_predictor(fit_train_test):
    model, test = fit_train_test
    with pytest.raises(ValueError, match="missing predictor"):
        get_regression_points(model, newdata=test.select("price"))


def test_points_newdata_bad_id(fit_train_test):
    model, test = fit_train_test
    with pytest.raises(ValueError, match="ID column"):
        get_regression_points(model, newdata=test, ID="nope")


def test_points_id_from_source_column(houses):
    h = houses.with_row_index("home_id")
    model = smf.ols("price ~ living_area", data=h.to_pandas()).fit()
    out = get_regression_points(model, ID="home_id")
    assert out.columns[0] == "home_id"
    assert out.height == h.height


def test_points_id_not_in_source(houses):
    model = smf.ols("price ~ living_area", data=houses.to_pandas()).fit()
    with pytest.raises(ValueError, match="ID column"):
        get_regression_points(model, ID="nope")


def test_points_id_requires_formula_model():
    import statsmodels.api as sm

    X = sm.add_constant(np.arange(1.0, 11.0))
    model = sm.OLS(np.arange(1.0, 11.0) * 2, X).fit()
    with pytest.raises(TypeError, match="formula-API"):
        get_regression_points(model, ID="whatever")


# ============================ get_regression_table ======================


def test_default_categorical_levels(houses):
    un = md.load_un_member_states_2024().to_pandas()
    model = smf.ols("life_expectancy_2022 ~ gdp_per_capita + C(continent)", data=un).fit()
    pretty = get_regression_table(model)["term"].to_list()
    raw = get_regression_table(model, default_categorical_levels=True)["term"].to_list()
    assert any(t.startswith("continent: ") for t in pretty)
    assert any("C(continent)[T." in t for t in raw)
    # intercept is mapped in both
    assert "intercept" in pretty and "intercept" in raw


# ============================ gg_parallel_slopes alpha ==================


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
def test_gg_parallel_slopes_alpha(engine):
    evals = md.load_evals()
    fig = md.gg_parallel_slopes(
        evals, response="score", explanatory="age", by="gender", alpha=0.3, engine=engine
    )
    assert isinstance(fig, go.Figure if engine == "plotly" else ggplot)
