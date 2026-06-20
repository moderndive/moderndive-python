"""Tests for the PR #144 parity additions:

- get_correlation() multiple predictors (long/wide) + quiet message
- glm support + exponentiate in the regression helpers
- in-formula transformation handling in get_regression_points()
- plot_3d_regression()
- View() via itables
plus the beginner-friendly messaging helpers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

import moderndive as md
from moderndive import (
    View,
    get_correlation,
    get_regression_points,
    get_regression_summaries,
    get_regression_table,
    plot_3d_regression,
)
from moderndive._messaging import ModernDiveMessage


@pytest.fixture(scope="module")
def mtcars():
    return sm.datasets.get_rdataset("mtcars").data


@pytest.fixture(scope="module")
def un():
    return md.load_un_member_states_2024()


# ============================ get_correlation ============================


def test_get_correlation_single_unchanged(un):
    out = get_correlation(un, "life_expectancy_2022 ~ gdp_per_capita")
    assert out.columns == ["cor"] and out.height == 1
    # x=/y= form matches the formula form
    by_kw = get_correlation(un, x="gdp_per_capita", y="life_expectancy_2022")
    assert out.item() == pytest.approx(by_kw.item())


def test_get_correlation_multi_long_and_quiet(un):
    f = "life_expectancy_2022 ~ gdp_per_capita + fertility_rate_2022"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        long = get_correlation(un, f)
    assert long.columns == ["predictor", "cor"]
    assert long["predictor"].to_list() == ["gdp_per_capita", "fertility_rate_2022"]
    assert any(isinstance(w.message, ModernDiveMessage) for w in caught)
    # quiet=True silences the message
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        get_correlation(un, f, quiet=True)
    assert not any(isinstance(w.message, ModernDiveMessage) for w in caught2)


def test_get_correlation_multi_wide(un):
    f = "life_expectancy_2022 ~ gdp_per_capita + fertility_rate_2022"
    wide = get_correlation(un, f, wide=True, quiet=True)
    assert wide.columns == ["gdp_per_capita", "fertility_rate_2022"]
    assert wide.height == 1
    # wide values equal the long values
    long = get_correlation(un, f, quiet=True)
    assert wide["gdp_per_capita"].item() == pytest.approx(
        long.filter(pl.col("predictor") == "gdp_per_capita")["cor"].item()
    )


def test_get_correlation_errors(un):
    with pytest.raises(ValueError, match="Provide a formula"):
        get_correlation(un)
    with pytest.raises(ValueError, match="not both"):
        get_correlation(un, "a ~ b", x="b", y="a")
    with pytest.raises(ValueError, match="look like"):
        get_correlation(un, "no tilde here")
    with pytest.raises(ValueError, match="at least one predictor"):
        get_correlation(un, "life_expectancy_2022 ~ ")
    with pytest.raises(ValueError, match="not found"):
        get_correlation(un, "life_expectancy_2022 ~ nope")


def test_get_correlation_accepts_pandas(un):
    out = get_correlation(un.to_pandas(), "life_expectancy_2022 ~ gdp_per_capita")
    assert out.columns == ["cor"]


# ============================ glm support ================================


@pytest.fixture(scope="module")
def glm_model(mtcars):
    return smf.glm("am ~ hp + wt", mtcars, family=sm.families.Binomial()).fit()


def test_glm_regression_table_plain_and_exponentiated(glm_model):
    plain = get_regression_table(glm_model)
    exp = get_regression_table(glm_model, exponentiate=True)
    assert plain.columns == [
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "lower_ci",
        "upper_ci",
    ]
    # exponentiate transforms estimate + CI but leaves std_error/statistic/p alone
    hp_plain = plain.filter(pl.col("term") == "hp")
    hp_exp = exp.filter(pl.col("term") == "hp")
    assert hp_exp["estimate"].item() == pytest.approx(np.exp(hp_plain["estimate"].item()), rel=1e-3)
    assert hp_exp["std_error"].item() == pytest.approx(hp_plain["std_error"].item())


def test_glm_summaries_shape(glm_model):
    out = get_regression_summaries(glm_model)
    assert out.columns == [
        "mse",
        "rmse",
        "deviance",
        "null_deviance",
        "aic",
        "bic",
        "log_lik",
        "df_residual",
        "df_null",
        "nobs",
    ]
    # no R^2 columns for a glm
    assert "r_squared" not in out.columns
    assert out["nobs"].item() == int(glm_model.nobs)


def test_glm_points_response_scale(glm_model):
    pts = get_regression_points(glm_model)
    assert pts.columns == ["ID", "am", "hp", "wt", "am_hat", "residual"]
    # fitted values are probabilities in [0, 1]; residual = y - p_hat
    assert pts["am_hat"].min() >= 0.0 and pts["am_hat"].max() <= 1.0
    row = pts.row(0, named=True)
    assert row["residual"] == pytest.approx(row["am"] - row["am_hat"], abs=1e-3)


# ============================ transformations ============================


def test_points_transformed_lhs(mtcars):
    model = smf.ols("np.log(mpg) ~ hp", mtcars).fit()
    pts = get_regression_points(model)
    assert pts.columns == ["ID", "log_mpg", "hp", "log_mpg_hat", "residual"]
    # outcome is on the log scale
    assert pts["log_mpg"].max() < 5  # log(mpg) ~ 2-3.5, never raw mpg


def test_points_transformed_rhs_shows_original_columns(mtcars):
    model = smf.ols("mpg ~ np.power(hp, 2) + wt", mtcars).fit()
    pts = get_regression_points(model)
    # original predictor columns, not poly/basis columns
    assert pts.columns == ["ID", "mpg", "hp", "wt", "mpg_hat", "residual"]


def test_points_ols_basic_and_ids(mtcars):
    model = smf.ols("mpg ~ wt", mtcars).fit()
    pts = get_regression_points(model)
    assert pts.columns == ["ID", "mpg", "wt", "mpg_hat", "residual"]
    assert pts["ID"].to_list() == list(range(1, pts.height + 1))


def test_regression_points_array_api_numpy():
    # Array-API numpy fit: no formula/data frame, just the design matrix.
    X = sm.add_constant(np.column_stack([np.arange(1.0, 7), np.array([2.0, 1, 4, 3, 6, 5])]))
    y = np.array([2.0, 4, 5, 4, 6, 7])
    model = sm.OLS(y, X).fit()
    pts = get_regression_points(model)
    # constant column dropped; design columns become predictors; "y" is the outcome
    assert pts.columns == ["ID", "y", "x1", "x2", "y_hat", "residual"]
    assert pts.height == 6
    # table also works on the bare-array fit (params has no index)
    tbl = get_regression_table(model)
    assert tbl["term"].to_list() == ["intercept", "x1", "x2"]


def test_regression_points_array_api_pandas_and_glm():
    # Pandas array-API keeps the named columns.
    Xp = sm.add_constant(pd.DataFrame({"wt": [1.0, 2, 3, 4, 5, 6], "hp": [2.0, 1, 4, 3, 6, 5]}))
    yp = pd.Series([2.0, 4, 5, 4, 6, 7], name="mpg")
    pts = get_regression_points(sm.OLS(yp, Xp).fit())
    assert pts.columns == ["ID", "mpg", "wt", "hp", "mpg_hat", "residual"]
    # Array-API GLM: fitted values on the response scale.
    am = pd.Series([0.0, 1, 0, 1, 1, 0], name="am")
    glm = sm.GLM(am, Xp, family=sm.families.Binomial()).fit()
    gpts = get_regression_points(glm)
    assert gpts.columns == ["ID", "am", "wt", "hp", "am_hat", "residual"]
    assert gpts["am_hat"].min() >= 0.0 and gpts["am_hat"].max() <= 1.0


def test_regression_helpers_reject_non_models():
    for fn in (get_regression_table, get_regression_points, get_regression_summaries):
        with pytest.raises(TypeError, match="fitted statsmodels"):
            fn("not a model")


# ============================ plot_3d_regression =========================


def test_plot_3d_regression_builds(un):
    fig = plot_3d_regression(un, "life_expectancy_2022 ~ gdp_per_capita + fertility_rate_2022")
    assert isinstance(fig, go.Figure)
    assert [t.type for t in fig.data] == ["scatter3d", "surface"]


def test_plot_3d_regression_accepts_pandas_and_n(un):
    fig = plot_3d_regression(
        un.to_pandas(), "life_expectancy_2022 ~ gdp_per_capita + fertility_rate_2022", n=10
    )
    # surface grid honors n
    assert np.asarray(fig.data[1].z).shape == (10, 10)


def test_plot_3d_regression_errors(un):
    f2 = "life_expectancy_2022 ~ gdp_per_capita + fertility_rate_2022"
    with pytest.raises(ValueError, match="integer"):
        plot_3d_regression(un, f2, n=1)
    with pytest.raises(ValueError, match="look like"):
        plot_3d_regression(un, "no tilde")
    with pytest.raises(ValueError, match="transformations"):
        plot_3d_regression(
            un, "np.log(life_expectancy_2022) ~ gdp_per_capita + fertility_rate_2022"
        )
    with pytest.raises(ValueError, match="exactly two"):
        plot_3d_regression(un, "life_expectancy_2022 ~ gdp_per_capita")
    with pytest.raises(ValueError, match="not found"):
        plot_3d_regression(un, "life_expectancy_2022 ~ gdp_per_capita + nope")


def test_plot_3d_regression_rejects_non_numeric(un):
    with pytest.raises(ValueError, match="must be numeric"):
        plot_3d_regression(un, "life_expectancy_2022 ~ gdp_per_capita + continent")


# ============================ View() =====================================


def test_view_renders_with_itables(un, monkeypatch):
    calls = {}

    def fake_render(frame, title):
        calls["title"] = title
        return "WIDGET"

    monkeypatch.setattr(md.view, "_render_datatable", fake_render)
    out = View(un.head(3), title="UN")
    assert out == "WIDGET" and calls["title"] == "UN"


def test_view_coerces_non_frame(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        md.view, "_render_datatable", lambda frame, title: captured.setdefault("f", frame)
    )
    View({"a": [1, 2], "b": [3, 4]})
    assert isinstance(captured["f"], pl.DataFrame)


def test_view_accepts_pandas(monkeypatch):
    import pandas as pd

    captured = {}
    monkeypatch.setattr(
        md.view, "_render_datatable", lambda frame, title: captured.setdefault("f", frame)
    )
    View(pd.DataFrame({"a": [1]}))
    assert isinstance(captured["f"], pd.DataFrame)


def test_view_fallback_without_itables(un, monkeypatch):
    monkeypatch.setattr(md.view, "_itables_available", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = View(un.head(2))
    assert isinstance(out, pl.DataFrame)
    assert any(isinstance(w.message, ModernDiveMessage) for w in caught)


# ============================ messaging ==================================


def test_messaging_helpers_format():
    from moderndive._messaging import helpful_error, inform

    msg = helpful_error("Something went wrong.", "do this", "or that")
    assert msg.startswith("Something went wrong.")
    assert "  → do this" in msg and "  → or that" in msg
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inform("Heads up.", "a hint")
    assert isinstance(caught[0].message, ModernDiveMessage)
    assert "  → a hint" in str(caught[0].message)
