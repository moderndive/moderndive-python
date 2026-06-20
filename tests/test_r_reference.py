"""Cross-package validation: assert the Python results equal R's, value for value.

Each expected value below was produced by the R ``moderndive`` / ``infer``
packages on the *same fixed dataset* defined here (so the check doesn't depend on
dataset-column parity or R being installed at test time). If an implementation
drifts from R, this fails. New-argument R validations (correlation methods,
prop_test vs prop.test, GOF vs scipy) live alongside their feature tests; this
module pins the core functions.
"""

from __future__ import annotations

import polars as pl
import pytest
import statsmodels.formula.api as smf

import moderndive as md

# A fixed dataset, identical to the one used in R to compute the constants below.
_DF = pl.DataFrame(
    {
        "y": [2.0, 4, 5, 4, 6, 7, 9, 8, 7, 10],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "z": [3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3],
    }
)


@pytest.fixture(scope="module")
def model():
    return smf.ols("y ~ x + z", data=_DF.to_pandas()).fit()


def test_get_regression_table_matches_r(model):
    # R: moderndive::get_regression_table(lm(y ~ x + z, df))
    tbl = md.get_regression_table(model)
    assert tbl["term"].to_list() == ["intercept", "x", "z"]
    expected = {
        "estimate": [2.102, 0.755, -0.015],
        "std_error": [0.867, 0.132, 0.162],
        "statistic": [2.424, 5.705, -0.09],
        "p_value": [0.046, 0.001, 0.931],
        "lower_ci": [0.052, 0.442, -0.398],
        "upper_ci": [4.151, 1.069, 0.369],
    }
    for col, vals in expected.items():
        assert tbl[col].to_list() == pytest.approx(vals, abs=1e-3)


def test_get_regression_summaries_matches_r(model):
    # R: moderndive::get_regression_summaries(...)
    s = md.get_regression_summaries(model)
    expected = {
        "r_squared": 0.838,
        "adj_r_squared": 0.792,
        "mse": 0.899575,
        "rmse": 0.948459,
        "sigma": 1.134,
        "statistic": 18.132,
        "p_value": 0.002,
        "df": 2,
        "nobs": 10,
    }
    for col, val in expected.items():
        assert s[col][0] == pytest.approx(val, abs=1e-3)


def test_get_correlation_matches_r():
    # R: get_correlation(df, y ~ x)  ->  0.9154346
    assert md.get_correlation(_DF, "y ~ x").item() == pytest.approx(0.9154346, abs=1e-6)


def test_get_correlation_rank_methods_match_r():
    # R: cor(df$x, df$y, method = "spearman" / "kendall")
    assert md.get_correlation(_DF, "y ~ x", method="spearman").item() == pytest.approx(
        0.91465115, abs=1e-6
    )
    assert md.get_correlation(_DF, "y ~ x", method="kendall").item() == pytest.approx(
        0.79566006, abs=1e-6
    )


def test_get_regression_points_newdata_matches_r():
    # R: m <- lm(y ~ x + z, df[1:7,]); predict(m, df[8:10,])
    train, test = _DF.head(7), _DF.tail(3)
    model = smf.ols("y ~ x + z", data=train.to_pandas()).fit()
    pts = md.get_regression_points(model, newdata=test)
    assert pts["y_hat"].to_list() == pytest.approx([9.2737, 10.3158, 11.386], abs=1e-3)
    assert pts["residual"].to_list() == pytest.approx([-1.2737, -3.3158, -1.386], abs=1e-3)


def test_goodness_of_fit_matches_r():
    # R: chisq.test(c(20, 10, 30), p = rep(1/3, 3))  ->  X-squared 10, df 2, p 0.00673795
    cat = pl.DataFrame({"g": ["A"] * 20 + ["B"] * 10 + ["C"] * 30})
    p = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    obs = cat.specify(response="g").hypothesize(null="point", p=p).calculate(stat="Chisq")
    assert float(obs) == pytest.approx(10.0, abs=1e-6)
    wrapper = md.chisq_test(cat, response="g", p=p)
    assert wrapper["chisq_df"][0] == 2
    assert wrapper["p_value"][0] == pytest.approx(0.00673795, abs=1e-7)


def test_pop_sd_matches_r():
    # R: sqrt(sum((x-mean(x))^2)/length(x))  ->  2.872281
    assert md.pop_sd(_DF["x"]) == pytest.approx(2.872281, abs=1e-6)


def test_prop_test_matches_r():
    # R: prop.test(c(10, 4), c(34, 16)) — chi-square (Yates), p, and CI
    rows = [("yes", "a")] * 10 + [("no", "a")] * 24 + [("yes", "b")] * 4 + [("no", "b")] * 12
    df = pl.DataFrame({"r": [x[0] for x in rows], "g": [x[1] for x in rows]})
    out = md.prop_test(df, formula="r ~ g", success="yes", order=("a", "b"))
    assert out["statistic"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["p_value"][0] == pytest.approx(1.0)
    assert out["lower_ci"][0] == pytest.approx(-0.26168, abs=1e-4)
    assert out["upper_ci"][0] == pytest.approx(0.34991, abs=1e-4)


def test_t_test_one_sample_matches_r():
    # R: t.test(df$y, mu = 5)
    out = md.t_test(_DF, response="y", mu=5)
    assert out["statistic"][0] == pytest.approx(1.52674, abs=1e-4)
    assert out["t_df"][0] == pytest.approx(9)
    assert out["p_value"][0] == pytest.approx(0.161171, abs=1e-5)
    assert out["lower_ci"][0] == pytest.approx(4.421971, abs=1e-4)
    assert out["upper_ci"][0] == pytest.approx(7.978029, abs=1e-4)
    assert out["estimate"][0] == pytest.approx(6.2, abs=1e-6)
