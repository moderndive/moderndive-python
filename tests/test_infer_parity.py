"""Tests for full infer parity: new stats, observe, assume, wrappers, aliases, paired."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import moderndive as md
from moderndive import assume, chisq_test, observe, prop_test, specify, t_test


def _yawn():
    return md.load_mythbusters_yawn()


# --- new calculate stats --------------------------------------------------

def test_count_and_prop_consistent():
    df = pl.DataFrame({"y": ["s"] * 7 + ["f"] * 3})
    spec = specify(df, response="y", success="s")
    assert float(spec.calculate(stat="count")) == 7.0
    assert float(spec.calculate(stat="prop")) == pytest.approx(0.7)


def test_ratio_and_odds_props_manual():
    # group a: 2/4 success (p=.5); group b: 1/4 (p=.25)
    df = pl.DataFrame(
        {
            "y": ["s", "s", "f", "f", "s", "f", "f", "f"],
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    spec = specify(df, formula="y ~ g", success="s")
    assert float(spec.calculate(stat="ratio of props", order=("a", "b"))) == pytest.approx(2.0)
    odds = spec.calculate(stat="odds ratio", order=("a", "b"))
    assert float(odds) == pytest.approx((0.5 / 0.5) / (0.25 / 0.75))


def test_chisq_equals_prop_test_z_squared():
    yawn = _yawn()
    chi = float(specify(yawn, formula="yawn ~ group").calculate(stat="Chisq"))
    z = float(prop_test(yawn, formula="yawn ~ group", success="yes", order=("seed", "control"))["statistic"][0])
    assert chi == pytest.approx(z**2, rel=1e-6)


def test_anova_f_equals_two_sample_t_squared():
    movies = md.load_movies_sample()
    f = float(specify(movies, formula="rating ~ genre").calculate(stat="F"))
    t = float(t_test(movies, formula="rating ~ genre", order=("Action", "Romance"))["statistic"][0])
    # Welch t differs slightly from pooled; use pooled via equal_var for the identity
    from scipy import stats as st
    a = movies.filter(pl.col("genre") == "Action")["rating"].to_numpy()
    b = movies.filter(pl.col("genre") == "Romance")["rating"].to_numpy()
    t_pooled = st.ttest_ind(a, b, equal_var=True).statistic
    assert f == pytest.approx(t_pooled**2, rel=1e-6)


def test_one_sample_t_needs_mu():
    df = pl.DataFrame({"x": [1.0, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        specify(df, response="x").calculate(stat="t")
    # with mu via observe it works
    val = observe(df, response="x", stat="t", null="point", mu=2.0)
    assert float(val) == pytest.approx((3.0 - 2.0) / (df["x"].std() / np.sqrt(5)))


def test_custom_stat_callable():
    df = pl.DataFrame({"x": [1.0, 2, 3, 4]})
    val = specify(df, response="x").calculate(stat=lambda r, e: float(r.max() - r.min()))
    assert float(val) == 3.0


# --- observe --------------------------------------------------------------

def test_observe_matches_pipeline():
    df = pl.DataFrame({"x": [2.0, 4, 6]})
    assert float(observe(df, response="x", stat="mean")) == float(
        specify(df, response="x").calculate(stat="mean")
    )


# --- assume (theoretical) -------------------------------------------------

def test_assume_t_pvalue_matches_scipy():
    th = assume("t", df=10)
    from scipy.stats import t as tdist

    assert float(th.get_p_value(2.0, "right")["p_value"][0]) == pytest.approx(float(tdist.sf(2.0, 10)))


def test_assume_visualize_builds():
    import matplotlib

    matplotlib.use("Agg")
    p = assume("z").visualize()
    from plotnine import ggplot

    assert isinstance(p, ggplot)


# --- wrappers -------------------------------------------------------------

def test_t_test_one_sample_tidy_columns():
    age = md.load_age_at_marriage()
    out = t_test(age, response="age", mu=23, alternative="greater")
    assert {"statistic", "t_df", "p_value", "estimate", "lower_ci", "upper_ci"} <= set(out.columns)


def test_chisq_test_df_and_stat():
    out = chisq_test(_yawn(), formula="yawn ~ group")
    assert out["chisq_df"][0] == 1
    assert out["statistic"][0] > 0


# --- bias-corrected CI ----------------------------------------------------

def test_bias_corrected_ci_brackets_estimate():
    age = md.load_age_at_marriage()
    est = float(observe(age, response="age", stat="mean"))
    boot = specify(age, response="age").generate(reps=500, type="bootstrap", seed=1).calculate(stat="mean")
    ci = boot.get_confidence_interval(level=0.95, type="bias-corrected", point_estimate=est)
    assert float(ci["lower_ci"][0]) < est < float(ci["upper_ci"][0])


# --- paired null ----------------------------------------------------------

def test_paired_null_centers_near_zero():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"diff": rng.normal(loc=2.0, size=40)})
    null = (
        specify(df, response="diff")
        .hypothesize(null="paired independence")
        .generate(reps=500, type="permute", seed=1)
        .calculate(stat="mean")
    )
    assert abs(float(null.stats.mean())) < 0.3  # centered at 0 despite observed mean 2


# --- aliases --------------------------------------------------------------

def test_aliases_point_to_canonical():
    assert md.get_pvalue is md.get_p_value
    assert md.get_ci is md.get_confidence_interval
    assert md.visualise is md.visualize
    assert md.shade_pvalue is md.shade_p_value
