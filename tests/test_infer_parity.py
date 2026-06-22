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
    # prop_test now defaults to the chi-square statistic (R parity); ask for the
    # z explicitly. Without continuity correction, chi-square == z**2.
    z = float(
        prop_test(
            yawn,
            formula="yawn ~ group",
            success="yes",
            order=("seed", "control"),
            z=True,
            correct=False,
        )["statistic"][0]
    )
    assert chi == pytest.approx(z**2, rel=1e-6)


def test_anova_f_equals_two_sample_t_squared():
    movies = md.load_movies_sample()
    f = float(specify(movies, formula="rating ~ genre").calculate(stat="F"))
    # ANOVA F equals the pooled two-sample t, squared (for two groups).
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

    assert float(th.get_p_value(2.0, "right")["p_value"][0]) == pytest.approx(
        float(tdist.sf(2.0, 10))
    )


def test_assume_visualize_builds():
    import matplotlib

    matplotlib.use("Agg")
    # plotly default returns a go.Figure-wrapping InferPlot; plotnine returns a ggplot.
    import plotly.graph_objects as go
    from plotnine import ggplot

    assert isinstance(assume("z").visualize().figure, go.Figure)
    assert isinstance(assume("z").visualize(engine="plotnine").gg, ggplot)


# --- wrappers -------------------------------------------------------------


def test_t_test_one_sample_tidy_columns():
    age = md.load_age_at_marriage()
    out = t_test(age, response="age", mu=23, alternative="greater")
    assert {"statistic", "t_df", "p_value", "estimate", "lower_ci", "upper_ci"} <= set(out.columns)


def test_chisq_test_df_and_stat():
    # Default is the uncorrected Pearson statistic (matches moderndive 0.1.0 and
    # the simulation-based calculate(stat="Chisq")) — strictly positive here.
    out = chisq_test(_yawn(), formula="yawn ~ group")
    assert out["chisq_df"][0] == 1
    assert out["statistic"][0] > 0
    # Opt into Yates' continuity correction (R's chisq.test default); on this weak
    # 2x2 association the corrected statistic is smaller.
    corrected = chisq_test(_yawn(), formula="yawn ~ group", correct=True)
    assert corrected["statistic"][0] < out["statistic"][0]


# --- bias-corrected CI ----------------------------------------------------


def test_bias_corrected_ci_brackets_estimate():
    age = md.load_age_at_marriage()
    est = float(observe(age, response="age", stat="mean"))
    boot = (
        specify(age, response="age")
        .generate(reps=500, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
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


# --- chi-square goodness-of-fit -------------------------------------------


def _finrela_uniform_p():
    gss = md.load_gss()
    levels = gss["finrela"].unique().to_list()
    return gss, {lvl: 1 / len(levels) for lvl in levels}


def test_gof_observed_matches_scipy():
    from scipy.stats import chisquare

    gss, p = _finrela_uniform_p()
    obs = float(
        specify(gss, response="finrela").hypothesize(null="point", p=p).calculate(stat="Chisq")
    )
    counts = gss.select("finrela").drop_nulls()["finrela"].value_counts()
    observed = {r["finrela"]: r["count"] for r in counts.iter_rows(named=True)}
    total = sum(observed.values())
    f_obs = [observed[lvl] for lvl in p]
    f_exp = [total * prob for prob in p.values()]
    assert obs == pytest.approx(float(chisquare(f_obs, f_exp).statistic))


def test_gof_null_distribution_and_pvalue():
    gss, p = _finrela_uniform_p()
    obs = specify(gss, response="finrela").hypothesize(null="point", p=p).calculate(stat="Chisq")
    null = (
        specify(gss, response="finrela")
        .hypothesize(null="point", p=p)
        .generate(reps=300, type="draw", seed=1)
        .calculate(stat="Chisq")
    )
    assert null.data.height == 300
    pv = float(md.get_p_value(null, obs_stat=obs, direction="greater")["p_value"][0])
    assert 0.0 <= pv <= 1.0
    # "simulate" alias produces the same draws as "draw"
    null2 = (
        specify(gss, response="finrela")
        .hypothesize(null="point", p=p)
        .generate(reps=300, type="simulate", seed=1)
        .calculate(stat="Chisq")
    )
    assert null.data["stat"].to_list() == null2.data["stat"].to_list()


def test_gof_chisq_test_wrapper_matches_pipeline():
    gss, p = _finrela_uniform_p()
    obs = float(
        specify(gss, response="finrela").hypothesize(null="point", p=p).calculate(stat="Chisq")
    )
    out = chisq_test(gss, response="finrela", p=p)
    assert out["statistic"][0] == pytest.approx(obs)
    assert out["chisq_df"][0] == len(p) - 1
    assert md.chisq_stat(gss, response="finrela", p=p) == pytest.approx(obs)


def test_gof_observe_shortcut():
    gss, p = _finrela_uniform_p()
    val = observe(gss, response="finrela", stat="Chisq", null="point", p=p)
    assert float(val) > 0


def test_gof_errors():
    from moderndive.infer.statistics import compute_statistic

    gss, p = _finrela_uniform_p()
    # univariate Chisq without a p dict
    with pytest.raises(ValueError, match="goodness-of-fit"):
        specify(gss, response="finrela").calculate(stat="Chisq")
    # p must sum to 1
    resp = gss["finrela"].drop_nulls().to_numpy()
    bad = dict(p)
    first = next(iter(bad))
    bad[first] = bad[first] + 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        compute_statistic(resp, None, "Chisq", p=bad)
    # p missing a level present in the data
    missing = {k: v for k, v in list(p.items())[:-1]}
    total = sum(missing.values())
    missing = {k: v / total for k, v in missing.items()}
    with pytest.raises(ValueError, match="missing a probability"):
        compute_statistic(resp, None, "Chisq", p=missing)
    # wrapper needs explanatory or p
    with pytest.raises(ValueError, match="goodness-of-fit"):
        chisq_test(gss, response="finrela")
