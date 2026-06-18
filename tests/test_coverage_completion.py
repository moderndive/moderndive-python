"""Targeted tests to exercise the remaining branches (drives coverage to 100%).

Grouped by module; each test names the behavior it pins down.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from plotnine import ggplot

import moderndive as md
from moderndive import assume, observe, specify
from moderndive.infer import resample as _resample
from moderndive.infer.core import _parse_formula
from moderndive.infer.statistics import compute_statistic, needs_explanatory, needs_success

# ============================ core.py ====================================


def test_parse_formula_requires_tilde():
    with pytest.raises(ValueError):
        _parse_formula("y plus x")


def test_parse_formula_null_variants():
    assert _parse_formula("y ~ NULL") == ("y", None)
    assert _parse_formula("y ~ 1") == ("y", None)
    assert _parse_formula("y ~ x") == ("y", "x")


def test_specify_errors():
    df = pl.DataFrame({"y": [1.0, 2.0], "x": [3.0, 4.0]})
    with pytest.raises(ValueError):  # both formula and response
        specify(df, formula="y ~ x", response="y")
    with pytest.raises(ValueError):  # neither
        specify(df)
    with pytest.raises(ValueError):  # response missing
        specify(df, response="nope")
    with pytest.raises(ValueError):  # explanatory missing
        specify(df, response="y", explanatory="nope")


def test_hypothesize_invalid_null():
    df = pl.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        specify(df, response="y").hypothesize(null="bogus")


def test_hypothesise_alias_method():
    df = pl.DataFrame({"y": ["s", "f"], "x": ["a", "b"]})
    h = specify(df, formula="y ~ x", success="s").hypothesise(null="independence")
    assert h.null == "independence"


def test_hypothesis_generate_default_types():
    df = pl.DataFrame({"y": [1.0, 2, 3, 4], "g": ["a", "a", "b", "b"]})
    # point null -> bootstrap default
    g_point = specify(df, response="y").hypothesize(null="point", mu=2).generate(reps=3, seed=1)
    assert g_point.type == "bootstrap"
    # independence -> permute default
    g_ind = specify(df, formula="y ~ g").hypothesize(null="independence").generate(reps=3, seed=1)
    assert g_ind.type == "permute"


def test_generate_invalid_type():
    df = pl.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        specify(df, response="y").generate(reps=3, type="nonsense")


def test_fit_requires_bootstrap_or_permute_for_generated():
    # draw-type generated has no fit() path
    df = pl.DataFrame({"y": ["s"] * 5 + ["f"] * 5})
    gen = (
        specify(df, response="y", success="s")
        .hypothesize(null="point", p=0.5)
        .generate(reps=3, type="draw", seed=1)
    )
    with pytest.raises(ValueError):
        gen.fit()


def test_assume_method_on_specification():
    df = pl.DataFrame({"y": [1.0, 2, 3]})
    th = specify(df, response="y").assume("t", df=2)
    assert th.distribution == "t"


def test_observed_statistic_dunders():
    df = pl.DataFrame({"y": [2.0, 4.0]})
    obs = specify(df, response="y").calculate(stat="mean")
    assert float(obs) == 3.0
    assert obs.to_frame()["stat"][0] == 3.0
    assert "ObservedStatistic" in repr(obs)


def test_distribution_aliases():
    df = pl.DataFrame({"y": np.arange(20.0)})
    boot = (
        specify(df, response="y").generate(reps=50, type="bootstrap", seed=1).calculate(stat="mean")
    )
    assert "lower_ci" in boot.get_ci(level=0.9).columns
    # get_pvalue alias + visualise alias
    assert "p_value" in boot.get_pvalue(obs_stat=10.0, direction="two-sided").columns
    assert isinstance(boot.visualise(), ggplot)


def test_fitresult_display_and_helpers():
    sar = md.load_saratoga_houses()
    obs = specify(sar, formula="price ~ living_area").fit()
    assert obs.head(1).height == 1
    assert obs._repr_html_().startswith("<")
    assert "term" in repr(obs)
    assert not obs.is_distribution
    boot = (
        specify(sar, formula="price ~ living_area")
        .generate(reps=20, type="bootstrap", seed=1)
        .fit()
    )
    assert boot.is_distribution
    assert isinstance(boot.visualize(), ggplot)


def test_calculate_with_callable_labels_stat():
    df = pl.DataFrame({"y": [1.0, 2, 3]})
    boot = (
        specify(df, response="y")
        .generate(reps=10, type="bootstrap", seed=1)
        .calculate(stat=lambda r, e: float(r.mean()))
    )
    assert boot.stat == "stat"


def test_observe_without_null_path():
    df = pl.DataFrame({"y": [1.0, 2, 3]})
    assert float(observe(df, response="y", stat="mean")) == 2.0


# ============================ statistics.py ==============================


def test_all_one_variable_stats():
    r = np.array([1.0, 2, 3, 4, 5])
    assert compute_statistic(r, None, "sum") == 15
    assert compute_statistic(r, None, "sd") == pytest.approx(r.std(ddof=1))
    assert compute_statistic(r, None, "median") == 3


def test_diff_in_medians_and_ratio_of_means():
    r = np.array([10.0, 12, 2, 4])
    g = np.array(["a", "a", "b", "b"])
    assert compute_statistic(r, g, "diff in medians", order=("a", "b")) == pytest.approx(11 - 3)
    assert compute_statistic(r, g, "ratio of means", order=("a", "b")) == pytest.approx(11 / 3)


def test_two_sample_t_and_z():
    r = np.array([1.0, 2, 3, 9, 10, 11])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    assert compute_statistic(r, g, "t", order=("a", "b")) < 0  # group a < group b
    y = np.array(["s", "s", "f", "s", "f", "f"])
    z = compute_statistic(y, g, "z", order=("a", "b"), success="s")
    assert isinstance(z, float)


def test_stat_helpers_and_errors():
    assert needs_explanatory("slope") and not needs_explanatory("mean")
    assert needs_success("prop") and not needs_success("mean")
    with pytest.raises(ValueError):
        compute_statistic(np.array([1.0]), None, "unknown-stat")
    with pytest.raises(ValueError):  # bivariate without explanatory
        compute_statistic(np.array([1.0, 2.0]), None, "slope")
    with pytest.raises(ValueError):  # order required
        compute_statistic(np.array([1.0, 2.0]), np.array(["a", "b"]), "diff in means")


# ============================ resample.py ================================


def test_shift_for_point_null_centers_mean_and_median():
    r = np.array([1.0, 2, 3, 4])
    shifted = _resample.shift_for_point_null(r, stat="mean", mu=10.0, p=None)
    assert np.mean(shifted) == pytest.approx(10.0)
    shifted_med = _resample.shift_for_point_null(r, stat="median", mu=0.0, p=None)
    assert np.median(shifted_med) == pytest.approx(0.0)
    # proportions / no mu -> unchanged
    assert np.array_equal(_resample.shift_for_point_null(r, stat="prop", mu=None, p=0.5), r)


def test_point_null_mean_bootstrap_centers_distribution():
    df = pl.DataFrame({"x": np.arange(30.0)})
    null = (
        specify(df, response="x")
        .hypothesize(null="point", mu=100.0)
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    assert float(null.stats.mean()) == pytest.approx(100.0, abs=1.0)


# ============================ intervals.py ===============================


def test_se_confidence_interval_value():
    df = pl.DataFrame({"x": np.arange(50.0)})
    boot = (
        specify(df, response="x")
        .generate(reps=300, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    ci = boot.get_confidence_interval(level=0.95, type="se", point_estimate=24.5)
    assert float(ci["lower_ci"][0]) < 24.5 < float(ci["upper_ci"][0])


def test_invalid_ci_type():
    df = pl.DataFrame({"x": np.arange(20.0)})
    boot = (
        specify(df, response="x")
        .generate(reps=100, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    with pytest.raises(ValueError):
        boot.get_confidence_interval(type="bogus")
    with pytest.raises(ValueError):  # bias-corrected needs point estimate
        boot.get_confidence_interval(type="bias-corrected")


# ============================ pvalue.py ==================================


def test_pvalue_left_and_invalid_direction():
    df = pl.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "g": ["a", "a", "a", "b", "b", "b"]})
    null = (
        specify(df, formula="y ~ g")
        .hypothesize(null="independence")
        .generate(reps=200, type="permute", seed=1)
        .calculate(stat="diff in means", order=("a", "b"))
    )
    small = float(np.min(null.stats)) - 1.0
    assert float(null.get_p_value(obs_stat=small, direction="left")["p_value"][0]) == pytest.approx(
        0.0
    )
    with pytest.raises(ValueError):
        null.get_p_value(obs_stat=0.0, direction="sideways")


def test_fit_pvalue_directions():
    sar = md.load_saratoga_houses()
    obs = specify(sar, formula="price ~ living_area").fit()
    null = (
        specify(sar, formula="price ~ living_area")
        .hypothesize(null="independence")
        .generate(reps=100, type="permute", seed=1)
        .fit()
    )
    for direction in ("right", "left", "two-sided"):
        out = null.get_p_value(obs_stat=obs, direction=direction)
        assert "p_value" in out.columns


# ============================ theoretical.py =============================


def test_assume_distributions_and_directions():
    from scipy.stats import chi2, norm
    from scipy.stats import f as fdist

    assert float(assume("z").get_p_value(1.0, "right")["p_value"][0]) == pytest.approx(norm.sf(1.0))
    assert float(assume("z").get_p_value(-1.0, "left")["p_value"][0]) == pytest.approx(
        norm.cdf(-1.0)
    )
    assert float(assume("z").get_p_value(1.0, "two-sided")["p_value"][0]) == pytest.approx(
        2 * norm.sf(1.0)
    )
    assert float(assume("F", df=(2, 20)).get_p_value(3.0, "right")["p_value"][0]) == pytest.approx(
        fdist.sf(3.0, 2, 20)
    )
    assert float(assume("Chisq", df=3).get_p_value(5.0, "right")["p_value"][0]) == pytest.approx(
        chi2.sf(5.0, 3)
    )


def test_assume_errors_and_visualize():
    with pytest.raises(ValueError):
        assume("weibull").get_p_value(1.0, "right")
    with pytest.raises(ValueError):
        assume("t", df=5).get_p_value(1.0, "diagonal")
    assert isinstance(assume("t", df=5).visualize(), ggplot)


# ============================ viz.py =====================================


def test_shade_p_value_left_and_two_sided_and_ci_tuple():
    from moderndive import shade_confidence_interval, shade_p_value, visualize

    df = pl.DataFrame({"x": np.arange(40.0)})
    boot = (
        specify(df, response="x")
        .generate(reps=100, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    assert isinstance(visualize(boot) + shade_p_value(obs_stat=10.0, direction="left"), ggplot)
    assert isinstance(visualize(boot) + shade_p_value(obs_stat=5.0, direction="two-sided"), ggplot)
    # endpoints as a plain tuple
    assert isinstance(visualize(boot) + shade_confidence_interval(endpoints=(10.0, 30.0)), ggplot)


# ============================ wrappers.py ================================


def test_wrapper_t_test_two_sample_and_t_stat():
    from moderndive import t_stat, t_test

    movies = md.load_movies_sample()
    out = t_test(movies, formula="rating ~ genre", order=("Action", "Romance"))
    assert "statistic" in out.columns
    assert isinstance(t_stat(movies, formula="rating ~ genre", order=("Action", "Romance")), float)
    with pytest.raises(ValueError):  # two-sample needs order
        t_test(movies, formula="rating ~ genre")


def test_wrapper_prop_test_paths():
    from moderndive import prop_test

    yawn = md.load_mythbusters_yawn()
    # one-sample, all alternatives
    for alt in ("greater", "less", "two-sided"):
        out = prop_test(yawn, response="yawn", success="yes", p=0.3, alternative=alt)
        assert 0 <= float(out["p_value"][0]) <= 1
    # two-sample needs order
    with pytest.raises(ValueError):
        prop_test(yawn, formula="yawn ~ group", success="yes")
    assert (
        "statistic"
        in prop_test(yawn, formula="yawn ~ group", success="yes", order=("seed", "control")).columns
    )


def test_wrapper_chisq_needs_explanatory():
    from moderndive import chisq_stat, chisq_test

    yawn = md.load_mythbusters_yawn()
    assert isinstance(chisq_stat(yawn, formula="yawn ~ group"), float)
    with pytest.raises(ValueError):
        chisq_test(yawn, response="yawn")


# ============================ theory.py ==================================


def test_theory_module_functions():
    age = md.load_age_at_marriage()["age"]
    assert "statistic" in md.theory.t_test_one_sample(age, mu=23).columns
    assert "lower_ci" in md.theory.t_confidence_interval(age).columns
    a = np.array([1.0, 2, 3, 4])
    b = np.array([2.0, 3, 4, 5])
    assert "statistic" in md.theory.t_test_two_sample(a, b).columns
    out = md.theory.prop_test_two_sample(
        successes=(60, 40), totals=(100, 100), alternative="greater"
    )
    assert float(out["estimate"][0]) == pytest.approx(0.2)
    assert (
        md.theory.prop_test_two_sample((40, 60), (100, 100), alternative="less")["p_value"][0] <= 1
    )


# ============================ modeling.py ================================


def test_tidy_summary_accepts_pandas():
    import pandas as pd

    out = md.tidy_summary(pd.DataFrame({"a": [1.0, 2, 3]}))
    assert out["mean"][0] == pytest.approx(2.0)


# ============================ plots.py ===================================


def test_pairplot_returns_figure():
    from matplotlib.figure import Figure

    coffee = md.load_coffee_quality()
    fig = md.pairplot(coffee, columns=["total_cup_points", "aroma", "flavor"])
    assert isinstance(fig, Figure)
    # default columns (auto-detect numeric) + hue path
    fig2 = md.pairplot(
        coffee.select("total_cup_points", "aroma", "continent_of_origin"), hue="continent_of_origin"
    )
    assert isinstance(fig2, Figure)


# ============================ sampling.py ================================


def test_rep_sample_n_alias_and_replace_error():
    df = pl.DataFrame({"x": range(10)})
    assert md.rep_sample_n(df, n=3, reps=2, seed=1).height == 6
    with pytest.raises(ValueError):
        md.rep_slice_sample(df, n=20, replace=False)


# ============================ data/__init__.py ===========================


def test_load_dataset_dispatches_derived():
    out = md.load_dataset("spotify_metal_deephouse")
    assert out.height == 2000


# ============================ final-line mop-up ==========================


def test_fit_with_response_explanatory_builds_formula():
    # _full_formula via response/explanatory (not a formula string)
    sar = md.load_saratoga_houses()
    obs = specify(sar, response="price", explanatory="living_area").fit()
    assert set(obs.data["term"].to_list()) == {"intercept", "living_area"}


def test_distribution_len():
    df = pl.DataFrame({"x": np.arange(15.0)})
    boot = (
        specify(df, response="x").generate(reps=42, type="bootstrap", seed=1).calculate(stat="mean")
    )
    assert len(boot) == 42


def test_one_sample_z_compute_and_missing_p():
    df = pl.DataFrame({"y": ["s"] * 6 + ["f"] * 4})
    val = observe(df, response="y", success="s", stat="z", null="point", p=0.5)
    assert isinstance(float(val), float)
    with pytest.raises(ValueError):  # z without p
        compute_statistic(np.array(["s", "f"], dtype=object), None, "z", success="s")


def test_unknown_stat_with_explanatory_hits_final_raise():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    g = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError):
        compute_statistic(r, g, "totally-bogus", order=("a", "b"))


def test_to_pandas_both_branches():
    import pandas as pd

    from moderndive.modeling import _to_pandas

    assert isinstance(_to_pandas(pl.DataFrame({"a": [1]})), pd.DataFrame)
    pdf = pd.DataFrame({"a": [1]})
    assert _to_pandas(pdf) is pdf


def test_prop_test_two_sample_two_sided():
    out = md.theory.prop_test_two_sample(
        successes=(60, 40), totals=(100, 100), alternative="two-sided"
    )
    assert 0 <= float(out["p_value"][0]) <= 1
