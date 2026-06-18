"""Tests for the infer grammar: statistics, resampling, intervals, p-values."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import moderndive as md
from moderndive import get_confidence_interval, get_p_value, specify
from moderndive.infer.statistics import compute_statistic


# --- statistics on known small inputs -------------------------------------

def test_mean_and_median():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert float(specify(df, response="x").calculate(stat="mean")) == 2.5
    assert float(specify(df, response="x").calculate(stat="median")) == 2.5


def test_prop_uses_success():
    df = pl.DataFrame({"y": ["yes", "yes", "no", "no", "yes"]})
    obs = specify(df, response="y", success="yes").calculate(stat="prop")
    assert float(obs) == pytest.approx(3 / 5)


def test_diff_in_props_manual():
    # metal: 2/3 popular; deep-house: 1/3 popular; diff = 1/3
    df = pl.DataFrame(
        {
            "genre": ["metal", "metal", "metal", "deep-house", "deep-house", "deep-house"],
            "pop": ["popular", "popular", "not", "popular", "not", "not"],
        }
    )
    obs = specify(df, formula="pop ~ genre", success="popular").calculate(
        stat="diff in props", order=("metal", "deep-house")
    )
    assert float(obs) == pytest.approx(1 / 3)


def test_diff_in_means_manual():
    df = pl.DataFrame({"y": [10.0, 12.0, 1.0, 3.0], "g": ["a", "a", "b", "b"]})
    obs = specify(df, formula="y ~ g").calculate(stat="diff in means", order=("a", "b"))
    assert float(obs) == pytest.approx(11.0 - 2.0)


def test_slope_is_exact_for_linear_data():
    x = np.arange(10, dtype=float)
    df = pl.DataFrame({"x": x, "y": 2.0 * x + 5.0})
    obs = specify(df, formula="y ~ x").calculate(stat="slope")
    assert float(obs) == pytest.approx(2.0)


def test_correlation_perfect():
    x = np.arange(10, dtype=float)
    df = pl.DataFrame({"x": x, "y": 3.0 * x - 1.0})
    obs = specify(df, formula="y ~ x").calculate(stat="correlation")
    assert float(obs) == pytest.approx(1.0)


def test_unknown_stat_raises():
    with pytest.raises(ValueError):
        compute_statistic(np.array([1.0, 2.0]), None, "bogus")


# --- formula parsing -------------------------------------------------------

def test_formula_null_explanatory():
    df = pl.DataFrame({"weight": [1.0, 2.0]})
    spec = specify(df, formula="weight ~ NULL")
    assert spec.response == "weight"
    assert spec.explanatory is None


def test_specify_rejects_missing_column():
    df = pl.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError):
        specify(df, response="nope")


# --- resampling reproducibility -------------------------------------------

def test_bootstrap_reproducible_under_seed():
    df = pl.DataFrame({"x": np.arange(50, dtype=float)})
    a = specify(df, response="x").generate(reps=200, type="bootstrap", seed=42).calculate(stat="mean")
    b = specify(df, response="x").generate(reps=200, type="bootstrap", seed=42).calculate(stat="mean")
    assert np.array_equal(a.stats, b.stats)


def test_permute_reproducible_under_seed():
    df = pl.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "g": ["a", "a", "a", "b", "b", "b"]})
    kw = dict(formula="y ~ g")
    a = specify(df, **kw).hypothesize(null="independence").generate(reps=200, type="permute", seed=7).calculate(stat="diff in means", order=("a", "b"))
    b = specify(df, **kw).hypothesize(null="independence").generate(reps=200, type="permute", seed=7).calculate(stat="diff in means", order=("a", "b"))
    assert np.array_equal(a.stats, b.stats)


def test_permute_requires_explanatory():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        specify(df, response="x").generate(reps=10, type="permute", seed=1)


# --- intervals & p-values --------------------------------------------------

def test_percentile_ci_brackets_point_estimate():
    almonds = md.load_almonds_sample_100()
    xbar = float(specify(almonds, response="weight").calculate(stat="mean"))
    boot = specify(almonds, response="weight").generate(reps=1000, type="bootstrap", seed=76).calculate(stat="mean")
    ci = get_confidence_interval(boot, level=0.95, type="percentile")
    lo, hi = float(ci["lower_ci"][0]), float(ci["upper_ci"][0])
    assert lo < xbar < hi


def test_se_ci_requires_point_estimate():
    df = pl.DataFrame({"x": np.arange(30, dtype=float)})
    boot = specify(df, response="x").generate(reps=200, type="bootstrap", seed=1).calculate(stat="mean")
    with pytest.raises(ValueError):
        get_confidence_interval(boot, type="se")


def test_p_value_directions():
    # Null distribution symmetric around 0; obs at 0 -> right/left ~0.5, two-sided ~1.
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"y": rng.normal(size=400), "g": ["a"] * 200 + ["b"] * 200})
    null = specify(df, formula="y ~ g").hypothesize(null="independence").generate(reps=1000, type="permute", seed=3).calculate(stat="diff in means", order=("a", "b"))
    p_two = float(get_p_value(null, obs_stat=0.0, direction="two-sided")["p_value"][0])
    assert 0.0 <= p_two <= 1.0
    # An obs far in the right tail should give a small right p-value.
    big = float(np.max(null.stats)) + 1.0
    p_right = float(get_p_value(null, obs_stat=big, direction="right")["p_value"][0])
    assert p_right == pytest.approx(0.0)


def test_observed_stat_matches_r_reference():
    """Deterministic observed statistics validated against the R book (see plan)."""
    spotify = md.load_spotify_metal_deephouse()
    obs = specify(spotify, formula="popular_or_not ~ track_genre", success="popular").calculate(
        stat="diff in props", order=("metal", "deep-house")
    )
    assert float(obs) == pytest.approx(0.034, abs=1e-6)

    almonds = md.load_almonds_sample_100()
    xbar = float(specify(almonds, response="weight").calculate(stat="mean"))
    assert xbar == pytest.approx(3.682, abs=1e-3)
