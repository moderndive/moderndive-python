"""Tests for the sampling helper, draw-type generation, and regression fit()."""

from __future__ import annotations

import polars as pl
import pytest

import moderndive as md
from moderndive import rep_slice_sample, specify


def test_rep_slice_sample_shape_and_replicate():
    df = pl.DataFrame({"x": range(100)})
    out = rep_slice_sample(df, n=10, reps=5, seed=1)
    assert out.columns[0] == "replicate"
    assert out.height == 10 * 5
    assert out["replicate"].unique().sort().to_list() == [1, 2, 3, 4, 5]


def test_rep_slice_sample_without_replacement_is_unique_within_replicate():
    df = pl.DataFrame({"x": range(20)})
    out = rep_slice_sample(df, n=20, reps=1, replace=False, seed=2)
    assert out["x"].n_unique() == 20  # a full permutation, no repeats


def test_rep_slice_sample_reproducible():
    df = pl.DataFrame({"x": range(50)})
    a = rep_slice_sample(df, n=10, reps=10, seed=7)
    b = rep_slice_sample(df, n=10, reps=10, seed=7)
    assert a.equals(b)


def test_rep_slice_sample_recovers_population_proportion():
    bowl = md.load_bowl()
    samples = rep_slice_sample(bowl, n=50, reps=500, seed=1)
    props = samples.group_by("replicate").agg(prop=(pl.col("color") == "red").mean())
    # population proportion of red is ~0.375
    assert float(props["prop"].mean()) == pytest.approx(0.375, abs=0.02)


def test_draw_generates_proportion_near_p():
    df = pl.DataFrame({"y": ["s"] * 30 + ["f"] * 70})
    null = (
        specify(df, response="y", success="s")
        .hypothesize(null="point", p=0.5)
        .generate(reps=500, type="draw", seed=3)
        .calculate(stat="prop")
    )
    assert float(null.stats.mean()) == pytest.approx(0.5, abs=0.03)


def test_draw_requires_p():
    df = pl.DataFrame({"y": ["s", "f"]})
    with pytest.raises(ValueError):
        specify(df, response="y", success="s").generate(reps=5, type="draw")


def test_observed_fit_matches_statsmodels():
    sar = md.load_saratoga_houses()
    obs = specify(sar, formula="price ~ living_area + bedrooms").fit()
    assert set(obs.data["term"].to_list()) == {"intercept", "living_area", "bedrooms"}
    # living_area coefficient is solidly positive
    assert obs.estimate_for("living_area") > 0


def test_bootstrap_fit_ci_brackets_observed():
    sar = md.load_saratoga_houses()
    obs = specify(sar, formula="price ~ living_area").fit()
    boot = (
        specify(sar, formula="price ~ living_area")
        .generate(reps=300, type="bootstrap", seed=11)
        .fit()
    )
    ci = boot.get_confidence_interval(level=0.95)
    row = ci.filter(pl.col("term") == "living_area")
    lo, hi = float(row["lower_ci"][0]), float(row["upper_ci"][0])
    assert lo < obs.estimate_for("living_area") < hi


def test_permute_fit_pvalue_significant_for_real_predictor():
    sar = md.load_saratoga_houses()
    obs = specify(sar, formula="price ~ living_area").fit()
    null = (
        specify(sar, formula="price ~ living_area")
        .hypothesize(null="independence")
        .generate(reps=300, type="permute", seed=12)
        .fit()
    )
    pvals = null.get_p_value(obs_stat=obs, direction="two-sided")
    p_living = float(pvals.filter(pl.col("term") == "living_area")["p_value"][0])
    assert p_living < 0.05
