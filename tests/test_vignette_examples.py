"""Vignette smoke test: every infer `calculate(stat=...)` form on `gss`.

This mirrors the R `infer` "Full pipeline examples" (observed_stat_examples)
vignette and runs each statistic against the bundled `gss` dataset, so a
behavioral regression or a newly-missing capability is caught in CI rather than
only by manually re-reading the vignette. It complements the structural
parity-drift tooling (which tracks dataset/function *names*, not behavior).

If R `infer` gains a new `stat`, add a case here so the gap is visible.
"""

from __future__ import annotations

import math

import pytest

import moderndive as md
from moderndive import assume, chisq_test, get_confidence_interval, get_p_value, prop_test, t_test


@pytest.fixture(scope="module")
def gss():
    return md.load_gss()


# --- observed statistics, by variable type (the "stat menu") -------------

# (label, callable producing a scalar observed statistic)
_OBSERVED = {
    # one numerical
    "mean": lambda g: g.specify(response="hours").calculate(stat="mean"),
    "median": lambda g: g.specify(response="age").calculate(stat="median"),
    "sd": lambda g: g.specify(response="age").calculate(stat="sd"),
    "sum": lambda g: g.specify(response="hours").calculate(stat="sum"),
    "t (one-sample)": lambda g: (
        g.specify(response="hours").hypothesize(null="point", mu=40).calculate(stat="t")
    ),
    # one categorical
    "prop": lambda g: g.specify(response="sex", success="female").calculate(stat="prop"),
    "count": lambda g: g.specify(response="sex", success="female").calculate(stat="count"),
    "z (one-prop)": lambda g: (
        g.specify(response="sex", success="female")
        .hypothesize(null="point", p=0.5)
        .calculate(stat="z")
    ),
    # two categorical (2 levels)
    "diff in props": lambda g: g.specify(formula="college ~ sex", success="degree").calculate(
        stat="diff in props", order=("male", "female")
    ),
    "ratio of props": lambda g: g.specify(formula="college ~ sex", success="degree").calculate(
        stat="ratio of props", order=("male", "female")
    ),
    "odds ratio": lambda g: g.specify(formula="college ~ sex", success="degree").calculate(
        stat="odds ratio", order=("male", "female")
    ),
    "z (two-prop)": lambda g: g.specify(formula="college ~ sex", success="degree").calculate(
        stat="z", order=("male", "female")
    ),
    # two categorical (test of independence)
    "Chisq (independence)": lambda g: g.specify(formula="finrela ~ sex").calculate(stat="Chisq"),
    # numerical ~ categorical (2 levels)
    "diff in means": lambda g: g.specify(formula="age ~ college").calculate(
        stat="diff in means", order=("degree", "no degree")
    ),
    "diff in medians": lambda g: g.specify(formula="age ~ college").calculate(
        stat="diff in medians", order=("degree", "no degree")
    ),
    "ratio of means": lambda g: g.specify(formula="age ~ college").calculate(
        stat="ratio of means", order=("degree", "no degree")
    ),
    "t (two-sample)": lambda g: g.specify(formula="age ~ college").calculate(
        stat="t", order=("degree", "no degree")
    ),
    # numerical ~ categorical (3+ levels)
    "F": lambda g: g.specify(formula="age ~ partyid").calculate(stat="F"),
    # two numerical
    "correlation": lambda g: g.specify(formula="hours ~ age").calculate(stat="correlation"),
    "slope": lambda g: g.specify(formula="hours ~ age").calculate(stat="slope"),
}


@pytest.mark.parametrize("label", list(_OBSERVED))
def test_observed_statistic_is_finite(gss, label):
    value = float(_OBSERVED[label](gss))
    assert math.isfinite(value)


def test_goodness_of_fit_stat(gss):
    # one categorical (3+ levels) vs hypothesized proportions
    levels = gss["finrela"].unique().to_list()
    p = {lvl: 1 / len(levels) for lvl in levels}
    obs = gss.specify(response="finrela").hypothesize(null="point", p=p).calculate(stat="Chisq")
    assert math.isfinite(float(obs)) and float(obs) > 0


# --- full pipelines: null distribution + p-value, and bootstrap CI -------


def test_pipeline_permutation_pvalue(gss):
    obs = gss.specify(formula="age ~ college").calculate(
        stat="diff in means", order=("degree", "no degree")
    )
    null = (
        gss.specify(formula="age ~ college")
        .hypothesize(null="independence")
        .generate(reps=200, type="permute", seed=1)
        .calculate(stat="diff in means", order=("degree", "no degree"))
    )
    pv = float(get_p_value(null, obs_stat=obs, direction="two-sided")["p_value"][0])
    assert 0.0 <= pv <= 1.0


def test_pipeline_bootstrap_ci(gss):
    boot = (
        gss.specify(response="hours")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    ci = get_confidence_interval(boot, level=0.95, type="percentile")
    assert float(ci["lower_ci"][0]) < float(ci["upper_ci"][0])


def test_pipeline_simulation_gof_pvalue(gss):
    levels = gss["finrela"].unique().to_list()
    p = {lvl: 1 / len(levels) for lvl in levels}
    obs = gss.specify(response="finrela").hypothesize(null="point", p=p).calculate(stat="Chisq")
    null = (
        gss.specify(response="finrela")
        .hypothesize(null="point", p=p)
        .generate(reps=200, type="draw", seed=1)
        .calculate(stat="Chisq")
    )
    pv = float(get_p_value(null, obs_stat=obs, direction="greater")["p_value"][0])
    assert 0.0 <= pv <= 1.0


# --- theory-based and wrappers (the other infer vignettes) ----------------


def test_theory_assume_matches_directionality(gss):
    obs_t = gss.specify(response="hours").hypothesize(null="point", mu=40).calculate(stat="t")
    pv = float(
        assume("t", df=gss.height - 1).get_p_value(obs_t, direction="two-sided")["p_value"][0]
    )
    assert 0.0 <= pv <= 1.0


def test_wrappers_t_prop_chisq(gss):
    assert "statistic" in t_test(gss, response="hours", mu=40).columns
    assert (
        "statistic" in t_test(gss, formula="age ~ college", order=("degree", "no degree")).columns
    )
    assert (
        "statistic"
        in prop_test(
            gss, formula="college ~ sex", success="degree", order=("male", "female")
        ).columns
    )
    assert "statistic" in chisq_test(gss, formula="finrela ~ sex").columns
    # goodness-of-fit wrapper
    levels = gss["finrela"].unique().to_list()
    p = {lvl: 1 / len(levels) for lvl in levels}
    assert "statistic" in chisq_test(gss, response="finrela", p=p).columns
