"""Tests for the infer R argument-parity additions:

- hypothesize/observe(med=) — median point null
- prop_test(z=, correct=, conf_int=, conf_level=) — validated vs R's prop.test
- rep_slice_sample(prop=, weight_by=) / rep_sample_n(prob=)
- generate(variables=)
- shade_p_value/shade_confidence_interval(fill=), visualize(dens_color=)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
from plotnine import ggplot

import moderndive as md
from moderndive import (
    get_p_value,
    observe,
    prop_test,
    rep_sample_n,
    rep_slice_sample,
    shade_confidence_interval,
    shade_p_value,
    visualize,
)

# ============================ median point null =========================


def test_median_point_null_centers_and_pvalue():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(10.0, 3.0, 300)})
    null = (
        df.specify(response="x")
        .hypothesize(null="point", med=8.0)
        .generate(reps=500, type="bootstrap", seed=1)
        .calculate(stat="median")
    )
    assert float(null.stats.mean()) == pytest.approx(8.0, abs=0.5)
    obs = observe(df, response="x", stat="median")
    pv = float(get_p_value(null, obs_stat=obs, direction="two-sided")["p_value"][0])
    assert 0.0 <= pv <= 1.0


def test_observe_med_via_hypothesise_alias():
    df = pl.DataFrame({"x": [1.0, 2, 3, 4, 5]})
    h = df.specify(response="x").hypothesise(null="point", med=3.0)
    assert h.med == 3.0


# ============================ prop_test (vs R prop.test) ================


def _two_group():
    rows = (
        [("yes", "seed")] * 10
        + [("no", "seed")] * 24
        + [("yes", "control")] * 4
        + [("no", "control")] * 12
    )
    return pl.DataFrame({"yawn": [r[0] for r in rows], "group": [r[1] for r in rows]})


def test_prop_test_two_sample_matches_r():
    out = prop_test(_two_group(), formula="yawn ~ group", success="yes", order=("seed", "control"))
    # R prop.test(c(10,4), c(34,16)): X-squared 0, p 1, CI [-0.26168, 0.34991]
    assert out["statistic"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["chisq_df"][0] == 1
    assert out["p_value"][0] == pytest.approx(1.0)
    assert out["lower_ci"][0] == pytest.approx(-0.26168, abs=1e-4)
    assert out["upper_ci"][0] == pytest.approx(0.34991, abs=1e-4)


def test_prop_test_two_sample_no_correction_and_z():
    g = _two_group()
    nc = prop_test(
        g, formula="yawn ~ group", success="yes", order=("seed", "control"), correct=False
    )
    assert nc["statistic"][0] == pytest.approx(0.10504, abs=1e-4)  # R X-squared, correct=FALSE
    z = prop_test(g, formula="yawn ~ group", success="yes", order=("seed", "control"), z=True)
    assert "chisq_df" not in z.columns
    assert {"statistic", "p_value", "estimate", "alternative", "lower_ci", "upper_ci"} <= set(
        z.columns
    )


def test_prop_test_one_sample_matches_r():
    df = pl.DataFrame({"x": ["yes"] * 14 + ["no"] * 36})
    out = prop_test(df, response="x", success="yes", p=0.3)
    # R prop.test(14, 50, p=0.3): X-squared 0.02381, p 0.87737, Wilson CI [0.1667, 0.4271]
    assert out["statistic"][0] == pytest.approx(0.02381, abs=1e-4)
    assert out["p_value"][0] == pytest.approx(0.87737, abs=1e-4)
    assert out["lower_ci"][0] == pytest.approx(0.1667, abs=1e-3)
    assert out["upper_ci"][0] == pytest.approx(0.4271, abs=1e-3)
    # one-sided alternative matches R's "less" p-value
    less = prop_test(df, response="x", success="yes", p=0.3, alternative="less")
    assert less["p_value"][0] == pytest.approx(0.43869, abs=1e-4)


def test_prop_test_one_sample_no_correction_ci():
    df = pl.DataFrame({"x": ["yes"] * 14 + ["no"] * 36})
    out = prop_test(df, response="x", success="yes", p=0.3, correct=False)
    # uncorrected Wilson score interval for x=14, n=50
    assert out["lower_ci"][0] == pytest.approx(0.1747, abs=1e-3)
    assert out["upper_ci"][0] == pytest.approx(0.4167, abs=1e-3)


def test_prop_test_conf_int_false_drops_ci():
    df = pl.DataFrame({"x": ["yes"] * 14 + ["no"] * 36})
    out = prop_test(df, response="x", success="yes", p=0.3, conf_int=False)
    assert "lower_ci" not in out.columns and "upper_ci" not in out.columns


def test_prop_test_two_sample_needs_order():
    with pytest.raises(ValueError, match="order"):
        prop_test(_two_group(), formula="yawn ~ group", success="yes")


# ============================ rep sampling ==============================


def test_rep_slice_sample_prop():
    bowl = md.load_bowl()
    out = rep_slice_sample(bowl, prop=0.1, reps=3, seed=1)
    assert out["replicate"].n_unique() == 3
    assert out.filter(pl.col("replicate") == 1).height == round(0.1 * bowl.height)


def test_rep_slice_sample_weight_by_column_and_sequence():
    df = pl.DataFrame({"x": list(range(10)), "w": [0.0] * 9 + [1.0]})
    # only the last row has weight → every draw is x == 9
    by_col = rep_slice_sample(df, n=5, replace=True, weight_by="w", seed=1)
    assert set(by_col["x"].to_list()) == {9}
    by_seq = rep_slice_sample(df, n=5, replace=True, weight_by=[0.0] * 9 + [1.0], seed=1)
    assert set(by_seq["x"].to_list()) == {9}


def test_rep_sample_n_prob():
    df = pl.DataFrame({"x": list(range(10))})
    out = rep_sample_n(df, n=5, replace=True, prob=[0.0] * 9 + [1.0], seed=1)
    assert set(out["x"].to_list()) == {9}


def test_rep_slice_sample_n_xor_prop():
    bowl = md.load_bowl()
    with pytest.raises(ValueError, match="exactly one"):
        rep_slice_sample(bowl, n=5, prop=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        rep_slice_sample(bowl)


def test_rep_slice_sample_bad_weights():
    df = pl.DataFrame({"x": [1, 2, 3], "w": [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="positive weights"):
        rep_slice_sample(df, n=2, replace=True, weight_by="w")


def test_rep_slice_sample_without_replacement_too_big():
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="without replacement"):
        rep_slice_sample(df, n=5)


# ============================ generate(variables=) ======================


def test_generate_variables_permutes_chosen_column():
    gss = md.load_gss()
    # permuting the response gives a valid null distribution for the diff in means
    null = (
        gss.specify(formula="age ~ college")
        .hypothesize(null="independence")
        .generate(reps=200, type="permute", variables="age", seed=1)
        .calculate(stat="diff in means", order=("degree", "no degree"))
    )
    assert null.data.height == 200
    assert abs(float(null.stats.mean())) < 1.0  # centered near 0 under the null


def test_generate_variables_must_be_a_model_variable():
    gss = md.load_gss()
    with pytest.raises(ValueError, match="must be the response or explanatory"):
        gss.specify(formula="age ~ college").hypothesize(null="independence").generate(
            reps=5, type="permute", variables="nope"
        )


# ============================ shade fill + dens_color ===================


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
def test_shade_fill_both_engines(engine):
    boot = (
        md.load_age_at_marriage()
        .specify(response="age")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    from moderndive import get_confidence_interval

    ci = get_confidence_interval(boot, type="percentile")
    p_ci = visualize(boot, engine=engine) + shade_confidence_interval(
        ci, color="navy", fill="lightblue"
    )
    p_pv = visualize(boot, engine=engine) + shade_p_value(
        obs_stat=float(boot.data["stat"].mean()),
        direction="right",
        color="darkred",
        fill="mistyrose",
    )
    cls = go.Figure if engine == "plotly" else ggplot
    assert isinstance(p_ci.figure, cls) and isinstance(p_pv.figure, cls)


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
@pytest.mark.parametrize("method", ["theoretical", "both"])
def test_dens_color_both_engines(engine, method):
    boot = (
        md.load_age_at_marriage()
        .specify(response="age")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    p = visualize(boot, engine=engine, method=method, dens_color="green")
    cls = go.Figure if engine == "plotly" else ggplot
    assert isinstance(p.figure, cls)
