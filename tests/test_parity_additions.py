"""Tests for the R-parity additions: new functions, plot helpers, infer gaps, datasets."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
import statsmodels.formula.api as smf
from plotnine import aes, geom_point, ggplot

import moderndive as md
from moderndive import (
    geom_parallel_slopes,
    get_correlation,
    get_regression_summaries,
    gg_categorical_model,
    gg_parallel_slopes,
    pop_sd,
    specify,
)

# ============================ WS1: functions =============================


def test_get_correlation_formula_and_kwargs():
    df = pl.DataFrame({"y": [1.0, 2, 3, 4, 5], "x": [2.0, 4, 5, 4, 5]})
    by_formula = float(get_correlation(df, "y ~ x")["cor"][0])
    by_kwargs = float(get_correlation(df, x="x", y="y")["cor"][0])
    expected = float(np.corrcoef(df["x"], df["y"])[0, 1])
    assert by_formula == pytest.approx(expected)
    assert by_kwargs == pytest.approx(expected)


def test_get_correlation_accepts_pandas_and_drops_nulls():
    pdf = pl.DataFrame({"y": [1.0, 2, None, 4], "x": [1.0, 2, 3, 4]}).to_pandas()
    assert "cor" in get_correlation(pdf, "y ~ x").columns


def test_get_correlation_errors():
    df = pl.DataFrame({"y": [1.0, 2], "x": [3.0, 4]})
    with pytest.raises(ValueError):  # no formula / no x,y
        get_correlation(df)
    with pytest.raises(ValueError):  # both formula and kwargs
        get_correlation(df, "y ~ x", x="x", y="y")
    with pytest.raises(ValueError):  # malformed formula
        get_correlation(df, "y plus x")
    with pytest.raises(ValueError):  # empty side of the formula
        get_correlation(df, "y ~ ")
    with pytest.raises(ValueError):  # missing column
        get_correlation(df, "y ~ nope")


def test_pop_sd_matches_ddof0():
    assert pop_sd([1, 2, 3, 4, 5]) == pytest.approx(np.sqrt(2.0))
    assert pop_sd(pl.Series([1.0, 2, 3, None])) == pytest.approx(np.std([1.0, 2, 3], ddof=0))
    assert pop_sd(np.array([1.0, np.nan, 3.0])) == pytest.approx(np.std([1.0, 3.0], ddof=0))


def test_get_regression_summaries_columns_and_values():
    sar = md.load_saratoga_houses()
    model = smf.ols("price ~ living_area", data=sar.to_pandas()).fit()
    out = get_regression_summaries(model)
    assert out.columns == [
        "r_squared",
        "adj_r_squared",
        "mse",
        "rmse",
        "sigma",
        "statistic",
        "p_value",
        "df",
        "nobs",
    ]
    assert out["r_squared"][0] == pytest.approx(round(model.rsquared, 3))
    assert out["nobs"][0] == int(model.nobs)
    assert out["rmse"][0] == pytest.approx(round(np.sqrt(model.ssr / model.nobs), 3))


# ============================ WS2: plot helpers ==========================


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
def test_gg_parallel_slopes_both_engines(engine):
    ev = md.load_evals()
    fig = gg_parallel_slopes(ev, "score", "age", "gender", engine=engine)
    if engine == "plotly":
        assert isinstance(fig, go.Figure)
    else:
        assert isinstance(fig, ggplot)


def test_geom_parallel_slopes_adds_plotnine_layers():
    ev = md.load_evals()
    base = (
        ggplot(ev.select("score", "age", "gender").to_pandas(), aes("age", "score", color="gender"))
        + geom_point()
    )
    layered = base + geom_parallel_slopes(ev, "score", "age", "gender")
    assert len(layered.layers) > len(base.layers)
    # explicit color path (single-color lines)
    layered2 = base + geom_parallel_slopes(ev, "score", "age", "gender", color="black")
    assert len(layered2.layers) > len(base.layers)


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
def test_gg_categorical_model_both_engines(engine):
    ev = md.load_evals()
    fig = gg_categorical_model(ev, "score", "rank", engine=engine)
    if engine == "plotly":
        assert isinstance(fig, go.Figure)
    else:
        assert isinstance(fig, ggplot)


def test_plot_helpers_reject_bad_engine():
    ev = md.load_evals()
    with pytest.raises(ValueError):
        gg_parallel_slopes(ev, "score", "age", "gender", engine="bogus")
    with pytest.raises(ValueError):
        gg_categorical_model(ev, "score", "rank", engine="bogus")


def test_pairplot_plotly_engine_and_bad_engine():
    coffee = md.load_coffee_quality()
    fig = md.pairplot(coffee, columns=["total_cup_points", "aroma", "flavor"], engine="plotly")
    assert isinstance(fig, go.Figure)
    with pytest.raises(ValueError):
        md.pairplot(coffee, engine="bogus")


# ============================ WS4: infer gaps ============================


def test_simulate_alias_equals_draw():
    df = pl.DataFrame({"x": ["s"] * 30 + ["f"] * 70})
    common = dict(reps=100, seed=1)
    draw = (
        specify(df, response="x", success="s")
        .hypothesize(null="point", p=0.5)
        .generate(type="draw", **common)
        .calculate(stat="prop")
    )
    sim = (
        specify(df, response="x", success="s")
        .hypothesize(null="point", p=0.5)
        .generate(type="simulate", **common)
        .calculate(stat="prop")
    )
    assert draw.data["stat"].to_list() == sim.data["stat"].to_list()


def test_one_sample_mean_z_with_sigma():
    df = pl.DataFrame({"w": np.arange(1.0, 51.0)})
    n = df.height
    obs = (
        specify(df, response="w").hypothesize(null="point", mu=25.0, sigma=5.0).calculate(stat="z")
    )
    expected = (float(df["w"].mean()) - 25.0) / (5.0 / np.sqrt(n))
    assert float(obs) == pytest.approx(expected)
    # null distribution of the sigma-z via bootstrap point null
    null = (
        specify(df, response="w")
        .hypothesize(null="point", mu=25.0, sigma=5.0)
        .generate(reps=50, type="bootstrap", seed=1)
        .calculate(stat="z")
    )
    assert null.data.height == 50


def test_sigma_z_requires_mu():
    from moderndive.infer.statistics import compute_statistic

    with pytest.raises(ValueError):
        compute_statistic(np.arange(10.0), None, "z", sigma=2.0)


def test_visualize_method_both_uses_density_overlay():
    boot = (
        specify(md.load_age_at_marriage(), response="age")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    # plotly "both": histogram trace + overlay scatter (the normal curve)
    fig = md.visualize(boot, engine="plotly", method="both").figure
    assert any(t.type == "scatter" for t in fig.data)


# ============================ WS5: datasets ==============================

# (dataset, expected (rows, cols)) — verified against the R packages.
_NEW_DATASETS = {
    "DD_vs_SB": (2048, 6),
    "MA_schools": (332, 4),
    "alaska_flights": (714, 19),
    "amazon_books": (325, 13),
    "avocados": (18249, 13),
    "babies": (1236, 24),
    "bowl_sample_1": (50, 1),
    "bowl_samples": (10, 5),
    "coffee_ratings": (1339, 43),
    "early_january_weather": (358, 15),
    "ev_charging": (3395, 24),
    "evals": (463, 14),
    "ipf_lifts": (41152, 16),
    "ma_traffic_2020_vs_2019": (264, 5),
    "mario_kart_auction": (143, 12),
    "mass_traffic_2020": (874, 11),
    "orig_pennies_sample": (40, 2),
    "pennies": (800, 2),
    "pennies_resamples": (1750, 3),
    "pennies_sample": (50, 2),
    "promotions": (48, 3),
    "promotions_shuffled": (48, 3),
    "spotify_52_original": (52, 6),
    "spotify_52_shuffled": (52, 6),
    "gss": (500, 11),
}


@pytest.mark.parametrize("name,dims", _NEW_DATASETS.items())
def test_new_datasets_load_with_expected_dims(name, dims):
    df = md.load_dataset(name)
    assert (df.height, df.width) == dims


def test_new_datasets_registered_and_have_loaders():
    available = set(md.available_datasets())
    for name in _NEW_DATASETS:
        assert name in available
        assert callable(getattr(md, f"load_{name}"))
