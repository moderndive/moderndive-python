"""Tests for dataset loaders and theory-based helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import moderndive as md
from moderndive import theory


def test_loaders_return_polars_with_expected_shapes():
    cases = {
        "almonds_sample_100": (100, ["weight"]),
        "spotify_by_genre": (6000, ["track_genre", "popular_or_not"]),
        "gapminder_2007": (142, ["country", "continent", "lifeExp", "gdpPercap"]),
        "envoy_flights": (357, ["dep_delay", "arr_delay"]),
        "airports": (1255, ["faa", "name", "lat", "lon"]),
        "planes": (4840, ["tailnum", "manufacturer", "model"]),
        "drinks": (193, ["country", "beer_servings", "wine_servings"]),
        "dem_score": (96, ["country", "1952", "1992"]),
    }
    for name, (nrows, cols) in cases.items():
        df = md.load_dataset(name)
        assert isinstance(df, pl.DataFrame)
        assert df.height == nrows
        for col in cols:
            assert col in df.columns


def test_spotify_metal_deephouse_derivation():
    df = md.load_spotify_metal_deephouse()
    assert set(df["track_genre"].unique().to_list()) == {"metal", "deep-house"}
    assert df.height == 2000  # 1000 per genre


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        md.load_dataset("does_not_exist")


def test_t_confidence_interval_matches_manual():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    out = theory.t_confidence_interval(x, level=0.95)
    # Manual: mean=5, sd=2.138, n=8 -> t*=2.365, se=0.756
    assert out["lower_ci"][0] == pytest.approx(5.0 - 2.365 * 0.7559, abs=1e-2)
    assert out["upper_ci"][0] == pytest.approx(5.0 + 2.365 * 0.7559, abs=1e-2)


def test_prop_test_two_sample_sign():
    out = theory.prop_test_two_sample(successes=(60, 40), totals=(100, 100), alternative="greater")
    assert out["estimate"][0] == pytest.approx(0.2)
    assert 0.0 <= out["p_value"][0] <= 1.0
