"""Smoke tests that the plotnine visualization layers compose and render."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from plotnine import ggplot

import moderndive as md
from moderndive import (
    get_confidence_interval,
    shade_confidence_interval,
    shade_p_value,
    specify,
    visualize,
)


def test_visualize_returns_ggplot():
    almonds = md.load_almonds_sample_100()
    boot = (
        specify(almonds, response="weight")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    p = visualize(boot)
    assert isinstance(p, ggplot)


def test_shade_layers_increase_layer_count_and_render(tmp_path):
    almonds = md.load_almonds_sample_100()
    boot = (
        specify(almonds, response="weight")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )
    ci = get_confidence_interval(boot, level=0.95, type="percentile")
    base = visualize(boot)
    shaded = base + shade_confidence_interval(endpoints=ci)
    assert len(shaded.layers) > len(base.layers)
    out = tmp_path / "ci.png"
    shaded.save(out, width=5, height=3, dpi=70, verbose=False)
    assert out.exists() and out.stat().st_size > 0


def test_shade_p_value_renders(tmp_path):
    spotify = md.load_spotify_metal_deephouse()
    obs = specify(spotify, formula="popular_or_not ~ track_genre", success="popular").calculate(
        stat="diff in props", order=("metal", "deep-house")
    )
    null = (
        specify(spotify, formula="popular_or_not ~ track_genre", success="popular")
        .hypothesize(null="independence")
        .generate(reps=200, type="permute", seed=2)
        .calculate(stat="diff in props", order=("metal", "deep-house"))
    )
    p = visualize(null, bins=25) + shade_p_value(obs_stat=obs, direction="right")
    out = tmp_path / "pval.png"
    p.save(out, width=5, height=3, dpi=70, verbose=False)
    assert out.exists() and out.stat().st_size > 0
