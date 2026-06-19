"""Dual-engine visualization tests (plotly default + plotnine), shading, save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import plotly.graph_objects as go
import pytest
from plotnine import ggplot

import moderndive as md
from moderndive import (
    get_confidence_interval,
    shade_confidence_interval,
    shade_p_value,
    specify,
    visualize,
)
from moderndive.infer.viz import InferPlot, ShadeSpec, visualize_fit


def _fit_obs_null_ci():
    sar = md.load_saratoga_houses()
    f = "price ~ living_area + bedrooms"
    obs = specify(sar, formula=f).fit()
    dist = specify(sar, formula=f).generate(reps=60, type="bootstrap", seed=1).fit()
    null = (
        specify(sar, formula=f)
        .hypothesize(null="independence")
        .generate(reps=60, type="permute", seed=1)
        .fit()
    )
    return obs, dist, null, dist.get_confidence_interval(level=0.95)


def _boot():
    almonds = md.load_almonds_sample_100()
    return (
        specify(almonds, response="weight")
        .generate(reps=200, type="bootstrap", seed=1)
        .calculate(stat="mean")
    )


def _null_diff():
    spotify = md.load_spotify_metal_deephouse()
    obs = specify(
        spotify, formula="popular_or_not ~ track_genre", success="popular"
    ).calculate(stat="diff in props", order=("metal", "deep-house"))
    null = (
        specify(spotify, formula="popular_or_not ~ track_genre", success="popular")
        .hypothesize(null="independence")
        .generate(reps=200, type="permute", seed=2)
        .calculate(stat="diff in props", order=("metal", "deep-house"))
    )
    return obs, null


def test_visualize_default_engine_is_plotly():
    p = visualize(_boot())
    assert isinstance(p, InferPlot) and p.engine == "plotly"
    assert isinstance(p.figure, go.Figure)


def test_visualize_plotnine_engine_returns_ggplot():
    p = visualize(_boot(), engine="plotnine")
    assert p.engine == "plotnine"
    assert isinstance(p.gg, ggplot)


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
@pytest.mark.parametrize("method", ["simulation", "theoretical", "both"])
def test_visualize_all_methods_both_engines(engine, method):
    p = visualize(_boot(), engine=engine, method=method)
    assert isinstance(p, InferPlot)


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
@pytest.mark.parametrize("direction", ["right", "left", "two-sided"])
def test_shade_p_value_compose_both_engines(engine, direction):
    obs, null = _null_diff()
    # operator form
    p = visualize(null, bins=25, engine=engine) + shade_p_value(obs_stat=obs, direction=direction)
    assert isinstance(p, InferPlot)
    # keyword form
    p2 = visualize(
        null, bins=25, engine=engine, shade_pvalue={"obs_stat": float(obs), "direction": direction}
    )
    assert isinstance(p2, InferPlot)


@pytest.mark.parametrize("engine", ["plotly", "plotnine"])
def test_shade_confidence_interval_both_engines(engine):
    boot = _boot()
    ci = get_confidence_interval(boot, level=0.95, type="percentile")
    base = visualize(boot, engine=engine)
    # DataFrame endpoints (operator) + tuple endpoints (keyword), custom color
    shaded = base + shade_confidence_interval(endpoints=ci, color="green")
    assert isinstance(shaded, InferPlot)
    assert isinstance(visualize(boot, engine=engine, shade_ci=(0.5, 0.6)), InferPlot)


def test_plotnine_shade_adds_layers():
    boot = _boot()
    ci = get_confidence_interval(boot, level=0.95, type="percentile")
    base = visualize(boot, engine="plotnine")
    shaded = base + shade_confidence_interval(endpoints=ci)
    assert len(shaded.gg.layers) > len(base.gg.layers)


def test_plotnine_add_arbitrary_layer():
    from plotnine import labs

    p = visualize(_boot(), engine="plotnine") + labs(title="custom")
    assert isinstance(p.gg, ggplot)


def test_visualize_fit_both_engines():
    sar = md.load_saratoga_houses()
    boot = (
        specify(sar, formula="price ~ living_area")
        .generate(reps=50, type="bootstrap", seed=1)
        .fit()
    )
    assert isinstance(boot.visualize().figure, go.Figure)
    assert isinstance(boot.visualize(engine="plotnine").gg, ggplot)


def test_save_plotly_html_and_plotnine_png(tmp_path):
    boot = _boot()
    html = tmp_path / "p.html"
    visualize(boot, engine="plotly").save(html)
    assert html.exists() and html.stat().st_size > 0
    png = tmp_path / "p.png"
    visualize(boot, engine="plotnine").save(png, width=5, height=3, dpi=70, verbose=False)
    assert png.exists() and png.stat().st_size > 0


def test_invalid_engine_and_method():
    boot = _boot()
    with pytest.raises(ValueError):
        visualize(boot, engine="bogus")
    with pytest.raises(ValueError):
        visualize(boot, method="bogus")


def test_inferplot_errors_and_reprs():
    boot = _boot()
    # .gg unavailable on plotly engine
    with pytest.raises(AttributeError):
        _ = visualize(boot, engine="plotly").gg
    # adding a non-ShadeSpec to a plotly InferPlot is a TypeError
    with pytest.raises(TypeError):
        visualize(boot, engine="plotly") + 42
    # bad coercion of shade_pvalue keyword
    with pytest.raises(TypeError):
        visualize(boot, engine="plotly", shade_pvalue=42)
    # repr / _repr_html_ delegate to the figure
    p = visualize(boot, engine="plotly")
    assert isinstance(repr(p), str)
    assert isinstance(p._repr_html_(), str)


def test_shade_spec_is_frozen_dataclass():
    spec = shade_p_value(obs_stat=1.0, direction="right")
    assert isinstance(spec, ShadeSpec) and spec.kind == "p_value"
    ci_spec = shade_confidence_interval((1.0, 2.0))
    assert ci_spec.kind == "confidence_interval" and ci_spec.lower == 1.0


def test_shade_spec_passthrough_via_keywords():
    boot = _boot()
    # passing ready-made ShadeSpecs to the keyword args hits the passthrough branch
    p = visualize(
        boot,
        engine="plotly",
        shade_pvalue=shade_p_value(obs_stat=float(boot.data["stat"].mean()), direction="right"),
        shade_ci=shade_confidence_interval((0.5, 0.6)),
    )
    assert isinstance(p, InferPlot)


def test_inferplot_show_plotnine():
    # show() delegates to the underlying figure (Agg backend → no window)
    visualize(_boot(), engine="plotnine").show()


def test_plotly_save_image_calls_write_image(monkeypatch, tmp_path):
    p = visualize(_boot(), engine="plotly")
    captured = {}
    monkeypatch.setattr(p.figure, "write_image", lambda path: captured.setdefault("path", path))
    p.save(tmp_path / "fig.png")
    assert captured["path"].endswith("fig.png")


def test_shade_p_value_invalid_direction_raises():
    boot = _boot()
    with pytest.raises(ValueError):
        visualize(boot, engine="plotly") + shade_p_value(obs_stat=1.0, direction="sideways")


def test_apply_shade_on_empty_plotly_figure():
    # a figure with no x-bearing traces exercises the data-range fallback
    empty = InferPlot(go.Figure(), "plotly")
    shaded = empty + shade_confidence_interval((1.0, 2.0))
    assert isinstance(shaded.figure, go.Figure)


# ---- per-facet shading for faceted regression-fit plots (both engines) ----


@pytest.mark.parametrize("direction", ["right", "left", "two-sided"])
def test_fit_per_facet_pvalue_plotly(direction):
    obs, _dist, null, _ci = _fit_obs_null_ci()
    p = visualize_fit(null, engine="plotly") + shade_p_value(obs_stat=obs, direction=direction)
    assert p.terms == ["intercept", "living_area", "bedrooms"]
    # one shaded region per term (plus vlines); shapes scale with the 3 facets
    assert len(p.figure.layout.shapes) >= 3 * 3 if direction == "two-sided" else 3 * 2


def test_fit_per_facet_ci_plotly():
    _obs, dist, _null, ci = _fit_obs_null_ci()
    p = visualize_fit(dist, engine="plotly") + shade_confidence_interval(ci)
    # 3 terms x (1 vrect + 2 vlines) = 9 shapes
    assert len(p.figure.layout.shapes) == 9


def test_fit_per_facet_pvalue_plotnine():
    obs, _dist, null, _ci = _fit_obs_null_ci()
    base = visualize_fit(null, engine="plotnine")
    shaded = base + shade_p_value(obs_stat=obs, direction="two-sided")
    assert len(shaded.gg.layers) > len(base.gg.layers)
    assert isinstance(shaded.gg, ggplot)


def test_fit_per_facet_ci_plotnine_keyword():
    _obs, dist, _null, ci = _fit_obs_null_ci()
    # keyword form on FitResult.visualize, custom color
    g = dist.visualize(engine="plotnine", shade_ci=ci)
    base = visualize_fit(dist, engine="plotnine")
    assert len(g.gg.layers) > len(base.gg.layers)


def test_fit_shade_pvalue_keyword_dict_form():
    obs, _dist, null, _ci = _fit_obs_null_ci()
    p = null.visualize(engine="plotly", shade_pvalue={"obs_stat": obs, "direction": "right"})
    assert len(p.figure.layout.shapes) >= 3


def test_per_term_obs_accepts_table_and_dict():
    obs, _dist, _null, _ci = _fit_obs_null_ci()
    # term/estimate frame, term/stat frame, and dict all yield per-term specs
    spec_df = shade_p_value(obs.data, direction="right")
    spec_stat = shade_p_value(obs.data.rename({"estimate": "stat"}), direction="right")
    spec_dict = shade_p_value({"living_area": 90.0, "bedrooms": -7000.0}, direction="right")
    for spec in (spec_df, spec_stat, spec_dict):
        assert spec.per_term is not None and spec.obs_stat is None


def test_per_term_shade_on_nonfacet_raises():
    obs, _dist, _null, _ci = _fit_obs_null_ci()
    boot = _boot()
    with pytest.raises(TypeError):
        visualize(boot, engine="plotly") + shade_p_value(obs_stat=obs, direction="right")


def test_per_facet_skips_unknown_terms():
    import polars as pl

    _obs, dist, _null, ci = _fit_obs_null_ci()
    # add a term not present in the facets; it must be silently skipped (both engines)
    extra = pl.concat([ci, ci.head(1).with_columns(pl.lit("ghost").alias("term"))])
    p = visualize_fit(dist, engine="plotly") + shade_confidence_interval(extra)
    assert len(p.figure.layout.shapes) == 9  # ghost term ignored
    g = visualize_fit(dist, engine="plotnine") + shade_confidence_interval(extra)
    assert isinstance(g.gg, ggplot)
    # p-value path: a dict with a term not among the facets is skipped too
    obs_with_ghost = {"living_area": 90.0, "ghost": 1.0}
    pp = visualize_fit(dist, engine="plotly") + shade_p_value(obs_with_ghost, direction="right")
    assert len(pp.figure.layout.shapes) == 2  # only living_area shaded (1 vline + 1 rect)
    gp = visualize_fit(dist, engine="plotnine") + shade_p_value(obs_with_ghost, direction="right")
    assert isinstance(gp.gg, ggplot)


def test_subplot_xrange_fallback():
    from moderndive.infer.viz._plotly import _subplot_xrange

    assert _subplot_xrange(go.Figure(), 1) == (-1.0, 1.0)
