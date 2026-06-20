# Changelog

## Unreleased

Full parity with the R `moderndive` and `infer` packages.

- **Chi-square goodness-of-fit** (closes the last `infer` vignette gap):
  `specify(response=cat).hypothesize(null="point", p={level: prob, ...})` with
  `generate(type="draw")` and `calculate(stat="Chisq")` now runs a one-variable
  goodness-of-fit test, and `chisq_test(data, response=, p=)` is the one-line
  wrapper. `hypothesize(p=...)` accepts a `{level: probability}` mapping.

- **Parity with R `moderndive` PR #144**:
  - `get_correlation()` now accepts multiple right-hand-side predictors
    (`"y ~ x1 + x2"`) — long output by default, `wide=True` for one column per
    predictor; a suppressible (`quiet=True`) note points to a full pairwise matrix.
  - `get_regression_table()`, `get_regression_points()`, and
    `get_regression_summaries()` now accept fitted **`glm()`** models. GLM points
    are on the response scale (e.g. probabilities), summaries are GLM-shaped
    (`deviance`, `null_deviance`, `aic`, `bic`, `log_lik`, …), and the table gains
    an `exponentiate=` argument for odds/rate ratios.
  - `get_regression_points()` handles **in-formula transformations**: a
    transformed outcome like `np.log(mpg)` is shown on the model scale as
    `log_mpg`/`log_mpg_hat`, and transformed predictors (`poly()`, `scale()`,
    `I()`) are shown as their original columns rather than leaking basis matrices.
  - New **`plot_3d_regression(data, "z ~ x + y")`**: interactive 3D scatter with a
    fitted regression plane (plotly).
  - New **`View()`**: renders a data frame as an interactive, searchable table via
    the optional `itables` package (`pip install "moderndive[view]"`) — the Python
    counterpart of R's `DT::datatable()` — with a graceful fallback when itables
    isn't installed.
  - `geom_categorical_model()` is available as an alias of `gg_categorical_model()`.
  - Messages and errors are now beginner-friendly, in the `infer` style (a short
    summary line followed by `→` hint bullets); informational notes use a
    dedicated, suppressible `ModernDiveMessage` category.
- **`count_missing()`**: a beginner-friendly helper that counts `null` values per
  column and returns a tidy `column`/`n_missing` data frame sorted from most to
  fewest missing — a gentler alternative to `df.select(pl.all().is_null().sum())`.
- **`InferPlot` now renders in Jupyter/Quarto for both engines**: added
  `_repr_mimebundle_`, which delegates to the wrapped figure. Previously only
  `_repr_html_` was implemented, so plotnine-engine plots
  (`visualize(..., engine="plotnine")`) rendered blank in notebooks and Quarto
  because a `ggplot` returns `None` from `_repr_html_` (it renders via
  `_repr_mimebundle_`). plotly-engine plots were unaffected.
- **Per-facet shading for regression-fit plots**: `shade_p_value` and
  `shade_confidence_interval` now accept per-term values (an observed `FitResult`,
  a `term`-keyed CI/p-value table, or a dict) so each facet of a faceted
  `visualize_fit()` plot is shaded from its own observed statistic / interval —
  in both the plotly and plotnine engines. (Previously shading was scalar-only
  and couldn't shade per facet, so faceted multiple-regression inference plots
  rendered without the overlay.)
- **Dual-engine plotting**: every plotting function (`visualize`,
  `shade_p_value`, `shade_confidence_interval`, `pairplot`, and the new model
  plots) now takes `engine="plotly"` (the new default) or `engine="plotnine"`.
  `visualize()` returns an `InferPlot` that composes with shading via `+` for
  both engines; `shade_*` return an engine-neutral `ShadeSpec`. **Breaking**:
  the default return type is now a plotly figure — pass `engine="plotnine"`
  (or `engine="seaborn"` for `pairplot`) for the previous behavior. `plotly` is
  a new dependency; static image export needs the optional `moderndive[image]`
  extra (kaleido).
- New `moderndive` functions: `get_correlation`, `pop_sd`,
  `get_regression_summaries`, and the model-plot helpers `gg_parallel_slopes`,
  `geom_parallel_slopes`, `gg_categorical_model`.
- New `infer` features: `visualize(method="both"|"theoretical")` overlay,
  `generate(type="simulate")` as an alias for `"draw"`, and `sigma=` in
  `hypothesize()` for a one-sample mean `z` statistic.
- 25 additional datasets for full dataset parity (`pennies*`, `promotions*`,
  `evals`, `avocados`, `babies`, `ipf_lifts`, `coffee_ratings`, `gss`, and more).

## 0.1.0 (unreleased)

Initial release of the Python companion package for ModernDive.

- Tidy simulation-inference grammar mirroring R `infer`: `specify`,
  `hypothesize`, `generate` (bootstrap / permute / draw), `calculate`, `fit`,
  `observe`, `assume`, with full `stat=` vocabulary (mean, median, sum, sd,
  prop, count, diff in means/medians/props, ratio of means/props, odds ratio,
  slope, correlation, t, z, F, Chisq, and custom callables).
- Confidence intervals (percentile, SE, bias-corrected) and simulation p-values,
  including per-term inference for multiple regression via `fit()`.
- Theory-based wrapper tests: `t_test`, `prop_test`, `chisq_test`, `t_stat`,
  `chisq_stat`, plus the `theory` module.
- Regression helpers: `get_regression_table`, `get_regression_points`,
  `tidy_summary`.
- Visualization: `visualize` / `shade_p_value` / `shade_confidence_interval`
  (plotnine), `pairplot` (seaborn scatterplot matrix).
- `rep_slice_sample` / `rep_sample_n` for sampling activities.
- 30+ bundled datasets via `load_*()` loaders.
- 100% test coverage, enforced in CI.
