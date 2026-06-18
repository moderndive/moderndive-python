# Changelog

## Unreleased

Full parity with the R `moderndive` and `infer` packages.

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
