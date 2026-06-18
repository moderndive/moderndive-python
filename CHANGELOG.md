# Changelog

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
