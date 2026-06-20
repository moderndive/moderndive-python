# moderndive (Python) — NEWS

## 0.1.0

First substantive release of the Python companion to **ModernDive: Statistical
Inference via Data Science**, a faithful port of the R `moderndive` and `infer`
packages to a modern Python data-science stack (polars, plotly, plotnine,
statsmodels). The package targets **full functional, dataset, and
argument-level parity** with the two R packages, with a few deliberate
Python/teaching-oriented choices documented in
[R compatibility notes](doc/r-compatibility.md).

### Inference grammar (infer)

- The tidy pipeline `specify → hypothesize → generate → calculate`, plus
  `fit()`, `observe()`, and `assume()` (theoretical `t`/`z`/`F`/`Chisq`).
- `specify()` is also a DataFrame method: `df.specify(...)` mirrors R's
  `df %>% specify(...)`.
- Full `calculate(stat=)` vocabulary: `mean`, `median`, `sum`, `sd`, `prop`,
  `count`, `diff in means`/`medians`/`props`, `ratio of means`/`props`,
  `odds ratio`, `slope`, `correlation`, `t`, `z`, `F`, `Chisq`, **plus any
  custom callable** statistic.
- Chi-square **goodness-of-fit** via a `{level: probability}` point null
  (`hypothesize(null="point", p=...)`, `generate(type="draw")`).
- `hypothesize(med=)` (median point null) and `sigma=` (one-sample mean `z`).
- `generate(type="simulate")` alias for `"draw"`, and `generate(variables=)` to
  choose the permuted column.
- Confidence intervals (`percentile`, `se`, `bias-corrected`) and
  simulation p-values, including **per-term** inference for multiple regression
  via `fit()`.

### Theory-based wrappers

- `t_test`, `prop_test`, `chisq_test`, `t_stat`, `chisq_stat`, plus the
  `theory` module. `prop_test` mirrors R's `prop.test` (chi-square default,
  Yates correction, Wilson-score CI for one proportion; `z=`, `correct=`,
  `conf_int=`, `conf_level=`). `chisq_test(response=, p=)` runs goodness-of-fit.

### Regression & summary helpers (moderndive)

- `get_regression_table` (with `conf_level`, `exponentiate`,
  `default_categorical_levels`), `get_regression_points` (with `newdata=`,
  `ID=`), `get_regression_summaries`, `get_correlation`
  (`method=pearson/spearman/kendall`, multi-predictor, `wide=`, `na_rm=`),
  `pop_sd`, `tidy_summary`, and `count_missing`.
- OLS **and** GLM models are supported (GLM points on the response scale,
  GLM-shaped summaries), via both the formula and array statsmodels APIs.

### Dual-engine plotting (plotly default, plotnine optional)

- Every plot takes `engine="plotly"` (default, interactive) or
  `engine="plotnine"`: `visualize` / `shade_p_value` / `shade_confidence_interval`
  (with `fill=`, `dens_color=`, `method="both"/"theoretical"`), `pairplot`,
  `gg_parallel_slopes` / `geom_parallel_slopes` (with `alpha=`),
  `gg_categorical_model` / `geom_categorical_model`, and `plot_3d_regression`.
- `View()` renders an interactive table via the optional `itables` package.

### Datasets

- 58 bundled datasets via `load_*()` loaders returning polars DataFrames —
  the R `moderndive`/`infer`, `nycflights23`, `gapminder`, ISLR2, and
  FiveThirtyEight data used in the book.

### Quality & tooling

- Beginner-friendly, `infer`-style messages and errors (summary + `→` hints;
  suppressible `ModernDiveMessage` notes).
- 100% test coverage enforced in CI, including a vignette smoke test and
  R-validated reference values.
- Parity-drift CI against the upstream R packages; docs on Read the Docs with
  runnable examples.
