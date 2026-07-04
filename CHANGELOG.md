# Changelog

<!--
Conventions: group entries under ### Added / ### Changed / ### Fixed.
ANY behavior change that could alter existing users' results is a BREAKING change
and MUST go in a dedicated, top-of-version "### ⚠️ Breaking changes" section that
states (a) exactly what changed, (b) how to restore the previous behavior, and
(c) why. Breaking changes require a minor/major version bump, never a patch.
-->

## 0.2.0

Ongoing **R data-parity** line: bundling the datasets the R `moderndive`
ecosystem exposes (including the end-of-chapter exercise data) so the Python
edition and its exercises can match the R book one-for-one. More datasets will be
added here as chapters are ported.

### Added — datasets

- `load_olympic_athletes()` — Olympic athletes 1896-2016, one row per
  athlete-event (~315k rows; the `olympicAthletes` R package's `olympic_athletes`).
- `load_medal_table()` — Olympic medal counts per Games × NOC
  (`olympicAthletes::medal_table`).
- `load_editions()` — metadata for every Olympic Games edition
  (`olympicAthletes::editions`).
- The `olympicAthletes` **convenience subsets** used by the exercises:
  `load_athletics_athletes()`, `load_gymnastics_athletes()`,
  `load_basketball_athletes()`, `load_volleyball_athletes()`,
  `load_curling_athletes()`, `load_art_competitions_athletes()`,
  `load_team_sport_athletes()`, `load_olympic_athletes_2024()` (Paris 2024),
  `load_usa_summer_medals()`, `load_paris_2024_top_medals()`, and
  `load_season_counts()` — exported verbatim from `olympicAthletes` 0.5.7 for
  exact parity.

### Added

- `tidy_summary()` gains an `interpolation=` parameter controlling how `Q1`/`Q3`
  are computed. The default is unchanged from 0.1.0 (`"nearest"`); pass
  `interpolation="linear"` for R's `quantile()` type 7 — also NumPy's default and
  the quartiles drawn by Plotly/ggplot2 boxplots. **Non-breaking** (default
  preserved).
- `chisq_test()` gains a `correct=` parameter for Yates' continuity correction on
  the test of independence. The default is unchanged from 0.1.0 (`correct=False`,
  the uncorrected Pearson statistic, matching the simulation-based
  `calculate(stat="Chisq")`); pass `correct=True` to match R's
  `chisq.test`/`prop_test`. As in R, the correction only affects 2x2 tables.
  **Non-breaking** (default preserved).

## 0.1.0 (2026-06-20)

Initial release of the Python companion to **ModernDive: Statistical Inference
via Data Science** — a faithful port of the R `moderndive` and `infer` packages,
targeting full functional, dataset, and argument-level parity on a modern Python
stack (polars, plotly, plotnine, statsmodels).

### Inference grammar (infer)

- Tidy pipeline `specify → hypothesize → generate → calculate`, plus `fit()`
  (per-term inference for multiple regression), `observe()`, and `assume()`
  (theoretical `t`/`z`/`F`/`Chisq`). `specify()` is also a DataFrame method, so
  `df.specify(...)` mirrors R's `df %>% specify(...)`.
- Full `calculate(stat=)` vocabulary: `mean`, `median`, `sum`, `sd`, `prop`,
  `count`, `diff in means`/`medians`/`props`, `ratio of means`/`props`,
  `odds ratio`, `slope`, `correlation`, `t`, `z`, `F`, `Chisq`, **plus any
  custom callable** statistic.
- `hypothesize()` point nulls for the mean (`mu`), median (`med`), proportion
  (`p`), and one-sample mean `z` (`sigma`); a `{level: probability}` mapping
  drives the chi-square **goodness-of-fit** test.
- `generate()` bootstrap / permute / draw (with `"simulate"` as an alias for
  `"draw"`), and `variables=` to choose the permuted column.
- Confidence intervals (`percentile`, `se`, `bias-corrected`) and simulation
  p-values via `get_confidence_interval` / `get_p_value`. British-spelling and
  short aliases included.

### Theory-based wrappers

- `t_test`, `prop_test`, `chisq_test`, `t_stat`, `chisq_stat`, plus the
  `moderndive.theory` module. `prop_test` mirrors R's `prop.test` (chi-square by
  default with Yates correction, `z=`, `conf_int=`/`conf_level=`, and a
  Wilson-score CI for one proportion); `chisq_test(response=, p=)` runs
  goodness-of-fit.

### Regression & summary helpers (moderndive)

- `get_regression_table` (`conf_level`, `exponentiate`,
  `default_categorical_levels`), `get_regression_points` (`newdata=`, `ID=`,
  in-formula-transformation handling), `get_regression_summaries`,
  `get_correlation` (`method=pearson/spearman/kendall`, multiple predictors,
  `wide=`, `na_rm=`), `pop_sd`, `tidy_summary`, and `count_missing`.
- Both OLS and **GLM** models are supported (GLM points on the response scale,
  GLM-shaped summaries), via the formula and array statsmodels APIs.

### Dual-engine plotting (plotly default, plotnine optional)

- Every plot takes `engine="plotly"` (default, interactive) or
  `engine="plotnine"`: `visualize` / `shade_p_value` / `shade_confidence_interval`
  (with `fill=`, `dens_color=`, `method="both"/"theoretical"`, and per-facet
  shading for `visualize_fit`), `pairplot`, `gg_parallel_slopes` /
  `geom_parallel_slopes` (`alpha=`), `gg_categorical_model` /
  `geom_categorical_model`, and `plot_3d_regression`. `visualize()` returns an
  `InferPlot` that composes with shading via `+` and renders in Jupyter/Quarto
  for both engines. `View()` renders an interactive table via the optional
  `itables` package. Static image export uses the optional `moderndive[image]`
  extra (kaleido); both extras are optional so the core stays pure-Python and
  Pyodide-installable.

### Datasets

- 58 bundled datasets via `load_*()` loaders returning polars DataFrames — the R
  `moderndive`/`infer`, `nycflights23`, `gapminder`, ISLR2, and FiveThirtyEight
  data used in the book.

### Quality & tooling

- Beginner-friendly, `infer`-style messages and errors (a summary line + `→`
  hint bullets; suppressible `ModernDiveMessage` notes).
- 100% statement coverage enforced in CI, including a vignette smoke test and
  R-validated reference values; parity-drift CI against the upstream R packages;
  docs on Read the Docs with runnable examples.
