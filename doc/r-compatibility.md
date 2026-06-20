# R compatibility notes

This package aims to reproduce the **numerical results** of the R `moderndive`
and `infer` packages, not just their API. In several places that means making a
deliberate choice that differs from the "default" or "typical" Python/SciPy/
statsmodels behavior. They're collected here so the differences are intentional
and discoverable.

## Statistics

- **`pop_sd` divides by `n`** (population SD, `ddof=0`), unlike the sample
  `sd`/`calculate(stat="sd")` and `numpy.std`'s default which use `n − 1`. This
  matches R `moderndive::pop_sd`.
- **`get_regression_summaries`**: `mse` is the mean squared residual using `n`
  in the denominator (so `rmse = sqrt(mse)`), while `sigma` is the residual
  standard error using `n − p`. Both match the R package; note `mse` is *not*
  statsmodels' `mse_resid` (which uses `n − p`).
- **GLM summaries** use the log-likelihood-based BIC (`bic_llf`), not
  statsmodels' deviance-based `bic`, to align with `broom::glance`.
- **`get_p_value` two-sided** = `2 × min(left, right)` capped at 1 — the `infer`
  convention, which can differ slightly from a symmetric-tail p-value.
- **`F` and `Chisq`** are treated as inherently one-sided (right tail) for
  p-values regardless of the `direction` argument, matching `infer`.

## `prop_test` (mirrors R's `prop.test`, not a plain z-test)

- **Chi-square statistic by default** (with a `chisq_df` column), like R's
  `prop.test`. Pass `z=True` for the signed z-statistic that a "typical" Python
  two-proportion test would report.
- **Yates' continuity correction** is on by default (`correct=True`), as in R.
- **Confidence intervals** match R: a **Wilson score** interval for one
  proportion (not the Wald interval `statsmodels` returns by default), and a
  Wald interval **widened by the continuity correction** for a two-proportion
  difference.

## Correlation

- **`get_correlation` drops nulls by default** (`na_rm=True`) so beginners get a
  number rather than `nan`. R's `na.rm` defaults to `FALSE`; pass `na_rm=False`
  to match R exactly.
- `method="spearman"`/`"kendall"` use the SciPy implementations and match R's
  `cor(method=...)`.

## Regression points / tables

- **In-formula transformations** are reshaped to match R: a transformed outcome
  like `np.log(mpg)` is shown on the model scale as `log_mpg`/`log_mpg_hat`, and
  transformed predictors (`poly`, `scale`, `I`) show their original columns
  rather than the patsy basis matrix.
- **`get_regression_table`** prettifies categorical term names
  (`income[T.High]` → `income: High`) by default; `default_categorical_levels=True`
  keeps the raw statsmodels names.

## Datasets

- **Datetime columns are stored in UTC.** R's nycflights `time_hour` is stored in
  `America/New_York`; the bundled Parquet stores the identical instants in UTC, so
  a displayed hour differs by the UTC offset (the integer `hour` column matches R).
- **`early_january_2023_weather` is derived from `weather`.** The R dataset ships
  `temp`/`dewp`/`humid`/`pressure` as all-`NA`; this package recomputes the table
  from `weather` (Newark, first 15 days of Jan 2023) so those columns hold real
  values.

## Plotting

- **plotly is the default engine** (the book is moving to interactive plots);
  pass `engine="plotnine"` anywhere for grammar-of-graphics output. R returns
  ggplot2 objects.
- `visualize`'s default `bins` is 20 (R's is 15).
- Two-sided p-value shading mirrors the observed statistic about 0, matching
  `infer`'s `shade_p_value`.

## Reproducibility

- Pass `seed=` to `generate()` / `rep_slice_sample()` for reproducible draws
  (R uses `set.seed()`). Identical seeds will **not** reproduce R's exact random
  draws — only the statistical behavior matches, not the specific RNG stream.
