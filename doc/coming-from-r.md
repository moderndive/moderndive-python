# Coming from R

If you know the R `moderndive` and `infer` packages, this page maps the API to
Python. The grammar is the same; the main differences are Python method-chaining
(`.hypothesize()` instead of the `|>`/`%>%` pipe) and polars DataFrames.

## The infer pipeline

```r
# R
mtcars %>%
  specify(response = mpg) %>%
  hypothesize(null = "point", mu = 20) %>%
  generate(reps = 1000, type = "bootstrap") %>%
  calculate(stat = "mean")
```

```python
# Python — verbs are methods on the returned objects
(
    specify(data, response="mpg")
    .hypothesize(null="point", mu=20)
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
```

`specify(formula="y ~ x")` works just like R's formula interface; `success=` marks
the success level for categorical responses.

## Function map

| R (`infer` / `moderndive`)        | Python (`moderndive`)                                  |
| --------------------------------- | ------------------------------------------------------ |
| `specify()`                       | `specify()`                                            |
| `hypothesize()` / `hypothesise()` | `.hypothesize()` / `.hypothesise()`                    |
| `generate()`                      | `.generate()`                                          |
| `calculate()`                     | `.calculate()`                                         |
| `fit()`                           | `.fit()`                                               |
| `assume()`                        | `assume()`                                             |
| `observe()`                       | `observe()`                                            |
| `get_p_value()` / `get_pvalue()`  | `get_p_value()` / `get_pvalue()`                       |
| `get_confidence_interval()` / `get_ci()` | `get_confidence_interval()` / `get_ci()`        |
| `visualize()` / `visualise()`     | `visualize()` / `visualise()`                          |
| `shade_p_value()` / `shade_pvalue()` | `shade_p_value()` / `shade_pvalue()`                |
| `shade_confidence_interval()` / `shade_ci()` | `shade_confidence_interval()` / `shade_ci()` |
| `t_test()`, `prop_test()`, `chisq_test()` | same names                                     |
| `t_stat()`, `chisq_stat()`        | same names                                             |
| `rep_sample_n()` / `rep_slice_sample()` | `rep_sample_n()` / `rep_slice_sample()`          |
| `get_regression_table()`          | `get_regression_table(model)`                          |
| `get_regression_points()`         | `get_regression_points(model)`                         |
| `get_regression_summaries()`      | `get_regression_summaries(model)`                      |
| `get_correlation()`               | `get_correlation(data, "y ~ x")`                       |
| `pop_sd()`                        | `pop_sd()`                                             |
| `geom_parallel_slopes()`          | `geom_parallel_slopes()` (plotnine) / `gg_parallel_slopes(engine=...)` |
| `geom_categorical_model()`        | `gg_categorical_model(engine=...)`                     |
| `tidy_summary()`                  | `tidy_summary()`                                       |

## Key differences

- **Pipe → methods.** `x %>% hypothesize(...)` becomes `x.hypothesize(...)`.
- **Models.** Where R passes an `lm()` object, Python passes a fitted
  [statsmodels](https://www.statsmodels.org) model:
  `smf.ols("y ~ x", data=df.to_pandas()).fit()`.
- **Plotting engine.** R returns ggplot2; Python defaults to **plotly**
  (interactive) with `engine="plotnine"` available everywhere. Plots compose with
  `+` in both engines.
- **DataFrames.** Inputs/outputs are polars; pass `.to_pandas()` when a downstream
  tool needs pandas.
- **Reproducibility.** Pass `seed=` to `generate()` (R uses `set.seed()`).

## Same datasets

Most R `moderndive`/`infer` datasets are bundled here under the same name —
`load_promotions()`, `load_pennies()`, `load_gss()`, etc. See {doc}`datasets`.
