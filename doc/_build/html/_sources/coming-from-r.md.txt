# Coming from R

If you know the R `moderndive` and `infer` packages, this page maps the API to
Python. The grammar is the same; the main differences are Python method-chaining
(`.hypothesize()` instead of the `|>`/`%>%` pipe) and polars DataFrames.

## The infer pipeline

```r
# R
pennies %>%
  specify(response = year) %>%
  hypothesize(null = "point", mu = 1995) %>%
  generate(reps = 1000, type = "bootstrap") %>%
  calculate(stat = "mean")
```

```python
# Python — verbs are methods on the returned objects
(
    specify(md.load_pennies(), response="year")
    .hypothesize(null="point", mu=1995)
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
```

`specify(formula="y ~ x")` works just like R's formula interface; `success=` marks
the success level for categorical responses.

## Most names are identical

The overwhelming majority of functions keep **the same name** (including the
British-spelling and short-form aliases). They're called just as in R — as
methods on the pipeline where applicable:

> `specify`, `hypothesize`/`hypothesise`, `generate`, `calculate`, `fit`,
> `assume`, `observe`, `get_p_value`/`get_pvalue`,
> `get_confidence_interval`/`get_ci`, `visualize`/`visualise`,
> `shade_p_value`/`shade_pvalue`, `shade_confidence_interval`/`shade_ci`,
> `t_test`, `prop_test`, `chisq_test`, `t_stat`, `chisq_stat`,
> `rep_sample_n`/`rep_slice_sample`, `get_regression_table`,
> `get_regression_points`, `get_regression_summaries`, `get_correlation`,
> `pop_sd`, `tidy_summary`, `geom_parallel_slopes`.

## What's actually different

| R | Python | Why |
| --- | --- | --- |
| `x %>% f(...)` / `x \|> f(...)` | `x.f(...)` (method chaining) | no pipe operator in Python |
| `geom_categorical_model()` | `gg_categorical_model(engine=...)` | renamed to match `gg_parallel_slopes` |
| ggplot2 `geom_*` layers | plotly by default; `engine="plotnine"` for ggplot-style | dual-engine plotting |
| `lm(y ~ x, data)` object | a fitted **statsmodels** model: `smf.ols("y ~ x", data=df.to_pandas()).fit()` | regression backend |
| `get_correlation(df, y ~ x)` | `get_correlation(df, "y ~ x")` *or* `get_correlation(df, x="x", y="y")` | formula passed as a string |

## Other things to know

- **Plots compose with `+`** in both engines, and default to **plotly**
  (interactive); pass `engine="plotnine"` anywhere for ggplot-style output.
- **DataFrames** are polars in and out; pass `.to_pandas()` when a downstream tool
  needs pandas.
- **Reproducibility:** pass `seed=` to `generate()` (R uses `set.seed()`).

## Same datasets

Most R `moderndive`/`infer` datasets are bundled here under the same name —
`load_pennies()`, `load_promotions()`, `load_gss()`, etc. See {doc}`datasets`.
