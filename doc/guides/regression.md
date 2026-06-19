---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} python
:tags: [remove-input]
import matplotlib
matplotlib.use("Agg")
import plotly.io as pio
pio.renderers.default = "png"
```

# Regression

The regression helpers turn a fitted [statsmodels](https://www.statsmodels.org)
model into tidy polars tables — the analog of R `moderndive`'s
`get_regression_table()` / `get_regression_points()` / `get_regression_summaries()`.

Fit models with statsmodels' formula API, then tidy them:

```{code-cell} python
import statsmodels.formula.api as smf
import moderndive as md
from moderndive import (
    get_regression_table, get_regression_points, get_regression_summaries, get_correlation,
)

houses = md.load_saratoga_houses()
model = smf.ols("price ~ living_area + bedrooms", data=houses.to_pandas()).fit()
```

## Coefficient table (with confidence intervals)

```{code-cell} python
get_regression_table(model)
```


Change the confidence level with `conf_level=` (e.g. `0.99`).

## Fitted values & residuals

```{code-cell} python
get_regression_points(model).head()
# columns: ID, price, living_area, bedrooms, price_hat, residual
```

## Model-fit summaries

```{code-cell} python
get_regression_summaries(model)
# r_squared, adj_r_squared, mse, rmse, sigma, statistic (F), p_value, df, nobs
```

## Correlation

```{code-cell} python
get_correlation(houses, "price ~ living_area")   # ≈ 0.759
# or: get_correlation(houses, x="living_area", y="price")
```

## Visualizing models

Two ports of R `moderndive`'s ggplot helpers, both dual-engine:

```{code-cell} python
from moderndive import gg_parallel_slopes, gg_categorical_model

evals = md.load_evals()

# Parallel-slopes model: one common slope, a separate intercept per group
gg_parallel_slopes(evals, response="score", explanatory="age", by="gender")            # plotly
gg_parallel_slopes(evals, response="score", explanatory="age", by="gender",
                   engine="plotnine")

# Regression with a single categorical predictor
gg_categorical_model(evals, response="score", explanatory="rank")
```

For the plotnine engine you can also drop the parallel-slopes lines onto your own
`ggplot` with `geom_parallel_slopes()`.

## Inference for regression coefficients

`fit()` runs the regression on every bootstrap/permutation replicate, giving a
distribution per coefficient. Pair it with `visualize_fit` and **per-facet**
shading:

```{code-cell} python
from moderndive import specify
from moderndive.infer.viz import visualize_fit
from moderndive import shade_confidence_interval, shade_p_value

f = "price ~ living_area + bedrooms"
obs_fit = houses.specify(formula=f).fit()

# Bootstrap distribution of each coefficient → per-term CIs
boot_fit = houses.specify(formula=f).generate(reps=1000, type="bootstrap", seed=1).fit()
boot_fit.get_confidence_interval(level=0.95)          # one row per term
visualize_fit(boot_fit) + shade_confidence_interval(boot_fit.get_confidence_interval())

# Null distribution → per-term p-values, each facet shaded at its own estimate
null_fit = (
    houses.specify(formula=f).hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=1).fit()
)
null_fit.get_p_value(obs_stat=obs_fit, direction="two-sided")
visualize_fit(null_fit) + shade_p_value(obs_stat=obs_fit, direction="two-sided")
```

Per-facet shading works in both engines: `shade_p_value` / `shade_confidence_interval`
accept the observed `FitResult` or a term-keyed table, and each facet is shaded
from its own value.
