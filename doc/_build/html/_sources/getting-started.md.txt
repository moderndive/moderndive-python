# Getting started

This page walks through a complete analysis end to end, then points you at the
task guides for more depth.

## Install

```bash
pip install moderndive
```

`moderndive` returns [polars](https://pola.rs) DataFrames, but every function also
accepts pandas DataFrames as input.

## Load a dataset

All datasets ship with the package and load with `load_<name>()`:

```python
import moderndive as md

promotions = md.load_promotions()
promotions.head()
```

```text
shape: (5, 3)
┌─────┬──────────┬────────┐
│ id  ┆ decision ┆ gender │
│ i64 ┆ str      ┆ str    │
╞═════╪══════════╪════════╡
│ 1   ┆ promoted ┆ male   │
│ 2   ┆ promoted ┆ male   │
│ …   ┆ …        ┆ …      │
└─────┴──────────┴────────┘
```

List everything that's available with `md.available_datasets()` (58 datasets), and
see {doc}`datasets` for a thematic tour.

## A first summary

`tidy_summary` gives a per-variable five-number summary (numeric) or counts
(categorical):

```python
from moderndive import tidy_summary

tidy_summary(md.load_almonds_sample_100(), columns=["weight"])
```

## The inference pipeline

The core grammar mirrors R `infer`. You build a pipeline and read it like a
sentence:

```python
from moderndive import specify, observe, get_p_value

# 1. The observed statistic
obs = observe(
    promotions, formula="decision ~ gender", success="promoted",
    stat="diff in props", order=("male", "female"),
)

# 2. A null distribution: specify → hypothesize → generate → calculate
null = (
    specify(promotions, formula="decision ~ gender", success="promoted")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=42)
    .calculate(stat="diff in props", order=("male", "female"))
)

# 3. Summarize
get_p_value(null, obs_stat=obs, direction="right")   # ≈ 0.025
```

Each verb has a focused guide: {doc}`guides/sampling`,
{doc}`guides/confidence-intervals`, and {doc}`guides/hypothesis-testing`.

## Visualizing — choose your engine

Plots default to **plotly** (interactive). Pass `engine="plotnine"` for
grammar-of-graphics output. The composition syntax is identical:

```python
from moderndive import visualize, shade_p_value

# Interactive plotly figure
visualize(null) + shade_p_value(obs_stat=obs, direction="right")

# Same plot, plotnine
visualize(null, engine="plotnine") + shade_p_value(obs_stat=obs, direction="right")
```

See {doc}`guides/plotting` for shading, confidence-interval overlays, theoretical
overlays, and the regression-model plots.

## Regression

```python
import statsmodels.formula.api as smf
from moderndive import get_regression_table

houses = md.load_saratoga_houses()
model = smf.ols("price ~ living_area + bedrooms", data=houses.to_pandas()).fit()
get_regression_table(model)
```

Full details in {doc}`guides/regression`.
