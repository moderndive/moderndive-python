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

```{code-cell} python
import moderndive as md

yawn = md.load_mythbusters_yawn()
yawn.head()
```


List everything that's available with `md.available_datasets()` (58 datasets), and
see {doc}`datasets` for a thematic tour.

To explore a dataset in an interactive, searchable/sortable table, use `View()`
(install the optional extra with `pip install "moderndive[view]"`):

```python
md.View(yawn)
```

## A first summary

`tidy_summary` gives a per-variable five-number summary (numeric) or counts
(categorical):

```{code-cell} python
from moderndive import tidy_summary

tidy_summary(md.load_almonds_sample_100(), columns=["weight"])
```

`count_missing` reports how many `null` values each column has, sorted worst-first
— handy for a quick data-quality check:

```{code-cell} python
from moderndive import count_missing

count_missing(md.load_evals())
```

## The inference pipeline

The core grammar mirrors R `infer`. You build a pipeline and read it like a
sentence:

```{code-cell} python
from moderndive import specify, observe, get_p_value

# 1. The observed statistic: do "seeded" people yawn more than the control group?
obs = observe(
    yawn, formula="yawn ~ group", success="yes",
    stat="diff in props", order=("seed", "control"),
)

# 2. A null distribution: specify → hypothesize → generate → calculate
null = (
    yawn.specify(formula="yawn ~ group", success="yes")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=42)
    .calculate(stat="diff in props", order=("seed", "control"))
)

# 3. Summarize
get_p_value(null, obs_stat=obs, direction="right")
```

Each verb has a focused guide: {doc}`guides/sampling`,
{doc}`guides/confidence-intervals`, and {doc}`guides/hypothesis-testing`.

## Visualizing — choose your engine

Plots default to **plotly** (interactive). Pass `engine="plotnine"` for
grammar-of-graphics output. The composition syntax is identical:

```{note}
The plots shown in this documentation are **static images**. Running the code
yourself yields **interactive** plotly figures by default.
```

```{code-cell} python
from moderndive import visualize, shade_p_value

# Interactive plotly figure
visualize(null) + shade_p_value(obs_stat=obs, direction="right")

# Same plot, plotnine
visualize(null, engine="plotnine") + shade_p_value(obs_stat=obs, direction="right")
```

See {doc}`guides/plotting` for shading, confidence-interval overlays, theoretical
overlays, and the regression-model plots.

## Regression

```{code-cell} python
import statsmodels.formula.api as smf
from moderndive import get_regression_table

houses = md.load_saratoga_houses()
model = smf.ols("price ~ living_area + bedrooms", data=houses.to_pandas()).fit()
get_regression_table(model)
```

Full details in {doc}`guides/regression`.
