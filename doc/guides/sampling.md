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

# Sampling

Sampling activities (the "bowl" of balls, tactile samples) are how ModernDive
builds intuition for sampling variation. `rep_slice_sample` takes repeated samples
and stacks them with a `replicate` column — the analog of R
`moderndive::rep_slice_sample()` / `infer::rep_sample_n()`.

## The bowl

```{code-cell} python
import moderndive as md
import polars as pl

bowl = md.load_bowl()      # 2400 red/white balls
bowl.head()
```

## One virtual sample

```{code-cell} python
from moderndive import rep_slice_sample

sample = rep_slice_sample(bowl, n=50, seed=1)
# proportion red in this sample
sample.select((pl.col("color") == "red").mean().alias("prop_red"))
```

## Many samples → a sampling distribution

Take 1000 samples of size 50 and compute the proportion red in each:

```{code-cell} python
samples = rep_slice_sample(bowl, n=50, reps=1000, seed=1)

prop_red = (
    samples
    .group_by("replicate")
    .agg((pl.col("color") == "red").mean().alias("prop_red"))
)
prop_red.head()
```

That `prop_red` column is a **sampling distribution**. Visualize its spread the
same way you would any distribution (see {doc}`confidence-intervals` for building
one with the inference pipeline instead).

## With vs. without replacement

`rep_slice_sample` samples **without** replacement by default (like dealing from a
deck). Pass `replace=True` for bootstrap-style resampling:

```{code-cell} python
rep_slice_sample(bowl, n=50, reps=1000, replace=True, seed=1)
```

## Tactile samples

The hand-collected counterparts are bundled too:

```{code-cell} python
md.load_tactile_prop_red()   # 33 groups' samples of 50 balls
```

```{seealso}
`rep_sample_n` is an alias of `rep_slice_sample` (the older infer name). Both are
documented in the {doc}`../api`.
```
