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
pio.renderers.default = "png"  # render plotly figures as static images in the docs
```

# moderndive (Python)

The Python companion package for **ModernDive: Statistical Inference via Data
Science** — a faithful port of the R [`moderndive`](https://moderndive.github.io/moderndive/)
and [`infer`](https://infer.tidymodels.org) packages to a modern Python
data-science stack ([polars](https://pola.rs), [plotly](https://plotly.com/python/),
[plotnine](https://plotnine.org), [statsmodels](https://www.statsmodels.org)).

If you teach or learn statistical inference the *tidy* way — `specify` →
`hypothesize` → `generate` → `calculate` → `visualize` — this package gives you
the same grammar in Python, plus regression helpers and the book's datasets.

## Why moderndive?

- **A tidy inference grammar.** Bootstrap confidence intervals and
  permutation/​simulation hypothesis tests read like sentences, mirroring R `infer`.
- **Regression helpers** that return tidy tables: `get_regression_table`,
  `get_regression_points`, `get_regression_summaries`, `get_correlation`.
- **Dual-engine plots.** Every plot takes `engine="plotly"` (default, interactive)
  or `engine="plotnine"` (grammar-of-graphics) — same code, your choice of output.
- **58 bundled datasets** via `load_*()` loaders returning polars DataFrames.
- **polars-first**, works with pandas too.

## Installation

```bash
pip install moderndive
```

Optional extra for saving plotly figures as static images (PNG/SVG):

```bash
pip install "moderndive[image]"   # adds kaleido
```

## 30-second example

Are tracks more likely to be popular in *metal* than in *deep house*? Compute the
observed difference in "popular" rates, then shuffle the genre labels 1000 times
to build a null distribution and read off a p-value.

```{code-cell} python
import moderndive as md
from moderndive import specify, observe, get_p_value, visualize, shade_p_value

spotify = md.load_spotify_metal_deephouse()

# Observed difference in "popular" proportions (metal − deep-house)
obs = observe(
    spotify, formula="popular_or_not ~ track_genre", success="popular",
    stat="diff in props", order=("metal", "deep-house"),
)
obs
```

```{code-cell} python
# Null distribution under "genre doesn't matter" (permutation)
null = (
    spotify.specify(formula="popular_or_not ~ track_genre", success="popular")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=76)
    .calculate(stat="diff in props", order=("metal", "deep-house"))
)
get_p_value(null, obs_stat=obs, direction="right")
```

```{code-cell} python
# Visualize it (interactive plotly by default; engine="plotnine" also works)
visualize(null) + shade_p_value(obs_stat=obs, direction="right")
```

```{note}
Plots throughout this documentation are rendered as **static images**. When you
run the code yourself, the default `engine="plotly"` produces **interactive**
figures (hover, zoom, pan); `engine="plotnine"` gives static grammar-of-graphics
plots.
```

## Where to next

- New here? Start with {doc}`getting-started`.
- Coming from R? See {doc}`coming-from-r` for a function-by-function map.
- Browse the task guides below, or jump to the {doc}`api`.

```{toctree}
:maxdepth: 1
:caption: Get started

getting-started
coming-from-r
datasets
r-compatibility
```

```{toctree}
:maxdepth: 1
:caption: Guides

guides/sampling
guides/confidence-intervals
guides/hypothesis-testing
guides/regression
guides/theory-based
guides/plotting
guides/infer-examples
guides/messages
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
