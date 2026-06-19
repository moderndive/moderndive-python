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

Did promotion decisions depend on the applicant's (perceived) gender? Shuffle the
labels 1000 times to build a null distribution and read off a p-value:

```python
import moderndive as md
from moderndive import specify, observe, get_p_value, visualize, shade_p_value

promotions = md.load_promotions()

# Observed difference in promotion rates (male − female)
obs = observe(
    promotions, formula="decision ~ gender", success="promoted",
    stat="diff in props", order=("male", "female"),
)

# Null distribution under "gender doesn't matter" (permutation)
null = (
    specify(promotions, formula="decision ~ gender", success="promoted")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=42)
    .calculate(stat="diff in props", order=("male", "female"))
)

get_p_value(null, obs_stat=obs, direction="right")   # ≈ 0.025

# Visualize it (interactive plotly by default; engine="plotnine" also works)
visualize(null) + shade_p_value(obs_stat=obs, direction="right")
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
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
