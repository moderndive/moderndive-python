# moderndive (Python)

<img src="https://raw.githubusercontent.com/moderndive/moderndive-python/main/doc/_static/moderndive-logo.png" align="right" height="160" alt="ModernDive hex logo" />

[![Tests](https://github.com/moderndive/moderndive-python/actions/workflows/tests.yml/badge.svg)](https://github.com/moderndive/moderndive-python/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/moderndive/moderndive-python/branch/main/graph/badge.svg)](https://codecov.io/gh/moderndive/moderndive-python)
[![Docs](https://readthedocs.org/projects/moderndive/badge/?version=latest)](https://moderndive.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The Python companion package for **ModernDive: Statistical Inference via Data
Science** — a faithful port of the R [`moderndive`](https://moderndive.github.io/moderndive/)
and [`infer`](https://infer.tidymodels.org) packages to a modern Python
data-science stack ([polars](https://pola.rs), [plotly](https://plotly.com/python/),
[plotnine](https://plotnine.org), [statsmodels](https://www.statsmodels.org)).

📖 **Documentation (with runnable examples): <https://moderndive.readthedocs.io>**

It is intentionally **pure-Python** (no compiled extensions) so it installs under
[Pyodide](https://pyodide.org) via `micropip` for in-browser execution.

## Installation

```bash
pip install moderndive          # from PyPI (once published)
# or, from source:
pip install git+https://github.com/moderndive/moderndive-python
```

## What's inside

- **A tidy simulation-inference grammar** mirroring R `infer`:
  `specify → hypothesize → generate → calculate`, plus `fit()` for multiple
  regression, `observe()`, and `assume()` (theoretical t/z/F/Chisq). `specify()`
  is also available as a DataFrame method, so you can write
  `df.specify(...)` just like R's `df %>% specify(...)`. `calculate(stat=...)`
  takes the full infer vocabulary **or any custom callable** test statistic.
  Summaries via `get_p_value` / `get_confidence_interval` (percentile, SE,
  bias-corrected); British-spelling and short aliases included.
- **Dual-engine plots**: `visualize` / `shade_p_value` / `shade_confidence_interval`
  (and every plot helper) take `engine="plotly"` (default, interactive) or
  `engine="plotnine"` — same code, your choice of output.
- **Theory-based wrapper tests**: `t_test`, `prop_test`, `chisq_test`,
  `t_stat`, `chisq_stat`, plus the `moderndive.theory` module.
- **Regression & summary helpers** mirroring R `moderndive`: `get_regression_table`,
  `get_regression_points`, `get_regression_summaries`, `get_correlation`,
  `pop_sd`, `tidy_summary`, `count_missing` (built on `statsmodels` where
  relevant, returning `polars` frames), plus the model plots
  `gg_parallel_slopes` / `geom_parallel_slopes` and
  `gg_categorical_model` / `geom_categorical_model`, and `pairplot`
  (the `GGally::ggpairs` analog).
- **Sampling**: `rep_slice_sample` / `rep_sample_n` for sampling-distribution
  activities.
- **58 datasets**: `load_*()` loaders returning `polars` DataFrames (the
  `moderndive`/`infer`, `nycflights23`, `gapminder`, ISLR2, and FiveThirtyEight
  datasets used in the book).

## Quick start

```python
import moderndive as md
from moderndive import get_p_value, visualize, shade_p_value

spotify = md.load_spotify_metal_deephouse()

# Observed difference in popularity rates (metal − deep house)
obs = (
    spotify
    .specify(formula="popular_or_not ~ track_genre", success="popular")
    .calculate(stat="diff in props", order=("metal", "deep-house"))
)

# Permutation null distribution
null = (
    spotify
    .specify(formula="popular_or_not ~ track_genre", success="popular")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=76)
    .calculate(stat="diff in props", order=("metal", "deep-house"))
)

get_p_value(null, obs_stat=obs, direction="right")

# Visualize — interactive plotly by default; engine="plotnine" for ggplot-style
visualize(null) + shade_p_value(obs_stat=obs, direction="right")
```

## Development

This repo uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create the environment
make test                    # run the test suite (enforces 100% coverage)
make build-data              # rebuild the bundled Parquet datasets (needs R; see tools/)
make build                   # build the wheel/sdist
```

The test suite is held at **100% statement coverage** (enforced in CI via
`--cov-fail-under=100`).

## License

MIT. The ModernDive book *content* is licensed CC-BY-NC-SA 4.0; this *software
package* is MIT-licensed.
