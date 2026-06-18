# moderndive (Python)

[![Tests](https://github.com/moderndive/moderndive-python/actions/workflows/tests.yml/badge.svg)](https://github.com/moderndive/moderndive-python/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/moderndive/moderndive-python/branch/main/graph/badge.svg)](https://codecov.io/gh/moderndive/moderndive-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The Python companion package for **ModernDive: Statistical Inference via Data
Science** — a faithful port of the R [`moderndive`](https://moderndive.github.io/moderndive/)
and [`infer`](https://infer.tidymodels.org) packages to a modern Python
data-science stack ([polars](https://pola.rs), [plotnine](https://plotnine.org),
[statsmodels](https://www.statsmodels.org)).

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
  regression, `observe()`, `assume()` (theoretical t/z/F/Chisq), the
  visualization layers `visualize` / `shade_p_value` / `shade_confidence_interval`,
  and the getters `get_p_value` / `get_confidence_interval` (percentile, SE,
  bias-corrected). British-spelling and short aliases are included.
- **Theory-based wrapper tests**: `t_test`, `prop_test`, `chisq_test`,
  `t_stat`, `chisq_stat`, plus the `moderndive.theory` module.
- **Regression helpers** mirroring R `moderndive`: `get_regression_table`,
  `get_regression_points`, `tidy_summary` (built on `statsmodels`, returning
  `polars` frames).
- **Sampling**: `rep_slice_sample` / `rep_sample_n` for sampling-distribution
  activities.
- **Plots**: `pairplot` (a seaborn scatterplot-matrix, the `GGally::ggpairs` analog).
- **Datasets**: `load_*()` loaders returning `polars` DataFrames (the `nycflights23`,
  `gapminder`, `moderndive`, ISLR2, and FiveThirtyEight datasets used in the book).

## Quick start

```python
import moderndive as md
from moderndive import specify, get_p_value, visualize, shade_p_value

spotify = md.load_spotify_metal_deephouse()

# observed difference in popularity rates (metal − deep house)
obs = specify(spotify, formula="popular_or_not ~ track_genre", success="popular") \
    .calculate(stat="diff in props", order=("metal", "deep-house"))

# permutation null distribution + p-value
null = (
    specify(spotify, formula="popular_or_not ~ track_genre", success="popular")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=76)
    .calculate(stat="diff in props", order=("metal", "deep-house"))
)
get_p_value(null, obs_stat=obs, direction="right")
visualize(null, bins=25) + shade_p_value(obs_stat=obs, direction="right")
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
