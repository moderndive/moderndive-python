# moderndive (Python)

The Python companion package for **ModernDive: Statistical Inference via Data
Science** — a faithful port of the R `moderndive` and `infer` packages to a
modern Python data-science stack (polars, plotnine, statsmodels).

```{toctree}
:maxdepth: 2
:caption: Contents

self
api
```

## Installation

```bash
pip install moderndive
```

## Quick start

```python
import moderndive as md
from moderndive import specify, get_p_value, visualize, shade_p_value

spotify = md.load_spotify_metal_deephouse()

obs = specify(spotify, formula="popular_or_not ~ track_genre", success="popular") \
    .calculate(stat="diff in props", order=("metal", "deep-house"))

null = (
    specify(spotify, formula="popular_or_not ~ track_genre", success="popular")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=76)
    .calculate(stat="diff in props", order=("metal", "deep-house"))
)
get_p_value(null, obs_stat=obs, direction="right")
```

See the {doc}`api` for the full reference.
