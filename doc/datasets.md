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

# Datasets

`moderndive` bundles 58 datasets (the R `moderndive` + `infer` data, plus a few
derived tables). Each loads with `load_<name>()` and returns a polars DataFrame.

```{code-cell} python
import moderndive as md

md.available_datasets()          # sorted list of every dataset name
md.load_dataset("pennies")       # load by name (string)
md.load_pennies()                # or via the generated loader
```

## By topic

**Sampling & simulation**
: `bowl`, `bowl_sample_1`, `bowl_samples`, `tactile_prop_red`,
  `almonds_bowl`, `almonds_sample`, `almonds_sample_100`,
  `pennies`, `pennies_sample`, `pennies_resamples`, `orig_pennies_sample`

**Hypothesis testing & CIs**
: `mythbusters_yawn`, `movies_sample`, `spotify_by_genre`, `spotify_52_original`,
  `spotify_52_shuffled`, `spotify_metal_deephouse`, `offshore`, `age_at_marriage`,
  `zinc_tidy`, `cle_sac`, `gss`

**Regression**
: `evals`, `saratoga_houses`, `house_prices`, `coffee_quality`, `coffee_ratings`,
  `credit`, `MA_schools`, `un_member_states_2024`, `mario_kart_auction`,
  `amazon_books`, `old_faithful_2024`

**Flights & weather (nycflights23 / 13)**
: `flights`, `weather`, `airlines`, `airports`, `planes`, `envoy_flights`,
  `alaska_flights`, `early_january_weather`, `early_january_2023_weather`

**Other example data**
: `gapminder`, `gapminder_2007`, `drinks`, `airline_safety`, `dem_score`,
  `us_births_1994_2003`, `steves_episodes`, `avocados`, `babies`, `ipf_lifts`,
  `ev_charging`, `DD_vs_SB`, `ma_traffic_2020_vs_2019`, `mass_traffic_2020`

## Tips

- Every loader returns a polars DataFrame; call `.to_pandas()` if a function
  (e.g. statsmodels) expects pandas.
- Datetime columns (e.g. `time_hour` in the weather/flights data) are stored in
  UTC.
- `load_dataset("<name>")` raises a helpful `ValueError` listing valid names if
  the dataset doesn't exist.
