"""Dataset loaders for the Python edition of ModernDive.

Each ``load_<name>()`` returns a :class:`polars.DataFrame` read from a Parquet
file bundled inside the package. Parquet preserves dtypes, so datetime and
categorical columns load correctly without re-parsing.

The bundled Parquet files are produced from the R packages' datasets by
``scripts/build_data.py`` (see the repo root). Loader functions are generated
from the ``_REGISTRY`` below so adding a dataset is a one-line change.
"""

from __future__ import annotations

import sys
from importlib.resources import files

import polars as pl

# name -> one-line description (used in the generated loader's docstring).
_REGISTRY: dict[str, str] = {
    # Getting started / visualization / wrangling / tidy
    "flights": "All flights departing NYC in 2023 (nycflights23).",
    "weather": "Hourly weather for NYC airports, 2023 (nycflights23).",
    "airlines": "Airline carrier-code lookup table (nycflights23).",
    "airports": "Airport metadata: FAA code, name, location (nycflights23).",
    "planes": "Aircraft metadata keyed by tail number (nycflights23).",
    "envoy_flights": "Envoy Air (carrier MQ) flights from NYC in 2023.",
    "early_january_2023_weather": "Hourly Newark weather, Jan 1-15 2023.",
    "gapminder": "Gapminder country-year development indicators.",
    "gapminder_2007": "Gapminder, 2007 only.",
    "drinks": "Alcohol servings per person by country (fivethirtyeight).",
    "airline_safety": "Airline safety records, 1985-2014 (fivethirtyeight).",
    "dem_score": "Democracy scores by country and year, wide format.",
    # Regression
    "un_member_states_2024": "UN member states (2024): demographics and economy.",
    "credit": "Credit-card holder data (ISLR2 Credit).",
    # Sampling / inference
    "bowl": "A bowl of 2400 red and white balls (sampling activity).",
    "tactile_prop_red": "33 hand-collected samples of 50 balls (proportion red).",
    "almonds_bowl": "The full 'bowl' of 5000 almonds (sampling population).",
    "almonds_sample": "A single sample of 25 almonds with their weights.",
    "almonds_sample_100": "A single sample of 100 almonds with their weights.",
    "mythbusters_yawn": "MythBusters yawning experiment (50 people).",
    "spotify_by_genre": "Spotify tracks across six genres with a popular_or_not label.",
    "old_faithful_2024": "Old Faithful eruptions, 2024: duration and waiting time.",
    "movies_sample": "A sample of 68 movies with rating and genre.",
    "coffee_quality": "Coffee-quality ratings with sensory and origin variables.",
    # Tell your story
    "house_prices": "Seattle/King County house sales (Kaggle).",
    "us_births_1994_2003": "Daily US births, 1994-2003 (fivethirtyeight).",
    # Appendix B inference examples
    "saratoga_houses": "Saratoga house prices with size, bedrooms, bathrooms.",
    "steves_episodes": "Rick Steves' Europe episodes with IMDb ratings.",
    "offshore": "California voters' opinions on offshore drilling.",
    "age_at_marriage": "Ages at first marriage (one-mean example).",
    "zinc_tidy": "Paired zinc concentrations (surface vs. bottom).",
    "cle_sac": "Personal incomes in Cleveland vs. Sacramento.",
}

# Derived datasets (computed from a bundled one rather than stored as Parquet).
_DERIVED = {"spotify_metal_deephouse"}


def available_datasets() -> list[str]:
    """Return the sorted names of all loadable datasets."""
    return sorted(set(_REGISTRY) | _DERIVED)


def load_dataset(name: str) -> pl.DataFrame:
    """Load a dataset by name, returning a polars DataFrame."""
    if name == "spotify_metal_deephouse":
        return load_spotify_metal_deephouse()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(available_datasets())}")
    path = files("moderndive.data").joinpath(f"{name}.parquet")
    with path.open("rb") as handle:
        return pl.read_parquet(handle)


def load_spotify_metal_deephouse() -> pl.DataFrame:
    """Spotify tracks restricted to the ``metal`` and ``deep-house`` genres.

    Derived in Chapter 9 from :func:`load_spotify_by_genre` by filtering to the
    two genres and selecting the columns of interest.
    """
    return (
        load_dataset("spotify_by_genre")
        .filter(pl.col("track_genre").is_in(["metal", "deep-house"]))
        .select(
            "track_id",
            "track_genre",
            "artists",
            "track_name",
            "popularity",
            "popular_or_not",
        )
    )


# --- Generate a load_<name>() function for every registered dataset ----------


def _make_loader(_name: str, _doc: str):
    def _loader() -> pl.DataFrame:
        return load_dataset(_name)

    _loader.__name__ = f"load_{_name}"
    _loader.__qualname__ = f"load_{_name}"
    _loader.__doc__ = _doc
    return _loader


_module = sys.modules[__name__]
for _ds_name, _ds_doc in _REGISTRY.items():
    setattr(_module, f"load_{_ds_name}", _make_loader(_ds_name, _ds_doc))

__all__ = [
    "load_dataset",
    "available_datasets",
    "load_spotify_metal_deephouse",
    *[f"load_{name}" for name in _REGISTRY],
]
