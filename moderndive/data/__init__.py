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
    # End-of-chapter exercises (olympicAthletes)
    "olympic_athletes": "Olympic athletes 1896-2026: one row per athlete-event (olympicAthletes).",
    "medal_table": "Olympic medal counts per Games x NOC, 1896-2026 (olympicAthletes).",
    "editions": "Metadata for every Olympic Games edition, 1896-2026 (olympicAthletes).",
    # olympicAthletes convenience subsets (sport/Games slices of olympic_athletes)
    "athletics_athletes": "Athletics athlete-events (olympicAthletes subset).",
    "gymnastics_athletes": "Gymnastics athlete-events (olympicAthletes subset).",
    "basketball_athletes": "Basketball athlete-events (olympicAthletes subset).",
    "volleyball_athletes": "Volleyball athlete-events (olympicAthletes subset).",
    "curling_athletes": "Curling athlete-events (olympicAthletes subset).",
    "art_competitions_athletes": "Art Competitions athlete-events, 1912-1948 (olympicAthletes subset).",
    "team_sport_athletes": "Team-sport athlete-events (olympicAthletes subset).",
    "olympic_athletes_2024": "Athlete-events at the Paris 2024 Summer Games (olympicAthletes subset).",
    "usa_summer_medals": "USA medal counts by Summer Games (olympicAthletes).",
    "paris_2024_top_medals": "Top medal-winning NOCs at Paris 2024, long format (olympicAthletes).",
    "season_counts": "Athlete-event counts by Olympic season (olympicAthletes).",
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
    # --- Full parity with the R packages (Workstream 5) ---------------------
    # Flights / weather
    "alaska_flights": "Alaska Airlines flights from NYC in 2013 (nycflights13).",
    "early_january_weather": "Hourly NYC weather, Jan 1-15 2013 (nycflights13).",
    # Sampling: bowl and pennies activities
    "bowl_sample_1": "A single tactile sample of 50 balls from the bowl.",
    "bowl_samples": "Ten tactile samples of balls by group (red/white/green counts).",
    "pennies": "Years and ages of 800 pennies (sampling population).",
    "pennies_sample": "A sample of 50 pennies with their mint years.",
    "orig_pennies_sample": "An original sample of 40 pennies and their ages in 2011.",
    "pennies_resamples": "Bootstrap resamples of the 50-penny sample.",
    # Hypothesis testing examples
    "promotions": "Resume gender-bias promotion experiment (48 resumes).",
    "promotions_shuffled": "Promotions data with gender shuffled (permutation example).",
    "spotify_52_original": "52 Spotify tracks (metal vs. deep-house) for testing.",
    "spotify_52_shuffled": "spotify_52_original with genre labels shuffled.",
    "gss": "General Social Survey subset, 500 respondents (infer).",
    # Regression and modeling
    "evals": "UT Austin teaching evaluations with instructor beauty scores.",
    "MA_schools": "Massachusetts high schools: SAT math, size, and disadvantage.",
    "amazon_books": "325 books with Amazon and list prices and physical traits.",
    "mario_kart_auction": "143 eBay auctions of Mario Kart for the Wii.",
    "babies": "Child Health and Development birth-weight study (1236 births).",
    "coffee_ratings": "Coffee Quality Institute ratings with sensory scores.",
    "ev_charging": "Electric-vehicle charging sessions with energy and time.",
    "ipf_lifts": "International Powerlifting Federation meet results.",
    "avocados": "Weekly US Hass avocado prices and volumes by region.",
    # Traffic and demographics
    "DD_vs_SB": "Dunkin' Donuts vs. Starbucks shop counts by US county.",
    "ma_traffic_2020_vs_2019": "Massachusetts traffic change, 2020 vs. 2019.",
    "mass_traffic_2020": "Massachusetts traffic-volume and crash counts, 2020.",
    # End-of-chapter exercises (R companion packages)
    "planets": "Confirmed exoplanets: mass, radius, orbit, discovery (exoplanetdata).",
    "volcanoes": "Holocene volcanoes of the world (volcanoes).",
    "eruptions": "Documented volcanic eruptions with VEI and dates (volcanoes).",
    "bob_ross": "Element tags for every Joy of Painting episode (fivethirtyeight).",
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
