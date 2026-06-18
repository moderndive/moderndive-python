"""Build bundled Parquet datasets for the moderndive companion package.

Reads the intermediate CSVs exported from the R packages (see
``ModernDive_book/scripts/export_pilot_data.R``) and writes typed Parquet files
into ``packages/moderndive/src/moderndive/data/``.

Gapminder is sourced from the Python ``gapminder`` package (not exported from R).

Usage:
    uv run python scripts/build_data.py [CSV_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

CSV_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/pilot_csv")
OUT_DIR = Path(__file__).resolve().parents[1] / "moderndive/data"

# Columns that should be parsed as datetimes when present.
_DATETIME_COLS = {"time_hour", "time", "sched_dep_time_hour"}


def _read_csv(name: str) -> pl.DataFrame:
    return pl.read_csv(
        CSV_DIR / f"{name}.csv",
        try_parse_dates=True,
        infer_schema_length=10000,
        null_values=["NA", ""],
    )


def _write(df: pl.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.parquet"
    df.write_parquet(out, compression="zstd")
    size_kb = out.stat().st_size / 1024
    print(f"{name:30s} {df.height:7d} rows  {df.width:2d} cols  {size_kb:8.1f} KiB")


def main() -> None:
    from_r = [
        "envoy_flights",
        "early_january_2023_weather",
        "weather",
        "flights",
        "airlines",
        "airports",
        "planes",
        "spotify_by_genre",
        "almonds_sample_100",
        "almonds_bowl",
        "drinks",
        "airline_safety",
        "dem_score",
        "un_member_states_2024",
        "credit",
        # Ch 7 sampling
        "bowl",
        "tactile_prop_red",
        "almonds_sample",
        # Ch 8 confidence intervals
        "mythbusters_yawn",
        # Ch 10 inference for regression
        "old_faithful_2024",
        "movies_sample",
        "coffee_quality",
        # Ch 11 tell your story
        "house_prices",
        "us_births_1994_2003",
        # Appendix B inference examples
        "saratoga_houses",
        "steves_episodes",
        "offshore",
        "age_at_marriage",
        "zinc_tidy",
        "cle_sac",
    ]
    for name in from_r:
        _write(_read_csv(name), name)

    # Gapminder ships a CSV inside the `gapminder` PyPI package. We read that file
    # directly (the package's Python import is broken on 3.14: it uses the removed
    # pkg_resources). The data matches the canonical R gapminder dataset.
    import importlib.util

    spec = importlib.util.find_spec("gapminder")
    if spec and spec.submodule_search_locations:
        csv_path = Path(spec.submodule_search_locations[0]) / "gapminder.csv"
        if csv_path.exists():
            gm = pl.read_csv(csv_path, infer_schema_length=10000)
            # Normalize column names to the R gapminder names.
            rename = {"gdpPercap": "gdpPercap", "lifeExp": "lifeExp", "pop": "pop"}
            gm = gm.rename({k: v for k, v in rename.items() if k in gm.columns})
            _write(gm, "gapminder")
            _write(gm.filter(pl.col("year") == 2007), "gapminder_2007")
            return
    print("WARNING: gapminder.csv not found; skipping gapminder")


if __name__ == "__main__":
    main()
