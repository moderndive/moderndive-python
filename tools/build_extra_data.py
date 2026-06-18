"""Build the 25 extra bundled Parquet datasets (Workstream 5, full parity).

Reads the intermediate CSVs exported by ``tools/export_extra_data.R`` (default
``/tmp/extra_csv``) and writes typed zstd Parquet files into
``moderndive/data/``. Mirrors the options used in ``tools/build_data.py``.

Usage:
    uv run python tools/build_extra_data.py [CSV_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

CSV_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/extra_csv")
OUT_DIR = Path(__file__).resolve().parents[1] / "moderndive/data"

EXTRA = [
    "DD_vs_SB",
    "MA_schools",
    "alaska_flights",
    "amazon_books",
    "avocados",
    "babies",
    "bowl_sample_1",
    "bowl_samples",
    "coffee_ratings",
    "early_january_weather",
    "ev_charging",
    "evals",
    "ipf_lifts",
    "ma_traffic_2020_vs_2019",
    "mario_kart_auction",
    "mass_traffic_2020",
    "orig_pennies_sample",
    "pennies",
    "pennies_resamples",
    "pennies_sample",
    "promotions",
    "promotions_shuffled",
    "spotify_52_original",
    "spotify_52_shuffled",
    "gss",
]


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
    for name in EXTRA:
        _write(_read_csv(name), name)


if __name__ == "__main__":
    main()
