"""Tests for the Pyodide/WebAssembly polars shims applied under emscripten."""

from __future__ import annotations

import polars as pl

import moderndive as md


def test_apply_pyodide_polars_shims(tmp_path):
    """The shim routes Parquet IO through pyarrow and to_pandas through a dict.

    We apply it explicitly (it only auto-applies under Pyodide) and restore the
    originals afterwards so the rest of the suite uses real polars.
    """
    orig_read, orig_scan = pl.read_parquet, pl.scan_parquet
    orig_to_pandas = pl.DataFrame.to_pandas
    try:
        md._apply_pyodide_polars_shims()

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "t.parquet"
        # write via the original writer (writing isn't shimmed)
        df.write_parquet(path)

        # read_parquet now goes through pyarrow
        got = pl.read_parquet(path)
        assert got.sort("a").to_dicts() == df.to_dicts()

        # scan_parquet returns a LazyFrame backed by the same data
        scanned = pl.scan_parquet(path).collect()
        assert scanned.sort("a").to_dicts() == df.to_dicts()

        # to_pandas goes through a dict round-trip
        pdf = df.to_pandas()
        assert list(pdf.columns) == ["a", "b"]
        assert pdf["a"].tolist() == [1, 2, 3]
    finally:
        pl.read_parquet = orig_read
        pl.scan_parquet = orig_scan
        pl.DataFrame.to_pandas = orig_to_pandas

    # originals restored
    assert pl.read_parquet is orig_read
    assert pl.DataFrame.to_pandas is orig_to_pandas
