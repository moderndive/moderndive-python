"""Tests for the Pyodide/WebAssembly shims applied under emscripten."""

from __future__ import annotations

import base64

import polars as pl
from plotnine import aes, geom_point, ggplot

import moderndive as md


def test_png_to_html_img():
    """Wraps base64 str and raw bytes in a data-URI <img>, None for empty."""
    assert md._png_to_html_img(None) is None
    assert md._png_to_html_img("") is None
    assert md._png_to_html_img("QUJD").startswith('<img src="data:image/png;base64,QUJD"')
    raw = b"\x89PNG\r\n"
    expected = base64.b64encode(raw).decode()
    assert expected in md._png_to_html_img(raw)


def test_apply_pyodide_shims(tmp_path):
    """The shims route Parquet/to_pandas through pyarrow/dict and make a plotnine
    ggplot render via _repr_html_. Applied explicitly (only auto-applies under
    Pyodide); originals restored afterwards so the rest of the suite is unaffected.
    """
    orig_read, orig_scan = pl.read_parquet, pl.scan_parquet
    orig_to_pandas = pl.DataFrame.to_pandas
    had_html = "_repr_html_" in ggplot.__dict__
    try:
        md._apply_pyodide_shims()

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / "t.parquet"
        df.write_parquet(path)

        assert pl.read_parquet(path).sort("a").to_dicts() == df.to_dicts()
        assert pl.scan_parquet(path).collect().sort("a").to_dicts() == df.to_dicts()

        pdf = df.to_pandas()
        assert list(pdf.columns) == ["a", "b"]
        assert pdf["a"].tolist() == [1, 2, 3]

        # plotnine ggplot now renders an embedded-PNG <img> via _repr_html_
        plot = ggplot(pdf, aes(x="a", y="b")) + geom_point()
        html = plot._repr_html_()
        assert html.startswith('<img src="data:image/png;base64,')
    finally:
        pl.read_parquet = orig_read
        pl.scan_parquet = orig_scan
        pl.DataFrame.to_pandas = orig_to_pandas
        if not had_html:
            del ggplot._repr_html_

    assert pl.read_parquet is orig_read
    assert "_repr_html_" not in ggplot.__dict__
