"""``View()``: render a data frame as an interactive table.

The Python analog of R ``moderndive::View()``. In a notebook or Quarto document
it renders an interactive DataTables widget (search / sort / paginate) via the
optional ``itables`` package — the Python counterpart of R's ``DT::datatable()``
(both wrap the same DataTables.net library). If ``itables`` isn't installed it
falls back to returning the data frame for normal display.
"""

from __future__ import annotations

import importlib.util

import polars as pl

from ._messaging import inform

__all__ = ["View"]


def _itables_available() -> bool:
    """Whether the optional ``itables`` dependency is importable (wrapped for testing)."""
    return importlib.util.find_spec("itables") is not None


def _render_datatable(frame, title):  # pragma: no cover - exercised via View() in tests
    """Render ``frame`` as an interactive DataTables widget via itables."""
    from itables import show

    return show(frame, caption=title)


def _as_frame(x):
    """Coerce ``x`` to a polars/pandas frame for display (leave frames untouched)."""
    if isinstance(x, pl.DataFrame):
        return x
    if type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame":
        return x
    return pl.DataFrame(x)


def View(x, title: str | None = None):
    """Display a data frame as an interactive table (search, sort, paginate).

    In a notebook / Quarto context this renders an interactive table via the
    optional ``itables`` package (install with ``pip install "moderndive[view]"``).
    Without ``itables`` it returns the data frame so it still displays. Accepts a
    polars or pandas DataFrame (or anything coercible to one). ``title`` is shown
    as the table caption.
    """
    frame = _as_frame(x)
    if _itables_available():
        return _render_datatable(frame, title)
    inform(
        "itables isn't installed, so View() can't render an interactive table.",
        'Install it with: pip install "moderndive[view]".',
        "Returning the data frame so it still displays.",
    )
    return frame
