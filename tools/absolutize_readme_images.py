"""Rewrite relative README figure paths to absolute GitHub raw URLs.

Quarto writes README.md plot images as relative paths (``README_files/...``),
which render on GitHub but not on PyPI (which needs absolute URLs). Run after
``quarto render README.qmd`` so the generated plot shows everywhere. Idempotent.
"""

from __future__ import annotations

import pathlib

_BASE = "https://raw.githubusercontent.com/moderndive/moderndive-python/main/"
_README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def main() -> None:
    text = _README.read_text()
    updated = text.replace("](README_files/", f"]({_BASE}README_files/")
    if updated != text:
        _README.write_text(updated)
        print("Rewrote README_files image paths to absolute GitHub raw URLs.")
    else:
        print("No relative README_files image paths to rewrite.")


if __name__ == "__main__":
    main()
