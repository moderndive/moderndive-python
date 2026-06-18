"""Repeated sampling helpers for the sampling activities (Chapter 7).

Mirrors the R ``moderndive`` functions ``rep_slice_sample()`` /
``rep_sample_n()``: draw ``reps`` samples of size ``n`` from a data frame and
stack them into one long data frame with a ``replicate`` column (1..reps), so a
grouped summary computes one statistic per virtual sample.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["rep_slice_sample", "rep_sample_n"]


def rep_slice_sample(
    data: pl.DataFrame,
    n: int,
    reps: int = 1,
    replace: bool = False,
    seed: int | None = None,
) -> pl.DataFrame:
    """Take ``reps`` samples of size ``n`` from ``data``.

    Returns a polars DataFrame with a leading ``replicate`` column identifying
    which sample each row belongs to. Set ``replace=True`` for sampling with
    replacement (e.g. bootstrap-style). Pass ``seed`` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_rows = data.height
    if not replace and n > n_rows:
        raise ValueError(
            f"cannot take a sample of size {n} without replacement from {n_rows} rows"
        )

    samples = []
    for replicate in range(1, reps + 1):
        idx = rng.choice(n_rows, size=n, replace=replace)
        sample = data[idx.tolist()].with_columns(
            pl.lit(replicate, dtype=pl.Int64).alias("replicate")
        )
        samples.append(sample)

    combined = pl.concat(samples)
    # Put `replicate` first.
    return combined.select(["replicate", *data.columns])


# The older moderndive name is a thin alias with the same behavior.
def rep_sample_n(
    data: pl.DataFrame,
    n: int,
    reps: int = 1,
    replace: bool = False,
    seed: int | None = None,
) -> pl.DataFrame:
    """Alias for :func:`rep_slice_sample` (older moderndive name)."""
    return rep_slice_sample(data, n=n, reps=reps, replace=replace, seed=seed)
