"""Repeated sampling helpers for the sampling activities (Chapter 7).

Mirrors the R ``moderndive`` functions ``rep_slice_sample()`` /
``rep_sample_n()``: draw ``reps`` samples from a data frame and stack them into
one long data frame with a ``replicate`` column (1..reps), so a grouped summary
computes one statistic per virtual sample.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._messaging import helpful_error

__all__ = ["rep_slice_sample", "rep_sample_n"]


def _weights(data: pl.DataFrame, weight_by) -> np.ndarray | None:
    """Normalize ``weight_by`` (a column name or a sequence) into probabilities."""
    if weight_by is None:
        return None
    w = data[weight_by].to_numpy() if isinstance(weight_by, str) else np.asarray(weight_by, float)
    total = w.sum()
    if total <= 0:
        raise ValueError(helpful_error("weight_by must contain positive weights that sum to > 0."))
    return w / total


def rep_slice_sample(
    data: pl.DataFrame,
    n: int | None = None,
    *,
    prop: float | None = None,
    reps: int = 1,
    replace: bool = False,
    weight_by=None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Take ``reps`` samples from ``data``.

    Give the sample size as either ``n`` (a count) or ``prop`` (a fraction of the
    rows, e.g. ``prop=0.5``). Returns a polars DataFrame with a leading
    ``replicate`` column identifying which sample each row belongs to. Set
    ``replace=True`` for sampling with replacement (bootstrap-style). ``weight_by``
    gives unequal selection probabilities — a column name or a sequence of
    weights. Pass ``seed`` for reproducibility.
    """
    if (n is None) == (prop is None):
        raise ValueError(
            helpful_error(
                "Specify exactly one of n= (a count) or prop= (a fraction).",
                "e.g. rep_slice_sample(df, n=50) or rep_slice_sample(df, prop=0.5).",
            )
        )
    n_rows = data.height
    size = n if n is not None else int(round(prop * n_rows))
    if not replace and size > n_rows:
        raise ValueError(
            f"cannot take a sample of size {size} without replacement from {n_rows} rows"
        )
    probs = _weights(data, weight_by)
    rng = np.random.default_rng(seed)

    samples = []
    for replicate in range(1, reps + 1):
        idx = rng.choice(n_rows, size=size, replace=replace, p=probs)
        sample = data[idx.tolist()].with_columns(
            pl.lit(replicate, dtype=pl.Int64).alias("replicate")
        )
        samples.append(sample)

    combined = pl.concat(samples)
    return combined.select(["replicate", *data.columns])


def rep_sample_n(
    data: pl.DataFrame,
    n: int,
    *,
    reps: int = 1,
    replace: bool = False,
    prob=None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Take ``reps`` samples of size ``n`` (older moderndive name).

    Like :func:`rep_slice_sample`, but the sample size is always the count ``n``
    and unequal selection weights are passed as ``prob`` (a column name or a
    sequence), matching the R ``rep_sample_n`` signature.
    """
    return rep_slice_sample(data, n=n, reps=reps, replace=replace, weight_by=prob, seed=seed)
