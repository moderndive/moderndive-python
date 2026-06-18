"""Seedable resampling utilities (numpy) for the infer grammar.

The per-replicate resampling *plans* (bootstrap indices, permutations, draws) are
built in :mod:`moderndive.infer.core`; this module holds the shared helpers.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Construct a numpy Generator. ``None`` yields a fresh, unseeded generator."""
    return np.random.default_rng(seed)


def shift_for_point_null(
    response: np.ndarray, *, stat: str, mu: float | None, p: float | None
) -> np.ndarray:
    """Shift the response so the bootstrap is centered at the hypothesized value.

    Mirrors infer's point-null behavior: for a ``"mean"``/``"median"`` point null
    we translate the observed values so their center equals ``mu`` before
    bootstrapping, so the resulting null distribution is centered at ``mu``.
    """
    if stat in ("mean", "median") and mu is not None:
        center = np.mean(response) if stat == "mean" else np.median(response)
        return response - center + mu
    # For proportions the point null is handled by drawing, not shifting; return as-is.
    return response
