"""The infer grammar: specify / hypothesize / generate / calculate / fit (+ summaries).

Mirrors the R ``infer`` package, including British-spelling and short-form
aliases. Entry point is :func:`specify`; the returned objects expose
``.hypothesize()``, ``.generate()``, ``.calculate()``, ``.fit()``, and
``.assume()`` methods.
"""

from __future__ import annotations

from .core import (
    Distribution,
    FitResult,
    GeneratedReplicates,
    Hypothesis,
    ObservedStatistic,
    Specification,
    observe,
    register_dataframe_accessor,
    specify,
)
from .intervals import get_confidence_interval
from .pvalue import get_p_value
from .theoretical import TheoreticalDistribution, assume
from .viz import shade_confidence_interval, shade_p_value, visualize
from .wrappers import chisq_stat, chisq_test, prop_test, t_stat, t_test

# Attach `.specify()` to polars/pandas DataFrames so `df.specify(...)` works,
# mirroring R's `df %>% specify(...)`.
register_dataframe_accessor()

# infer-parity aliases (British spellings + short forms).
# (`hypothesise`/`visualise` also exist as methods on Specification/Distribution.)
visualise = visualize
get_pvalue = get_p_value
get_ci = get_confidence_interval
shade_pvalue = shade_p_value
shade_ci = shade_confidence_interval

__all__ = [
    # verbs
    "specify",
    "observe",
    "assume",
    # classes
    "Specification",
    "Hypothesis",
    "GeneratedReplicates",
    "Distribution",
    "FitResult",
    "ObservedStatistic",
    "TheoreticalDistribution",
    # getters / viz
    "get_confidence_interval",
    "get_p_value",
    "visualize",
    "shade_p_value",
    "shade_confidence_interval",
    # aliases
    "visualise",
    "get_pvalue",
    "get_ci",
    "shade_pvalue",
    "shade_ci",
    # wrapper tests
    "t_test",
    "t_stat",
    "prop_test",
    "chisq_test",
    "chisq_stat",
]
