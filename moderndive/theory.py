"""Theory-based inference wrappers (scipy.stats).

The book deliberately teaches simulation-based inference first, then ties results
back to the traditional theory-based methods (t-distribution CIs, the two-sample
test, normal approximations). These helpers provide those theory-based companions
so the chapters can draw the simulation-vs-theory comparison.

All functions return small polars frames with tidy column names.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "t_test_one_sample",
    "t_test_two_sample",
    "t_confidence_interval",
    "prop_test_two_sample",
]


def _arr(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.drop_nulls().to_numpy()
    return np.asarray(x, dtype=float)


def t_test_one_sample(x, mu: float = 0.0, alternative: str = "two-sided") -> pl.DataFrame:
    """One-sample t-test of H0: mean == ``mu``."""
    from scipy import stats

    a = _arr(x)
    res = stats.ttest_1samp(a, popmean=mu, alternative=alternative)
    return pl.DataFrame(
        {
            "statistic": [float(res.statistic)],
            "df": [float(a.size - 1)],
            "p_value": [float(res.pvalue)],
        }
    )


def t_confidence_interval(x, level: float = 0.95) -> pl.DataFrame:
    """Theory-based t confidence interval for a single mean."""
    from scipy import stats

    a = _arr(x)
    n = a.size
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - (1 - level) / 2, df=n - 1))
    return pl.DataFrame({"lower_ci": [mean - tcrit * se], "upper_ci": [mean + tcrit * se]})


def t_test_two_sample(
    x, y, alternative: str = "two-sided", equal_var: bool = False
) -> pl.DataFrame:
    """Two-sample (Welch by default) t-test of equal means."""
    from scipy import stats

    a, b = _arr(x), _arr(y)
    res = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    return pl.DataFrame({"statistic": [float(res.statistic)], "p_value": [float(res.pvalue)]})


def prop_test_two_sample(
    successes: tuple[int, int], totals: tuple[int, int], alternative: str = "two-sided"
) -> pl.DataFrame:
    """Two-sample z-test for a difference in proportions (normal approximation)."""
    from scipy import stats

    x1, x2 = successes
    n1, n2 = totals
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    if alternative in ("greater", "right"):
        p = float(stats.norm.sf(z))
    elif alternative in ("less", "left"):
        p = float(stats.norm.cdf(z))
    else:
        p = float(2 * stats.norm.sf(abs(z)))
    return pl.DataFrame({"estimate": [p1 - p2], "statistic": [float(z)], "p_value": [p]})
