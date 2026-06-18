"""Theory-based distributions for the infer grammar (``assume()``).

Mirrors R ``infer::assume()``: set a theoretical sampling distribution (``t``,
``z``, ``F``, ``Chisq``) that can be visualized and used to compute a p-value
without simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_RIGHT = {"right", "greater"}
_LEFT = {"left", "less"}
_BOTH = {"two-sided", "two_sided", "both", "two sided"}
_SYMMETRIC = {"t", "z"}


@dataclass(frozen=True)
class TheoreticalDistribution:
    """A named theoretical distribution (from :func:`assume`)."""

    distribution: str
    df: object | None = None  # scalar for t/Chisq; (df1, df2) for F

    def _dist(self):
        from scipy import stats

        name = self.distribution.lower()
        if name in ("t", "two-sample t"):
            return stats.t(df=self.df)
        if name == "z":
            return stats.norm()
        if name == "f":
            df1, df2 = self.df
            return stats.f(df1, df2)
        if name in ("chisq", "chi-squared", "chi-square"):
            return stats.chi2(df=self.df)
        raise ValueError(f"unknown theoretical distribution {self.distribution!r}")

    def get_p_value(self, obs_stat, direction: str) -> pl.DataFrame:
        """Theory-based p-value for a (standardized) observed statistic."""
        dist = self._dist()
        obs = float(obs_stat)
        d = direction.lower()
        name = self.distribution.lower()
        if name in ("f", "chisq", "chi-squared", "chi-square"):
            p = float(dist.sf(obs))  # these tests are inherently one-sided (right)
        elif d in _RIGHT:
            p = float(dist.sf(obs))
        elif d in _LEFT:
            p = float(dist.cdf(obs))
        elif d in _BOTH:
            p = float(2 * min(dist.cdf(obs), dist.sf(obs)))
        else:
            raise ValueError("direction must be right/greater, left/less, or two-sided")
        return pl.DataFrame({"p_value": [min(p, 1.0)]})

    def visualize(self, bins: int = 100):
        """Plot the theoretical density curve (a plotnine ggplot)."""
        import pandas as pd
        from plotnine import aes, geom_line, ggplot, labs, theme_light

        dist = self._dist()
        lo, hi = dist.ppf(0.001), dist.ppf(0.999)
        x = np.linspace(lo, hi, 400)
        pdf = pd.DataFrame({"x": x, "density": dist.pdf(x)})
        return (
            ggplot(pdf, aes(x="x", y="density"))
            + geom_line()
            + labs(
                x="statistic", y="density", title=f"Theoretical {self.distribution} distribution"
            )
            + theme_light()
        )


def assume(distribution: str, df: object | None = None) -> TheoreticalDistribution:
    """Set a theoretical distribution (``"t"``, ``"z"``, ``"F"``, ``"Chisq"``).

    ``df`` is the degrees of freedom: a scalar for ``t``/``Chisq``, a
    ``(df1, df2)`` tuple for ``F``, and unused for ``z``.
    """
    return TheoreticalDistribution(distribution=distribution, df=df)
