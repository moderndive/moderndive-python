"""Tidy theory-based test wrappers, mirroring R ``infer``.

``t_test()``, ``prop_test()``, ``chisq_test()`` and the statistic-only
``t_stat()`` / ``chisq_stat()``. Each accepts a polars DataFrame plus either a
``formula="y ~ x"`` or ``response=``/``explanatory=`` columns, and returns a
small polars DataFrame of results (built on :mod:`scipy.stats`).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .core import _parse_formula


def _resolve(formula, response, explanatory):
    if formula is not None:
        return _parse_formula(formula)
    return response, explanatory


def t_test(
    data: pl.DataFrame,
    *,
    formula: str | None = None,
    response: str | None = None,
    explanatory: str | None = None,
    order: tuple[object, object] | None = None,
    alternative: str = "two-sided",
    mu: float = 0.0,
    conf_level: float = 0.95,
) -> pl.DataFrame:
    """One-sample (no explanatory) or two-sample (Welch) t-test, tidy output."""
    from scipy import stats

    resp, expl = _resolve(formula, response, explanatory)
    if expl is None:
        x = data[resp].drop_nulls().to_numpy().astype(float)
        res = stats.ttest_1samp(x, popmean=mu, alternative=alternative)
        n = x.size
        se = x.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(1 - (1 - conf_level) / 2, df=n - 1)
        mean = x.mean()
        return pl.DataFrame(
            {
                "statistic": [float(res.statistic)],
                "t_df": [float(n - 1)],
                "p_value": [float(res.pvalue)],
                "alternative": [alternative],
                "estimate": [float(mean)],
                "lower_ci": [float(mean - tcrit * se)],
                "upper_ci": [float(mean + tcrit * se)],
            }
        )
    if order is None:
        raise ValueError("two-sample t_test requires order=(group1, group2)")
    g1, g2 = order
    sub = data.select(resp, expl).drop_nulls()
    a = sub.filter(pl.col(expl) == g1)[resp].to_numpy().astype(float)
    b = sub.filter(pl.col(expl) == g2)[resp].to_numpy().astype(float)
    res = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return pl.DataFrame(
        {
            "statistic": [float(res.statistic)],
            "p_value": [float(res.pvalue)],
            "alternative": [alternative],
            "estimate": [float(a.mean() - b.mean())],
        }
    )


def t_stat(data: pl.DataFrame, **kwargs) -> float:
    """The t statistic only (see :func:`t_test`)."""
    kwargs.pop("conf_level", None)
    return float(t_test(data, **kwargs)["statistic"][0])


def prop_test(
    data: pl.DataFrame,
    *,
    formula: str | None = None,
    response: str | None = None,
    explanatory: str | None = None,
    success: object | None = None,
    order: tuple[object, object] | None = None,
    p: float | None = None,
    alternative: str = "two-sided",
) -> pl.DataFrame:
    """One- or two-proportion z-test (normal approximation), tidy output."""
    from scipy import stats

    resp, expl = _resolve(formula, response, explanatory)
    if expl is None:
        col = data[resp].drop_nulls()
        n = col.len()
        x = int((col == success).sum())
        p0 = 0.5 if p is None else p
        phat = x / n
        se = np.sqrt(p0 * (1 - p0) / n)
        z = (phat - p0) / se
    else:
        if order is None:
            raise ValueError("two-proportion prop_test requires order=(group1, group2)")
        g1, g2 = order
        sub = data.select(resp, expl).drop_nulls()
        a = sub.filter(pl.col(expl) == g1)[resp]
        b = sub.filter(pl.col(expl) == g2)[resp]
        xa, xb = int((a == success).sum()), int((b == success).sum())
        na, nb = a.len(), b.len()
        ppool = (xa + xb) / (na + nb)
        se = np.sqrt(ppool * (1 - ppool) * (1 / na + 1 / nb))
        z = (xa / na - xb / nb) / se
    if alternative in _GREATER:
        pval = float(stats.norm.sf(z))
    elif alternative in _LESS:
        pval = float(stats.norm.cdf(z))
    else:
        pval = float(2 * stats.norm.sf(abs(z)))
    return pl.DataFrame({"statistic": [float(z)], "p_value": [pval], "alternative": [alternative]})


def chisq_test(
    data: pl.DataFrame,
    *,
    formula: str | None = None,
    response: str | None = None,
    explanatory: str | None = None,
    p: dict | None = None,
) -> pl.DataFrame:
    """Tidy chi-squared test.

    With an explanatory variable, this is a **test of independence**. With only a
    response and a ``p={level: probability, ...}`` mapping, it is a **goodness-of-fit**
    test against those hypothesized proportions. Returns ``statistic``,
    ``chisq_df``, ``p_value``.
    """
    from scipy import stats

    resp, expl = _resolve(formula, response, explanatory)
    if expl is None:
        if p is None:
            raise ValueError(
                "chisq_test needs either an explanatory variable (test of independence) "
                "or p={level: probability, ...} (goodness-of-fit)."
            )
        counts = data.select(resp).drop_nulls()[resp].value_counts()
        observed = {row[resp]: row["count"] for row in counts.iter_rows(named=True)}
        total = sum(observed.values())
        levels = list(p.keys())
        f_obs = [observed.get(lvl, 0) for lvl in levels]
        f_exp = [total * p[lvl] for lvl in levels]
        chi2, pval = stats.chisquare(f_obs, f_exp)
        return pl.DataFrame(
            {"statistic": [float(chi2)], "chisq_df": [len(levels) - 1], "p_value": [float(pval)]}
        )
    sub = data.select(resp, expl).drop_nulls()
    table = sub.to_pandas().pivot_table(index=resp, columns=expl, aggfunc="size", fill_value=0)
    chi2, pval, dof, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    return pl.DataFrame(
        {"statistic": [float(chi2)], "chisq_df": [int(dof)], "p_value": [float(pval)]}
    )


def chisq_stat(data: pl.DataFrame, **kwargs) -> float:
    """The chi-squared statistic only (see :func:`chisq_test`)."""
    return float(chisq_test(data, **kwargs)["statistic"][0])


_GREATER = {"greater", "right"}
_LESS = {"less", "left"}
