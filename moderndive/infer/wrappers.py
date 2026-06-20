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


def _wilson_cc(x: int, n: int, conf_level: float, correct: bool) -> tuple[float, float]:
    """Wilson score interval for one proportion (R prop.test's CI; cc = Yates)."""
    from scipy import stats

    z = float(stats.norm.ppf((1 + conf_level) / 2))
    phat = x / n
    if correct:
        lo = (
            2 * x + z**2 - 1 - z * np.sqrt(z**2 - 2 - 1 / n + 4 * phat * (n * (1 - phat) + 1))
        ) / (2 * (n + z**2))
        hi = (
            2 * x + z**2 + 1 + z * np.sqrt(z**2 + 2 - 1 / n + 4 * phat * (n * (1 - phat) - 1))
        ) / (2 * (n + z**2))
        return max(0.0, lo), min(1.0, hi)
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


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
    z: bool = False,
    correct: bool = True,
    conf_int: bool = True,
    conf_level: float = 0.95,
) -> pl.DataFrame:
    """Tidy one- or two-proportion test, mirroring R ``infer::prop_test``.

    By default reports the **chi-square** statistic (like R's ``prop.test``) with a
    ``chisq_df`` column; pass ``z=True`` for the signed **z** statistic instead.
    ``correct`` applies Yates' continuity correction. With ``conf_int=True``
    (default) the output includes a ``conf_level`` confidence interval — on the
    proportion (one-sample) or on the difference in proportions (two-sample).
    """
    from scipy import stats

    resp, expl = _resolve(formula, response, explanatory)

    if expl is None:
        col = data[resp].drop_nulls()
        n = col.len()
        x = int((col == success).sum())
        phat = x / n
        p0 = 0.5 if p is None else p
        diff = phat - p0
        cc = min(0.5 / n, abs(diff)) if correct else 0.0
        zstat = np.sign(diff) * (abs(diff) - cc) / np.sqrt(p0 * (1 - p0) / n)
        estimate = phat
        ci_bounds = _wilson_cc(x, n, conf_level, correct)  # R uses Wilson for one proportion
    else:
        if order is None:
            raise ValueError("two-proportion prop_test requires order=(group1, group2)")
        g1, g2 = order
        sub = data.select(resp, expl).drop_nulls()
        a = sub.filter(pl.col(expl) == g1)[resp]
        b = sub.filter(pl.col(expl) == g2)[resp]
        xa, xb = int((a == success).sum()), int((b == success).sum())
        na, nb = a.len(), b.len()
        pa, pb = xa / na, xb / nb
        diff = pa - pb
        ppool = (xa + xb) / (na + nb)
        cc = min(0.5 * (1 / na + 1 / nb), abs(diff)) if correct else 0.0
        zstat = np.sign(diff) * (abs(diff) - cc) / np.sqrt(ppool * (1 - ppool) * (1 / na + 1 / nb))
        se_est = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        estimate = diff
        crit = float(stats.norm.ppf(1 - (1 - conf_level) / 2))
        half = crit * se_est + cc  # R's prop.test widens the 2-sample diff CI by the correction
        ci_bounds = (diff - half, diff + half)

    if alternative in _GREATER:
        pval = float(stats.norm.sf(zstat))
    elif alternative in _LESS:
        pval = float(stats.norm.cdf(zstat))
    else:
        pval = float(2 * stats.norm.sf(abs(zstat)))

    statistic = float(zstat) if z else float(zstat**2)
    out = {"statistic": [statistic]}
    if not z:
        out["chisq_df"] = [1]
    out["p_value"] = [pval]
    out["estimate"] = [float(estimate)]
    out["alternative"] = [alternative]
    if conf_int:
        out["lower_ci"] = [float(ci_bounds[0])]
        out["upper_ci"] = [float(ci_bounds[1])]
    return pl.DataFrame(out)


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
