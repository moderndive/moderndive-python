"""Summary/test statistics for the infer grammar.

Each statistic is computed on plain numpy arrays so the resampling kernels can
call them in a tight loop. ``response`` is always required; ``explanatory`` is
required for the two-group / bivariate statistics. ``mu`` / ``p`` carry the
hypothesized value (from ``hypothesize()``) for the standardized ``t``/``z``
statistics. Mirrors the ``stat=`` vocabulary of R ``infer``'s ``calculate()``.
"""

from __future__ import annotations

import numpy as np

# Statistics that need an explanatory variable.
_BIVARIATE = frozenset(
    {
        "diff in means",
        "diff in medians",
        "diff in props",
        "ratio of means",
        "ratio of props",
        "odds ratio",
        "slope",
        "correlation",
        "F",
        "Chisq",
    }
)
# Statistics that need a `success` level (categorical response).
_NEEDS_SUCCESS = frozenset({"prop", "count", "diff in props", "ratio of props", "odds ratio", "z"})

SUPPORTED_STATS = frozenset(
    {
        "mean",
        "median",
        "sum",
        "sd",
        "prop",
        "count",
        "diff in means",
        "diff in medians",
        "diff in props",
        "ratio of means",
        "ratio of props",
        "odds ratio",
        "slope",
        "correlation",
        "t",
        "z",
        "F",
        "Chisq",
    }
)


def needs_explanatory(stat: str) -> bool:
    return stat in _BIVARIATE


def needs_success(stat: str) -> bool:
    return stat in _NEEDS_SUCCESS


def _prop(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else float("nan")


def _groups(response, explanatory, order):
    if order is None:
        raise ValueError("this statistic requires `order=(group1, group2)`")
    g1, g2 = order
    return response[explanatory == g1], response[explanatory == g2]


def compute_statistic(
    response: np.ndarray,
    explanatory: np.ndarray | None,
    stat,
    *,
    success: object | None = None,
    order: tuple[object, object] | None = None,
    mu: float | None = None,
    p: float | None = None,
    sigma: float | None = None,
) -> float:
    """Compute a single statistic from response (+ optional explanatory) arrays.

    ``stat`` may also be a callable taking ``(response, explanatory)`` and
    returning a float (infer's custom-statistic feature). ``sigma`` is the known
    population SD for a one-sample ``z`` statistic on a mean.
    """
    if callable(stat):
        return float(stat(response, explanatory))

    n = response.shape[0]

    # --- one-variable statistics -----------------------------------------
    if stat == "mean":
        return float(np.mean(response))
    if stat == "median":
        return float(np.median(response))
    if stat == "sum":
        return float(np.sum(response))
    if stat == "sd":
        return float(np.std(response, ddof=1))
    if stat == "prop":
        return _prop(response == success)
    if stat == "count":
        return float(np.sum(response == success))
    if stat == "t" and explanatory is None:
        if mu is None:
            raise ValueError("stat 't' (one sample) requires hypothesize(mu=...)")
        s = np.std(response, ddof=1)
        return float((np.mean(response) - mu) / (s / np.sqrt(n)))
    if stat == "z" and explanatory is None:
        if sigma is not None:  # one-sample z on a mean with known population SD
            if mu is None:
                raise ValueError("stat 'z' on a mean requires hypothesize(mu=..., sigma=...)")
            return float((np.mean(response) - mu) / (sigma / np.sqrt(n)))
        if p is None:
            raise ValueError("stat 'z' (one proportion) requires hypothesize(p=...)")
        phat = _prop(response == success)
        return float((phat - p) / np.sqrt(p * (1 - p) / n))

    # --- bivariate statistics --------------------------------------------
    if explanatory is None:
        raise ValueError(f"stat {stat!r} requires an explanatory variable")

    if stat in ("diff in means", "diff in medians", "ratio of means"):
        a, b = _groups(response, explanatory, order)
        fn = np.median if stat == "diff in medians" else np.mean
        if stat == "ratio of means":
            return float(np.mean(a) / np.mean(b))
        return float(fn(a) - fn(b))

    if stat in ("diff in props", "ratio of props", "odds ratio"):
        a, b = _groups(response, explanatory, order)
        pa, pb = _prop(a == success), _prop(b == success)
        if stat == "ratio of props":
            return float(pa / pb)
        if stat == "odds ratio":
            return float((pa / (1 - pa)) / (pb / (1 - pb)))
        return pa - pb

    if stat == "t":  # two-sample (Welch) t for a difference in means
        a, b = _groups(response, explanatory, order)
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(va / a.size + vb / b.size)
        return float((np.mean(a) - np.mean(b)) / se)

    if stat == "z":  # two-proportion pooled z
        a, b = _groups(response, explanatory, order)
        xa, xb = np.sum(a == success), np.sum(b == success)
        na, nb = a.size, b.size
        ppool = (xa + xb) / (na + nb)
        se = np.sqrt(ppool * (1 - ppool) * (1 / na + 1 / nb))
        return float((xa / na - xb / nb) / se)

    if stat == "slope":
        x = explanatory.astype(float)
        y = response.astype(float)
        xm = x.mean()
        return float(np.sum((x - xm) * (y - y.mean())) / np.sum((x - xm) ** 2))

    if stat == "correlation":
        return float(np.corrcoef(explanatory.astype(float), response.astype(float))[0, 1])

    if stat == "F":  # one-way ANOVA F statistic (numerical ~ categorical)
        return _anova_f(response.astype(float), explanatory)

    if stat == "Chisq":  # chi-square statistic of independence (cat ~ cat)
        return _chisq_independence(response, explanatory)

    raise ValueError(f"Unknown stat {stat!r}. Supported: {', '.join(sorted(SUPPORTED_STATS))}")


def _anova_f(y: np.ndarray, group: np.ndarray) -> float:
    levels = np.unique(group)
    grand = y.mean()
    k, n = levels.size, y.size
    ss_between = sum((group == g).sum() * (y[group == g].mean() - grand) ** 2 for g in levels)
    ss_within = sum(((y[group == g] - y[group == g].mean()) ** 2).sum() for g in levels)
    df_between, df_within = k - 1, n - k
    return float((ss_between / df_between) / (ss_within / df_within))


def _chisq_independence(response: np.ndarray, explanatory: np.ndarray) -> float:
    r_levels = np.unique(response)
    c_levels = np.unique(explanatory)
    observed = np.array(
        [[np.sum((response == r) & (explanatory == c)) for c in c_levels] for r in r_levels],
        dtype=float,
    )
    row_tot = observed.sum(axis=1, keepdims=True)
    col_tot = observed.sum(axis=0, keepdims=True)
    expected = row_tot @ col_tot / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(terms.sum())
