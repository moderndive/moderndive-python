"""Core classes for the infer grammar: specify / hypothesize / generate / calculate / fit.

Mirrors the R ``infer`` pipeline. Because polars DataFrames have no ``.specify``
method, the pipeline entry point is the module-level :func:`specify` function:

    bootstrap_means = (
        specify(almonds_sample_100, response="weight")
        .generate(reps=1000, type="bootstrap")
        .calculate(stat="mean")
    )

    null = (
        specify(spotify, formula="popular_or_not ~ track_genre", success="popular")
        .hypothesize(null="independence")
        .generate(reps=1000, type="permute")
        .calculate(stat="diff in props", order=("metal", "deep-house"))
    )

    # Multiple regression: fit() instead of calculate()
    boot_fits = (
        specify(coffee, formula="total_cup_points ~ aroma + flavor")
        .generate(reps=1000, type="bootstrap")
        .fit()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import resample as _resample
from . import statistics as _stats


def _parse_formula(formula: str) -> tuple[str, str | None]:
    """Parse ``"y ~ x"`` → (response, explanatory). ``x`` may be ``NULL``/``None``.

    For multiple terms (``"y ~ a + b"``) the explanatory is returned as the whole
    right-hand side string; single-variable stats use it as a column name, while
    :meth:`GeneratedReplicates.fit` uses the full formula instead.
    """
    if "~" not in formula:
        raise ValueError(f"formula {formula!r} must contain '~' (e.g. 'y ~ x')")
    lhs, rhs = (side.strip() for side in formula.split("~", 1))
    explanatory = None if rhs in ("", "NULL", "None", "1", ".") else rhs
    return lhs, explanatory


@dataclass(frozen=True)
class Specification:
    """Result of :func:`specify`: the chosen response (+ optional explanatory)."""

    data: pl.DataFrame
    response: str
    explanatory: str | None = None
    success: object | None = None
    formula: str | None = None

    @property
    def _response_values(self) -> np.ndarray:
        return self.data[self.response].to_numpy()

    @property
    def _explanatory_values(self) -> np.ndarray | None:
        # Single-column explanatory only (used by single-variable statistics).
        if self.explanatory is None or self.explanatory not in self.data.columns:
            return None
        return self.data[self.explanatory].to_numpy()

    def _full_formula(self) -> str:
        if self.formula is not None:
            return self.formula
        if self.explanatory is None:
            return f"{self.response} ~ 1"
        return f"{self.response} ~ {self.explanatory}"

    def hypothesize(
        self, null: str, *, mu: float | None = None, p: float | None = None
    ) -> Hypothesis:
        if null not in ("point", "independence", "paired independence"):
            raise ValueError(
                "null must be 'point', 'independence', or 'paired independence'"
            )
        return Hypothesis(spec=self, null=null, mu=mu, p=p)

    def generate(
        self, reps: int, type: str = "bootstrap", *, seed: int | None = None
    ) -> GeneratedReplicates:
        return _generate(self, hypothesis=None, reps=reps, type=type, seed=seed)

    def calculate(
        self,
        stat,
        *,
        order: tuple[object, object] | None = None,
        mu: float | None = None,
        p: float | None = None,
    ) -> ObservedStatistic:
        """Compute the observed statistic (no resampling)."""
        value = _stats.compute_statistic(
            self._response_values,
            self._explanatory_values,
            stat,
            success=self.success,
            order=order,
            mu=mu,
            p=p,
        )
        label = stat if isinstance(stat, str) else "stat"
        return ObservedStatistic(value=value, stat=label)

    def assume(self, distribution: str, df: object | None = None):
        """Set a theoretical sampling distribution (see :func:`assume`)."""
        from .theoretical import assume as _assume

        return _assume(distribution, df=df)

    # British-spelling alias (infer parity).
    def hypothesise(self, null: str, *, mu=None, p=None):
        return self.hypothesize(null, mu=mu, p=p)

    def fit(self) -> "FitResult":
        """Fit the observed regression (ordinary least squares) for the formula."""
        import statsmodels.formula.api as smf

        from ..modeling import _clean_term

        model = smf.ols(self._full_formula(), data=self.data.to_pandas()).fit()
        terms = [_clean_term(t) for t in model.params.index]
        df = pl.DataFrame({"term": terms, "estimate": model.params.to_numpy()})
        return FitResult(data=df)


@dataclass(frozen=True)
class Hypothesis:
    spec: Specification
    null: str
    mu: float | None = None
    p: float | None = None

    def generate(
        self, reps: int, type: str | None = None, *, seed: int | None = None
    ) -> GeneratedReplicates:
        if type is None:
            type = "bootstrap" if self.null == "point" else "permute"
        return _generate(self.spec, hypothesis=self, reps=reps, type=type, seed=seed)

    def calculate(
        self,
        stat,
        *,
        order: tuple[object, object] | None = None,
    ) -> ObservedStatistic:
        """Observed statistic that needs the hypothesized value (e.g. stat='t'/'z')."""
        return self.spec.calculate(stat, order=order, mu=self.mu, p=self.p)


@dataclass(frozen=True)
class GeneratedReplicates:
    """Materialized resampling plan; supports both ``calculate()`` and ``fit()``.

    ``plans`` holds one numpy array per replicate:
    - bootstrap: row indices (with replacement) into the data,
    - permute: a permutation of the row positions (used to shuffle a column),
    - draw: a simulated response array under a point null proportion.
    """

    spec: Specification
    type: str
    null: str | None
    plans: list[np.ndarray] = field(repr=False)
    shifted_response: np.ndarray | None = field(default=None, repr=False)
    hyp_mu: float | None = None
    hyp_p: float | None = None

    # --- single-variable / two-group statistics ---------------------------
    def calculate(
        self, stat, *, order: tuple[object, object] | None = None
    ) -> Distribution:
        spec = self.spec
        resp_full = self.shifted_response
        if resp_full is None:
            resp_full = spec._response_values
        expl_full = spec._explanatory_values
        paired = self.null == "paired independence"

        values = []
        for plan in self.plans:
            if self.type == "bootstrap":
                resp = resp_full[plan]
                expl = None if expl_full is None else expl_full[plan]
            elif self.type == "permute" and paired:
                resp = resp_full * plan  # randomly flip the sign of each difference
                expl = None
            elif self.type == "permute":
                resp = resp_full
                expl = None if expl_full is None else expl_full[plan]
            else:  # draw
                resp = plan  # the simulated response array
                expl = None
            values.append(
                _stats.compute_statistic(
                    resp, expl, stat, success=spec.success, order=order,
                    mu=self.hyp_mu, p=self.hyp_p,
                )
            )
        df = pl.DataFrame(
            {
                "replicate": np.arange(1, len(values) + 1, dtype=np.int64),
                "stat": np.asarray(values, dtype=float),
            }
        )
        label = stat if isinstance(stat, str) else "stat"
        return Distribution(data=df, stat=label, null=self.null, type=self.type)

    # --- multiple regression ---------------------------------------------
    def fit(self) -> "FitResult":
        """Fit OLS on each replicate, returning per-term estimates (long form)."""
        import statsmodels.formula.api as smf

        from ..modeling import _clean_term

        spec = self.spec
        formula = spec._full_formula()
        base = spec.data
        response = spec.response

        rows_replicate, rows_term, rows_estimate = [], [], []
        for i, plan in enumerate(self.plans, start=1):
            if self.type == "bootstrap":
                rep_data = base[plan.tolist()]
            elif self.type == "permute":
                # Shuffle the response column to break association under the null.
                shuffled = base[response].to_numpy()[plan]
                rep_data = base.with_columns(pl.Series(response, shuffled))
            else:
                raise ValueError("fit() supports type='bootstrap' or 'permute'")
            model = smf.ols(formula, data=rep_data.to_pandas()).fit()
            for term, est in zip(model.params.index, model.params.to_numpy()):
                rows_replicate.append(i)
                rows_term.append(_clean_term(term))
                rows_estimate.append(float(est))

        df = pl.DataFrame(
            {"replicate": rows_replicate, "term": rows_term, "estimate": rows_estimate}
        )
        return FitResult(data=df, null=self.null, type=self.type)


def _generate(
    spec: Specification,
    *,
    hypothesis: Hypothesis | None,
    reps: int,
    type: str,
    seed: int | None,
) -> GeneratedReplicates:
    if type not in ("bootstrap", "permute", "draw"):
        raise ValueError("type must be 'bootstrap', 'permute', or 'draw'")
    rng = _resample.make_rng(seed)
    n = spec.data.height
    null = None if hypothesis is None else hypothesis.null
    hyp_mu = None if hypothesis is None else hypothesis.mu
    hyp_p = None if hypothesis is None else hypothesis.p
    shifted = None
    plans: list[np.ndarray] = []

    if type == "permute" and null == "paired independence":
        # Randomly flip the sign of each paired difference.
        for _ in range(reps):
            plans.append(rng.choice([-1.0, 1.0], size=n))
    elif type == "permute":
        if spec.explanatory is None and spec.formula is None:
            raise ValueError("type='permute' requires an explanatory variable")
        for _ in range(reps):
            plans.append(rng.permutation(n))
    elif type == "draw":
        if hypothesis is None or hypothesis.p is None:
            raise ValueError("type='draw' requires hypothesize(null='point', p=...)")
        success = spec.success
        nonsuccess = "\x00not_success"
        p0 = hypothesis.p
        for _ in range(reps):
            draws = rng.random(n) < p0
            plans.append(np.where(draws, success, nonsuccess).astype(object))
    else:  # bootstrap
        if (
            hypothesis is not None
            and hypothesis.null == "point"
            and hypothesis.mu is not None
        ):
            shifted = _resample.shift_for_point_null(
                spec._response_values, stat="mean", mu=hypothesis.mu, p=None
            )
        for _ in range(reps):
            plans.append(rng.integers(0, n, size=n))

    return GeneratedReplicates(
        spec=spec,
        type=type,
        null=null,
        plans=plans,
        shifted_response=shifted,
        hyp_mu=hyp_mu,
        hyp_p=hyp_p,
    )


@dataclass(frozen=True)
class ObservedStatistic:
    """A single observed statistic. Behaves like a float and prints like infer."""

    value: float
    stat: str

    def __float__(self) -> float:
        return float(self.value)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame({"stat": [self.value]})

    def __repr__(self) -> str:
        return f"ObservedStatistic(stat={self.stat!r}, value={self.value:.6g})"


@dataclass(frozen=True)
class Distribution:
    """A simulated distribution of statistics (one row per replicate)."""

    data: pl.DataFrame
    stat: str
    null: str | None = None
    type: str = "bootstrap"

    @property
    def stats(self) -> np.ndarray:
        return self.data["stat"].to_numpy()

    def __len__(self) -> int:
        return self.data.height

    def get_confidence_interval(
        self,
        level: float = 0.95,
        type: str = "percentile",
        *,
        point_estimate: float | None = None,
    ) -> pl.DataFrame:
        from .intervals import get_confidence_interval

        return get_confidence_interval(
            self, level=level, type=type, point_estimate=point_estimate
        )

    def get_p_value(self, obs_stat, direction: str) -> pl.DataFrame:
        from .pvalue import get_p_value

        return get_p_value(self, obs_stat=obs_stat, direction=direction)

    def visualize(self, bins: int = 20, **kwargs):
        from .viz import visualize

        return visualize(self, bins=bins, **kwargs)

    # infer-parity aliases.
    def get_pvalue(self, obs_stat, direction: str) -> pl.DataFrame:
        return self.get_p_value(obs_stat, direction)

    def get_ci(self, level: float = 0.95, type: str = "percentile", **kw) -> pl.DataFrame:
        return self.get_confidence_interval(level=level, type=type, **kw)

    def visualise(self, bins: int = 20, **kwargs):
        return self.visualize(bins=bins, **kwargs)


@dataclass(frozen=True)
class FitResult:
    """Regression coefficients: observed (one row per term) or a distribution of
    them across replicates (``replicate``, ``term``, ``estimate``)."""

    data: pl.DataFrame
    null: str | None = None
    type: str = "bootstrap"

    @property
    def is_distribution(self) -> bool:
        return "replicate" in self.data.columns

    # Display niceties so a FitResult shows its table in notebooks/Quarto.
    def head(self, n: int = 5) -> pl.DataFrame:
        return self.data.head(n)

    def _repr_html_(self) -> str:
        return self.data._repr_html_()

    def __repr__(self) -> str:
        return repr(self.data)

    def estimate_for(self, term: str) -> float:
        row = self.data.filter(pl.col("term") == term)
        return float(row["estimate"][0])

    def get_confidence_interval(self, level: float = 0.95) -> pl.DataFrame:
        from .intervals import get_fit_confidence_interval

        return get_fit_confidence_interval(self, level=level)

    def get_p_value(self, obs_stat: "FitResult", direction: str = "two-sided") -> pl.DataFrame:
        from .pvalue import get_fit_p_value

        return get_fit_p_value(self, obs_stat=obs_stat, direction=direction)

    def visualize(self, bins: int = 20):
        from .viz import visualize_fit

        return visualize_fit(self, bins=bins)


def specify(
    data: pl.DataFrame,
    *,
    response: str | None = None,
    explanatory: str | None = None,
    formula: str | None = None,
    success: object | None = None,
) -> Specification:
    """Specify the response (and optional explanatory) variable(s) for inference.

    Two equivalent forms, mirroring R ``infer``:

    - ``specify(df, response="weight")``
    - ``specify(df, formula="popular_or_not ~ track_genre", success="popular")``

    Multi-term formulas (``"y ~ a + b"``) are supported for :meth:`fit`.
    """
    full_formula = formula
    if formula is not None:
        if response is not None or explanatory is not None:
            raise ValueError("pass either `formula` or `response`/`explanatory`, not both")
        response, explanatory = _parse_formula(formula)
    if response is None:
        raise ValueError("specify() needs a `response` or a `formula`")
    if response not in data.columns:
        raise ValueError(f"column {response!r} not found in data")
    # Validate single-column explanatory only (multi-term handled by fit()).
    if (
        explanatory is not None
        and "+" not in explanatory
        and "*" not in explanatory
        and explanatory not in data.columns
    ):
        raise ValueError(f"column {explanatory!r} not found in data")
    return Specification(
        data=data,
        response=response,
        explanatory=explanatory,
        success=success,
        formula=full_formula,
    )


def observe(
    data: pl.DataFrame,
    *,
    response: str | None = None,
    explanatory: str | None = None,
    formula: str | None = None,
    success: object | None = None,
    stat: str = "mean",
    order: tuple[object, object] | None = None,
    null: str | None = None,
    mu: float | None = None,
    p: float | None = None,
) -> ObservedStatistic:
    """Shortcut for ``specify() |> [hypothesize()] |> calculate()``.

    Mirrors R ``infer::observe()`` — compute an observed statistic in one call.
    """
    spec = specify(
        data, response=response, explanatory=explanatory, formula=formula, success=success
    )
    if null is not None:
        return spec.hypothesize(null=null, mu=mu, p=p).calculate(stat, order=order)
    return spec.calculate(stat, order=order, mu=mu, p=p)
