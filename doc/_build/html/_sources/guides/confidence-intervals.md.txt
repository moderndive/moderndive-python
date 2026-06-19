# Bootstrapping & confidence intervals

Resample your data with replacement to build a **bootstrap distribution**, then
read a confidence interval off it.

## A bootstrap distribution for a mean

```python
import moderndive as md
from moderndive import specify, get_confidence_interval, visualize, shade_confidence_interval

almonds = md.load_almonds_sample_100()

boot = (
    specify(almonds, response="weight")
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
```

## Confidence intervals — three methods

```python
# Percentile method (default)
get_confidence_interval(boot, level=0.95, type="percentile")
# shape: (1, 2) → lower_ci ≈ 3.61, upper_ci ≈ 3.75

# Standard-error method (needs the point estimate)
point = float(specify(almonds, response="weight").calculate(stat="mean"))
get_confidence_interval(boot, level=0.95, type="se", point_estimate=point)

# Bias-corrected
get_confidence_interval(boot, level=0.95, type="bias-corrected", point_estimate=point)
```

Distribution objects also expose the getter as a method:

```python
boot.get_confidence_interval(level=0.95, type="percentile")
```

## Visualize the interval

```python
ci = get_confidence_interval(boot, level=0.95, type="percentile")

# plotly (default) — interactive
visualize(boot) + shade_confidence_interval(ci)

# plotnine
visualize(boot, engine="plotnine") + shade_confidence_interval(ci)

# one-call keyword form
visualize(boot, shade_ci=ci)
```

## A confidence interval for a proportion

```python
import polars as pl

mythbusters = md.load_mythbusters_yawn()

boot_prop = (
    specify(mythbusters, response="yawn", success="yes")
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="prop")
)
get_confidence_interval(boot_prop, level=0.95, type="percentile")
```

## A confidence interval for a difference

Add an explanatory variable and an `order=` to compare two groups:

```python
boot_diff = (
    specify(mythbusters, formula="yawn ~ group", success="yes")
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="diff in props", order=("seed", "control"))
)
boot_diff.get_confidence_interval(level=0.95)
```

```{seealso}
For *theory-based* intervals (the formula-based `t` interval) see
{doc}`theory-based`.
```
