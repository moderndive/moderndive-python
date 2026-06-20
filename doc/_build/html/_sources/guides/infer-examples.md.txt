---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} python
:tags: [remove-input]
import matplotlib
matplotlib.use("Agg")
import plotly.io as pio
pio.renderers.default = "png"
```

# Pipeline examples (the infer "stat menu")

This page follows the R `infer`
[observed-statistic examples](https://infer.tidymodels.org/articles/observed_stat_examples.html)
vignette: the `calculate(stat=...)` forms organized by the **types of variables**
involved, using the same `gss` dataset (a sample from the US General Social
Survey) that ships with the package.

```{note}
This covers the statistics the package currently supports. One example from the R
vignette is **not** yet available here: the chi-square **goodness-of-fit** test
(one categorical variable with 3+ levels against a vector of hypothesized
proportions). See the note in the chi-square section below.
```

```{code-cell} python
import moderndive as md
from moderndive import get_p_value, get_confidence_interval, assume

gss = md.load_gss()
gss.head()
```

Throughout, the pattern is the same as in R `infer`:

- the **observed statistic** comes from `specify() → [hypothesize()] → calculate()`;
- a **null distribution** comes from adding `generate()` before `calculate()`;
- a **bootstrap distribution** (for confidence intervals) comes from
  `generate(type="bootstrap")` with no `hypothesize()`.

## One numerical variable

```{code-cell} python
# mean
gss.specify(response="hours").calculate(stat="mean")
```

```{code-cell} python
# median, then standard deviation
print(gss.specify(response="hours").calculate(stat="median"))
print(gss.specify(response="hours").calculate(stat="sd"))
```

**Standardized mean (one-sample `t`)** — needs a hypothesized `mu`:

```{code-cell} python
gss.specify(response="hours").hypothesize(null="point", mu=40).calculate(stat="t")
```

A full one-mean test — null distribution and p-value:

```{code-cell} python
obs_mean = gss.specify(response="hours").calculate(stat="mean")
null_mean = (
    gss.specify(response="hours")
    .hypothesize(null="point", mu=40)
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
get_p_value(null_mean, obs_stat=obs_mean, direction="two-sided")
```

…and a bootstrap confidence interval for the mean:

```{code-cell} python
boot_mean = (
    gss.specify(response="hours")
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
get_confidence_interval(boot_mean, level=0.95, type="percentile")
```

## One categorical variable

```{code-cell} python
# proportion of a "success" level, then the raw count
print(gss.specify(response="sex", success="female").calculate(stat="prop"))
print(gss.specify(response="sex", success="female").calculate(stat="count"))
```

**Standardized proportion (one-sample `z`)** — needs a hypothesized `p`:

```{code-cell} python
gss.specify(response="sex", success="female").hypothesize(null="point", p=0.5).calculate(stat="z")
```

A one-proportion test by *simulation* (`draw`, alias `simulate`):

```{code-cell} python
obs_prop = gss.specify(response="sex", success="female").calculate(stat="prop")
null_prop = (
    gss.specify(response="sex", success="female")
    .hypothesize(null="point", p=0.5)
    .generate(reps=1000, type="draw", seed=1)
    .calculate(stat="prop")
)
get_p_value(null_prop, obs_stat=obs_prop, direction="two-sided")
```

```{note}
The chi-square **goodness-of-fit** test (one categorical variable with 3+ levels
against a vector of hypothesized proportions) is the one infer example this
package does not yet support — `calculate(stat="Chisq")` here requires an
explanatory variable. Use a two-variable chi-square test of independence (below).
```

## Two categorical variables (2 levels each)

```{code-cell} python
# difference in proportions of "degree" between the sexes
gss.specify(formula="college ~ sex", success="degree").calculate(
    stat="diff in props", order=("male", "female")
)
```

```{code-cell} python
# ratio of proportions and the odds ratio
spec = gss.specify(formula="college ~ sex", success="degree")
print(spec.calculate(stat="ratio of props", order=("male", "female")))
print(spec.calculate(stat="odds ratio", order=("male", "female")))
```

**Standardized difference in proportions (`z`)**:

```{code-cell} python
gss.specify(formula="college ~ sex", success="degree").calculate(
    stat="z", order=("male", "female")
)
```

## Two categorical variables (chi-square test of independence)

```{code-cell} python
obs_chisq = gss.specify(formula="finrela ~ sex").calculate(stat="Chisq")
obs_chisq
```

```{code-cell} python
null_chisq = (
    gss.specify(formula="finrela ~ sex")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=1)
    .calculate(stat="Chisq")
)
get_p_value(null_chisq, obs_stat=obs_chisq, direction="greater")
```

A chi-square test is inherently one-sided; the theoretical distribution agrees:

```{code-cell} python
n_levels_finrela = gss["finrela"].n_unique()
df_chisq = (n_levels_finrela - 1) * (gss["sex"].n_unique() - 1)
assume("Chisq", df=df_chisq).get_p_value(obs_chisq, direction="greater")
```

## One numerical and one categorical variable (2 levels)

```{code-cell} python
# difference in mean age by college degree
gss.specify(formula="age ~ college").calculate(
    stat="diff in means", order=("degree", "no degree")
)
```

```{code-cell} python
# difference in medians, ratio of means, and the two-sample t
spec = gss.specify(formula="age ~ college")
print(spec.calculate(stat="diff in medians", order=("degree", "no degree")))
print(spec.calculate(stat="ratio of means", order=("degree", "no degree")))
print(spec.calculate(stat="t", order=("degree", "no degree")))
```

Null distribution and p-value for the difference in means:

```{code-cell} python
obs_diff = gss.specify(formula="age ~ college").calculate(
    stat="diff in means", order=("degree", "no degree")
)
null_diff = (
    gss.specify(formula="age ~ college")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=1)
    .calculate(stat="diff in means", order=("degree", "no degree"))
)
get_p_value(null_diff, obs_stat=obs_diff, direction="two-sided")
```

## One numerical and one categorical variable (3+ levels): ANOVA F

```{code-cell} python
gss.specify(formula="age ~ partyid").calculate(stat="F")
```

## Two numerical variables

```{code-cell} python
# Pearson correlation, then the regression slope
print(gss.specify(formula="hours ~ age").calculate(stat="correlation"))
print(gss.specify(formula="hours ~ age").calculate(stat="slope"))
```

A permutation test for the slope, visualized:

```{code-cell} python
from moderndive import visualize, shade_p_value

obs_slope = gss.specify(formula="hours ~ age").calculate(stat="slope")
null_slope = (
    gss.specify(formula="hours ~ age")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=1)
    .calculate(stat="slope")
)
visualize(null_slope) + shade_p_value(obs_stat=obs_slope, direction="two-sided")
```

## Custom statistics

Any of the above can be swapped for a callable returning a single number — see the
custom-statistic example in {doc}`hypothesis-testing`.
