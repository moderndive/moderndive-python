# Plotting: plotly & plotnine

Every plotting function in `moderndive` takes an `engine=` argument:

- `engine="plotly"` (**default**) — interactive figures (`plotly.graph_objects.Figure`).
- `engine="plotnine"` — grammar-of-graphics figures (`plotnine.ggplot`).

The composition syntax is identical across engines, so you can switch a whole
analysis by changing one argument.

## visualize() returns an InferPlot

`visualize()` returns a small `InferPlot` wrapper that you compose with shading via
`+` in **both** engines:

```python
import moderndive as md
from moderndive import specify, get_confidence_interval, visualize, shade_confidence_interval

boot = (
    specify(md.load_almonds_sample_100(), response="weight")
    .generate(reps=1000, type="bootstrap", seed=1)
    .calculate(stat="mean")
)
ci = get_confidence_interval(boot, level=0.95, type="percentile")

p = visualize(boot) + shade_confidence_interval(ci)   # plotly InferPlot
p.figure          # the underlying plotly Figure
p.show()          # display it
```

For the plotnine engine, the raw `ggplot` is available via `.gg`:

```python
g = visualize(boot, engine="plotnine") + shade_confidence_interval(ci)
g.gg              # the underlying plotnine ggplot
```

### Keyword form

If you'd rather not use `+`, pass the shading inline (handy for plotly):

```python
visualize(boot, shade_ci=ci)
visualize(null, shade_pvalue={"obs_stat": obs, "direction": "right"})
```

## Shading p-values

```python
from moderndive import observe, shade_p_value

promotions = md.load_promotions()
obs = observe(promotions, formula="decision ~ gender", success="promoted",
              stat="diff in props", order=("male", "female"))
null = (
    specify(promotions, formula="decision ~ gender", success="promoted")
    .hypothesize(null="independence")
    .generate(reps=1000, type="permute", seed=42)
    .calculate(stat="diff in props", order=("male", "female"))
)

visualize(null) + shade_p_value(obs_stat=obs, direction="right")
```

## Simulation vs. theory overlays

```python
visualize(boot, method="simulation")    # histogram (default)
visualize(boot, method="theoretical")   # normal-approximation curve
visualize(boot, method="both")          # histogram + curve overlaid
```

## Faceted regression-coefficient plots

`visualize_fit` shows one panel per term, and shading is **per-facet**:

```python
from moderndive.infer.viz import visualize_fit

f = "price ~ living_area + bedrooms"
houses = md.load_saratoga_houses()
boot_fit = specify(houses, formula=f).generate(reps=1000, type="bootstrap", seed=1).fit()

visualize_fit(boot_fit) + shade_confidence_interval(boot_fit.get_confidence_interval())
```

## Model plots

```python
from moderndive import gg_parallel_slopes, gg_categorical_model, pairplot

evals = md.load_evals()
gg_parallel_slopes(evals, response="score", explanatory="age", by="gender")
gg_categorical_model(evals, response="score", explanatory="rank")

# Scatterplot matrix (~ GGally::ggpairs)
pairplot(md.load_coffee_quality(), columns=["total_cup_points", "aroma", "flavor"])
```

## Saving figures

```python
p = visualize(boot)            # plotly
p.save("dist.html")            # interactive HTML (no extra deps)
p.save("dist.png")             # static image — needs: pip install "moderndive[image]"

g = visualize(boot, engine="plotnine")
g.save("dist.png", width=6, height=4, dpi=150)
```

`pairplot(..., engine="seaborn")` (alias `"plotnine"`) returns a Matplotlib figure
if you prefer the seaborn-backed scatterplot matrix.
