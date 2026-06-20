# API reference

## Inference grammar

```{eval-rst}
.. autofunction:: moderndive.specify
.. autofunction:: moderndive.observe
.. autofunction:: moderndive.assume
.. autoclass:: moderndive.infer.core.Specification
   :members: hypothesize, generate, calculate, fit, assume
.. autoclass:: moderndive.infer.core.Hypothesis
   :members: generate, calculate
.. autoclass:: moderndive.infer.core.GeneratedReplicates
   :members: calculate, fit
.. autoclass:: moderndive.infer.core.Distribution
   :members: get_confidence_interval, get_p_value, visualize
.. autoclass:: moderndive.infer.core.FitResult
   :members: get_confidence_interval, get_p_value, visualize, estimate_for
.. autoclass:: moderndive.infer.theoretical.TheoreticalDistribution
   :members:
```

## Getters and visualization

```{eval-rst}
.. autofunction:: moderndive.get_p_value
.. autofunction:: moderndive.get_confidence_interval
.. autofunction:: moderndive.visualize
.. autofunction:: moderndive.shade_p_value
.. autofunction:: moderndive.shade_confidence_interval
```

## Theory-based tests

```{eval-rst}
.. autofunction:: moderndive.t_test
.. autofunction:: moderndive.t_stat
.. autofunction:: moderndive.prop_test
.. autofunction:: moderndive.chisq_test
.. autofunction:: moderndive.chisq_stat
.. automodule:: moderndive.theory
   :members:
```

## Regression & summary helpers

```{eval-rst}
.. autofunction:: moderndive.get_regression_table
.. autofunction:: moderndive.get_regression_points
.. autofunction:: moderndive.get_regression_summaries
.. autofunction:: moderndive.get_correlation
.. autofunction:: moderndive.pop_sd
.. autofunction:: moderndive.tidy_summary
.. autofunction:: moderndive.count_missing
```

## Sampling and plots

All plotting helpers accept ``engine="plotly"`` (default) or ``engine="plotnine"``.

```{eval-rst}
.. autofunction:: moderndive.rep_slice_sample
.. autofunction:: moderndive.rep_sample_n
.. autofunction:: moderndive.pairplot
.. autofunction:: moderndive.gg_parallel_slopes
.. autofunction:: moderndive.geom_parallel_slopes
.. autofunction:: moderndive.gg_categorical_model
.. autofunction:: moderndive.geom_categorical_model
.. autofunction:: moderndive.plot_3d_regression
```

## Viewing data

```{eval-rst}
.. autofunction:: moderndive.View
```

## Datasets

```{eval-rst}
.. autofunction:: moderndive.load_dataset
.. autofunction:: moderndive.data.available_datasets
```

Each dataset also has a convenience loader ``moderndive.load_<name>()`` returning
a polars ``DataFrame``. Call :func:`moderndive.data.available_datasets` for the
full list.
