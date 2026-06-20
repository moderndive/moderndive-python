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
```

# Helpful messages & errors

`moderndive` is built for people *learning* statistics, so its messages and errors
are written to teach, not just to report. They follow the same style as the R
`infer` package: a short **summary line**, then one or more `→` **hint bullets**
telling you what to do next.

There are two kinds:

- **Errors** (`ValueError` / `TypeError`) — something can't proceed, raised with a
  fix-it hint.
- **Informational notes** — a heads-up that doesn't stop your code, emitted as a
  suppressible `ModernDiveMessage` warning.

## Errors that tell you how to fix them

Mistype a column name and you get the available columns back, not a cryptic
`KeyError`:

```{code-cell} python
import moderndive as md
from moderndive import get_correlation

houses = md.load_saratoga_houses()
try:
    get_correlation(houses, "price ~ livng_area")  # typo
except ValueError as e:
    print(e)
```

Pass the wrong kind of object and the error names the fix:

```{code-cell} python
from moderndive import get_regression_table

try:
    get_regression_table("not a fitted model")
except TypeError as e:
    print(e)
```

```{code-cell} python
from moderndive import plot_3d_regression

# plot_3d_regression needs exactly two numeric predictors
try:
    plot_3d_regression(houses, "price ~ living_area")
except ValueError as e:
    print(e)
```

## Informational notes (and how to silence them)

Some functions point out something useful without stopping. For example, asking
for correlations against several predictors notes that you're getting
predictor-vs-outcome correlations only (not a full pairwise matrix):

```{code-cell} python
import warnings
from moderndive._messaging import ModernDiveMessage

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    get_correlation(houses, "price ~ living_area + bedrooms + bathrooms")

for w in caught:
    if issubclass(w.category, ModernDiveMessage):
        print(w.message)
```

These notes are designed to be **easy to turn off** once you know what they mean.
Many functions take a `quiet=True` flag:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    get_correlation(houses, "price ~ living_area + bedrooms", quiet=True)

print("messages emitted:", sum(issubclass(w.category, ModernDiveMessage) for w in caught))
```

Or silence every moderndive note globally with a standard `warnings` filter:

```python
import warnings
from moderndive._messaging import ModernDiveMessage

warnings.filterwarnings("ignore", category=ModernDiveMessage)
```

## Why this matters

For a newcomer, the difference between `KeyError: 'livng_area'` and "*Column(s)
not found in the data: livng_area. → Available columns: …*" is the difference
between getting stuck and fixing it in seconds. Because the informational notes
are their own warning category, they're equally easy to **see** while learning and
to **mute** once you're comfortable.
