"""Sphinx configuration for the moderndive (Python) documentation."""

from importlib.metadata import version as _version

project = "moderndive"
author = "Chester Ismay, Albert Y. Kim, and Arturo Valdivia"
copyright = "2026, ModernDive"
try:
    release = _version("moderndive")
except Exception:  # pragma: no cover - docs build without install
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",  # MyST markdown + execution of {code-cell} blocks (supersedes myst_parser)
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "deflist"]

# --- myst-nb execution -------------------------------------------------------
# Pages carrying a Jupytext frontmatter header run their {code-cell} blocks at
# build time so real outputs (and plots) appear on the site. Plots are rendered
# as static PNGs: plotnine via matplotlib, plotly via kaleido (the renderer is
# set to "png" in each page's hidden setup cell). We tell myst-nb to prefer the
# PNG over plotly's heavy interactive text/html so pages stay small.
nb_execution_mode = "auto"
nb_execution_raise_on_error = True
nb_execution_timeout = 120
nb_output_stderr = "remove"
nb_mime_priority_overrides = [("html", "image/png", 0)]
# Heavy scientific deps are installed in the RTD build; nothing to mock.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "plotnine": ("https://plotnine.org", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

html_theme = "furo"
html_title = f"moderndive {version}"
html_static_path = ["_static"]
html_logo = "_static/moderndive-logo.png"
html_favicon = "_static/moderndive-logo.png"
