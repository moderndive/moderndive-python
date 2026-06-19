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
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "deflist"]
# Heavy scientific deps are installed in the RTD build; nothing to mock.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "plotnine": ("https://plotnine.org", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

html_theme = "furo"
html_title = f"moderndive {version}"
html_static_path = ["_static"]
html_logo = "_static/moderndive-logo.png"
html_favicon = "_static/moderndive-logo.png"
