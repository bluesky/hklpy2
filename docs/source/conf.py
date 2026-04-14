# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# flake8: noqa

import os
import pathlib
import sys
import tomllib
from importlib.metadata import version

root_path = pathlib.Path(__file__).parent.parent.parent

with open(root_path / "pyproject.toml", "rb") as fp:
    toml = tomllib.load(fp)
metadata = toml["project"]

sys.path.insert(0, str(root_path / "src"))

# imports here for sphinx to build the documents without many WARNINGS.
import hklpy2

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = metadata["name"]
github_url = metadata["urls"]["source"]
copyright = toml["tool"]["copyright"]["copyright"]
author = metadata["authors"][0]["name"]
description = metadata["description"]
today_fmt = "%Y-%m-%d %H:%M"

# -- Special handling for version numbers ---------------------------------------------------
# https://github.com/pypa/setuptools_scm#usage-from-sphinx
release = hklpy2.__version__
version = ".".join(release.split(".")[:2])

# version_match: used by the version switcher to highlight the current version.
# Set DOC_VERSION in CI to "latest" (main branch) or the tag (e.g. "0.3.1").
# Falls back to release so local builds still work.
switcher_version_match = os.environ.get("DOC_VERSION", release)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_nb",  # includes "myst_parser"
    "nbsphinx",
    "autoapi.extension",
]
extensions.append("sphinx_tabs.tabs")  # this must be last

exclude_patterns = [
    "**.ipynb_checkpoints",
    "dev_*.ipynb",
    "dev_*.py",
]
myst_enable_extensions = ["colon_fence"]
source_suffix = ".rst .md".split()
templates_path = ["_templates"]

# myst-nb notebook execution when building docs
nb_execution_mode = "off"

autoapi_dirs = ["../.."]
autoapi_ignore = [
    ".dev/*",
    "**/.logs/*",
    "**/dev_*",
    "**/docs/*",
    "**/examples/*",
    "**/scripts/**",
    "*tests*",
    "**/conftest.py",
    "**/_version.py",
]
autoapi_add_toctree_entry = False  # added manually to index.rst
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autoapi_member_order = "alphabetical"
autoapi_template_dir = "_templates/autoapi"


def autoapi_skip_member(app, what, name, obj, skip, options):
    """Skip logger instances from the autoapi output."""
    if what == "data" and name.endswith(".logger"):
        return True
    return skip


def _shorten_type_str(s: str) -> str:
    """Replace dotted.module.ClassName with ClassName in type annotation strings.

    Used in literal code spans where Sphinx's domain resolver does not run.
    Bare names (``str``, ``int``, ``None``) are left unchanged.
    """
    import re

    return re.sub(
        r"(?<![.\w])([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)",
        lambda m: m.group(0).split(".")[-1],
        s,
    )


def _tilde_type_str(s: str) -> str:
    """Prepend ~ to dotted.module.ClassName in type annotation strings.

    Used inside ``.. py:*::`` directives so Sphinx resolves the full cross-
    reference but displays only the short name (the ``~`` prefix convention).
    Bare names (``str``, ``int``, ``None``) are left unchanged.
    """
    import re

    return re.sub(
        r"(?<![.\w~])([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)",
        lambda m: "~" + m.group(0),
        s,
    )


def _prepare_jinja_env(jinja_env) -> None:
    """Register custom Jinja2 filters for autoapi templates."""
    jinja_env.filters["shorten_type"] = _shorten_type_str
    jinja_env.filters["tilde_type"] = _tilde_type_str


def setup(app):
    """Connect autoapi events and register Jinja2 filters."""
    app.connect("autoapi-skip-member", autoapi_skip_member)
    app.config.autoapi_prepare_jinja_env = _prepare_jinja_env


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_context = {
    "github_user": "bluesky",
    "github_repo": "hklpy2",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "github_url": "https://github.com/bluesky/hklpy2",
    "use_edit_page_button": True,
    "navbar_align": "content",
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "switcher": {
        # Use the custom domain so all versions fetch from the same origin.
        "json_url": "https://blueskyproject.io/hklpy2/latest/_static/switcher.json",
        "version_match": switcher_version_match,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
}

rst_prolog = """
.. |hklpy| replace:: **hklpy (v1)**
.. |hklpy.url| replace:: https://blueskyproject.io/hklpy
.. |hklpy2| replace:: **hklpy2**
.. |libhkl| replace:: **Hkl/Soleil**
.. |ophyd| replace:: **ophyd**
.. |solver| replace:: **Solver**
.. |spec| replace:: **SPEC**
"""


# -- Options for autodoc ---------------------------------------------------

autodoc_exclude_members = ",".join(
    """
    __weakref__
    _component_kinds
    _device_tuple
    _required_for_connection
    _sig_attrs
    _sub_devices
    calc_class
    component_names
    logger
    """.split()
)
autodoc_default_options = {
    "members": True,
    # 'member-order': 'bysource',
    "private-members": True,
    # "special-members": True,
    "undoc-members": True,
    "exclude-members": autodoc_exclude_members,
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_mock_imports = """
    bluesky
    bluesky_tiled_plugins
    coverage
    dask
    gi
    numba
    numba_backend
    ophyd
    pandas
    pint
    sparse
    tiled
    tqdm
""".split()

extlinks = {
    "issue": (f"{github_url}/issues/%s", "issue #%s"),
    "pr": (f"{github_url}/pull/%s", "PR #%s"),
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
inheritance_graph_attrs = {"rankdir": "LR"}
inheritance_node_attrs = {"fontsize": 24}
autosummary_generate = True
