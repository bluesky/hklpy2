# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# flake8: noqa

import pathlib
import sys
import tomllib
from importlib.metadata import version

root_path = pathlib.Path(__file__).parent.parent.parent

with open(root_path / "pyproject.toml", "rb") as fp:
    toml = tomllib.load(fp)
metadata = toml["project"]

sys.path.insert(0, str(root_path))

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
release = version(project)
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
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

exclude_patterns = ["**.ipynb_checkpoints"]
myst_enable_extensions = ["colon_fence"]
source_suffix = ".rst .md".split()
templates_path = ["_templates"]

# myst-nb notebook execution when building docs
nb_execution_mode = "off"

autoapi_dirs = ["../.."]
autoapi_ignore = [
    "*tests*",
    "dev_*",
    "**/examples/*",
    "docs/source/examples/*",
]

# # where the generated rst files will be written (relative to the Sphinx source dir)
# autoapi_output_dir = "autoapi"   # e.g. docs/source/autoapi

# # KEEP the generated files (default: False)
# autoapi_keep_files = True

# # optional: add the autoapi tree to the toctree
# autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_static_path = ["_static"]

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
}

rst_prolog = """
.. |hklpy| replace:: **hklpy (v1)**
.. |hklpy.url| replace:: https://blueskyproject.io/hklpy
.. |hklpy2| replace:: **hklpy2**
.. |libhkl| replace:: **Hkl/Soleil**
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
    coverage
    dask
    databroker
    gi
    numba
    numba_backend
    ophyd
    pandas
    pint
    sparse
    tqdm
""".split()

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
inheritance_graph_attrs = {"rankdir": "LR"}
inheritance_node_attrs = {"fontsize": 24}
autosummary_generate = True
