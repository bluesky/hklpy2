"""
(Re)create the `diffractometers.rst` document.

The file is only written when the installed solver versions differ from those
recorded in the existing file.  Version information is stored in a
machine-readable sentinel comment at the top of the file::

    .. solvers: hkl_soleil=5.1.2 th_tth=0.0.14

This sentinel is also displayed on the page as a human-readable table so
users can verify which library versions produced the geometry tables.

Automation
----------
Call :func:`main` from a Sphinx ``builder-inited`` event in ``conf.py``
to keep the file current automatically on every docs build.
"""

import logging
import pathlib
import re
from collections import defaultdict

from pyRestTable import Table

import hklpy2

logger = logging.getLogger(__name__)

DOCS_DIR = pathlib.Path(__file__).parent / "source"
GEO_DOC = DOCS_DIR / "diffractometers.rst"
H1, H2, H3, H4 = "= - ^ ~".split()
PAGE_TITLE = "Diffractometers"

PREFACE = """
Tables are provided for the different geometries (sorted by number of real axes)
and then, for each geometry, the calculation engines, modes of operation, pseudo
axes required, and any additional parameters required by the
:meth:`~hklpy2.backends.base.SolverBase.mode`.  The mode defines which axes will
be computed, which will be held constant, and any relationships between axes.

.. seealso::

   :ref:`concepts.lattice` — crystal lattice parameters and the seven crystal
   systems, including the :ref:`lattice.crystal-systems` reference table.
"""

# Regex to extract the sentinel line from an existing file.
_SENTINEL_RE = re.compile(r"^\.\. solvers:\s*(.+)$", re.MULTILINE)


def solver_versions() -> dict[str, str]:
    """Return a dict of {solver_name: version} for all registered solvers."""
    versions = {}
    for sname in sorted(hklpy2.solvers()):
        Solver = hklpy2.get_solver(sname)
        if not Solver.geometries():
            continue
        gname = sorted(Solver.geometries())[0]
        versions[sname] = Solver(gname).version
    return versions


def sentinel_string(versions: dict[str, str]) -> str:
    """Render solver versions as a compact sentinel string."""
    return " ".join(f"{k}={v}" for k, v in sorted(versions.items()))


def is_stale(versions: dict[str, str]) -> bool:
    """Return True if diffractometers.rst is missing or has different solver versions."""
    if not GEO_DOC.exists():
        logger.info("diffractometers.rst does not exist — will generate.")
        return True
    text = GEO_DOC.read_text()
    m = _SENTINEL_RE.search(text)
    if m is None:
        logger.info("diffractometers.rst has no solver sentinel — will regenerate.")
        return True
    current = sentinel_string(versions)
    recorded = m.group(1).strip()
    if current != recorded:
        logger.info(
            "diffractometers.rst is stale: recorded=%r current=%r — will regenerate.",
            recorded,
            current,
        )
        return True
    logger.info(
        "diffractometers.rst is up to date (solvers: %s) — skipping regeneration.",
        current,
    )
    return False


def title(text: str, underchar: str = H1, both: bool = False) -> str:
    bars = underchar * len(text)
    result = f"{text}\n{bars}\n"
    if both:
        result = f"{bars}\n{result}"
    return result


def page_header(versions: dict[str, str]) -> str:
    text = [
        f".. author: {pathlib.Path(__file__).name}",
        f".. solvers: {sentinel_string(versions)}",
        "",
        ".. _geometries:",
        "",
        title(PAGE_TITLE, underchar=H1, both=True),
        ".. index:: geometries",
        "",
    ]
    return "\n".join(text)


def solver_versions_table(versions: dict[str, str]) -> str:
    """RST table of solver names and versions displayed on the page."""
    table = Table()
    table.labels = ["Solver", "Version"]
    for sname, ver in sorted(versions.items()):
        table.addRow((sname, ver))
    text = [
        ".. _geometries.solver-versions:",
        "",
        title("Solver Versions", H1),
        ".. index:: geometries; solver versions",
        "",
        "The geometry tables below were generated from these installed solver versions:",
        "",
        str(table),
    ]
    return "\n".join(text)


def rst_anchor(sname: str, gname: str) -> str:
    replacement = "-"
    for c in [" ", ".", "_"]:
        gname = gname.replace(c, replacement)
    return f"geometries-{sname}-{gname}".lower()


def table_of_reals() -> str:
    # Count the reals.
    circles = defaultdict(list)
    for sname in sorted(hklpy2.solvers()):
        Solver = hklpy2.get_solver(sname)
        geometries = Solver.geometries()
        if len(geometries) == 0:
            continue
        for gname in sorted(Solver.geometries()):
            solver = Solver(gname)
            n = len(solver.real_axis_names)
            anchor = f":ref:`{sname}, {gname} <{rst_anchor(sname, gname)}>`"
            circles[n].append(anchor)

    # Build the table, sorted by number of reals.
    table = Table()
    table.labels = ["#reals", "solver, geometry"]
    for n, anchors in sorted(circles.items()):
        for anchor in anchors:
            table.addRow((n, anchor))

    # Build the report.
    text = [
        ".. _geometries.number_of_reals:",
        "",
        title("Geometries, by number of real axes", H1),
        ".. index:: geometries; by number of reals",
        "",
        "The different diffractometer geometries are distinguished,",
        "primarily, by the number of real axes.  This",
        "table is sorted first by the number of real axes, then by",
        "solver and geometry names.",
        "",
        str(table),
    ]
    return "\n".join(text)


def geometry_summary_table(solver_name, geometry_name: str) -> str:
    text = [
        f".. _{rst_anchor(solver_name, geometry_name)}:",
        "",
        title(f"solver={solver_name!r}, geometry={geometry_name!r}", H2),
        f".. index:: geometries; {solver_name}; {geometry_name}",
        "",
        str(hklpy2.get_solver(solver_name)(geometry=geometry_name).summary),
    ]
    return "\n".join(text)


def all_summary_tables() -> str:
    text = [
        ".. _geometries.summary_tables:",
        "",
        title("Available Solver Geometry Tables", H1),
        ".. index:: geometries; tables",
        "",
        ".. seealso:: :func:`hklpy2.user.solver_summary()`",
        "",
    ]

    for sname in sorted(hklpy2.solvers()):
        Solver = hklpy2.get_solver(sname)
        geometries = Solver.geometries()
        if len(geometries) == 0:
            continue
        for gname in sorted(Solver.geometries()):
            text.append(geometry_summary_table(sname, gname))

    return "\n".join(text)


def linter(text: str) -> str:
    """
    Clean up items that would be corrected on pre-commit linting.

    * trailing-whitespace
    """
    text = "\n".join(
        [
            line.rstrip()
            #
            for line in text.strip().splitlines()
        ]
    )
    return f"{text}\n"  # always end with blank line


def main() -> bool:
    """
    Generate ``diffractometers.rst`` if solver versions have changed.

    Returns
    -------
    bool
        ``True`` if the file was (re)generated, ``False`` if it was already
        up to date and no write was performed.
    """
    versions = solver_versions()
    if not is_stale(versions):
        return False

    text = [
        page_header(versions),
        PREFACE,
        solver_versions_table(versions),
        table_of_reals(),
        all_summary_tables(),
    ]
    GEO_DOC.write_text(linter("\n".join(text)))
    logger.info("diffractometers.rst written (solvers: %s).", sentinel_string(versions))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
