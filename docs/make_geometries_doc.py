"""
(Re)create the `geometries.rst` document.
"""

import datetime
import pathlib
from collections import defaultdict

from pyRestTable import Table

import hklpy2

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


def title(text: str, underchar: str = H1, both: bool = False) -> str:
    bars = underchar * len(text)
    result = f"{text}\n{bars}\n"
    if both:
        result = f"{bars}\n{result}"
    return result


def page_header():
    text = [
        f".. author: {__file__.split('/')[-1]}",
        f".. date: {datetime.datetime.now()}",
        "",
        ".. _geometries:",
        "",
        title(PAGE_TITLE, underchar=H1, both=True),
        ".. index:: geometries",
        "",
    ]
    return "\n".join(text)


def rst_anchor(sname: str, gname: str) -> str:
    replacement = "-"
    for c in [" ", ".", "_"]:
        gname = gname.replace(c, replacement)
    return f"geometries-{sname}-{gname}".lower()


def table_of_reals():
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


def all_summary_tables():
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


def main():
    text = [
        page_header(),
        PREFACE,
        table_of_reals(),
        all_summary_tables(),
    ]
    with open(GEO_DOC, "w") as f:
        f.write(linter("\n".join(text)))


if __name__ == "__main__":
    main()
