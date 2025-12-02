"""
diffractometers
"""

import math
import pathlib

from ophyd import Component as Cpt
from ophyd import Kind
from ophyd import SoftPositioner

from ..diffract import DiffractometerBase
from ..diffract import Hklpy2PseudoAxis
from ..diffract import diffractometer_class_factory
from ..misc import load_yaml_file

E4CV_CONFIG_FILE = pathlib.Path(__file__).parent / "e4cv_orient.yml"
HN = Kind.hinted | Kind.normal


def e4cv_config():
    return load_yaml_file(E4CV_CONFIG_FILE)


def add_oriented_vibranium_to_e4cv(e4cv):
    e4cv.add_sample("vibranium", 2 * math.pi, digits=3, replace=True)
    e4cv.beam.wavelength.put(1.54)
    e4cv.add_reflection(
        (4, 0, 0), dict(omega=-145.451, chi=0, phi=0, tth=69.066), name="r400"
    )
    r040 = e4cv.add_reflection((0, 4, 0), (-145.451, 0, 90, 69.066), name="r040")
    r004 = e4cv.add_reflection((0, 0, 4), (-145.451, 90, 0, 69.066), name="r004")
    e4cv.core.calc_UB(r040, r004)

    for constraint in e4cv.core.constraints.values():
        if "limits" in dir(constraint):
            constraint.limits = (-180.2, 180.2)  # just a little different


Fourc = diffractometer_class_factory()  # E4CV, hkl_soleil, hkl engine


class AugmentedFourc(Fourc):
    """Test case."""

    # define a few more axes,
    # extra parameters for some geometries/engines/modes

    h2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    k2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    l2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    psi = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)

    # and a few more axes not used by 4-circle code

    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    mu = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    nu = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    omicron = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)


class MultiAxis99NoSolver(DiffractometerBase):
    """Test case.  9 pseudo axes and 9 real axes."""

    _pseudo = "p1 p2".split()
    _real = "r1 r2 r3 r4".split()

    p1 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p3 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p4 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p5 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p6 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p7 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p8 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    p9 = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741

    r1 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r2 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r3 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r4 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r5 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r6 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r7 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r8 = Cpt(SoftPositioner, init_pos=0, kind=HN)
    r9 = Cpt(SoftPositioner, init_pos=0, kind=HN)

    # Should fail if no solver identified


class MultiAxis99(MultiAxis99NoSolver):
    """Fix by calling constructor with a solver & geometry."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            solver="no_op",  # no-op accepts ANY geometry name.
            geometry="multi-axis",
            **kwargs,
        )


class NoOpTh2Th(DiffractometerBase):
    """Test case."""

    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741

    th = Cpt(SoftPositioner, limits=(-90, 90), init_pos=0, kind=HN)
    tth = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            solver="no_op",  # no-op accepts ANY geometry name.
            geometry="powder",
            **kwargs,
        )


class TwoC(DiffractometerBase):
    """Test case with custom names and additional axes."""

    _pseudo = "q".split()
    _real = "theta ttheta".split()

    # sorted alphabetically
    another = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)  # noqa: E741
    horizontal = Cpt(SoftPositioner, limits=(-10, 855), init_pos=0, kind=HN)
    theta = Cpt(SoftPositioner, limits=(-90, 90), init_pos=0, kind=HN)
    ttheta = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    vertical = Cpt(SoftPositioner, limits=(-10, 855), init_pos=0, kind=HN)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            solver="th_tth",
            geometry="TH TTH Q",
            **kwargs,
        )
