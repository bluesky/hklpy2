# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
diffractometers
"""

import math
import pathlib

from ophyd import Component as Cpt
from ophyd import Kind, SoftPositioner

from ..diffract import (
    DiffractometerBase,
    Hklpy2PseudoAxis,
    diffractometer_class_factory,
)
from ..utils import load_yaml_file

E4CV_CONFIG_FILE = pathlib.Path(__file__).parent / "e4cv_orient.yml"
HN = Kind.hinted | Kind.normal


def e4cv_config():
    return load_yaml_file(E4CV_CONFIG_FILE)


def add_oriented_vibranium_to_e4cv(e4cv):
    e4cv.add_sample("vibranium", 2 * math.pi, digits=3, replace=True)
    e4cv.beam.wavelength.put(1.54)
    e4cv.add_reflection(
        (4, 0, 0), {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066}, name="r400"
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

    h2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    k2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    l2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    psi = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)

    # and a few more axes not used by 4-circle code

    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    mu = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    nu = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    omicron = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)


class MultiAxis99NoSolver(DiffractometerBase):
    """Test case.  9 pseudo axes and 9 real axes."""

    _pseudo = ["p1", "p2"]
    _real = ["r1", "r2", "r3", "r4"]

    p1 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p2 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p3 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p4 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p5 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p6 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p7 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p8 = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    p9 = Cpt(Hklpy2PseudoAxis, "", kind=HN)

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

    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)

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

    _pseudo = ["q"]
    _real = ["theta", "ttheta"]

    # sorted alphabetically
    another = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    q = Cpt(Hklpy2PseudoAxis, "", kind=HN)
    horizontal = Cpt(SoftPositioner, limits=(-10, 855), init_pos=0, kind=HN)
    theta = Cpt(SoftPositioner, limits=(-90, 90), init_pos=0, kind=HN)
    ttheta = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=HN)
    vertical = Cpt(SoftPositioner, limits=(-10, 855), init_pos=0, kind=HN)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver="th_tth", geometry="TH TTH Q", **kwargs)
