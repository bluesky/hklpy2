"""
Simplified interface for |hklpy2| diffractometer users.

.. autosummary::

    ~add_reflection
    ~add_sample
    ~cahkl
    ~cahkl_table
    ~calc_UB
    ~get_diffractometer
    ~list_samples
    ~or_swap
    ~remove_reflection
    ~pa
    ~remove_sample
    ~scan_extra
    ~set_diffractometer
    ~set_lattice
    ~set_wavelength
    ~setor
    ~solver_summary
    ~wh

.. seealso:: :ref:`user_guide.quickstart`
"""

import uuid
from collections.abc import Iterable
from typing import Any, Dict, List
from typing import Optional
from typing import Union

from bluesky.protocols import Movable
from bluesky.protocols import Readable
from pyRestTable import Table

from .blocks.lattice import Lattice
from .blocks.reflection import Reflection
from .diffract import DiffractometerBase
from .misc import AnyAxesType
from .misc import AxesDict
from .misc import AxesTuple
from .misc import NoForwardSolutions
from .ops import CoreError

__all__ = """
    add_reflection
    add_sample
    cahkl
    cahkl_table
    calc_UB
    get_diffractometer
    list_samples
    or_swap
    pa
    remove_reflection
    remove_sample
    set_diffractometer
    set_lattice
    set_wavelength
    setor
    solver_summary
    wh
""".split()

import logging

logger = logging.getLogger(__name__)

Number = Union[int, float]


class _SelectedDiffractometer:
    """
    Module class to maintain the diffractometer selection.

    .. autosummary::

        ~_selection
        ~diffractometer
    """

    _selection = None
    """The current diffractometer."""

    @property
    def diffractometer(self) -> DiffractometerBase:
        if self._selection is None:
            raise ValueError(
                "No diffractometer selected."
                " Call 'set_diffractometer(diffr)' where"
                " 'diffr' is a diffractometer instance."
            )
        return self._selection

    @diffractometer.setter
    def diffractometer(self, diffractometer: DiffractometerBase) -> None:
        """Name the diffractometer to be used."""
        if not isinstance(diffractometer, DiffractometerBase):
            if diffractometer is not None:
                raise TypeError(
                    f"{diffractometer} must be an hklpy2 'DiffractometerBase' subclass."
                )
        self._selection = diffractometer


_choice = _SelectedDiffractometer()  # selected diffractometer geometry


def add_sample(
    name: str,
    a: float,
    b: float = None,
    c: float = None,
    alpha: float = 90,
    beta: float = None,
    gamma: float = None,
    digits: int = 4,
    replace: bool = False,
):
    """
    Add (and select) a new crystal sample.

    EXAMPLE:

    .. code-block:: python

        >>> add_sample("example", 2, 4, 5)
        Sample(name='example',
            lattice=Lattice(a=2, b=4, c=5, system='orthorhombic'))
    .. seealso:: :func:`~hklpy2.user.list_samples` :func:`~hklpy2.user.remove_sample`
    """
    diffractometer = _choice.diffractometer
    if name in diffractometer.samples:
        logger.warning(
            (
                f"Sample {name!r} is already defined."
                "  Add 'replace=True' to redefine this sample name."
                "  Call 'set_lattice(a, ...)' to define a new lattice."
            ),
        )
    else:
        diffractometer.add_sample(
            name,
            a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            digits=digits,
            replace=replace,
        )
    return diffractometer.sample


def cahkl(h: float, k: float, l: float) -> list | str:  # noqa: E741
    """
    Calculate motor positions for specified 'h, k l' - DOES NOT MOVE motors.

    Returns a 'RealPos' object or the text of the exception.

    EXAMPLE:

    .. code-block:: python

        >>> cahkl(1,-1, 1)
        Hklpy2DiffractometerRealPos(
            omega=12.254918848391,
            chi=-35.26440860898,
            phi=45.015980687529,
            tth=24.509837696782)
    """
    diffractometer = get_diffractometer()
    try:
        return diffractometer._forward_solution(
            diffractometer.real_position,
            diffractometer.core.forward(pseudos=(h, k, l)),
        )
    except NoForwardSolutions as exinfo:
        return str(exinfo)


def cahkl_table(*reflections: list[AxesTuple], digits=4):
    """
    Print a table with motor positions for each reflection given.

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> cahkl_table((1, 1, 0), (1, 1, 1))
        ======= = ====== ========= ====== ======
        (hkl)   # omega  chi       phi    tth
        ======= = ====== ========= ====== ======
        (1 1 0) 1 45.0   45.0      90.0   90.0
        (1 1 0) 2 -45.0  -45.0     -90.0  -90.0
        (1 1 0) 3 45.0   135.0     -90.0  90.0
        (1 1 0) 4 -135.0 -45.0     -90.0  90.0
        (1 1 0) 5 -45.0  -135.0    90.0   -90.0
        (1 1 0) 6 -135.0 -135.0    90.0   90.0
        (1 1 1) 1 60.0   35.2644   45.0   120.0
        (1 1 1) 2 -60.0  -35.2644  -135.0 -120.0
        (1 1 1) 3 -60.0  -144.7356 45.0   -120.0
        (1 1 1) 4 -120.0 -35.2644  -135.0 120.0
        (1 1 1) 5 -120.0 -144.7356 45.0   120.0
        (1 1 1) 6 60.0   144.7356  -135.0 120.0
        ======= = ====== ========= ====== ======

    Parameters
    ----------
    reflections : list(tuple(number,number,number))
        This is a list of reflections where
        each reflection is a tuple of 3 numbers
        specifying (h, k, l) of the reflection
        to compute the ``forward()`` computation.

        Example:  ``[(1,0,0), (1,1,1)]``
    digits : int
        Number of digits to roundoff each position
        value.  Default is 5.
    """

    def brief(data: Dict[str, Number]) -> List[str]:
        def fmt(v: Number) -> str:
            if isinstance(v, int):
                return str(v)
            fv = round(float(v), digits)
            if fv == 0.0:
                return "0"
            # remove trailing ".0" for whole numbers
            s = f"{fv:.{digits}f}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s

        return [fmt(v) for v in data.values()]

    # def brief(input: dict[str, (float | int)]) -> list[(float | int)]:
    #     return [round(v, digits) for v in input.values()]

    core = get_diffractometer().core
    reals = get_diffractometer().real_axis_names
    table = Table()
    table.labels = ["(hkl)", "#"] + reals
    for r in reflections:
        r = core.standardize_pseudos(r)
        rstr = "(" + " ".join([str(v) for v in brief(r)]) + ")"
        i = 0
        for solution in core.forward(r):
            i += 1
            s = core.standardize_reals(solution)
            table.addRow([rstr, i] + brief(s))
    print(table)


def calc_UB(
    r1: Union[Reflection, str],
    r2: Union[Reflection, str],
    wavelength: float = None,
) -> list[list[float]]:
    """
    Compute the UB matrix with two reflections.

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> r400 = setor(name='r400', 4, 0, 0, omega=-145.451, chi=0, phi=0, tth=69.066)
        >>> r004 = setor(name='r004', 0, 0, 4, omega=-145.451, chi=90, phi=0, tth=69.066)
        >>> calc_UB(r400, r004)
        [[-0.000279252712, 0.999999913446, -0.000279252646],
        [0.0, -0.000279400627, -1.000000132342],
        [-1.000000087766, -0.000280008582, 2.82915e-07]]

    """
    return get_diffractometer().core.calc_UB(r1, r2)


def solver_summary(write=True):
    """
    Table of diffractometer solver's modes, axes, ...

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> import hklpy2
        >>> from hklpy2.user import *
        >>> e4cv = hklpy2.creator(name="e4cv")
        >>> set_diffractometer(e4cv)
        >>> solver_summary()
        ========= ================== ================== ==================== ==================== ===============
        engine    mode               pseudo(s)          real(s)              writable(s)          extra(s)
        ========= ================== ================== ==================== ==================== ===============
        hkl       bissector          h, k, l            omega, chi, phi, tth omega, chi, phi, tth
        hkl       constant_omega     h, k, l            omega, chi, phi, tth chi, phi, tth
        hkl       constant_chi       h, k, l            omega, chi, phi, tth omega, phi, tth
        hkl       constant_phi       h, k, l            omega, chi, phi, tth omega, chi, tth
        hkl       double_diffraction h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
        hkl       psi_constant       h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2, psi
        psi       psi                psi                omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
        q         q                  q                  tth                  tth
        incidence incidence          incidence, azimuth omega, chi, phi                           x, y, z
        emergence emergence          emergence, azimuth omega, chi, phi, tth                      x, y, z
        ========= ================== ================== ==================== ==================== ===============

    .. seealso:: :ref:`geometries.summary_tables`,
        :meth:`hklpy2.backends.base.SolverBase.summary()`
    """
    table = get_diffractometer().core.solver_summary
    if write:
        print(table)
    else:
        return table


def get_diffractometer():
    """
    Return the currently-selected diffractometer (or ``None``).

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> get_diffractometer()
        Hklpy2Diffractometer(
            prefix='',
            name='e4cv',
            settle_time=0.0,
            timeout=None, egu='',
            limits=(0, 0),
            source='computed',
            read_attrs=['h', 'h.readback', 'h.setpoint', 'k', 'k.readback', 'k.setpoint', 'l', 'l.readback', 'l.setpoint', 'omega', 'chi', 'phi', 'tth'],
            configuration_attrs=['geometry', 'solver', 'wavelength', 'h', 'k', 'l'],
            concurrent=True)
    """
    try:
        return _choice.diffractometer
    except ValueError:
        return None


def list_samples(full=False):
    """
    Summarize diffractometer's samples.

    Current sample appears first (with prefix ``"> "``).

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> list_samples()
        > Sample(name='vibranium', lattice=Lattice(a=6.2832, system='cubic'))
        Sample(name='sample', lattice=Lattice(a=1, system='cubic'))

    .. seealso:: :func:`~hklpy2.user.add_sample` :func:`~hklpy2.user.remove_sample`
    """

    def display(sample, preface=""):
        if full:
            print(f"{preface}{sample}")
        else:
            print(f"{preface}{sample!r}")

    diffractometer = _choice.diffractometer
    current_sample = diffractometer.sample

    # always show the default sample first
    display(current_sample, "> ")

    # now, show any other samples
    for sample in diffractometer.samples.values():
        if sample != current_sample:
            preface = "\n" if full else ""
            display(sample, preface)


def or_swap() -> list[list[float]]:
    """
    Swap the first 2 ORienting reflections, re-compute & return new [UB].

    .. note:: The SPEC user community knows this function as ``or_swap``
        (swap the first two orienting reflections).

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> # define 2 reflections
        >>> r400 = setor(4, 0, 0, tth=69.0966, omega=-145.451, chi=0, phi=0, wavelength=1.54)
        >>> r040 = setor(0, 4, 0, tth=69.0966, omega=-145.451, chi=0, phi=90, wavelength=1.54)
        >>> # calculate UB
        >>> calc_UB(r400, r040)
        >>> # swap the two reflections (and recalculate UB)
        >>> or_swap()

    .. seealso:: :func:`~hklpy2.user.setor`
    """
    diffractometer = _choice.diffractometer
    reflections = diffractometer.sample.reflections.swap()[:2]
    return calc_UB(*reflections)


def pa(digits=4):
    """
    Report (all) the diffractometer settings.

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> pa()
        diffractometer='e4cv'
        HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')
        Sample(name='vibranium', lattice=Lattice(a=6.2832, system='cubic'))
        U=[[0.000278604397, -0.99999996119, -3.9081e-08], [1.6307e-08, 3.9086e-08, -1.0], [0.99999996119, 0.000278604397, 1.6317e-08]]
        UB=[[0.000278604432, -0.999999952659, -1.87102e-07], [1.6307e-08, 3.9086e-08, -1.000000171333], [1.000000087947, 0.000279360313, -1.88574e-07]]
        Reflection(name='r400', geometry='E4CV', pseudos={'h': 4, 'k': 0, 'l': 0}, reals={'omega': -145.451, 'chi': 0, 'phi': 0, 'tth': 69.066}, wavelength=1.54, digits=4)
        Reflection(name='r040', geometry='E4CV', pseudos={'h': 0, 'k': 4, 'l': 0}, reals={'omega': -145.451, 'chi': 0, 'phi': 90, 'tth': 69.066}, wavelength=1.54, digits=4)
        Reflection(name='r004', geometry='E4CV', pseudos={'h': 0, 'k': 0, 'l': 4}, reals={'omega': -145.451, 'chi': 90, 'phi': 0, 'tth': 69.066}, wavelength=1.54, digits=4)
        Orienting reflections: ['r040', 'r004']
        constraint: -180.2 <= omega <= 180.2
        constraint: -180.2 <= chi <= 180.2
        constraint: -180.2 <= phi <= 180.2
        constraint: -180.2 <= tth <= 180.2
        h=0, k=0, l=0
        wavelength=1.54
        omega=0, chi=0, phi=0, tth=0

    .. seealso:: :func:`~hklpy2.user.wh`
    """
    _choice.diffractometer.wh(digits=digits, full=True)


def remove_reflection(name: str, error: bool = True) -> None:
    """
    Pop the named reflection and remove it from list of orienting reflections.

    EXAMPLE:

    .. code-block:: python

        >>> remove_reflection("r100")

    PARAMETERS

    name: str
        Reflection name to be removed.
    error: bool
        When ``True`` (default), ``KeyError`` is raised
        if ``name`` is not found.  Provide ``error=False`` to skip the exception.

    .. seealso:: :func:`~hklpy2.user.add_reflection`,
        :func:`~hklpy2.user.or_swap`,
        :func:`~hklpy2.user.setor`
    """
    try:
        get_diffractometer().sample.remove_reflection(name)
    except KeyError as exinfo:
        if error:
            raise exinfo


def remove_sample(name: str, error: bool = True) -> None:
    """
    Pop the named sample, set "selected" sample name to a valid one.

    EXAMPLE:

    .. code-block:: python

        >>> remove_sample("sample")

    PARAMETERS

    name: str
        Sample name to be removed.
    error: bool
        When ``error=True`` (default):

        =============================   =============
        and                             will raise
        =============================   =============
        ``name`` is not found.          ``KeyError``
        ``name`` is the only sample.    :class:`~hklpy2.ops.CoreError`
        =============================   =============

        Provide ``error=False`` to avoid raising an exception.

    .. seealso:: :func:`~hklpy2.user.add_sample` :func:`~hklpy2.user.list_samples`
    """
    try:
        get_diffractometer().core.remove_sample(name)
    except (KeyError, CoreError) as exinfo:
        if error:
            raise exinfo


def scan_extra(
    detectors: Iterable[Readable],
    *args: Union[Movable, Any],  # axis, start, finish, [...]
    num: Optional[int] = 2,
    pseudos: Optional[dict] = None,  # h, k, l
    reals: Optional[dict] = None,  # angles
    extras: Optional[dict] = {},
    fail_on_exception: Optional[bool] = False,
    md: Optional[dict] = None,
):
    """
    Scan diffractometer parameter(s) at constant pseudos or reals.

    Uses selected diffractometer.

    Example

    .. code-block:: python
        :linenos:

        >>> import hklpy2
        >>> from hklpy2.user import *
        >>> e6c = hklpy2.creator(name="e6c", geometry="e6c")
        >>> set_diffractometer(e6c)
        >>> e6c.core.mode = "psi_constant_vertical"
        >>> RE(
            scan_extra(
                [noisy_det, e6c],
                "psi", 0, 150,
                num=15,
                pseudos=dict(h=0, k=0, l=2),
                extras=dict(h2=1, k2=2, l2=0)
            )
        )

    Parameters

    detectors: Iterable[Readable]
        List of readable objects.
    args: Any
        Specification of scan axes.  The specification is a repeating
        pattern of axis (str), start (float), stop (float).

        In general:

        .. code-block:: python

            axis1, start1, stop1,
            axis2, start2, stop2,
            ...,
            axisN, startN, stopN

        Axis is any extra axis name supported by the current diffractometer
        geometry and mode.
    num: int
        Number of points.
    pseudos: dict
        Dictionary of pseudo axes positions to be held constant during the scan.
    reals: dict
        Dictionary of real axes positions to be held constant during the scan.
    extras: dict
        Dictionary of extra axes positions to be held constant during the scan.
    fail_on_exception: bool
        When True (deafult: False), scan will raise any exceptions.
        When False, all exceptions during the scan will be printed to console.
    md: dict
        Dictionary of user-supplied metadata.

    """
    yield from get_diffractometer().scan_extra(
        detectors,
        *args,
        num=num,
        psuedos=pseudos,
        reals=reals,
        extras=extras,
        fail_on_exception=fail_on_exception,
        md=md,
    )


def set_diffractometer(diffractometer: DiffractometerBase = None) -> None:
    """
    Declare the diffractometer to be used.

    EXAMPLE:

    .. code-block:: python

        >>> set_diffractometer(e4cv)

    .. seealso:: :func:`~hklpy2.user.get_diffractometer`
    """
    _choice.diffractometer = diffractometer


def set_wavelength(value: float, units=None):
    """
    Set the wavelength; if Signal has write access, changes control system.

    EXAMPLE:

    .. code-block:: python

        >>> set_wavelength(123.45, units="pm")
    """

    beam = _choice.diffractometer.beam
    if not beam.wavelength.write_access:
        raise TypeError(
            f"'set_wavelength()' not supported for {beam.wavelength.name!r},"
        )
    if units is not None:
        beam.wavelength_units.put(units)
    beam.wavelength.put(value)


def set_lattice(
    a: float,
    b: float = None,
    c: float = None,
    alpha: float = 90,
    beta: float = None,
    gamma: float = None,
    digits: int = 4,
):
    """
    Redefine the sample's lattice.

    EXAMPLE:

    .. code-block:: python

        >>> set_lattice(3, c=4, gamma=120)
    """
    _choice.diffractometer.sample.lattice = Lattice(
        a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        digits=digits,
    )


def setor(
    h,
    k,
    l,  # noqa: E741
    *reals: AnyAxesType,
    wavelength=None,
    name=None,
    **kwreals: AxesDict,
):  # noqa: E741
    """
    Define an ORienting reflection.

    Aliases: :func:`~hklpy2.user.add_reflection`, :func:`~hklpy2.user.setor`

    A reflection is defined by its reciprocal space coordinates (pseudos) and
    its motor positions (reals).  For convenience of the user, each reflection
    is named.

    .. note:: The SPEC user community knows this function as ``setor``.

    EXAMPLES::

        >>> setor(4, 0, 0)
        Reflection(name='r_4ad1', geometry='E4CV', pseudos={'h': 4, 'k': 0, 'l': 0},
            reals={'omega': -145.451, 'chi': 0, 'phi': 0, 'tth': 69.0966}, wavelength=1.54, digits=4)

        >>> setor(0, 4, 0, -145.451, 0, 90, 69.0966, name="r040")
        Reflection(name='r040', geometry='E4CV', pseudos={'h': 0, 'k': 4, 'l': 0},
            reals={'omega': -145.451, 'chi': 0, 'phi': 90, 'tth': 69.0966}, wavelength=1.54, digits=4)

        >>> setor(0, 0, 4, omega=-145.451, chi=90, phi=0, tth=69.0966, name="r004")
        Reflection(name='r004', geometry='E4CV', pseudos={'h': 0, 'k': 0, 'l': 4},
            reals={'omega': -145.451, 'chi': 90, 'phi': 0, 'tth': 69.0966}, wavelength=1.54, digits=4)

    PARAMETERS

    h, k, l: float
        Reciprocal-space coordinates of this reflection.
    reals: AnyAxesType
        (optional)
        Real-space values of this reflection.  Must provide all values in the
        order expected by the geometry.
        See *Positions* tip below.
    kwreals: AxesDict
        (optional)
        Real-space axis names and values of this reflection.  Must provide all
        axes expected by the geometry.
        See *Positions* tip below.
    wavelength: float
        (optional)
        Wavelength of this reflection.
        When not specified, use the current diffractometer value.
    name: str
        (optional)
        Reference text identifying this reflection.
        When not specified, a unique name will be assigned.

    .. tip:: Positions (``reals``, ``kwreals``, **or** omitted entirely):

      * Specified by values (in ``reals``).  Must use expected order.  Will
        skip ``kwreals`` if also provided.
      * Specified by names (in ``kwreals``).  Axes, can appear in any order.
      * ``reals`` and ``kwreals`` can be omitted entirely (use current values
        from diffractometer)

      See the examples above.
    """
    diffractometer = _choice.diffractometer
    if len(reals) > 0:  # Real motor positions as values in expected order.
        # NOTE: Will ignore any kwreals.
        rpos: AxesDict = diffractometer.core.standardize_reals(reals)
    elif len(kwreals) > 0:  # Real motor positions specified as dict, in any order.
        rpos: AxesDict = diffractometer.core.standardize_reals(kwreals)
    else:
        rpos: AxesDict = diffractometer.core.standardize_reals(None)

    # NOTE: hkl_soleil/libhkl gets the wavelength on a reflection from the diffractometer.
    # When the wavelength is set, it calls libhkl directly.
    # as self._hkl_geometry.wavelength_set(wavelength, self._units)
    # The code here uses that procedure.
    if wavelength not in (None, 0):
        diffractometer.beam.wavelength.put(wavelength)

    def make_name():
        while True:
            name = f"r_{str(uuid.uuid4())[:4]}"
            if name not in diffractometer.sample.reflections:
                return name

    name = name or make_name()
    refl = diffractometer.add_reflection((h, k, l), reals=rpos, name=name)
    return refl


def wh(digits=4):
    """
    Report (brief) where is the diffractometer.

    EXAMPLE:

    .. code-block:: python
        :linenos:

        >>> wh()
        h=0, k=0, l=0
        wavelength=1.0
        omega=0, chi=0, phi=0, tth=0

    .. seealso:: :func:`~hklpy2.user.pa`
    """
    _choice.diffractometer.wh(digits=digits, full=False)


add_reflection = setor
