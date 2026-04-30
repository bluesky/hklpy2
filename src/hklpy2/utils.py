"""
General-purpose utilities for |hklpy2|.

.. rubric:: Classes
.. autosummary::

    ~_SolverDirty

.. rubric:: Functions
.. autosummary::

    ~axes_to_dict
    ~benchmark
    ~check_value_in_list
    ~compare_float_dicts
    ~convert_units
    ~distance_between_pos_tuples
    ~flatten_lists
    ~istype
    ~load_yaml
    ~load_yaml_file
    ~pick_closest_solution
    ~pick_first_solution
    ~roundoff
    ~unique_name
    ~validate_and_canonical_unit
    ~validate_not_parallel

.. rubric:: Symbols
.. autosummary::

    ~DEFAULT_DIGITS
    ~DEFAULT_MOTOR_LABELS
    ~DEFAULT_START_KEY
    ~IDENTITY_MATRIX_3X3
    ~INTERNAL_ANGLE_UNITS
    ~INTERNAL_LENGTH_UNITS
    ~INTERNAL_XRAY_ENERGY_UNITS
    ~MISSING_HEADER_KEY_MSG
    ~PINT_ERRORS
    ~UREG
"""

import logging
import math
import numbers
import pathlib
import time
import uuid
import warnings
from collections.abc import Iterable
from enum import IntFlag
from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping
from typing import NamedTuple
from typing import Sequence
from typing import Type
from typing import Union

import numpy as np
import pint
import yaml
from deprecated.sphinx import versionadded
from deprecated.sphinx import versionchanged

from .exceptions import NoForwardSolutions
from .typing import AnyAxesType
from .typing import AxesArray
from .typing import AxesDict
from .typing import AxesList
from .typing import AxesTuple
from .typing import BlueskyPlanType
from .typing import Matrix3x3

if TYPE_CHECKING:
    from .backends.base import SolverBase  # noqa: F401
    from .diffract import DiffractometerBase  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    # Constants
    "DEFAULT_DIGITS",
    "DEFAULT_MOTOR_LABELS",
    "DEFAULT_START_KEY",
    "IDENTITY_MATRIX_3X3",
    "INTERNAL_ANGLE_UNITS",
    "INTERNAL_LENGTH_UNITS",
    "INTERNAL_XRAY_ENERGY_UNITS",
    "MISSING_HEADER_KEY_MSG",
    "PINT_ERRORS",
    "UREG",
    # Classes
    "_SolverDirty",
    # Functions
    "axes_to_dict",
    "benchmark",
    "check_value_in_list",
    "compare_float_dicts",
    "convert_units",
    "distance_between_pos_tuples",
    "flatten_lists",
    "istype",
    "load_yaml",
    "load_yaml_file",
    "pick_closest_solution",
    "pick_first_solution",
    "roundoff",
    "solver_summary",
    "unique_name",
    "validate_and_canonical_unit",
    "validate_not_parallel",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTITY_MATRIX_3X3: Matrix3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"""Identity matrix, 2-D, 3 rows, 3 columns."""

MISSING_HEADER_KEY_MSG: str = "Configuration is missing '_header' key."
"""Error message for missing _header key in configuration dicts."""

DEFAULT_DIGITS: int = 4
DEFAULT_START_KEY: str = "diffractometers"

INTERNAL_ANGLE_UNITS: str = "degrees"
INTERNAL_LENGTH_UNITS: str = "angstrom"
INTERNAL_XRAY_ENERGY_UNITS: str = "keV"

# Shared pint UnitRegistry to avoid recreating it repeatedly.
# Agents and hot paths should use this registry via helper functions below.
UREG = pint.UnitRegistry()

PINT_ERRORS = (pint.DimensionalityError, pint.UndefinedUnitError)
"""Exception from pint that we are trapping here."""

DEFAULT_MOTOR_LABELS: Sequence[str] = ["motors"]
"""Default labels applied to real-axis positioners."""

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


@versionadded(
    version="0.6.2",
    reason="Fine-grained dirty-domain tracking for the solver state.",
)
class _SolverDirty(IntFlag):
    """
    Bitfield identifying which solver state domains need to be re-pushed.

    Used by :class:`~hklpy2.ops.Core` to track, at fine granularity, which
    parts of the underlying solver's state have diverged from the canonical
    Python-side state and therefore must be re-pushed before the next
    ``forward()`` / ``inverse()`` call.

    Members:

    * ``SAMPLE`` -- the solver's sample state (lattice + reflections)
      needs a full re-push.  A SAMPLE re-push may also invalidate the
      solver's MODE / EXTRAS / U / UB state in some backends, so those
      domains are re-pushed implicitly whenever SAMPLE is re-pushed.
    * ``UB`` -- only the U and UB matrices need re-pushing.
    * ``MODE`` -- the solver mode needs to be re-set.  A MODE change may
      reset the solver's extra-axis values, so EXTRAS must be flagged
      whenever MODE is flagged.
    * ``EXTRAS`` -- the solver's extra-axis values need to be re-pushed.
    * ``WAVELENGTH`` -- the wavelength needs to be re-pushed.
    * ``ALL`` -- every domain (used for full re-sync).

    .. seealso:: :issue:`384`, :issue:`386`
    """

    SAMPLE = 1
    UB = 2
    MODE = 4
    EXTRAS = 8
    WAVELENGTH = 16
    ALL = SAMPLE | UB | MODE | EXTRAS | WAVELENGTH


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def axes_to_dict(input: AnyAxesType, names: list[str]) -> AxesDict:
    """
    Convert any acceptable axes input to standard form (dict).

    User could provide input in several forms:

    * dict: ``{"h": 0, "k": 1, "l": -1}``
    * namedtuple: ``(h=0.0, k=1.0, l=-1.0)``
    * ordered list: ``[0, 1, -1]  (for h, k, l)``
    * ordered tuple: ``(0, 1, -1)  (for h, k, l)``

    PARAMETERS:

    input : AnyAxesType
        Positions, specified as dict, list, or tuple.
    names : [str]
        Expected names of the axes, in order expected by the solver.
    """
    if not isinstance(names, list):
        raise TypeError(f"Expected a list of names, received {names=!r}")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"Each name should be text, received {name=!r}")
    if len(input) < len(names):
        raise ValueError(
            f"Expected at least {len(names)} axes,"
            # Always show received
            f" received {len(input)}."
        )
    if len(input) > len(names):
        warnings.warn(
            UserWarning(
                f" Extra inputs will be ignored. Expected {len(names)}."
                #
                f" Received {input=!r}, {names=!r}"
            )
        )

    axes = {}
    if istype(input, AxesDict):  # convert dict to ordered dict
        for name in names:
            value = input.get(name)
            if value is None:
                raise KeyError(
                    f"Missing axis {name!r}."
                    # Always show received
                    f" Received: {input=!r}"
                    # then
                    f" Expected: {names=!r}"
                )
            axes[name] = value

    elif istype(input, Union[AxesList, AxesTuple]):  # convert to ordered dict
        for name, value in zip(names, input):
            axes[name] = value

    elif istype(input, AxesArray) or isinstance(input, np.ndarray):
        # Accept numpy arrays (ndarray) of numeric values as an AxesArray.
        for name, value in zip(names, input):
            axes[name] = value

    else:
        raise TypeError(f"Unexpected type: {input!r}.  Expected 'AnyAxesType'.")

    for name, value in axes.items():
        # Accept Python ints/floats and numpy numeric scalar types (e.g. np.int64,
        # np.float64) by checking against numbers.Real.
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Expected a number. Received: {value!r}.")

    return axes


def check_value_in_list(title, value, examples, blank_ok=False) -> None:
    """Raise ValueError exception if value is not in the list of examples."""
    if blank_ok:
        examples.append("")
    if value not in examples:
        msg = f"{title} {value!r} unknown. Pick one of: {examples!r}"
        raise ValueError(msg)


def compare_float_dicts(a1, a2, tol=1e-4) -> bool:
    """
    Compare two dictionaries.  Values are all floats.
    """
    if tol <= 0:
        raise ValueError(f"Received {tol=}, should be tol >0.")

    if sorted(a1.keys()) != sorted(a2.keys()):
        return False

    tests = [True]
    for k, v in a1.items():
        if isinstance(v, float):
            if tol < 1:
                test = math.isclose(a1[k], a2[k], abs_tol=tol)
            else:
                test = round(a1[k], tol) == round(a2[k], tol)
        else:
            test = a1[k] == a2[k]
        if not test:
            return False  # no need to go further
    return False not in tests


def convert_units(value: float, old_units: str, new_units: str) -> float:
    """Convert 'value' from old units to new."""
    if old_units == new_units:
        return value
    return UREG.Quantity(value, old_units).to(new_units).magnitude


def distance_between_pos_tuples(pos1: NamedTuple, pos2: NamedTuple) -> float:
    """Return the RMS distance between 'pos1' and 'pos2'."""
    if len(pos1) != len(pos2):
        raise AttributeError(f"{pos1=} and {pos2=} are not the same length.")
    if len(pos1) == 0:
        rms = 0
    else:
        sum = 0
        for axis in pos1._fields:
            delta = getattr(pos1, axis) - getattr(pos2, axis)
            sum += delta * delta
        rms = math.sqrt(sum / len(pos1._fields))
    return rms


def flatten_lists(
    xs: Sequence[Union[bytes, Iterable, str]],
) -> BlueskyPlanType:
    """
    Convert nested lists into single list.

    https://stackoverflow.com/questions/2158395
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_lists(x)
        else:
            yield x


def istype(value: Any, annotation: Type) -> bool:
    """
    Check if 'value' matches the type 'annotation'.

    EXAMPLE::

        >>> istype({"a":1}, AxesDict)
        True
    """
    # https://stackoverflow.com/a/57813576/1046449
    from typeguard import TypeCheckError
    from typeguard import check_type

    try:
        check_type(value, annotation)
        return True
    except TypeCheckError:
        return False


def load_yaml(text: str) -> Mapping:
    """Load YAML from text."""
    return yaml.load(text, yaml.Loader)


def load_yaml_file(file: Union[pathlib.Path, str]) -> Mapping:
    """Return contents of a YAML file as a Python object."""
    path = pathlib.Path(file)
    if not path.exists():
        raise FileExistsError(f"YAML file '{path}' does not exist.")
    logger.debug("Loading YAML file %r", str(path))
    with open(path, "r") as f:
        return load_yaml(f.read())


@versionadded(
    version="0.1.4",
    reason="Alternative forward() solution picker using closest motor positions.",
)
def pick_closest_solution(
    position: NamedTuple,
    solutions: list[NamedTuple],
) -> NamedTuple:
    """
    Find the solution closest to the current real position.

    Used by :meth:`~hklpy2.diffract.DiffractometerBase.forward()` method to pick
    a solution from a list of possible solutions.  Assign to diffractometer's
    :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution` method.

    PARAMETERS

    position tuple :
        Current position.
    solutions list[tuple] :
        List of positions.

    .. seealso::
        :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution`,
        :func:`~hklpy2.utils.pick_first_solution`
    """
    if len(solutions) == 0:
        raise NoForwardSolutions("No solutions.")

    nearest = None
    separation = None
    for candidate in solutions:
        rms = distance_between_pos_tuples(position, candidate)
        if separation is None or rms < separation:
            separation = rms
            nearest = candidate
    return nearest


def pick_first_solution(
    position: NamedTuple,
    solutions: list[NamedTuple],
) -> NamedTuple:
    """
    Choose first solution from list.

    Used by :meth:`~hklpy2.diffract.DiffractometerBase.forward()` method to pick
    a solution from a list of possible solutions.  Assign to diffractometer's
    :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution` method.

    PARAMETERS

    position tuple :
        Current position.  (Required for general case, not used here.)
    solutions list[tuple] :
        List of positions.

    .. seealso::
        :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution`,
        :func:`~hklpy2.utils.pick_closest_solution`
    """
    if len(solutions) == 0:
        raise NoForwardSolutions("No solutions.")
    return solutions[0]


@versionchanged(
    version="0.6.2",
    reason=(
        "Accept structured inputs (namedtuple/Mapping/Sequence) and"
        " recursively round their numeric leaves; opaque inputs fall"
        " through to ``repr(value)``.  Scalar behavior is unchanged."
    ),
)
def roundoff(value, digits=4):
    """Round ``value`` to ``digits`` precision (fail-safe).

    The historical scalar contract is preserved: a scalar input
    returns a ``float`` (with ``-0`` collapsed to ``0``).

    Structured inputs are handled recursively and never raise:

    * ``namedtuple`` (has ``_fields`` and ``_asdict``) -> ``dict``
      mapping each field name to its recursively-rounded value.
    * :class:`~typing.Mapping` -> ``dict`` with the same keys and
      recursively-rounded values.
    * :class:`~typing.Sequence` (excluding ``str``/``bytes``) ->
      ``list`` of recursively-rounded values.
    * Anything else (including ``str``, ``bytes``, and objects without
      ``__round__``) -> ``repr(value)``.

    This keeps callers like :meth:`~hklpy2.diffract.DiffractometerBase.wh`
    safe when an auxiliary component reports a structured position
    (e.g. a nested ``ophyd.PseudoPositioner`` whose ``.position`` is a
    namedtuple).  See issue :issue:`385`.
    """
    # namedtuple: detect via duck-typing (_fields + _asdict) and
    # check before the generic Mapping/Sequence branches because a
    # namedtuple is also a Sequence.
    if hasattr(value, "_fields") and hasattr(value, "_asdict"):
        return {k: roundoff(v, digits) for k, v in value._asdict().items()}
    if isinstance(value, Mapping):
        return {k: roundoff(v, digits) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return repr(value)
    if isinstance(value, Sequence):
        return [roundoff(v, digits) for v in value]
    try:
        return round(value, ndigits=digits) or 0  # "-0" becomes "0"
    except TypeError:
        return repr(value)


def unique_name(prefix: str = "", length: int = 7) -> str:
    """
    Short, unique name, first 7 (at most) characters of a unique, random uuid.
    """
    return prefix + str(uuid.uuid4())[: max(1, min(length, 7))]


def validate_not_parallel(
    hkl: Sequence[float], hkl2: Sequence[float], tol: float = 1e-6
) -> None:
    """Raise ValueError if two vectors are parallel or anti-parallel.

    Parameters
    ----------
    hkl : sequence of float
        First vector.
    hkl2 : sequence of float
        Second vector.
    tol : float
        Tolerance on the cross-product magnitude below which the vectors are
        considered parallel.

    Raises
    ------
    ValueError
        If the cross product of *hkl* and *hkl2* has magnitude less than *tol*.
    """
    v1 = np.asarray(hkl, dtype=float)
    v2 = np.asarray(hkl2, dtype=float)
    cross = np.cross(v1, v2)
    if np.linalg.norm(cross) < tol:
        raise ValueError(
            f"hkl={tuple(hkl)} and hkl2={tuple(hkl2)} are parallel "
            "(or anti-parallel); psi is undefined. "
            "Choose a reference reflection that is not parallel to the scan reflection."
        )


def validate_and_canonical_unit(value: str, target_units: str) -> str:
    """Validate that *value* is a unit convertible to *target_units*.

    Returns a canonical string representation of the unit (via UREG).
    Raises ValueError on failure.
    """
    # Constructing the Unit will raise pint.UndefinedUnitError if unknown.
    unit = UREG.Unit(value)
    # Attempt a dimensional conversion; will raise pint.DimensionalityError if incompatible.
    UREG.Quantity(1, unit).to(target_units)
    # On success, preserve and return the original user-provided unit string so callers
    # (and tests) see the same spelling/casing that was provided.
    return value


def benchmark(
    diffractometer: "DiffractometerBase",
    n: int = 500,
    print: bool = True,
    snapshot: bool = True,
):
    """
    Assess ``forward()`` and ``inverse()`` throughput for a diffractometer.

    This function is **purely computational** — it does not move any motors or
    communicate with hardware.  It is safe to call on a live diffractometer.

    Uses the diffractometer's current real and pseudo-axis positions, and the
    current mode, as the benchmark inputs.

    Parameters
    ----------
    diffractometer :
        Any |hklpy2| diffractometer instance.
    n : int, optional
        Number of calls used to measure throughput.  Default: 500.
    print : bool, optional
        When ``True`` (default), print a human-readable report to stdout and
        return ``None``.  When ``False``, suppress all output and return a
        ``dict`` of results.
    snapshot : bool, optional
        When ``True`` (default), snapshot the diffractometer configuration
        and run all timing loops on a simulator built from that snapshot,
        leaving the original diffractometer (and its solver state)
        completely untouched.  When ``False``, run the timing loops directly
        on the supplied diffractometer (legacy behaviour).

    Returns
    -------
    None or dict
        ``None`` when *print* is ``True``; a ``dict`` when *print* is
        ``False``.  The dict contains:

        - ``"solver"`` — solver name
        - ``"geometry"`` — geometry name
        - ``"mode"`` — current solver mode
        - ``"wavelength"`` — current wavelength
        - ``"n"`` — number of calls measured
        - ``"forward_ops_per_sec"`` — ``forward()`` throughput
        - ``"forward_ms_per_call"`` — ``forward()`` latency in ms
        - ``"inverse_ops_per_sec"`` — ``inverse()`` throughput
        - ``"inverse_ms_per_call"`` — ``inverse()`` latency in ms
        - ``"fwd_inv_ratio"`` — ratio of forward to inverse ops/sec
        - ``"target_ops_per_sec"`` — minimum target (2,000)

    Example
    -------
    Print a report::

        from hklpy2.utils import benchmark
        benchmark(my_diffractometer)

    Capture results programmatically::

        from hklpy2.utils import benchmark
        results = benchmark(my_diffractometer, print=False)
        print(results["forward_ops_per_sec"])
    """
    TARGET = 2_000

    # Per #369: by default, snapshot the diffractometer configuration and
    # run all timing loops on a simulator built from that snapshot.  This
    # eliminates any risk of side effects on motor positions, sample state,
    # or solver state during the benchmark — especially important when
    # called on a live instrument.
    if snapshot:
        from .run_utils import simulator_from_config

        target = simulator_from_config(diffractometer)
    else:
        target = diffractometer

    # Use the first reflection's positions if available, as the origin (0,0,0)
    # has no forward() solution. Fall back to current position otherwise.
    reflections = list(target.sample.reflections.values())
    if reflections:
        pseudos = reflections[0].pseudos
        reals = reflections[0].reals
    else:
        pseudos = target.position._asdict()
        reals = target.real_position._asdict()
    solver = target.core.solver
    mode = solver.mode
    geometry = solver.geometry
    solver_name = solver.name
    wavelength = target.beam.wavelength.get()

    # forward() benchmark
    t0 = time.perf_counter()
    for _ in range(n):
        target.forward(**pseudos)
    fwd_elapsed = time.perf_counter() - t0
    fwd_ops = n / fwd_elapsed
    fwd_ms = fwd_elapsed * 1000 / n

    # inverse() benchmark
    t0 = time.perf_counter()
    for _ in range(n):
        target.inverse(**reals)
    inv_elapsed = time.perf_counter() - t0
    inv_ops = n / inv_elapsed
    inv_ms = inv_elapsed * 1000 / n

    fwd_inv_ratio = fwd_ops / inv_ops if inv_ops > 0 else float("inf")

    results = {
        "solver": solver_name,
        "geometry": geometry,
        "mode": mode,
        "wavelength": wavelength,
        "n": n,
        "forward_ops_per_sec": fwd_ops,
        "forward_ms_per_call": fwd_ms,
        "inverse_ops_per_sec": inv_ops,
        "inverse_ms_per_call": inv_ms,
        "fwd_inv_ratio": fwd_inv_ratio,
        "target_ops_per_sec": TARGET,
    }

    if not print:
        return results

    import builtins

    tolerance = 0.10
    threshold = TARGET * (1 - tolerance)

    def _status(ops):
        return "PASS" if ops >= threshold else "FAIL"

    builtins.print(
        f"Diffractometer benchmark"
        f"\n  solver:     {solver_name}"
        f"\n  geometry:   {geometry}"
        f"\n  mode:       {mode}"
        f"\n  wavelength: {wavelength}"
        f"\n  calls:      {n}"
        f"\n  snapshot:   {snapshot}"
        f"\n"
        f"\n  {'operation':<12} {'ops/sec':>10} {'ms/call':>11} {'fwd/inv':>9} {'status':>6}"  # noqa: E501
        f"\n  {'-' * 12} {'-' * 10} {'-' * 11} {'-' * 9} {'-' * 6}"
        f"\n  {'forward()':<12} {fwd_ops:>10.0f} {fwd_ms:>11.3f} {fwd_inv_ratio:>9.3f} {_status(fwd_ops):>6}"  # noqa: E501
        f"\n  {'inverse()':<12} {inv_ops:>10.0f} {inv_ms:>11.3f} {'':>9} {_status(inv_ops):>6}"  # noqa: E501
        f"\n"
        f"\n  target: {TARGET:,} ops/sec (+/-{tolerance:.0%})"
    )
    return None


@versionadded(
    version="0.6.1",
    reason="Solver summary helper that accepts any diffractometer instance.",
)
def solver_summary(
    diffractometer: "DiffractometerBase | None" = None,
    write: bool = True,
):
    """
    Table of the diffractometer solver's modes, axes, ...

    PARAMETERS

    diffractometer : DiffractometerBase, optional
        The diffractometer to summarize.  When ``None`` (default), the
        currently-selected diffractometer (set via
        :func:`~hklpy2.user.set_diffractometer`) is used.
    write : bool, optional
        When ``True`` (default), print the table and return ``None``.
        When ``False``, return the :class:`~pyRestTable.Table` object.

    RETURNS

    pyRestTable.Table or None

    EXAMPLE::

        >>> import hklpy2
        >>> diffract = hklpy2.creator(name="e4cv")
        >>> hklpy2.utils.solver_summary(diffract)

    SEE ALSO

        :func:`hklpy2.user.solver_summary`
    """
    if diffractometer is None:
        # Lazy import to avoid circularity (user imports from utils).
        from .user import get_diffractometer

        diffractometer = get_diffractometer()
        if diffractometer is None:
            raise ValueError(
                "No diffractometer selected and none provided."
                " Pass a diffractometer instance, or call"
                " 'set_diffractometer(diffr)' first."
            )

    table = diffractometer.core.solver_summary
    if write:
        print(table)
        return None
    return table
