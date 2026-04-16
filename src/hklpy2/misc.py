"""
Miscellaneous Support.

.. rubric: Functions
.. autosummary::

    ~axes_to_dict
    ~check_value_in_list
    ~compare_float_dicts
    ~convert_units
    ~distance_between_pos_tuples
    ~flatten_lists
    ~get_run_orientation
    ~get_solver
    ~istype
    ~list_orientation_runs
    ~load_yaml
    ~load_yaml_file
    ~pick_closest_solution
    ~pick_first_solution
    ~roundoff
    ~creator_from_config
    ~solver_factory
    ~solvers
    ~unique_name
    ~validate_and_canonical_unit
    ~validate_not_parallel

.. note::

    Device construction helpers have moved to :mod:`hklpy2.devices`.

.. rubric: Symbols
.. autosummary::

    ~IDENTITY_MATRIX_3X3
    ~SOLVER_ENTRYPOINT_GROUP

.. rubric: Custom Preprocessors
.. autosummary::

    ~ConfigurationRunWrapper

.. note::

    Exception classes have moved to :mod:`hklpy2.exceptions`.

"""

import logging
import math
import numbers
import pathlib
import sys

import uuid
import warnings
from collections.abc import Iterable
from importlib.metadata import entry_points

from deprecated.sphinx import versionadded
from deprecated.sphinx import versionchanged
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import NamedTuple
from typing import Sequence
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import pint
import tqdm
import yaml
from ophyd import Device

from .exceptions import NoForwardSolutions
from .exceptions import SolverError
from .typing import AnyAxesType
from .typing import AxesArray
from .typing import AxesDict
from .typing import AxesList
from .typing import AxesTuple
from .typing import BlueskyPlanType
from .typing import KeyValueMap
from .typing import Matrix3x3

if TYPE_CHECKING:
    from .backends.base import SolverBase  # noqa: F401

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
    "SOLVER_ENTRYPOINT_GROUP",
    "UREG",
    # Classes
    "ConfigurationRunWrapper",
    # Functions
    "axes_to_dict",
    "check_value_in_list",
    "compare_float_dicts",
    "convert_units",
    "creator_from_config",
    "distance_between_pos_tuples",
    "flatten_lists",
    "get_run_orientation",
    "get_solver",
    "istype",
    "list_orientation_runs",
    "load_yaml",
    "load_yaml_file",
    "pick_closest_solution",
    "pick_first_solution",
    "roundoff",
    "solver_factory",
    "solvers",
    "unique_name",
    "validate_and_canonical_unit",
]

# Constants and Structures

IDENTITY_MATRIX_3X3: Matrix3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"""Identity matrix, 2-D, 3 rows, 3 columns."""

MISSING_HEADER_KEY_MSG: str = "Configuration is missing '_header' key."
"""Error message for missing _header key in configuration dicts."""

SOLVER_ENTRYPOINT_GROUP: str = "hklpy2.solver"
"""Name by which |hklpy2| |solver| classes are grouped."""

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

# Custom preprocessors


class ConfigurationRunWrapper:
    """
    Write configuration of supported device(s) to a bluesky run.

    EXAMPLE::

        crw = ConfigurationRunWrapper(sim4c2)
        RE.preprocessors.append(crw.wrapper)
        RE(bp.rel_scan([noisy], m1, -1.2, 1.2, 11))

    Disable the preprocessor::

        crw.enable = False  # 'True' to enable

    Remove the last preprocessor::

        RE.preprocessors.pop()

    Add another diffractometer::

        crw.devices.append(e4cv)

    .. autosummary::

        ~device_names
        ~devices
        ~enable
        ~known_bases
        ~start_key
        ~validate
        ~wrapper
    """

    devices: Sequence[Device] = []
    """List of devices to be reported."""

    known_bases: Sequence[Device] = []
    """
    Known device base classes.

    Any device (base class) that reports its configuration dictionary in
    the `.read_configuration()` method can be added to this tuple.
    """

    start_key: str = DEFAULT_START_KEY
    """Top-level key in run's metadata dictionary."""

    def __init__(self, *devices, knowns=None) -> None:
        """
        Constructor.

        EXAMPLES::

            ConfigurationRunWrapper(sim4c)
            ConfigurationRunWrapper(e4cv, e6c)

        PARAMETERS

        devices : list
            List of supported objects to be reported.
        knowns : list
            List of base classes that identify supported objects.
            (default: :class:`hklpy2.DiffractometerBase`)
        """
        from .diffract import DiffractometerBase as hklpy2_DiffractometerBase

        self.enable = True
        self.known_bases = knowns or [hklpy2_DiffractometerBase]
        self.validate(devices)
        self.devices = list(devices)

    @property
    def device_names(self) -> list[str]:
        """Return list of configured device names."""
        return [dev.name for dev in self.devices]

    @property
    def enable(self) -> bool:
        """Is it permitted to write device configuration?"""
        return self._enable

    @enable.setter
    def enable(self, state: bool) -> None:
        """Set permit to write configuration."""
        self._enable = state

    def validate(self, devices: Sequence[Device]) -> None:
        """Verify all are recognized objects."""
        for dev in devices:
            if not isinstance(dev, tuple(self.known_bases)):
                raise TypeError(f"{dev} is not a recognized object.")

    def wrapper(self, plan: Iterator):
        """
        Bluesky plan wrapper (preprocessor).

        Writes device(s) configuration to start document metadata.

        Example::

            crw = ConfigurationRunWrapper(e4cv)
            RE.preprocessors.append(crw.wrapper)
        """
        from bluesky import preprocessors as bpp

        if not self._enable or len(self.devices) == 0:
            # Nothing to do here, move on.
            return (yield from plan)

        self.validate(self.devices)

        cfg = {dev.name: dev.configuration for dev in self.devices}

        return (yield from bpp.inject_md_wrapper(plan, {self.start_key: cfg}))


# Functions


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


def get_solver(solver_name: str) -> "SolverBase":
    """
    Load a Solver class from a named entry point.

    ::

        import hklpy2
        SolverClass = hklpy2.get_solver("hkl_soleil")
        libhkl_solver = SolverClass()
    """
    if solver_name not in solvers():
        raise SolverError(f"{solver_name=!r} unknown.  Pick one of: {solvers()!r}")
    logger.debug("Loading solver %r from entry points", solver_name)
    entries = entry_points(group=SOLVER_ENTRYPOINT_GROUP)
    return entries[solver_name].load()


@versionadded(
    version="0.2.3", reason="Retrieve diffractometer orientation from a Tiled run."
)
@versionchanged(version="0.4.0", reason="Exported from top-level ``hklpy2`` namespace.")
def get_run_orientation(
    run: Any,
    name=None,
    start_key: str = DEFAULT_START_KEY,
) -> KeyValueMap:
    """
    Return the orientation information dictionary from a run.

    EXAMPLE::

        In [3]: get_run_orientation(cat[9752], name="sim4c2")
        Out[3]:
        {'_header': {'datetime': '2025-02-27 15:54:33.364719',
        'hklpy2_version': '0.0.26.dev72+gcf9a65a.d20250227',
        'python_class': 'Hklpy2Diffractometer',
        'source_type': 'X-ray',
        'energy_units': 'keV',
        'energy': 12.398419843856837,
        'wavelength_units': 'angstrom',
        'wavelength': 1.0},
        'name': 'sim4c2',
        'axes': {'pseudo_axes': ['h', 'k', 'l'],
        'real_axes': ['omega', 'chi', 'phi', 'tth'],
        'axes_xref': {'h': 'h',
        'k': 'k',
        'l': 'l',
        'omega': 'omega',
        'chi': 'chi',
        'phi': 'phi',
        'tth': 'tth'},
        'extra_axes': {}},
        'sample_name': 'sample',
        'samples': {'sample': {'name': 'sample',
        'lattice': {'a': 1,
            'b': 1,
            'c': 1,
            'alpha': 90.0,
            'beta': 90.0,
            'gamma': 90.0},
        'reflections': {},
        'reflections_order': [],
        'U': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'UB': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'digits': 4}},
        'constraints': {'omega': {'label': 'omega',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'chi': {'label': 'chi',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'phi': {'label': 'phi',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'tth': {'label': 'tth',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'}},
        'solver': {'name': 'hkl_soleil',
        'description': "HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')",
        'geometry': 'E4CV',
        'real_axes': ['omega', 'chi', 'phi', 'tth'],
        'version': '5.1.2',
        'engine': 'hkl'}}


    Parameters
    ----------
    run : object
        Bluesky run object.
    name : str
        (optional)
        Name of the diffractometer. (default=None, returns all available.)
    start_key : str
        Metadata key where the orientation information is stored in the start
        document.  (default="diffractometers")
    """
    info = run.metadata["start"].get(start_key, {})
    if isinstance(name, str):
        info = info.get(name, {})
    return info


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


@versionadded(
    version="0.2.3", reason="List runs that contain diffractometer orientation data."
)
@versionchanged(version="0.4.0", reason="Exported from top-level ``hklpy2`` namespace.")
def list_orientation_runs(
    catalog: Any,
    limit: int = 10,
    start_key: str = DEFAULT_START_KEY,
    **kwargs: Mapping,
) -> pd.DataFrame:
    """
    List the runs with orientation information.

    EXAMPLE::

        In [42]: list_orientation_runs(cat, limit=5, date="_header.datetime")
        Out[42]:
            scan_id      uid  sample diffractometer geometry      solver                        date
        0      9752  41f71e9  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        1      9751  36e38bc  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        2      9750  62e425d  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        3      9749  18b11f0  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:53:55.958929
        4      9748  bf9912f  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:53:55.958929

    Returns
    -------
    Table of orientation runs: Pandas DataFrame object

    Parameters
    ----------
    catalog : object
        Catalog of bluesky runs.
    limit : int
        Limit the list to at most ``limit`` runs. (default=10)
        It could take a long time to search an entire catalog.
    start_key : str
        Metadata key where the orientation information is stored in the start
        document.  (default="diffractometers")
    **kwargs : dict[str:str]
        Keyword parameters describing data column names to be displayed. The
        value of each column name is the dotted path to the orientation
        information (in the start document's metadata).
    """
    buffer = []
    _count = 0
    columns = dict(
        sample="sample_name",
        diffractometer="name",
        geometry="solver.geometry",
        solver="solver.name",
    )
    columns.update(**kwargs)
    try:
        container = catalog.v2  # data broker catalog
    except AttributeError:
        container = catalog  # tiled Container
    limit = min(limit, len(container))
    with tqdm.tqdm(total=limit, file=sys.stdout, leave=False) as progress_bar:
        for full_uid in container:
            _count += 1
            run = container[full_uid]
            start_md = run.metadata.get("start", {})
            info = get_run_orientation(run, start_key=start_key)
            if info is not None:

                def get_subdict_value(biblio, full_key):
                    value = biblio
                    for key in full_key.split("."):
                        value = (value or {}).get(key)
                    return value

                for device in sorted(info):
                    orientation = info[device]
                    row = dict(
                        scan_id=start_md.get("scan_id", 0),
                        uid=full_uid[:7],
                    )
                    for f, addr in columns.items():
                        value = get_subdict_value(orientation, addr)
                        if value is not None:
                            row[f] = value
                    buffer.append(row)

            progress_bar.update()
            if _count >= limit:
                break
    return pd.DataFrame(buffer)


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
        :func:`~hklpy2.misc.pick_first_solution`
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
        :func:`~hklpy2.misc.pick_closest_solution`
    """
    if len(solutions) == 0:
        raise NoForwardSolutions("No solutions.")
    return solutions[0]


def roundoff(value: float, digits=4) -> float:
    """Round a number to specified precision."""
    return round(value, ndigits=digits) or 0  # "-0" becomes "0"


def solver_factory(
    solver_name: str,
    geometry: str,
    **kwargs: Mapping,
) -> "SolverBase":
    """
    Create a |solver| object with geometry and axes.
    """
    logger.debug(
        "Creating solver %r geometry=%r kwargs=%r", solver_name, geometry, kwargs
    )
    solver_class = get_solver(solver_name)
    return solver_class(geometry, **kwargs)


def solvers() -> Mapping[str, "SolverBase"]:
    """
    Dictionary of available Solver classes, mapped by entry point name.

    ::

        import hklpy2
        print(hklpy2.solvers())
    """
    return {ep.name: ep.value for ep in entry_points(group=SOLVER_ENTRYPOINT_GROUP)}


@versionadded(
    version="0.4.0",
    reason="Create a simulated diffractometer from a saved configuration.",
)
def creator_from_config(config: Union[dict, str, pathlib.Path]):
    """
    Create a simulated diffractometer from a saved configuration.

    Parses the configuration for the solver, geometry, and axis names, then
    constructs a simulator (all axes are soft positioners — no hardware
    connection) and restores the full orientation (samples, reflections, UB
    matrix, wavelength, constraints) from the configuration.

    PARAMETERS

    config : dict, str, or pathlib.Path
        Configuration dictionary, or path to a YAML configuration file
        previously saved with ``diffractometer.export()``.

    RETURNS

    DiffractometerBase
        A fully configured simulated diffractometer instance.

    EXAMPLE::

        >>> import hklpy2
        >>> sim = hklpy2.creator_from_config("e4cv-config.yml")
        >>> sim.wh()

    SEE ALSO

        :func:`~hklpy2.diffract.creator` — create a diffractometer from scratch.
    """
    from .diffract import creator

    if isinstance(config, (str, pathlib.Path)):
        logger.debug("creator_from_config: loading from file %r", str(config))
        config = load_yaml_file(config)
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a dict or path to a YAML file. Received: {type(config)!r}"
        )
    if "_header" not in config:
        raise KeyError(MISSING_HEADER_KEY_MSG)

    solver_cfg = config.get("solver", {})
    solver_name = solver_cfg.get("name", "hkl_soleil")
    geometry = solver_cfg.get("geometry", "E4CV")

    solver_kwargs: dict = {}
    engine = solver_cfg.get("engine")
    if engine is not None:
        solver_kwargs["engine"] = engine

    axes_cfg = config.get("axes", {})
    axes_xref = axes_cfg.get("axes_xref", {})
    pseudo_axes = axes_cfg.get("pseudo_axes", [])
    real_axes = [
        ax for ax in axes_cfg.get("real_axes", []) if ax not in set(pseudo_axes)
    ]

    # Sort diffractometer real axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name) and
    # solver.real_axes (solver canonical order).
    solver_real_order = solver_cfg.get("real_axes", [])
    if solver_real_order:
        solver_to_diff_real = {v: k for k, v in axes_xref.items() if k in real_axes}
        real_axes = [
            solver_to_diff_real[s]
            for s in solver_real_order
            if s in solver_to_diff_real
        ]

    # Sort diffractometer pseudo axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name).
    # The solver canonical pseudo order is derived from the xref values for pseudos.
    pseudo_set = set(pseudo_axes)
    solver_to_diff_pseudo = {v: k for k, v in axes_xref.items() if k in pseudo_set}
    # Preserve the solver-canonical order already encoded in axes_xref values;
    # fall back to the order in axes.pseudo_axes if no xref is available.
    pseudo_solver_order = [axes_xref.get(p, p) for p in pseudo_axes]
    pseudo_axes_ordered = [
        solver_to_diff_pseudo[s]
        for s in pseudo_solver_order
        if s in solver_to_diff_pseudo
    ] or pseudo_axes

    reals = {name: None for name in real_axes}

    # Pass _real and _pseudo so creator() maps axes in solver-expected order
    # even when diffractometer names differ from solver canonical names.
    diffractometer_name = config.get("name", geometry.lower())

    logger.debug(
        "creator_from_config: creating %r solver=%r geometry=%r",
        diffractometer_name,
        solver_name,
        geometry,
    )
    sim = creator(
        name=diffractometer_name,
        solver=solver_name,
        geometry=geometry,
        solver_kwargs=solver_kwargs,
        reals=reals,
        _real=real_axes if real_axes else None,
        _pseudo=pseudo_axes_ordered if pseudo_axes_ordered else None,
    )

    sim.restore(config)
    return sim


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
