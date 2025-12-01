"""
Miscellaneous Support.

.. rubric: Base classes
.. autosummary::

    ~VirtualPositionerBase

.. rubric: Functions
.. autosummary::

    ~axes_to_dict
    ~check_value_in_list
    ~compare_float_dicts
    ~convert_units
    ~define_real_axis
    ~dict_device_factory
    ~distance_between_pos_tuples
    ~dynamic_import
    ~flatten_lists
    ~get_run_orientation
    ~get_solver
    ~istype
    ~list_orientation_runs
    ~load_yaml
    ~load_yaml_file
    ~make_component
    ~make_dynamic_instance
    ~parse_factory_axes
    ~pick_closest_solution
    ~pick_first_solution
    ~roundoff
    ~solver_factory
    ~solvers
    ~unique_name
    ~validate_and_canonical_unit

.. rubric: Symbols
.. autosummary::

    ~IDENTITY_MATRIX_3X3
    ~SOLVER_ENTRYPOINT_GROUP

.. rubric: Custom Data Types
.. autosummary::

    ~AnyAxesType
    ~AxesArray
    ~AxesDict
    ~AxesList
    ~AxesTuple
    ~Matrix3x3
    ~NamedFloatDict
    ~NUMERIC

.. rubric: Custom Preprocessors
.. autosummary::

    ~ConfigurationRunWrapper

.. rubric: Custom Exceptions
.. autosummary::

    ~Hklpy2Error
    ~ConfigurationError
    ~ConstraintsError
    ~CoreError
    ~DiffractometerError
    ~LatticeError
    ~NoForwardSolutions
    ~ReflectionError
    ~SampleError
    ~SolverError
"""

import logging
import math
import numbers
import pathlib
import sys
import time
import uuid
import warnings
from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NamedTuple, Protocol
from typing import Sequence
from typing import Type
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import tqdm
import yaml
from bluesky.utils import Msg
from ophyd import Component
from ophyd import Device
from ophyd import EpicsMotor
from ophyd import PVPositioner
from ophyd import SoftPositioner

logger = logging.getLogger(__name__)

# Custom data types

if TYPE_CHECKING:
    from .backends.base import SolverBase

BlueskyPlanType = Iterator[Sequence[Msg]]
"""Type of a bluesky plan."""

KeyValueMap = Mapping[str, Any]
"""Dictionary for configuration and other."""

NUMERIC = Union[float, int]
"""Either integer or real number."""

AxesArray = npt.NDArray[np.floating]
"""Numpy array of axes values."""

AxesDict = dict[str, NUMERIC]
"""Dictionary of axes names and values."""

AxesList = list[NUMERIC]
"""List of axes values."""

AxesTuple = tuple[NUMERIC, ...]
"""Tuple of axes values."""

AnyAxesType = Union[AxesArray, AxesDict, AxesList, AxesTuple]
"""
Any of these types are used to describe both pseudo and real axes.

=============   =========================   ====================
description     example                     type annotation
=============   =========================   ====================
dict            {"h": 0, "k": 1, "l": -1}   AxesDict
namedtuple      (h=0.0, k=1.0, l=-1.0)      AxesTuple
numpy array     numpy.array([0, 1, -1])     AxesArray
ordered list    [0, 1, -1]                  AxesList
ordered tuple   (0, 1, -1)                  AxesTuple
=============   =========================   ====================
"""

Matrix3x3 = List[List[float]]
"""Python type annotation: mutable orientation & rotation matrices."""

NamedFloatDict = Mapping[str, NUMERIC]
"""Python type annotation: dictionary of named floats."""

# Constants and Structures

IDENTITY_MATRIX_3X3: Matrix3x3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"""Identity matrix, 2-D, 3 rows, 3 columns."""

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

# Custom exceptions


class Hklpy2Error(Exception):
    """Any exception from the |hklpy2| package."""


class ConfigurationError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.configure`."""


class ConstraintsError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.constraints`."""


class CoreError(Hklpy2Error):
    """Custom exceptions from :class:`hklpy2.ops.Core`."""


class DiffractometerError(Hklpy2Error):
    """Custom exceptions from :class:`hklpy2.diffract.DiffractometerBase`."""


class LatticeError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.lattice`."""


class NoForwardSolutions(Hklpy2Error):
    """A solver did not find any ``forward()`` solutions."""


class ReflectionError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.reflection`."""


class SampleError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.sample`."""


class SolverError(Hklpy2Error):
    """Custom exceptions from a |solver|."""


# Virtual positioner base class


class VirtualPositionerBase(SoftPositioner):
    """
    Base class for a diffractometer's virtual axis.

    This base class also serves as an example where the virtual axis is twice
    the value of the physical axis.  It is used as a Component of a
    'DiffractometerBase' definition.  The ``physical_name`` is the name of a
    sibling positioner attribute.
    """

    def __init__(self, *, physical_name: str = "", **kwargs) -> None:
        """Constructor.

        Subclass should override and add any additional kwargs, as needed.
        """
        if len(physical_name) == 0:
            raise ValueError("Must provide a value for 'physical_name'.")

        self._setup_finished: bool = False

        super().__init__(**kwargs)

        self.physical = getattr(self.parent, physical_name)

    def _setup_move(self, position: float, status: Any) -> None:
        """Move requested to position."""
        self._run_subs(sub_type=self.SUB_START, timestamp=time.time())

        self._started_moving = True
        self._moving = False

        if self._setup_finished:
            args = [self.inverse(position)]
            if not isinstance(self.physical, PVPositioner):
                args.append(status)
            self.physical._setup_move(*args)

        self._set_position(position)
        self._done_moving()

    def forward(self, physical: float) -> float:
        """
        Return virtual position from physical position.

        Subclass should override.
        """
        return 2 * physical  # Subclass should redefine.

    def inverse(self, virtual: float) -> float:
        """
        Return physical position from virtual position.

        Subclass should override.
        """
        return virtual / 2  # Subclass should redefine.

    def _cb_update_position(self, value: float, **kwargs) -> None:
        """Called when physical position is changed."""
        self._position = self.forward(value)

        # Update our position in diffractometer's internal cache.
        self.parent._real_cur_pos[self] = self._position

    def _finish_setup(self) -> None:
        """
        Complete the axis setup after diffractometer is built.

        This method is crucial for ensuring that the positioner is correctly
        initialized and ready to operate within the system, handling updates and
        constraints appropriately.

        Update our:

        * Position by subscription to readback changes.
        * Limits from physical axis.
        """
        try:
            physical = self.physical
        except AttributeError:
            # During initialization when 'self.physical'  isn't yet set up.
            return

        # Readback signal is in different locations.
        if isinstance(physical, SoftPositioner):
            # Includes PseudoPositioner subclass
            readback = physical
        elif isinstance(physical, EpicsMotor):
            readback = physical.user_readback
        elif isinstance(physical, PVPositioner):
            readback = physical.readback
        else:
            raise TypeError(f"Unknown 'readback' for {physical.name!r}.")

        if not self.parent.connected or self._setup_finished:
            return

        self._setup_finished = True
        # Call 'self._cb_update_position' when readback updates.
        readback.subscribe(self._cb_update_position)
        self._recompute_limits()

    def _recompute_limits(self) -> None:
        """Compute virtual axis limits from physical axis and refine constraints."""
        if self.parent.connected:
            self._limits = tuple(sorted(map(self.forward, self.physical.limits)))
            # Adjust constraints, only as needed.
            lo, hi = self.parent.core.constraints[self.attr_name].limits
            lo = max(lo, self._limits[0])
            hi = min(hi, self._limits[-1])
            self.parent.core.constraints[self.attr_name].limits = (lo, hi)

    def __getattribute__(self, name: str):
        """
        Run final setup automatically, on conditions.

        This is a special method in Python that is called
        whenever an attribute is accessed on an object. This method is
        overridden here to add custom behavior when accessing attributes,
        particularly the 'position' attribute.

        This implementation ensures that the setup process is completed before
        accessing the 'position' attribute, provided the object and its parent are
        connected. It adds robustness to the attribute access by handling
        potential errors gracefully and avoiding infinite recursion.

        This virtual positioner must subscribe to position updates of the
        physical positioner to which it is related.  Because that positioner
        might not be fully initialized and connected during construction of this
        virtual positioner, a final setup method must be called later.  The
        additional steps in this method ensure that final setup is called under
        the correct conditions.
        """

        if name == "position":  # Caution here to avoid recursion.
            if not self._setup_finished and self.connected:
                try:
                    if self.parent.connected:
                        # Run the final setup.
                        self._finish_setup()
                except (AttributeError, RecursionError):
                    pass  # Ignore, still not ready.

        # Return the actual attribute
        return object.__getattribute__(self, name)


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
        raise ValueError(f"received {tol=}, should be tol >0")

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


def define_real_axis(
    specs: Union[None, str, KeyValueMap],
    kwargs: KeyValueMap,
) -> tuple[str, Mapping]:
    """Return class and kwargs of a real axis from its 'specs'."""
    kwargs["labels"] += DEFAULT_MOTOR_LABELS

    if specs is None:
        class_name = "ophyd.SoftPositioner"
        kwargs.update({"limits": (-180, 180), "init_pos": 0})
    elif isinstance(specs, str):
        class_name = "ophyd.EpicsMotor"
        kwargs.update({"prefix": specs})
    elif isinstance(specs, dict):
        class_name = specs.pop("class", None)
        if class_name is None:
            raise KeyError("Expected 'class' key, received None")
        for label in specs.pop("labels", []):
            if label not in kwargs["labels"]:
                kwargs["labels"].append(label)
        kwargs.update(specs)
    else:
        raise TypeError(
            f"Incorrect type '{type(specs).__name__}' for {specs=!r}."
            #
            " Expected 'None', a PV name (str), or a dictionary specifying"
            " a custom configuration."
        )

    return class_name, kwargs


def dict_device_factory(data: KeyValueMap, **kwargs: KeyValueMap) -> Protocol:
    """
    Create a ``DictionaryDevice()`` class using the supplied dictionary.

    .. index:: factory; dict_device_factory, dict_device_factory
    """
    from ophyd import Signal

    component_dict = {
        k: Component(Signal, value=v, **kwargs)
        # metadata={"description": "solver extra axis"},
        # kind="hinted",
        for k, v in data.items()
    }
    fc = type("DictionaryDevice", (Device,), component_dict)
    return fc


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


def dynamic_import(full_path: str) -> type:
    """
    Import the object given its import path as text.

    Motivated by specification of class names for plugins
    when using ``apstools.devices.ad_creator()``.

    EXAMPLES::

        klass = dynamic_import("ophyd.EpicsMotor")
        m1 = klass("gp:m1", name="m1")

        creator = dynamic_import("hklpy2.diffract.creator")
        fourc = creator(name="fourc")

    From the `apstools <https://github.com/BCDA-APS/apstools>`_ package.
    """
    from importlib import import_module

    import_object = None

    if "." not in full_path:
        # fmt: off
        raise ValueError(
            "Must use a dotted path, no local imports."
            f" Received: {full_path!r}"
        )
        # fmt: on

    if full_path.startswith("."):
        # fmt: off
        raise ValueError(
            "Must use absolute path, no relative imports."
            f" Received: {full_path!r}"
        )
        # fmt: on

    module_name, object_name = full_path.rsplit(".", 1)
    module_object = import_module(module_name)
    import_object = getattr(module_object, object_name)

    return import_object


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
    entries = entry_points(group=SOLVER_ENTRYPOINT_GROUP)
    return entries[solver_name].load()


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
        Instance of a databroker catalog.
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
    limit = min(limit, len(catalog.v2))
    with tqdm.tqdm(total=limit, file=sys.stdout, leave=False) as progress_bar:
        for full_uid in catalog.v2:
            _count += 1
            run = catalog.v2[full_uid]
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
    return load_yaml(open(path, "r").read())


def make_component(call_name: str, **kwargs: Any) -> Component:
    """Create an Component for a custom ophyd Device class."""
    CallableObject = dynamic_import(call_name)
    return make_dynamic_instance("ophyd.Component", CallableObject, **kwargs)


def make_dynamic_instance(
    call_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Return an instance of the Python 'call_name'."""
    DynamicCallable = dynamic_import(call_name)
    if not callable(DynamicCallable):
        raise TypeError(f"{call_name!r} is not callable")
    return DynamicCallable(*args, **kwargs)


def parse_factory_axes(
    *,
    space: str = None,
    axes: Union[KeyValueMap, None, Sequence[str]] = None,
    order: Sequence[str] = None,
    canonical: Sequence[str] = None,
    labels: Sequence[str] = None,
    **_ignored: Any,
) -> Mapping[str, Union[Component, Sequence[str]]]:
    """
    Parse a set of axis specifications, return Device class attributes.

    Called from the diffract.diffractometer_class_factory().
    """
    if space not in ("pseudos", "reals"):
        raise KeyError(
            f"Unknown {space=!r}."
            #
            " Must be either 'pseudos' or 'reals'."
        )

    if order is not None and len(order) < len(canonical):
        raise ValueError(f"{len(order)=} must be >= {len(canonical)=}")
    order = order or []
    learn_order = len(order) == 0

    _axes = axes or []
    if len(_axes) == 0:
        _axes = list(canonical)
    if len(set(_axes)) < len(_axes):
        raise ValueError(f"Duplicates in axes={_axes}")
    if isinstance(_axes, list):  # make axes a dict now
        _axes = {k: None for k in _axes}

    if not learn_order and len(order) != len(set(order)):
        raise ValueError(f"Duplicates in {order=}.")

    attributes = {}
    for i, axis_name in enumerate(_axes):
        axis = _axes[axis_name]
        kwargs = dict(kind="hinted", labels=labels or [])
        class_name = "class_name"
        if space == "pseudos":
            class_name = "hklpy2.diffract.Hklpy2PseudoAxis"
            kwargs["prefix"] = ""
        elif space == "reals":
            class_name, kwargs = define_real_axis(axis, kwargs)
        if learn_order and i < len(canonical):
            order.append(axis_name)

        attributes[axis_name] = make_component(class_name, **kwargs)

    if len(order) > len(canonical):
        raise ValueError(
            f"Too many axes specified in {order=}."
            #
            f" Expected {len(canonical)}."
        )
    for axis_name in order:
        if axis_name not in _axes:
            raise KeyError(f"Unknown {axis_name=!r} in {order=!r}")

    # Define specific axis ordering attribute: '_pseudo' or '_real'
    attributes["_" + space.rstrip("s")] = order

    return attributes


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
    solver_class = get_solver(solver_name)
    return solver_class(geometry, **kwargs)


def solvers() -> Mapping[str, "SolverBase"]:
    """
    Dictionary of available Solver classes, mapped by entry point name.

    ::

        import hklpy2
        print(hklpy2.solvers())
    """
    # fmt: off
    entries = {
        ep.name: ep.value
        for ep in entry_points(group=SOLVER_ENTRYPOINT_GROUP)
    }
    # fmt: on
    return entries


def unique_name(prefix: str = "", length: int = 7) -> str:
    """
    Short, unique name, first 7 (at most) characters of a unique, random uuid.
    """
    return prefix + str(uuid.uuid4())[: max(1, min(length, 7))]


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
