"""
Ophyd device construction helpers for |hklpy2|.

These utilities are called during diffractometer creation to build ophyd
``Component`` and ``Device`` objects dynamically from axis specifications
and factory configurations.

.. autosummary::

    ~VirtualPositionerBase
    ~define_real_axis
    ~dict_device_factory
    ~dynamic_import
    ~make_component
    ~make_dynamic_instance
    ~parse_factory_axes
"""

import logging
import time
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

from deprecated.sphinx import versionadded
from ophyd import Component
from ophyd import Device
from ophyd import EpicsMotor
from ophyd import PVPositioner
from ophyd import SoftPositioner

from .typing import KeyValueMap

logger = logging.getLogger(__name__)

__all__ = [
    "VirtualPositionerBase",
    "define_real_axis",
    "dict_device_factory",
    "dynamic_import",
    "make_component",
    "make_dynamic_instance",
    "parse_factory_axes",
]

# Default labels applied to real-axis positioners.
# Default motor labels — kept in sync with utils.DEFAULT_MOTOR_LABELS.
_DEFAULT_MOTOR_LABELS: Sequence[str] = ["motors"]


@versionadded(version="0.1.4", reason="Base class for diffractometer virtual axes.")
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
        logger.debug(
            "VirtualPositionerBase._finish_setup() completed for %r", self.name
        )
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


def define_real_axis(
    specs: Union[None, str, KeyValueMap],
    kwargs: KeyValueMap,
) -> tuple[str, Sequence, Mapping]:
    """Return class and kwargs of a real axis from its 'specs'."""
    args = []
    kwargs["labels"] += _DEFAULT_MOTOR_LABELS

    if specs is None:
        class_name = "ophyd.SoftPositioner"
        kwargs.update({"limits": (-180, 180), "init_pos": 0})
    elif isinstance(specs, str):
        class_name = "ophyd.EpicsMotor"
        args.append(specs)  # PV (appends to parent's prefix)
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

    return class_name, args, kwargs


def dict_device_factory(data: KeyValueMap, **kwargs: Any) -> type:
    """
    Create a ``DictionaryDevice()`` class using the supplied dictionary.

    .. index:: factory; dict_device_factory, dict_device_factory
    """
    from ophyd import Signal

    component_dict = {k: Component(Signal, value=v, **kwargs) for k, v in data.items()}
    return type("DictionaryDevice", (Device,), component_dict)


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
    logger.debug("Imported %r from %r", object_name, module_name)

    return import_object


def make_component(call_name: str, *args: Any, **kwargs: Any) -> Component:
    """Create an Component for a custom ophyd Device class."""
    CallableObject = dynamic_import(call_name)
    return make_dynamic_instance(
        "ophyd.Component",
        CallableObject,
        *args,
        **kwargs,
    )


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
    space: Optional[str] = None,
    axes: Union[KeyValueMap, None, Sequence[str]] = None,
    order: Optional[Sequence[str]] = None,
    canonical: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
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
        args = []
        kwargs = dict(kind="hinted", labels=labels or [])
        class_name = "class_name"
        if space == "pseudos":
            class_name = "hklpy2.diffract.Hklpy2PseudoAxis"
        elif space == "reals":
            class_name, args, kwargs = define_real_axis(axis, kwargs)
        if learn_order and i < len(canonical):
            order.append(axis_name)

        attributes[axis_name] = make_component(class_name, *args, **kwargs)

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
