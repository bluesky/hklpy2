"""
Limitations on acceptable positions for computed 'forward()' solutions.

Computation of the real-space axis positions given a set of reciprocal-space
coordinates can have many solutions. One or more constraints (Constraint),
together with a choice of operating *mode*, can:

* Limit the range of ``forward()`` solutions accepted for that positioner.
* Declare the value to use when the positioner should be kept constant. (not
  implemented yet)

.. autosummary::

    ~DEFAULT_CUT_POINT
    ~RealAxisConstraints
    ~ConstraintBase
    ~LimitsConstraint
"""

import math
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..misc import ConfigurationError
from ..misc import ConstraintsError
from ..typing import NUMERIC
from ..typing import KeyValueMap

ENDPOINT_TOLERANCE: float = 1e-4  # for comparisons, less than motion step size
UNDEFINED_LABEL: str = "undefined"

DEFAULT_CUT_POINT: float = -180.0
"""
Default cut-point value (degrees).

A cut point ``c`` maps a computed angle into the range from ``c`` up to
(but not including) ``c + 360``.  The default of ``-180`` gives the
familiar range of -180 up to (but not including) +180 degrees.
"""


class ConstraintBase(ABC):
    """
    Base class for all constraints for selecting 'forward()' solutions.

    .. autosummary::

        ~_asdict
        ~valid
    """

    _fields: List[str] = []
    label: str = UNDEFINED_LABEL

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        content = [f"{k}={v}" for k, v in self._asdict().items()]
        return f"{self.__class__.__name__}({', '.join(content)})"

    def _asdict(self) -> KeyValueMap:
        """Return a new dict which maps field names to their values."""
        result = {k: getattr(self, k) for k in self._fields}
        result["class"] = self.__class__.__name__
        return result

    def _fromdict(self, config: KeyValueMap, core: Optional[Any] = None):
        """Redefine this constraint from a (configuration) dictionary."""
        from ..ops import Core

        if self.__class__.__name__ != config["class"]:
            raise ConfigurationError(
                f"Wrong configuration class {self.__class__.__name__}({self.label!r})."
                f" Received: {config!r}"
            )

        if isinstance(core, Core):
            # Validate with solver.
            axis = config["label"]
            axes_local = list(core.diffractometer.real_axis_names)
            axes_solver = list(core.solver.real_axis_names)
            if axis not in axes_local + axes_solver:
                raise KeyError(
                    f"Constraint label {axis=}"
                    f" not found in diffractometer reals: {axes_local}"
                    f" or solver's reals {axes_solver}."
                )

        for k in self._fields:
            if k in config:
                setattr(self, k, config[k])
            else:
                raise ConfigurationError(
                    f"Missing key for {self.__class__.__name__}({self.label!r})."
                    f" Expected key: {k!r}."
                    f" Received configuration: {config!r}"
                )

    @abstractmethod
    def valid(self, **values: Dict[str, NUMERIC]) -> bool:
        """
        Is this constraint satisifed by current value(s)?

        PARAMETERS

        values *dict*:
            Dictionary of current 'axis: value' pairs for comparison.
        """
        # return True


class LimitsConstraint(ConstraintBase):
    """
    Value must fall between low & high limits, after cut-point wrapping.

    Two mechanisms work together for each real axis:

    1. **Cut point** (:attr:`cut_point`): wraps the computed angle into
       a preferred 360-degree window before any limit check.  The cut
       point ``c`` maps an angle ``v`` to the equivalent angle in the
       range from ``c`` up to (but not including) ``c + 360``.  This
       controls *representation* — which 360-degree window the angle is
       expressed in — not whether the solution is accepted or rejected.

    2. **Limits** (:attr:`low_limit`, :attr:`high_limit`): filter out
       solutions whose (already-wrapped) axis value falls outside the
       configured range.  This controls *validity* — whether the
       physical motor can reach that position.

    The cut point is applied first (in :meth:`~hklpy2.ops.Core.forward`),
    then the limits check operates on the wrapped value.

    Parameters
    ----------
    low_limit : float
        Lowest acceptable value for this axis when computing real-space
        solutions from given reciprocal-space positions.
    high_limit : float
        Highest acceptable value for this axis when computing real-space
        solutions from given reciprocal-space positions.
    label : str
        Name of the axis for these limits.
    cut_point : float
        Angle (degrees) at which the 360-degree wrap begins.  The
        computed angle is mapped to the range from ``cut_point`` up to
        (but not including) ``cut_point + 360``.  Default is
        ``-180``, giving the range -180 up to (but not including) +180.
        Use ``0`` for the range 0 up to (but not including) 360.

    .. autosummary::

        ~apply_cut
        ~cut_point
        ~limits
        ~valid
    """

    def __init__(
        self,
        low_limit: Optional[float] = -180,
        high_limit: Optional[float] = 180,
        label: Optional[str] = None,
        cut_point: float = DEFAULT_CUT_POINT,
    ) -> None:
        if label is None:
            raise ConstraintsError("Must provide a value for 'label'.")

        self.label = label
        self._fields = "label low_limit high_limit cut_point".split()

        if low_limit is None:
            low_limit = -180
        if high_limit is None:
            high_limit = 180

        self.low_limit, self.high_limit = sorted(
            map(float, [low_limit, high_limit]),
        )
        self.cut_point = float(cut_point)

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        return (
            f"{self.low_limit} <= {self.label} <= {self.high_limit}"
            f" [cut={self.cut_point}]"
        )

    def _fromdict(self, config: KeyValueMap, core: Optional[Any] = None) -> None:
        """
        Redefine this constraint from a (configuration) dictionary.

        Tolerates missing ``cut_point`` key for backward compatibility
        with configurations saved before cut-point support was added;
        those default to :data:`DEFAULT_CUT_POINT`.
        """
        # Handle cut_point separately so we can supply a default when
        # loading older configurations that pre-date this field.
        saved_fields = self._fields
        self._fields = [f for f in self._fields if f != "cut_point"]
        super()._fromdict(config, core=core)
        self._fields = saved_fields
        self.cut_point = float(config.get("cut_point", DEFAULT_CUT_POINT))

    def apply_cut(self, value: float) -> float:
        """
        Map ``value`` into the range from ``cut_point`` up to (but not
        including) ``cut_point + 360``.

        For example, with the default cut point of ``-180``:

        - ``45.0``  →  ``45.0``   (already in range, unchanged)
        - ``200.0`` →  ``-160.0`` (wrapped down by 360)
        - ``-200.0`` → ``160.0``  (wrapped up by 360)
        - ``-180.0`` → ``-180.0`` (at the cut point, unchanged)
        - ``180.0``  → ``-180.0`` (at the open end, wraps to cut point)

        Parameters
        ----------
        value : float
            Angle in degrees to wrap.

        Returns
        -------
        float
            Equivalent angle in the range from ``cut_point`` up to
            (but not including) ``cut_point + 360``.
        """
        wrapped = self.cut_point + math.fmod(value - self.cut_point, 360.0) % 360.0
        # Floating-point arithmetic can produce a result indistinguishably
        # close to cut_point + 360 (the excluded upper bound).  Map it back.
        if abs(wrapped - (self.cut_point + 360.0)) < ENDPOINT_TOLERANCE:
            wrapped = self.cut_point
        return wrapped

    @property
    def limits(self) -> tuple[float, float]:
        """Return the low and high limits of this constraint."""
        return (self.low_limit, self.high_limit)

    @limits.setter
    def limits(self, values: tuple[float, float]) -> None:
        if len(values) != 2:
            raise ConstraintsError(f"Use exactly two values.  Received: {values!r}")
        self.low_limit, self.high_limit = sorted(map(float, values))

    def valid(self, **values: Dict[str, NUMERIC]) -> bool:
        """
        True if low <= value <= high.

        PARAMETERS

        reals *dict*:
            Dictionary of current 'axis: value' pairs for comparison.
            Values should already be cut-point-wrapped before calling
            this method (see :meth:`apply_cut` and
            :meth:`~hklpy2.ops.Core.forward`).
        """
        if self.label not in values:
            raise ConstraintsError(
                f"Supplied values ({values!r}) did not include this"
                f" constraint's label {self.label!r}."
            )

        value = values[self.label]
        # return self.low_limit <= values[self.label] <= self.high_limit
        return (
            (value + ENDPOINT_TOLERANCE) >= self.low_limit
            # .
            and (value - ENDPOINT_TOLERANCE) <= self.high_limit
        )


class RealAxisConstraints(dict):
    """
    Constraints for every (real) axis of the diffractometer.

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~valid
    """

    def __init__(self, reals: List[str]) -> None:
        for k in reals:
            self[k] = LimitsConstraint(label=k)

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        return str([str(c) for c in self.values()])

    def _asdict(self) -> KeyValueMap:
        """Return all constraints as a dictionary."""
        return {k: c._asdict() for k, c in self.items()}

    def _fromdict(self, config: KeyValueMap, core=None) -> None:
        """Redefine existing constraints from a (configuration) dictionary."""
        for k, v in config.items():
            self[k]._fromdict(v, core=core)

    def valid(self, **reals: Dict[str, NUMERIC]) -> bool:
        """Are all constraints satisfied?"""
        findings = [constraint.valid(**reals) for constraint in self.values()]
        return False not in findings
