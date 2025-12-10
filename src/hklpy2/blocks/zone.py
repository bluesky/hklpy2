"""
Crystallographic zone axis operations.

A *zone* is set of *lattice* planes which are all parallel to one line, called
the *zone axis*. A *zone* is defined by a *zone axis* (a unit vector), which can
be specified directly or computed from two vectors (normal to their respective
*lattice* planes) using their cross product.

.. autosummary::

    ~OrthonormalZone
    ~scan_zone
    ~zonespace
    ~zone_series
"""

import logging
from typing import Iterator
from typing import Sequence

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import Readable
from bluesky.utils import plan
from numpy.typing import NDArray
from pyRestTable import Table

from ..diffract import DiffractometerBase
from ..misc import BlueskyPlanType
from ..misc import INPUT_VECTOR
from ..misc import NoForwardSolutions

logger = logging.getLogger(__name__)


class OrthonormalZone:
    """
    An orthonormal (Cartesian) zone defined by a *zone axis*.

    The zone axis can be defined directly or computed from two vectors using
    their cross product.

    For crystallography, this class operates on the Cartesian reciprocal
    lattice, transformed from the crystal lattice h,k,l coordinates.

    Parameters
    ----------
    axis : INPUT_VECTOR, optional
        Direct specification of the zone axis vector
    b1 : INPUT_VECTOR, optional
        First vector for cross product calculation
    b2 : INPUT_VECTOR, optional
        Second vector for cross product calculation

    Raises
    ------
    ValueError
        If both axis and v1/v2 are provided, or if only one of v1/v2 is provided
    """

    def __init__(
        self,
        *,
        axis: INPUT_VECTOR = None,
        b1: INPUT_VECTOR = None,
        b2: INPUT_VECTOR = None,
    ) -> None:
        """Constructor"""
        self._axis = None

        if axis is not None and (b1 is not None or b2 is not None):
            raise ValueError("Cannot specify both 'axis' and 'b1/b2' parameters")

        if axis is not None:
            self.axis = axis
        elif b1 is not None and b2 is not None:
            self.define_axis(b1, b2)
        elif b1 is not None or b2 is not None:
            raise ValueError(
                "Both b1 and b2 must be provided to define axis from vectors"
            )

    def __repr__(self) -> str:
        """Text representation of a zone object."""
        try:
            axis = self.axis
        except ValueError:
            axis = "undefined"
        return f"{self.__class__.__name__}({axis=})"

    def _standardize_vector(self, vector: INPUT_VECTOR) -> NDArray[np.floating]:
        """
        Return a numpy array of 3 floats. Raise exception as needed.

        Parameters
        ----------
        vector : INPUT_VECTOR
            A 3-element sequence, array, or dict with 3 values

        Returns
        -------
        NDArray[np.floating]
            3-element numpy array of floats

        Raises
        ------
        ValueError
            If vector cannot be converted to 3-element array
        """
        arr = vector
        if isinstance(arr, dict):
            if not arr:
                raise ValueError("Cannot create vector from empty dictionary")
            arr = list(arr.values())
        arr = np.asarray(arr, dtype=float)

        if arr.shape != (3,):
            raise ValueError(
                "vector must be 1-D array-like vector of length 3."
                #
                f"  Received {vector} with shape {arr.shape}."
            )

        return arr

    @property
    def axis(self) -> NDArray[np.floating]:
        """Get the zone axis (unit) vector.

        Returns
        -------
        NDArray[np.floating]
            3-element numpy array representing the zone axis

        Raises
        ------
        ValueError
            If zone axis is undefined
        """
        if not self.axis_defined:
            raise ValueError("Zone axis is undefined.")
        return self._axis

    @axis.setter
    def axis(self, vector: INPUT_VECTOR) -> None:
        """Set the zone axis (unit) vector.

        Parameters
        ----------
        vector : INPUT_VECTOR
            A 3-element sequence, array, or dict representing the axis

        Raises
        ------
        ValueError
            If vector cannot be converted to 3-element array
        """
        vec = self._standardize_vector(vector)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError(f"Zero |axis| not allowed. Received {vector=}.")
        self._axis = vec / norm

    @property
    def axis_defined(self) -> bool:
        """Is the zone axis defined?"""
        return self._axis is not None

    def define_axis(
        self,
        b1: INPUT_VECTOR,
        b2: INPUT_VECTOR,
        normalize: bool = False,
    ) -> NDArray[np.floating]:
        """Define the zone axis from two vectors using cross product.

        Parameters
        ----------
        b1 : INPUT_VECTOR
            First vector (3-element)
        b2 : INPUT_VECTOR
            Second vector (3-element)
        normalize : bool, default False
            If True, normalize the resulting axis vector

        Returns
        -------
        NDArray[np.floating]
            The computed zone axis vector

        Raises
        ------
        ValueError
            If vectors are parallel, zero, or invalid
        """
        arr1 = self._standardize_vector(b1)
        arr2 = self._standardize_vector(b2)
        axis = np.cross(arr1, arr2)
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError(
                f"Vectors {b1=} and {b2=} are parallel or one is zero; "
                #
                "cross product is zero and cannot define a zone axis."
            )
        self.axis = axis / norm if normalize else axis
        return self.axis

    def in_zone(self, vector: INPUT_VECTOR, tol: float = 1e-5) -> bool:
        """Verify if 'vector' is in this zone.

        Parameters
        ----------
        vector : INPUT_VECTOR
            Vector to test for zone membership
        tol : float, default 1e-5
            Tolerance for dot product comparison

        Returns
        -------
        bool
            True if vector is in the zone (dot product with axis ≤ tol)

        Raises
        ------
        ValueError
            If zone axis is undefined or vector is invalid
        """
        if not self.axis_defined:
            raise ValueError("Cannot check zone membership: zone axis is undefined")

        arr = self._standardize_vector(vector)
        dot = float(np.dot(self.axis, arr))
        return abs(dot) <= float(tol)

    def vecspace(self, b1: INPUT_VECTOR, b2: INPUT_VECTOR, n: int) -> Iterator[NDArray]:
        """
        Generate 'n' vectors from 'b1' to 'b2' rotated around b1 x b2.

        Parameters
        ----------
        b1 : INPUT_VECTOR
            Starting vector
        b2 : INPUT_VECTOR
            Ending vector
        n : int
            Number of vectors to generate

        Yields
        ------
        NDArray
            Vectors interpolated between v1 and v2

        Raises
        ------
        ValueError
            If vectors are not in the zone
        """
        if n < 2:
            yield self._standardize_vector(b1)  # the trivial case
        else:
            if not self.axis_defined:
                self.define_axis(b1, b2)
            if not self.in_zone(b1):
                raise ValueError(f"{b1=} not in zone {self}.")
            if not self.in_zone(b2):
                raise ValueError(f"{b2=} not in zone {self}.")

            axis = self.axis / np.linalg.norm(self.axis)
            u1 = self._standardize_vector(b1)
            u2 = self._standardize_vector(b2)
            n1 = np.linalg.norm(u1)
            n2 = np.linalg.norm(u2)
            u1 /= n1
            u2 /= n2

            cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angles = np.linspace(0.0, np.arccos(cos_theta), n)

            def rodrigues_rotation(vec, unit_axis, theta):
                return (
                    vec * np.cos(theta)
                    + np.cross(unit_axis, vec) * np.sin(theta)
                    + unit_axis * (unit_axis @ vec) * (1 - np.cos(theta))
                )

            unit_vectors = np.stack(
                [rodrigues_rotation(u1, axis, t) for t in angles],
                axis=0,
            )
            magnitudes = np.linspace(n1, n2, n)[:, None]
            for vector, scale in zip(unit_vectors, magnitudes):
                yield vector * scale


def zonespace(
    diff: DiffractometerBase,
    hkl_1: INPUT_VECTOR,
    hkl_2: INPUT_VECTOR,
    n: int,
) -> Iterator[tuple[Sequence[float], Sequence[float]]]:
    """
    Generate pseudos and reals along a crystallographic zone.

    * Transforms crystallographic coordinates (hkl) to Cartesian space
      (b) using the sample's reciprocal lattice.
    * Creates a zone from b1 & b2
    * Yields corresponding diffractometer pseudo and real positions for
      'n' points (including hkl_1 and hkl_2 points) along the
      orthonormal zone, distributed evenly by angle.

    Parameters
    ----------
    diff : DiffractometerBase
        Diffractometer instance for sample & forward() calculations.
    hkl_1 : INPUT_VECTOR
        Starting vector of pseudos (h, k, l).
    hkl_2 : INPUT_VECTOR
        Ending vector of pseudos (h, k, l).
    n : int
        Number of interpolation points to generate (must be > 0).

    Yields
    ------
    tuple[Sequence[float], Sequence[float]]
        (pseudos, reals) for each valid point.
        Points where forward(pseudos) fails are logged and skipped.

    Notes
    -----
    * hkl_1 & hkl_2 vectors are transformed from Miller to orthonormal
      space using the sample's reciprocal lattice.
    * Failed forward() solutions are logged at debug level.
    """
    zone = OrthonormalZone()

    astar = diff.sample.lattice.cartesian_lattice_matrix
    astar_inv = np.linalg.inv(astar).T

    # Transform v1 & v2 from Miller space to Cartesian.
    b1 = astar.T @ zone._standardize_vector(hkl_1)
    b2 = astar.T @ zone._standardize_vector(hkl_2)

    # Step through the orthonormal zone space.
    for b_vector in zone.vecspace(b1, b2, n):
        # Convert each step back to crystallographic space.
        pseudos = (astar_inv @ b_vector).tolist()
        try:
            yield pseudos, list(diff.forward(pseudos))
        except NoForwardSolutions:
            logger.debug("no solution for forward(%s)", pseudos)


def zone_series(
    diff: DiffractometerBase,
    hkl_1: INPUT_VECTOR,
    hkl_2: INPUT_VECTOR,
    n: int,
) -> None:
    """
    Example: a series of diffractometer positions along a zone.

    Print 'n' positions along the zone from hkl_1 to hkl_2 inclusive.
    The series is computed using the orthonormal zone.
    Results are displayed in a formatted table showing both reciprocal
    space coordinates (pseudos) and real motor positions (reals).

    Parameters
    ----------
    diff : DiffractometerBase
        The diffractometer instance used for forward calculations.
    hkl_1 : INPUT_VECTOR
        Starting pseudo (h, k, l).
    hkl_2 : INPUT_VECTOR
        Ending pseudo (h, k, l).
    n : int
        Number of points to generate from hkl_1 to hkl_2 (inclusive).

    Examples
    --------
    >>> # Generate 5 points from (1,0,0) to (0,1,0)
    >>> zone_series(my_diffractometer, (1,0,0), (0,1,0), 5)
    """
    table = Table()
    table.labels = diff.pseudo_axis_names + diff.real_axis_names
    for pseudos, reals in zonespace(diff, hkl_1, hkl_2, n):
        table.addRow([f"{v:.4f}" for v in pseudos + reals])
    print(f"{hkl_1=} {hkl_2=} {n=}")
    print(table)


@plan
def scan_zone(
    detectors: Sequence[Readable],
    diff: DiffractometerBase,
    start: INPUT_VECTOR,
    finish: INPUT_VECTOR,
    num: int,
    md=None,
) -> BlueskyPlanType:
    """Scan zone."""  # TODO: elaborate
    _md = dict(plan_name="scan_zone").update(md or {})

    # TODO: Consider refactor using cycler and using bp.scan_nd.
    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner():
        # Compute sequence of pseudos & reals in the zone
        for pseudos, reals in zonespace(diff, start, finish, num):
            # move axes
            logger.debug("zone hkl=%s", pseudos)
            parms = []
            for k, v in zip(diff.real_axis_names, reals):
                parms.append(getattr(diff, k))
                parms.append(v)
            yield from bps.mv(*parms)

            # trigger
            group = "trigger_objects"
            for item in detectors:
                yield from bps.trigger(item, group=group)
            yield from bps.wait(group=group)

            # read
            yield from bps.create("primary")
            for item in detectors + [diff]:
                yield from bps.read(item)
            yield from bps.save()

    return (yield from inner())
