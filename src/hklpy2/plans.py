"""
Bluesky plans provided by |hklpy2|.

This module is the single, canonical location for all standalone |hklpy2|
plans.  New plans should be implemented here directly.

.. note::

    Diffractometer-method plans
    (:meth:`~hklpy2.diffract.DiffractometerBase.scan_extra`,
    :meth:`~hklpy2.diffract.DiffractometerBase.move_dict`, etc.) are not
    listed here because they require a diffractometer instance as ``self``.
    The free-function wrapper :func:`~hklpy2.user.scan_extra` is available
    in :mod:`hklpy2.user` for interactive use with a pre-selected
    diffractometer.

.. autosummary::

    ~scan_psi

.. rubric:: Solver compatibility for scan_psi

:func:`scan_psi` discovers the psi mode and psi axis name from the
diffractometer's solver at runtime using only the solver-agnostic API:

* ``diffractometer.core.modes`` — list of available mode names
* ``diffractometer.core.solver_extra_axis_names`` — extra axis names for
  the current mode

Auto-detection searches ``core.modes`` for a name that contains ``"psi"``.
If the solver exposes more than one such mode (e.g. ``"psi_constant"``
**and** ``"psi_constant_vertical"`` on an E6C geometry), auto-detection is
ambiguous and the caller must pass ``mode=`` explicitly.

Override parameters ``mode=`` and ``psi_axis=`` are provided as escape
hatches for future solvers whose naming conventions differ from the current
``hkl_soleil`` convention.
"""

import logging
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence

from bluesky.protocols import Readable
from bluesky.utils import plan
from deprecated.sphinx import versionadded

from .diffract import DiffractometerBase
from .misc import validate_not_parallel
from .typing import BlueskyPlanType

logger = logging.getLogger(__name__)

__all__ = [
    "scan_psi",
]


# ---------------------------------------------------------------------------
# Private helpers for scan_psi
# ---------------------------------------------------------------------------


def _find_psi_mode(
    diffractometer: DiffractometerBase, mode_override: Optional[str]
) -> str:
    """Return the name of the psi-capable mode, or raise.

    Parameters
    ----------
    diffractometer : DiffractometerBase
        The diffractometer to inspect.
    mode_override : str or None
        If not None, returned as-is after validating it exists in
        ``core.modes``.

    Raises
    ------
    ValueError
        If ``mode_override`` is not in ``core.modes``, or if auto-detection
        finds zero or more than one psi-capable mode.
    NotImplementedError
        If no psi-capable mode exists in ``core.modes``.
    """
    available = diffractometer.core.modes
    if mode_override is not None:
        if mode_override not in available:
            raise ValueError(
                f"mode={mode_override!r} not in available modes: {available}"
            )
        return mode_override

    candidates = [m for m in available if "psi" in m.lower()]
    if len(candidates) == 0:
        raise NotImplementedError(
            "No psi-capable mode found in available modes: "
            f"{available}. "
            "Pass mode= to specify the mode name explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple psi-capable modes found: {candidates}. "
            "Pass mode= to select one explicitly."
        )
    logger.debug("scan_psi: auto-detected psi mode %r", candidates[0])
    return candidates[0]


def _find_psi_axis(
    diffractometer: DiffractometerBase, psi_axis_override: Optional[str]
) -> str:
    """Return the name of the psi extra axis in the current mode, or raise.

    Must be called *after* the desired psi mode has been set on the
    diffractometer, so that ``core.solver_extra_axis_names`` reflects that
    mode.

    Parameters
    ----------
    diffractometer : DiffractometerBase
        The diffractometer (already in the psi mode).
    psi_axis_override : str or None
        If not None, returned as-is after validating it exists in
        ``core.solver_extra_axis_names``.

    Raises
    ------
    ValueError
        If the axis is not found, or auto-detection is ambiguous.
    """
    extra_axes = diffractometer.core.solver_extra_axis_names
    if psi_axis_override is not None:
        if psi_axis_override not in extra_axes:
            raise ValueError(
                f"psi_axis={psi_axis_override!r} not in extra axes for "
                f"current mode: {extra_axes}"
            )
        return psi_axis_override

    candidates = [ax for ax in extra_axes if "psi" in ax.lower()]
    if len(candidates) == 0:
        raise ValueError(
            f"No psi extra axis found in {extra_axes}. "
            "Pass psi_axis= to specify the axis name explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple psi-like extra axes found: {candidates}. "
            "Pass psi_axis= to select one explicitly."
        )
    logger.debug("scan_psi: auto-detected psi axis %r", candidates[0])
    return candidates[0]


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@versionadded(
    version="0.5.1",
    reason="Convenience plan for azimuthal (ψ) scans.",
)
@plan
def scan_psi(
    detectors: Sequence[Readable],
    diffractometer: DiffractometerBase,
    *,
    h: float,
    k: float,
    l: float,  # noqa: E741
    hkl2: Sequence[float],
    psi_start: float,
    psi_stop: float,
    num: int,
    mode: Optional[str] = None,
    psi_axis: Optional[str] = None,
    fail_on_exception: bool = False,
    md: Optional[Mapping[str, Any]] = None,
) -> BlueskyPlanType:
    """
    Scan the azimuthal angle (ψ) at a fixed *(h, k, l)* position.

    Rotates the sample about the scattering vector **Q** by sweeping the
    azimuthal extra parameter (conventionally named ``psi``) from
    *psi_start* to *psi_stop* in *num* steps, while holding *(h, k, l)*
    constant.

    The reference reflection *hkl2* defines the azimuthal zero (psi = 0).
    It must not be parallel (or anti-parallel) to *(h, k, l)*.

    .. rubric:: Behavior

    1. Discovers the psi-capable mode from ``diffractometer.core.modes``
       (any mode whose name contains ``"psi"``).  If the solver exposes more
       than one such mode the caller must pass ``mode=`` explicitly.
    2. Saves the current mode and restores it on exit (even after an error).
    3. Sets the discovered mode and discovers the psi extra axis name from
       ``core.solver_extra_axis_names`` (any axis whose name contains
       ``"psi"``).
    4. Validates that *hkl* and *hkl2* are not parallel.
    5. Calls :meth:`~hklpy2.diffract.DiffractometerBase.scan_extra` with
       the appropriate arguments.

    .. rubric:: Example

    .. code-block:: python

        from hklpy2 import creator, scan_psi
        fourc = creator()
        # ... set up sample and UB matrix ...
        RE(
            scan_psi(
                [det, fourc],
                fourc,
                h=2, k=2, l=0,
                hkl2=(1, -1, 0),
                psi_start=0, psi_stop=180,
                num=7,
            )
        )

    Parameters
    ----------
    detectors : Sequence[Readable]
        Ophyd devices to trigger and read at each measurement point.
    diffractometer : DiffractometerBase
        hklpy2 diffractometer object.
    h : float
        *h* component of the fixed reflection.
    k : float
        *k* component of the fixed reflection.
    l : float
        *l* component of the fixed reflection.
    hkl2 : Sequence[float]
        Reference reflection *(h2, k2, l2)* that defines psi = 0.
        Must not be parallel to *(h, k, l)*.
    psi_start : float
        Starting azimuthal angle (degrees).
    psi_stop : float
        Ending azimuthal angle (degrees).
    num : int
        Number of points (inclusive of endpoints).
    mode : str, optional
        Psi-capable mode name to use.  When *None* (default), the mode is
        discovered automatically by searching ``core.modes`` for a name
        containing ``"psi"``.  Pass explicitly when the solver exposes
        multiple psi modes (e.g. ``"psi_constant_vertical"`` on E6C).
    psi_axis : str, optional
        Name of the azimuthal extra axis to scan.  When *None* (default),
        the axis is discovered automatically by searching
        ``core.solver_extra_axis_names`` for a name containing ``"psi"``.
        Pass explicitly if the solver uses a different naming convention.
    fail_on_exception : bool, optional
        When *True*, any failed forward-calculation at a scan point raises
        immediately.  When *False* (default), the failure is printed and
        the scan continues.
    md : dict, optional
        User-supplied metadata added to the Bluesky run document.

    Raises
    ------
    NotImplementedError
        If no psi-capable mode is found in ``core.modes``.
    ValueError
        If mode or psi_axis discovery is ambiguous, *hkl2* is parallel to
        *(h, k, l)*, or an explicit override is not found in the solver.

    Notes
    -----
    * The current diffractometer mode is saved before the scan and restored
      afterward, even if the scan raises an exception.
    * The reference extra axes (h2, k2, l2 or equivalent) are identified as
      *all* extra axes in the psi mode that are not the psi axis itself.
      Exactly three such axes are expected; any other count raises
      ``ValueError``.
    * Currently only solvers that expose a psi-capable mode (mode name
      containing ``"psi"``) in their extra-axis API are supported.  This
      includes ``hkl_soleil`` geometries such as E4CV, E4CH, K4CV, E6C,
      K6C, APS POLAR, SOLEIL MARS, and PETRA3 P23.  Future solvers can
      support :func:`scan_psi` by exposing a similarly named mode and extra
      axis without any change to this function, or by passing ``mode=`` and
      ``psi_axis=`` explicitly.

    .. seealso::

        :meth:`~hklpy2.diffract.DiffractometerBase.scan_extra` — the
        lower-level plan this function wraps.
    """
    # --- Resolve pure-Python parameters before touching the diffractometer. ---
    # 1. Discover (or validate) the psi mode.
    psi_mode = _find_psi_mode(diffractometer, mode)

    # 2. Validate that hkl and hkl2 are not parallel (no side effects).
    validate_not_parallel((h, k, l), hkl2)

    # 3. Inner plan: all diffractometer side-effects and yielding happen here.
    @plan
    def _inner():
        # Activate the psi mode so solver_extra_axis_names reflects it.
        saved_mode = diffractometer.core.mode
        diffractometer.core.mode = psi_mode

        try:
            # Discover (or validate) the psi extra axis name.
            psi_ax = _find_psi_axis(diffractometer, psi_axis)

            # The remaining extra axes are the reference reflection axes
            # (h2, k2, l2 or equivalent).  All known psi-capable modes in
            # hkl_soleil have exactly 3 such axes; the count check below
            # guards against unexpected solver behaviour.
            all_extras = diffractometer.core.solver_extra_axis_names
            ref_axes = [ax for ax in all_extras if ax != psi_ax]
            if len(ref_axes) != 3:
                raise ValueError(
                    f"Expected exactly 3 reference extra axes (for h2, k2, l2 or "
                    f"equivalent), found {len(ref_axes)}: {ref_axes}. "
                    "Check the solver mode or pass psi_axis= to disambiguate."
                )

            # Build the extras dict for the reference reflection.
            ref_extras = dict(zip(ref_axes, hkl2))

            # Delegate to scan_extra.
            yield from diffractometer.scan_extra(
                detectors,
                psi_ax,
                psi_start,
                psi_stop,
                num=num,
                pseudos=dict(h=h, k=k, l=l),
                extras=ref_extras,
                fail_on_exception=fail_on_exception,
                md=md,
            )
        finally:
            # Always restore the mode the user had before calling scan_psi.
            if saved_mode and saved_mode in diffractometer.core.modes:
                diffractometer.core.mode = saved_mode

    yield from _inner()
