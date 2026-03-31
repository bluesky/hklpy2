.. _concepts.wavelength:

==================
Wavelength
==================

In diffraction, the wavelength of the incident radiation sets the radius of the
Ewald sphere. [#]_  Only :math:`hkl` reflections which lie within the Ewald sphere are
accessible to the experiment.

.. note:: While the *energy* of the incident beam may be interesting to
    diffractometer users at X-ray synchrotrons, *wavelength* is the general term
    used by diffraction science.

Here, a diffractometer (as a subclass of
:class:`~hklpy2.diffract.DiffractometerBase`) is a *positioner* that expects the
incident radiation to be *monochromatic*.  Knowledge of wavelength is essential
for diffractometer operations.  Simulators are provided for the general case
(:class:`~hklpy2.incident.Wavelength()`) for any type of monochromatic
radiation) and for the case of X-rays
(:class:`~hklpy2.incident.WavelengthXray()`) where photon energy is computed
from the wavelength.

When the wavelength (and possibly the energy) is known from EPICS, *read-only*
support is provided for the general case
(:class:`~hklpy2.incident.EpicsWavelengthRO()`) and for a monochromator
(:class:`~hklpy2.incident.EpicsMonochromatorRO()`) which provides both
wavelength and energy. Control of these EPICS PVs is beyond the scope of
diffractometer controls. Refer to EPICS for control of the monochromator or
wavelength PV.  Or, create a custom subclass of
:class:`~hklpy2.incident._WavelengthBase()`.

.. seealso:: The :mod:`~hklpy2.incident` module.

.. [#] https://dictionary.iucr.org/Ewald_sphere
