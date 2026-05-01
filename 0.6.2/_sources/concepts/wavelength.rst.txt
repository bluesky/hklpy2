.. _concepts.wavelength:

==================
Wavelength
==================

.. index:: !wavelength

In diffraction, the wavelength of the incident radiation sets the radius of the
Ewald sphere. [#]_  Only :math:`hkl` reflections which lie within the Ewald sphere are
accessible to the experiment.

.. note:: While the *energy* of the incident beam may be interesting to
    diffractometer users at X-ray synchrotrons, *wavelength* is the general term
    used by diffraction science.

A diffractometer (as a subclass of
:class:`~hklpy2.diffract.DiffractometerBase`) expects the incident radiation to
be *monochromatic*.  Wavelength is used directly in every
:meth:`~hklpy2.diffract.DiffractometerBase.forward` and
:meth:`~hklpy2.diffract.DiffractometerBase.inverse` calculation — it sets the
scale of the reciprocal lattice and determines which :math:`hkl` reflections
lie within the Ewald sphere and are therefore reachable.

|hklpy2| provides wavelength classes for several common situations:

- :class:`~hklpy2.incident.Wavelength` — general monochromatic source
  (simulated, any radiation type).
- :class:`~hklpy2.incident.WavelengthXray` — X-ray source with energy/wavelength
  conversion (default for most geometries).
- :class:`~hklpy2.incident.EpicsWavelengthRO` — read wavelength from an EPICS PV
  (read-only; control of the PV is outside the diffractometer).
- :class:`~hklpy2.incident.EpicsMonochromatorRO` — read both wavelength and
  energy from a monochromator EPICS PV (read-only).

.. seealso::

   :mod:`hklpy2.incident` — full API reference for wavelength classes.

   :ref:`guide.diffract` — how to connect wavelength to a diffractometer object.

.. [#] https://dictionary.iucr.org/Ewald_sphere
