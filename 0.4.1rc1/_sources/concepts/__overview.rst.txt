.. _overview:

========
Overview
========

|hklpy2| provides `ophyd <https://blueskyproject.io/ophyd>`_ diffractometer
devices.  Each diffractometer is a positioner which may be used with `bluesky
<https://blueskyproject.io/bluesky>`_ plans.

Any diffractometer may be provisioned with simulated axes; motors from an EPICS
control system are not required to use |hklpy2|.

Built from :class:`~hklpy2.diffract.DiffractometerBase()`, each diffractometer is
an `ophyd.PseudoPositioner
<https://blueskyproject.io/ophyd/positioners.html#pseudopositioner>`_ that
defines all the components of a diffractometer. The diffractometer
:ref:`geometry <geometries>` defines the names and order for the real motor
axes. Geometries are defined by backend  :ref:`concepts.solvers`. Some solvers
support different calculation engines (other than :math:`hkl`). It is common for a
geometry to support several operating *modes*.

.. seealso:: :ref:`glossary`, :ref:`v2_checklist`
