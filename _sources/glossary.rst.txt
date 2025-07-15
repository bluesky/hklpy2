..  The glossary is formatted as a reST "definition list".
    Follow the pattern.

    All glossary entries should be preceded by
    index entries.  Follow the pattern.

.. index:: !definition
.. index:: see: glossary; definition

.. _glossary:

==========
Glossary
==========

.. tip:: *Italics* are used in these definitions to identify
    other glossary entries.

..  index::
    !definition; axis
    !axis

:axis: Either a *pseudo*, *real*, or *extra*.

  :*physical*: Axis (either **pseudo** or **real**) which is not computed from
    other axes (except by the solver).  "Physical" for a *real* implies a direct
    connection to hardware,

  :*virtual*: Axis (either **pseudo** and **real**) which is computed from one
    or more other axes.

  :*extra*: See below for definition of **extra**.

..  index::
    !definition; backend
    !backend

:backend: Synonym for *solver*.

..  index::
    !definition; configuration
    !configuration

:configuration: Complete description of a *diffractometer's*
  settings.  Includes *solver*, *geometry* (& *engine*, if applicable),
  ordered lists of the *axis* names, dictionaries of *samples*
  (with *lattice* & *reflection(s)*).

..  index::
    !definition; constraint
    !constraint

:constraint: Limitations on acceptable positions for a *diffractometer's*
  computed ``forward()`` solutions (from :math:`hkl` to angles).  A *solver's*
  ``forward()`` computation returns a list of solutions, where a solution
  is the set of real-space angles that position the *diffractometer* to the
  desired :math:`hkl` value.  A constraint can be used to reject solutions for
  undesired angles.

..  index::
    !definition; core
    !core

:core: The |hklpy2| intermediate software adapter layer between
  :class:`~hklpy2.diffract.DiffractometerBase` (user-facing code) and a
  :class:`~hklpy2.backends.base.SolverBase`.

  Connects a *diffractometer* with  a |solver| library and
  one of its *geometries*.

..  index::
    !definition; crystal
    !crystal

:crystal: A homogeneous substance composed from a repeating three-dimensional
  pattern.  The pattern (*unit cell*) is characterized by its *lattice*.

..  index::
    !definition; detector
    !detector

:detector: Measures the intensity of diffracted radiation from the sample.

..  index::
    !definition; diffractometer
    !diffractometer

:diffractometer:
  Diffractometers, mechanical systems of *real* space rotation axes, are used in
  studies of the stucture of *crystalline* *samples*.  The structural features of
  interest are usually expressed in terms of reciprocal space (*pseudo*) axes.

  A diffractometer is a type of *goniometer*.  Generally, a diffractometer
  consists of two stacks of rotation axes, used to control the *orientation* of
  a *crystalline* *sample* and a *detector*.  In a study, while the sample is
  oriented and exposed to a controlled radiation source, the detector is
  oriented to measure the intensity of radiation diffracted by the sample in
  specific directions.

..  index::
    !definition; engine
    !engine

:engine: Some |solver| libraries provide coordinate transformations
  between *real* axes and different types of *pseudo* axes,
  such as for reflectometry or surface scattering.  The |solver| may provide
  an engine for each separate type of transformation (and related
  *pseudos*).

..  index::
    !definition; extra
    !extra

:extra: An additional axis used by a |solver| for operation of
  a *diffractometer*.
  For example, when rotating by angle :math:`\psi` around some arbitrary
  diffraction vector, :math:`(h_2,k_2,l_2)`, the extra axes are:
  ``"h2", "k2", "l2", "psi"``.

  An *extra* axis is not defined as a diffractometer `Component`.

..  index::
    !definition; geometry
    !geometry

:geometry: The set of *reals* (stacked rotation angles) which
  define a specific *diffractometer*. A common distinguishing feature is the
  number of axes in each stack. For example, the :ref:`E4CV
  <geometries-hkl_soleil-e4cv>`  geometry has 3 sample axes (``omega``, ``chi``,
  ``phi``) and 1 detector axis (``tth``). In some shorthand reference, this
  could be called "S3D1".

..  index::
    !definition; goniometer
    !goniometer

:goniometer: Mechanical system which allows an object to be rotated to
  a precise angular position.

..  index::
    !definition; lattice
    !lattice

:lattice: Characteristic dimensions of the parallelepiped representing the
  *sample* *crystal* structure.  For a three-dimensional crystal, the lengths of
  each side of the lattice are :math:`a`, :math:`b`, & :math:`c`, the angles
  between the sides are :math:`\alpha`, :math:`\beta`, & :math:`\gamma`

..  index::
    !definition; mode
    !mode

:mode: *Diffractometer* *geometry* operation mode for
  :meth:`~hklpy2.diffract.DiffractometerBase.forward()`.

  A *mode* (implemented by a |solver|), defines which axes will be
  modified by the
  :meth:`~hklpy2.diffract.DiffractometerBase.forward()` computation.

..  index::
    !definition; monochromatic
    !monochromatic

:monochromatic: Radiation of a single wavelength (or sufficiently narrow
  range), such that it may be characterized by a single floating point value.

..  index::
    !definition; OR
    !OR

:OR: Orienting Reflection, a *reflection* used to define the *sample*
  *orientation* (and compute the :math:`UB` matrix).

..  index::
    !definition; orientation
    !orientation

:orientation: Positioning of a *crystalline* sample's atomic planes
  (identified by a set of *pseudos*) within the laboratory reference
  frame (described by the *reals*).

..  index::
    !definition; pseudo
    !pseudo

:pseudo: Reciprocal-space axis, such as :math:`h`, :math:`k`, and :math:`l`.
  The engineering units (rarely examined for *crystalline* work) are reciprocal
  of the *wavelength* units.

  Note: **pseudo** axes are **virtual** axes, computed by the solver from
  **real** axes.

..  index::
    !definition; real
    !real

:real: Real-space axis (typically a rotation stage),
  such as ``omega`` (:math:`\omega`).
  The engineering units are expected to be in **degrees**.

..  index::
    !definition; reflection
    !reflection

:reflection: User-identified coordinates serving as a fiducial reference
  associating crystal orientation (reciprocal space, *pseudos*) and rotational
  axes (real space, *reals*). Reflections are used to orient a *sample* with a
  specific *diffractometer* geometry. In |hklpy2|, a reflection has a name, a
  set of *pseudos*, a set of *reals*, and a *wavelength*.

..  index::
    !definition; sample
    !sample

:sample: The named substance to be explored with the *diffractometer*.
  In |hklpy2|, a sample has a name, a *lattice*, and a list of *reflections*.

  The *axes* in a sample's *reflections* are specific to the *diffractometer*
  *geometry*.

  Consequently, the sample is defined for a specific |solver| and
  *geometry*.  The same sample cannot be used for other geometries.

..  index::
    !definition; solver
    !solver

:solver: The |hklpy2| interface layer to a backend |solver| library, such as
  |libhkl|. The library provides computations to transform coordinates between
  *pseudo* and *real* axes for a defined *diffractometer* *geometry*.  The
  library also provides one or more diffractometer geometries.

..  index::
    !U
    !UB
    !definition; U
    !definition; UB

:UB: Orientation matrix (3 x 3).

  :math:`U` Orientation matrix
    of the *crystal* *lattice* as mounted on the *diffractometer* *sample* holder.

  :math:`B` Transition matrix
    of a non-orthonormal (the reciprocal of the crystal) in an orthonormal system.

  :math:`UB` Orientation matrix
    of the *crystal* *lattice* in the laboratory reference frame.

..  index::
    !definition; unit cell
    !unit cell

:unit cell: The parallelepiped representing the smallest repeating structural
  pattern of the *crystal*, characterized by its *lattice* parameters.

..  index::
    !definition; virtual
    !real

:real: Virtual (computed) diffractometer **axis** (either **pseudo** or
    **real**), computed from one or more additional **diffractometer** axes.

..  index::
    !definition; wavelength
    !wavelength

:wavelength: The numerical value of the wavelength of the incident radiation.
  The radiation is expected to be *monochromatic* neutrons or X-rays.
  The engineering units of wavelength must be identical to those of the
  *crystalline* *lattice* length parameters.
