.. index:: see: glossary; definition

.. _glossary:

==========
Glossary
==========

.. tip:: *Italics* are used in these definitions to identify
    other glossary entries.

.. glossary::
    :sorted:

    auxiliary
        Axes that are not *pseudos* or *reals*.

    axis
        Either a *pseudo*, *real*, *extra*.

        :*physical*: Axis (either **pseudo** or **real**) which is not computed from
          other axes (except by the solver).  "Physical" for a *real* implies a direct
          connection to hardware,

        :*virtual*: Axis (either **pseudo** and **real**) which is computed from one
          or more other axes.

        :*extra*: See below for definition of **extra**.

    backend
        Synonym for *solver*.

    configuration
        Complete description of a *diffractometer's*
        settings.  Includes *solver*, *geometry* (& *engine*, if applicable),
        ordered lists of the *axis* names, dictionaries of *samples*
        (with *lattice* & *reflection(s)*).

    constraint
        Limitations on acceptable positions for a *diffractometer's*
        computed ``forward()`` solutions (from :math:`hkl` to angles).  A *solver's*
        ``forward()`` computation returns a list of solutions, where a solution
        is the set of real-space angles that position the *diffractometer* to the
        desired :math:`hkl` value.  A constraint can be used to reject solutions for
        undesired angles.

    core
        The |hklpy2| intermediate software adapter layer between
        :class:`~hklpy2.diffract.DiffractometerBase` (user-facing code) and a
        :class:`~hklpy2.backends.base.SolverBase`.

        Connects a *diffractometer* with  a |solver| library and
        one of its *geometries*.

    crystal
        A homogeneous substance composed from a repeating three-dimensional
        pattern.  The pattern (*unit cell*) is characterized by its *lattice*.

    crystal analyzer
        A passive *crystal* mounted on the detector arm that selects a
        specific scattered *wavelength* by Bragg diffraction, acting as a
        narrow bandpass filter.  The Bragg angle (often called ``atheta``)
        positions the crystal; the detector relocates to the analyzer arm
        (often ``attheta``) and points at the analyzer.  Treated by
        |hklpy2| as *auxiliary* positioners; does not affect
        crystallography calculations.  See :doc:`guides/how_analyzer`.

    cut point
        The angle at which a motor's reported position "wraps around."
        It sets the start of the 360-degree window used to express the
        angle — the physics is unchanged, only how the number is written.

        Two common choices:

        - ``cut_point = -180`` (default): angles reported in −180 up to
          (but not including) +180.
        - ``cut_point = 0``: angles reported in 0 up to (but not
          including) 360.

        Any finite number is acceptable.  ``inf``, ``-inf``, and ``nan``
        raise :exc:`~hklpy2.misc.ConstraintsError`.  The cut point does
        not need to fall within the axis limits — it is independent of
        ``low_limit`` and ``high_limit``.

        Applied *before* :term:`constraint` checking in the ``forward()``
        pipeline — controls *representation*, not whether a solution is
        accepted or rejected.

        Equivalent to SPEC ``cuts`` and diffcalc ``setcut``.

        See also: :term:`constraint`.

    detector
        Measures the intensity of diffracted radiation from the sample.

    diffractometer
        Diffractometers, mechanical systems of *real* space rotation axes, are used in
        studies of the stucture of *crystalline* *samples*.  The structural features of
        interest are usually expressed in terms of reciprocal space (*pseudo*) axes.

        A diffractometer is a type of *goniometer*.  Generally, a diffractometer
        consists of two stacks of rotation axes, used to control the *orientation* of
        a *crystalline* *sample* and a *detector*.  In a study, while the sample is
        oriented and exposed to a controlled radiation source, the detector is
        oriented to measure the intensity of radiation diffracted by the sample in
        specific directions.

    engine
        Some |solver| libraries provide coordinate transformations
        between *real* axes and different types of *pseudo* axes,
        such as for reflectometry or surface scattering.  The |solver| may provide
        an engine for each separate type of transformation (and related
        *pseudos*).

    entry point
        A Python packaging mechanism that allows a package to advertise
        a named object (such as a class or function) so other packages can discover
        and load it without a hard-coded import.  |hklpy2| uses the
        ``"hklpy2.solver"`` entry point group to locate installed |solver| plugins.

    extra
        An additional axis used by a |solver| for operation of
        a *diffractometer*.
        For example, when rotating by angle :math:`\psi` around some arbitrary
        diffraction vector, :math:`(h_2,k_2,l_2)`, the extra axes are:
        ``"h2", "k2", "l2", "psi"``.

        An *extra* axis is not defined as a diffractometer `Component`.

    geometry
        The set of *reals* (stacked rotation angles) which
        define a specific *diffractometer*. A common distinguishing feature is the
        number of axes in each stack. For example, the :ref:`E4CV
        <geometries-hkl_soleil-e4cv>`  geometry has 3 sample axes (``omega``, ``chi``,
        ``phi``) and 1 detector axis (``tth``). In some shorthand reference, this
        could be called "S3D1".

    goniometer
        Mechanical system which allows an object to be rotated to
        a precise angular position.

    lattice
        Characteristic dimensions of the parallelepiped representing the
        *sample* *crystal* structure.  For a three-dimensional crystal, the lengths of
        each side of the lattice are :math:`a`, :math:`b`, & :math:`c`, the angles
        between the sides are :math:`\alpha`, :math:`\beta`, & :math:`\gamma`

    mode
        *Diffractometer* *geometry* operation mode for
        :meth:`~hklpy2.diffract.DiffractometerBase.forward()`.

        A *mode* (implemented by a |solver|), defines which axes will be
        modified by the
        :meth:`~hklpy2.diffract.DiffractometerBase.forward()` computation.

    monochromatic
        Radiation of a single wavelength (or sufficiently narrow
        range), such that it may be characterized by a single floating point value.

    OR
        Orienting Reflection, a *reflection* used to define the *sample*
        *orientation* (and compute the :math:`UB` matrix).

    orientation
        Positioning of a *crystalline* sample's atomic planes
        (identified by a set of *pseudos*) within the laboratory reference
        frame (described by the *reals*).

    polarization analyzer
        A passive crystal mounted on the detector arm whose Bragg angle
        is close to 45° (so the scattering angle 2θ is near 90°),
        suppressing radiation polarized in the diffraction plane.
        Rotation around the beam selects which polarization direction is
        suppressed, allowing the beam polarization state to be probed.
        Treated by |hklpy2| as *auxiliary* positioners; does not affect
        crystallography calculations.  See :doc:`guides/how_polarizer`.

    preset
        .. index:: see: preset; presets

        A value assigned to a constant (read-only) real axis for use
        during ``forward()`` computations, in place of the current motor
        position.  Presets do not move any motor.  Each *mode* stores its
        own independent preset dictionary, defaulting to ``{}``.

        .. seealso:: :ref:`concepts.presets`, :ref:`how_presets`

    pseudo
        Reciprocal-space axis, such as :math:`h`, :math:`k`, and :math:`l`.
        The engineering units (rarely examined for *crystalline* work) are reciprocal
        of the *wavelength* units.

        Note: **pseudo** axes are **virtual** axes, computed by the solver from
        **real** axes.

    real
        Real-space axis (typically a rotation stage),
        such as ``omega`` (:math:`\omega`).
        The engineering units are expected to be in **degrees**.

    reflection
        User-identified coordinates serving as a fiducial reference
        associating crystal orientation (reciprocal space, *pseudos*) and rotational
        axes (real space, *reals*). Reflections are used to orient a *sample* with a
        specific *diffractometer* geometry. In |hklpy2|, a reflection has a name, a
        set of *pseudos*, a set of *reals*, and a *wavelength*.

    sample
        The named substance to be explored with the *diffractometer*.
        In |hklpy2|, a sample has a name, a *lattice*, and a list of *reflections*.

        The *axes* in a sample's *reflections* are specific to the *diffractometer*
        *geometry*.

        Consequently, the sample is defined for a specific |solver| and
        *geometry*.  The same sample cannot be used for other geometries.

    solver
        The |hklpy2| interface layer to a backend |solver| library, such as
        |libhkl|. The library provides computations to transform coordinates between
        *pseudo* and *real* axes for a defined *diffractometer* *geometry*.  The
        library also provides one or more diffractometer geometries.

    U
    UB
        Orientation matrix (3 x 3).

        :math:`U` Orientation matrix
          of the *crystal* *lattice* as mounted on the *diffractometer* *sample* holder.

        :math:`B` Transition matrix
          of a non-orthonormal (the reciprocal of the crystal) in an orthonormal system.

        :math:`UB` Orientation matrix
          of the *crystal* *lattice* in the laboratory reference frame.

    unit cell
        The parallelepiped representing the smallest repeating structural
        pattern of the *crystal*, characterized by its *lattice* parameters.

    virtual
        Virtual (computed) diffractometer *axis* (either *pseudo* or
        *real*), computed from one or more additional diffractometer axes.

    wavelength
        The numerical value of the wavelength of the incident radiation.
        The radiation is expected to be *monochromatic* neutrons or X-rays.
        The engineering units of wavelength must be identical to those of the
        *crystalline* *lattice* length parameters.

    zone
        A set of *lattice* planes which are all parallel to one line, called the
        *zone axis*.
