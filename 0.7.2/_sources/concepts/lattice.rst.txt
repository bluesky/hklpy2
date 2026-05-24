.. _concepts.lattice:

==================
Crystal Lattice
==================

.. index:: !lattice

A crystal lattice describes the periodic arrangement of atoms in a crystal.
Its geometry is characterised by six parameters: three unit-cell edge lengths
:math:`a, b, c` (in ångströms) and three inter-edge angles
:math:`\alpha, \beta, \gamma` (in degrees).

The lattice matters to diffractometer operation for two reasons:

1. **UB matrix** — the :math:`B` matrix is computed directly from the lattice
   parameters and transforms Miller indices :math:`(h, k, l)` into Cartesian
   reciprocal-space coordinates.  Without accurate lattice parameters the
   :math:`UB` matrix, and therefore every
   :meth:`~hklpy2.diffract.DiffractometerBase.forward` and
   :meth:`~hklpy2.diffract.DiffractometerBase.inverse` calculation, will be
   wrong.

2. **Crystal symmetry** — the crystal system (cubic, hexagonal, …) constrains
   which lattice parameters are independent.  |hklpy2| determines the crystal
   system from the supplied parameters automatically and reports it; if the
   parameters are inconsistent with any known system an exception is raised.

In |hklpy2|, lattice parameters are stored in an instance of
:class:`~hklpy2.blocks.lattice.Lattice`, which is part of the
:ref:`sample <concepts.sample>`.

.. seealso::

   :ref:`concepts.sample` — the sample that owns the lattice.

   :ref:`how_calc_ub` — computing the UB matrix from the lattice and orientation reflections.

   :ref:`glossary` — definitions of lattice, UB matrix, and related terms.

.. _lattice.crystal-systems:

.. rubric:: The Seven 3-D Crystal Systems (highest to lowest symmetry)

.. index:: crystal system

=============== =================================== = === === ===== ====== =====
system          command                             a b   c   alpha beta   gamma
=============== =================================== = === === ===== ====== =====
cubic           ``Lattice(5.)``                     5 *5* *5* *90*  *90*   *90*
hexagonal       ``Lattice(4., c=3., gamma=120)``    4 *4* 3   *90*  *90*   120
rhombohedral    ``Lattice(4., alpha=80.2)``         4 *4* *4* 80.2  *80.2* *80.2*
tetragonal      ``Lattice(4, c=3)``                 4 *4* 3   *90*  *90*   *90*
orthorhombic    ``Lattice(4, 5, 3)``                4 5   3   *90*  *90*   *90*
monoclinic      ``Lattice(4, 5, 3, beta=75)``       4 5   3   *90*  75     *90*
triclinic       ``Lattice(4, 5, 3, 75., 85., 95.)`` 4 5   3   75    85     95
=============== =================================== = === === ===== ====== =====

Default values are *italicized*.  It is not necessary to supply the default
parameters.

.. rubric:: Examples

.. code-block:: Python
    :linenos:

    >>> from hklpy2 import Lattice
    >>> Lattice(5.)
    Lattice(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)

    >>> Lattice(4., c=3., gamma=120)
    Lattice(a=4.0, b=4.0, c=3.0, alpha=90.0, beta=90.0, gamma=120)

    >>> Lattice(4., alpha=80.2)
    Lattice(a=4.0, b=4.0, c=4.0, alpha=80.2, beta=80.2, gamma=80.2)

    >>> Lattice(4, c=3)
    Lattice(a=4, b=4, c=3, alpha=90.0, beta=90.0, gamma=90.0)

    >>> Lattice(4, 5, 3)
    Lattice(a=4, b=5, c=3, alpha=90.0, beta=90.0, gamma=90.0)

    >>> Lattice(4, 5, 3, beta=75)
    Lattice(a=4, b=5, c=3, alpha=90.0, beta=75, gamma=90.0)

    >>> Lattice(4, 5, 3, 75., 85., 95.)
    Lattice(a=4, b=5, c=3, alpha=75.0, beta=85.0, gamma=95.0)

.. seealso::

   :ref:`lattice.crystal-systems` is also cross-referenced from :ref:`geometries`.

   `IUCr: Crystal system <https://dictionary.iucr.org/Crystal_system>`_

   `Wikipedia: Crystal system <https://en.wikipedia.org/wiki/Crystal_system>`_
