.. _concepts.migration_from_hklpy_v1:

=======================
Migration from |hklpy|
=======================

|hklpy2| is a complete redesign of the |hklpy| library with significant
architectural changes. The main difference is that |hklpy2| moves from a design
centered on |libhkl| to a Python-native data model with pluggable solvers.

For an AI-assisted description of the factors involved in migration, see
`DeepWiki <https://deepwiki.com/search/how-to-migrate-to-hklpy2-from_5bab86c6-f617-46c0-bdf0-79c7c62a3640?mode=fast>`_.

.. caution:: Be careful with the code examples presented by the AI-assisted
    description.  Often, they are not correct technically.

Simulated Eulerian 4-circle diffractometer
------------------------------------------

.. tabs::

    .. tab:: |hklpy2|

        .. code-block:: python
            :linenos:

            from hklpy2 import creator

            fourc = creator()

    .. tab:: |hklpy|

        .. code-block:: python
            :linenos:

            from hkl import E4CV
            from ophyd import PseudoSingle, SoftPositioner
            from ophyd import Component as Cpt

            class FourCircle(SimMixin, E4CV):
                """
                Our 4-circle.  Eulerian, vertical scattering orientation.
                """
                h = Cpt(PseudoSingle, "", kind="hinted")
                k = Cpt(PseudoSingle, "", kind="hinted")
                l = Cpt(PseudoSingle, "", kind="hinted")

                omega = Cpt(SoftPositioner, kind="hinted", init_pos=0)
                chi = Cpt(SoftPositioner, kind="hinted", init_pos=0)
                phi = Cpt(SoftPositioner, kind="hinted", init_pos=0)
                tth = Cpt(SoftPositioner, kind="hinted", init_pos=0)

            fourc = FourCircle("", name="fourc")

EPICS-based Eulerian 4-circle diffractometer
--------------------------------------------

.. tabs::

    .. tab:: |hklpy2|

        .. code-block:: python
            :linenos:

            from hklpy2 import creator

            fourc = creator(
                prefix="PREFIX:",
                name="fourc",
                geometry="E4CV",
                solver="hkl_soleil",
                reals=dict(
                    omega="m30",
                    chi="m31",
                    phi="m32",
                    tth="m29",
                )
            )

    .. tab:: |hklpy|

        .. code-block:: python
            :linenos:

            from hkl import E4CV
            from ophyd import EpicsMotor, PseudoSingle
            from ophyd import Component as Cpt

            class FourCircle(E4CV):
                """
                Our 4-circle.  Eulerian, vertical scattering orientation.
                """
                # the reciprocal axes h, k, l are defined by SimMixin
                h = Cpt(PseudoSingle, "", kind="hinted")
                k = Cpt(PseudoSingle, "", kind="hinted")
                l = Cpt(PseudoSingle, "", kind="hinted")

                omega = Cpt(EpicsMotor, "m30", kind="hinted")
                chi = Cpt(EpicsMotor, "m31", kind="hinted")
                phi = Cpt(EpicsMotor, "m32", kind="hinted")
                tth = Cpt(EpicsMotor, "m29", kind="hinted")

            fourc = FourCircle("PREFIX:", name="fourc")
