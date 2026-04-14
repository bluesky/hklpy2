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

.. _overview.architecture:

Package Architecture
--------------------

The diagram below shows how the major components of |hklpy2| fit together,
grouped by their role in the package.

.. graphviz::
    :caption: hklpy2 package architecture -- components grouped by layer.
    :align: center

    digraph hklpy2_architecture {
        graph [rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6,
               fontname="sans-serif", bgcolor="transparent", compound=true]
        node  [shape=box, style="rounded,filled", fontname="sans-serif",
               fontsize=10, margin="0.10,0.05"]
        edge  [fontname="sans-serif", fontsize=9, dir=none]

        subgraph cluster_bluesky {
            label="Bluesky\n(data acquisition)"
            style=dashed
            color="#888888"
            fontname="sans-serif"
            fontsize=10

            bluesky  [label="bluesky plans\n(RE, bps.mv, ...)",
                      fillcolor="#e0e0e0", color="#666666"]
        }

        subgraph cluster_user {
            label="User-facing"
            style=filled
            fillcolor="#f0f4ff"
            color="#3a6898"
            fontname="sans-serif"
            fontsize=10

            creator    [label="creator()\nfactory function",
                        fillcolor="#dce8f8", color="#3a6898"]
            usermod    [label="hklpy2.user\n(pa, wh, cahkl, setor, ...)",
                        fillcolor="#dce8f8", color="#3a6898"]
            diffract   [label="DiffractometerBase\n(ophyd PseudoPositioner)",
                        fillcolor="#dce8f8", color="#3a6898"]
            wavelength [label="Wavelength / beam\n(ophyd.Device)",
                        fillcolor="#dce8f8", color="#3a6898"]
        }

        subgraph cluster_core {
            label="Core  (diffractometer.core)"
            style=filled
            fillcolor="#fffbe8"
            color="#a07820"
            fontname="sans-serif"
            fontsize=10

            core [label="Core",
                  fillcolor="#fdf3dc", color="#a07820"]

            subgraph cluster_blocks {
                label="blocks"
                style=filled
                fillcolor="#fff8e0"
                color="#c09030"
                fontname="sans-serif"
                fontsize=9

                sample      [label="Sample\nLattice",
                             fillcolor="#fdf3dc", color="#a07820"]
                reflection  [label="Reflection",
                             fillcolor="#fdf3dc", color="#a07820"]
                constraints [label="Constraints\ncut_point",
                             fillcolor="#fdf3dc", color="#a07820"]
                presets     [label="Presets",
                             fillcolor="#fdf3dc", color="#a07820"]
                zone        [label="Zone",
                             fillcolor="#fdf3dc", color="#a07820"]
                configure   [label="Configuration",
                             fillcolor="#fdf3dc", color="#a07820"]

                { rank=same; sample; reflection; constraints }
                { rank=same; presets; zone; configure }
            }
        }

        subgraph cluster_solver {
            label="Solver adapter"
            style=filled
            fillcolor="#f4f0ff"
            color="#6a3a98"
            fontname="sans-serif"
            fontsize=10

            solverbase   [label="SolverBase\n(adapter interface)",
                          fillcolor="#e8e0f8", color="#6a3a98"]
            hklsolver    [label="HklSolver\n(libhkl adapter)",
                          fillcolor="#e8e0f8", color="#6a3a98"]
            thttthsolver [label="ThTthSolver",
                          fillcolor="#e8e0f8", color="#6a3a98"]
            noopsolver   [label="NoOpSolver",
                          fillcolor="#e8e0f8", color="#6a3a98"]
        }

        subgraph cluster_epics {
            label="EPICS\n(hardware controls)"
            style=dashed
            color="#888888"
            fontname="sans-serif"
            fontsize=10

            epics    [label="EPICS\nmotor axes",
                      fillcolor="#e0e0e0", color="#666666"]
        }

        subgraph cluster_backend {
            label="Backends\n(via entry points)"
            style=dashed
            color="#888888"
            fontname="sans-serif"
            fontsize=10

            libhkl   [label="libhkl",
                      fillcolor="#e0e0e0", color="#666666"]
            otherbk  [label="other backend\nlibraries ...",
                      fillcolor="#e0e0e0", color="#666666",
                      style="rounded,filled,dashed"]
        }

        bluesky    -> diffract
        epics      -> diffract   [style=dashed, color="#888888"]
        epics      -> wavelength [label="EPICS PVs\n(optional)",
                                  style=dashed, color="#888888"]
        creator    -> diffract   [xlabel="creates"]
        usermod    -> diffract
        diffract   -> wavelength [style=dashed, label=".beam"]

        diffract   -> core       [label=".core"]
        wavelength -> core       [xlabel="beam"]
        core       -> sample
        core       -> reflection
        core       -> constraints
        core       -> presets
        core       -> zone
        core       -> configure

        core       -> solverbase   [label="calculations"]
        solverbase -> hklsolver    [style=dashed, label="subclass"]
        solverbase -> thttthsolver [style=dashed]
        solverbase -> noopsolver   [style=dashed]
        hklsolver  -> libhkl
        solverbase -> otherbk      [style=dashed]
    }

.. seealso:: :ref:`glossary`
