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
    :caption: hklpy2 package architecture -- components grouped by layer (left to right).
    :align: center

    digraph hklpy2_architecture {
        graph [rankdir=LR, splines=ortho, nodesep=0.35, ranksep=0.7,
               fontname="sans-serif", bgcolor="transparent", compound=true]
        node  [shape=box, style="rounded,filled", fontname="sans-serif",
               fontsize=10, margin="0.10,0.05"]
        edge  [fontname="sans-serif", fontsize=9]

        subgraph cluster_bluesky_epics {
            label="Bluesky / EPICS"
            style=dashed
            color="#888888"
            fontname="sans-serif"
            fontsize=10
            rank=source

            bluesky  [label="bluesky plans\n(RE, bps.mv, ...)",
                      fillcolor="#e0e0e0", color="#666666"]
            epics    [label="EPICS / simulated\nmotor axes",
                      fillcolor="#e0e0e0", color="#666666"]
        }

        subgraph cluster_user {
            label="User-facing"
            style=filled
            fillcolor="#f0f4ff"
            color="#3a6898"
            fontname="sans-serif"
            fontsize=10

            diffract   [label="DiffractometerBase\n(ophyd PseudoPositioner)",
                        fillcolor="#dce8f8", color="#3a6898"]
            usermod    [label="hklpy2.user\n(pa, wh, cahkl, setor, ...)",
                        fillcolor="#dce8f8", color="#3a6898"]
            creator    [label="creator()\nfactory function",
                        fillcolor="#dce8f8", color="#3a6898"]
            wavelength [label="Wavelength / beam\n(ophyd Signal or subclass)",
                        fillcolor="#dce8f8", color="#3a6898"]
        }

        subgraph cluster_core {
            label="Core  (diffractometer.core)"
            style=filled
            fillcolor="#fffbe8"
            color="#a07820"
            fontname="sans-serif"
            fontsize=10

            core [label="Core\noperations hub",
                  fillcolor="#fdf3dc", color="#a07820"]

            subgraph cluster_blocks {
                label="blocks"
                style=filled
                fillcolor="#fff8e0"
                color="#c09030"
                fontname="sans-serif"
                fontsize=9

                sample      [label="Sample + Lattice",
                             fillcolor="#fdf3dc", color="#a07820"]
                reflection  [label="Reflection\n(UB calc)",
                             fillcolor="#fdf3dc", color="#a07820"]
                constraints [label="LimitsConstraint\n(cut_point + limits)",
                             fillcolor="#fdf3dc", color="#a07820"]
                presets     [label="Presets\n(constant axes)",
                             fillcolor="#fdf3dc", color="#a07820"]
                zone        [label="OrthonormalZone",
                             fillcolor="#fdf3dc", color="#a07820"]
                configure   [label="Configuration\n(save / restore)",
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

        subgraph cluster_backend {
            label="Backends"
            style=dashed
            color="#888888"
            fontname="sans-serif"
            fontsize=10

            backend  [label="backend library\n(libhkl, ...)",
                      fillcolor="#e0e0e0", color="#666666"]
        }

        // invisible chain enforces left-to-right cluster order:
        // Bluesky/EPICS -> User-facing -> Core -> Solver -> Backends
        bluesky -> diffract -> core -> solverbase -> backend [style=invis, weight=10]

        bluesky  -> diffract   [label="plans"]
        epics    -> diffract   [label="motor axes"]
        epics    -> wavelength [label="EPICS PVs\n(optional)",
                                style=dashed, color="#888888"]
        usermod  -> diffract   [label="convenience\nfunctions"]
        creator  -> diffract   [label="creates"]

        diffract   -> core     [label=".core"]
        wavelength -> core     [label="beam"]
        core -> sample
        core -> reflection
        core -> constraints
        core -> presets
        core -> zone
        core -> configure

        core       -> solverbase   [label="delegates\nforward()/inverse()"]
        solverbase -> hklsolver    [style=dashed, label="subclass"]
        solverbase -> thttthsolver [style=dashed]
        solverbase -> noopsolver   [style=dashed]
        hklsolver  -> backend      [label="calls"]
    }

.. seealso:: :ref:`glossary`
