(guide.design)=
# Architecture & Design Decisions

This page describes the architectural decisions behind *hklpy2* — why the
package is structured the way it is, and where it is headed.

## Package Architecture

![Package architecture overview](/_static/hklpy2-overview.svg)

The package is organised into five sections, flowing left to right:

- **External (left)** — Bluesky plans (RunEngine,
  [`bps.mv`](https://blueskyproject.io/bluesky/main/generated/bluesky.plan_stubs.mv.html),
  …) drive the diffractometer; EPICS provides real motor axes and optional
  PV-based wavelength control.  These are external to *hklpy2* and shown
  with dashed borders.
- **User-facing** — {class}`~hklpy2.diffract.DiffractometerBase` (an ophyd
  {class}`~ophyd.pseudopos.PseudoPositioner`),
  {class}`~hklpy2.wavelength.WavelengthBase`, the
  {func}`~hklpy2.misc.creator` factory, and the
  {mod}`hklpy2.user` convenience functions
  ({func}`~hklpy2.user.pa`, {func}`~hklpy2.user.wh`,
  {func}`~hklpy2.user.cahkl`, {func}`~hklpy2.user.setor`, …).
- **Core** — {class}`~hklpy2.ops.Core` manages the seven block classes
  ({class}`~hklpy2.blocks.sample.Sample`,
  {class}`~hklpy2.blocks.lattice.Lattice`,
  {class}`~hklpy2.blocks.reflection.Reflection`,
  {class}`~hklpy2.blocks.constraints.ConstraintBase`,
  Presets,
  {class}`~hklpy2.blocks.zone.OrthonormalZone`,
  {class}`~hklpy2.blocks.configure.Configuration`)
  and acts as the single point of contact between the diffractometer and its
  solver.
- **Solvers** — {class}`~hklpy2.backends.base.SolverBase` defines the
  adapter interface; built-in solvers
  ({class}`~hklpy2.backends.hkl_soleil.HklSolver`,
  {class}`~hklpy2.backends.th_tth_q.ThTthSolver`,
  {class}`~hklpy2.backends.no_op.NoOpSolver`) and additional solvers
  registered via Python entry points all subclass it.
- **Backend libraries (right)** — the third-party libraries that solvers
  delegate the heavy mathematics to (e.g. [Hkl/Soleil](https://people.debian.org/~picca/hkl/hkl.html)).  Like the external
  section on the left, these are outside *hklpy2* and shown with dashed
  borders.

See {ref}`overview.architecture` for detailed diagrams of each layer.

## Design Goals

### Why a new package?

Two specific user requests made it clear that incrementally patching *hklpy*
v1 was not viable.

The first was **named reflections**.  In v1, reflection storage was delegated
entirely to the backend library (Hkl/Soleil).  Adding user-visible names to
reflections would have required a deep refactor of every layer that touched
the backend — a change so invasive it would have broken the existing API
throughout.

The second was **wavelength handling**.  A review of how v1 managed (and
failed to manage) wavelength revealed that the tight coupling to Hkl/Soleil
made it impossible to support replaceable solver backends without rebuilding
the package from scratch.  Wavelength is a property of the beam, not of the
solver, and v1 had no clean way to express that separation.

Together these two issues confirmed that the goal of *replaceable solvers*
— the central design requirement of v2 — could not be achieved by refactoring
v1.  A new code base, with reflection management and wavelength control owned
by Python rather than delegated to the backend, was the only path forward.

### What changed

The redesign from *hklpy* v1 to *hklpy2* v2 addressed these shortcomings:

| hklpy v1 | hklpy2 v2 |
|---|---|
| Depends on **[Hkl/Soleil](https://people.debian.org/~picca/hkl/hkl.html)** (Linux x86-64 only) | Hkl/Soleil is one optional backend solver |
| All samples, lattices, reflections stored inside Hkl/Soleil | Samples, lattices, reflections stored in Python |
| Multiple confusing layers mirroring Hkl/Soleil internals | Two clear layers: Core and Solver |
| Difficult to add geometries or swap backends | Solvers are plugins, swappable at runtime |
| Difficult to use additional axes or parameters | Extra axes and parameters are first-class |
| No standard save/restore | Configuration block handles save/restore |

## Coordinate Systems

*hklpy2* works in two coordinate spaces simultaneously:

**Real space** — the physical rotation angles of the diffractometer motors
(e.g. `omega`, `chi`, `phi`, `tth`).  These are the *real* axes.  Their
names and order are defined by the diffractometer geometry provided by the
solver.

**Reciprocal space** — the crystallographic Miller indices {math}`(h, k, l)`
that identify planes in the crystal lattice.  These are the *pseudo* axes.
While Miller indices are conventionally integers, the pseudo axes in
*hklpy2* are floating-point values, allowing positions between Bragg peaks
(e.g. for continuous scanning along a reciprocal-space trajectory, diffuse
scattering, or incommensurate structures).
{class}`~hklpy2.diffract.DiffractometerBase` converts between the two spaces
via {meth}`~hklpy2.diffract.DiffractometerBase.forward` (hkl → angles) and
{meth}`~hklpy2.diffract.DiffractometerBase.inverse` (angles → hkl), delegating
the mathematics to Core and then to the solver.

### The UB matrix

The conversion between the two spaces requires knowing both the crystal
geometry and its orientation on the diffractometer.  This is encoded in the
{math}`UB` orientation matrix, which is the product of two matrices:

- {math}`B` — encodes the crystal lattice geometry (lengths and angles of
  the unit cell).  It transforms the sample's Miller indices {math}`(h, k,
  l)` into the diffractometer's reference frame (technically, to an
  orthonormal Cartesian basis aligned with the crystal axes).
- {math}`U` — encodes how the crystal is physically mounted on the
  diffractometer sample holder.  It transforms from the diffractometer's
  reference frame into the laboratory reference frame (technically, a
  rotation matrix from the crystal Cartesian frame to the reciprocal lab
  frame).
- {math}`UB` — the product of {math}`U` and {math}`B`; the single matrix
  used in all {meth}`~hklpy2.diffract.DiffractometerBase.forward` and
  {meth}`~hklpy2.diffract.DiffractometerBase.inverse` calculations to convert
  between Miller indices and diffractometer angles.

{math}`UB` is computed from two or more measured *orientation reflections* —
positions where the diffractometer angles and the corresponding {math}`(h, k,
l)` are both known.  See {func}`~hklpy2.user.calc_UB` and
{meth}`~hklpy2.ops.Core.calc_UB`.

See issue {issue}`192` for an open discussion on reconsidering coordinate
system transformations.

## Solver Plugin Design

A solver is a Python class that subclasses
{class}`~hklpy2.backends.base.SolverBase` and is registered
with the `"hklpy2.solver"` entry-point group in its package metadata.  This
allows solvers to be installed independently and discovered at runtime without
modifying *hklpy2* itself.

```toml
# pyproject.toml of a solver package
[project.entry-points."hklpy2.solver"]
my_solver = "my_package.solver:MySolver"
```

Built-in solvers shipped with *hklpy2*:

| Solver | Backend | Notes |
|---|---|---|
| {class}`~hklpy2.backends.hkl_soleil.HklSolver` (`hkl_soleil`) | [Hkl/Soleil](https://people.debian.org/~picca/hkl/hkl.html) | Linux x86-64 only; many geometry types |
| {class}`~hklpy2.backends.th_tth_q.ThTthSolver` (`th_tth`) | pure Python | Any OS; θ/2θ geometry with *Q* pseudo axis |
| {class}`~hklpy2.backends.no_op.NoOpSolver` (`no_op`) | none | Testing only; no useful geometries |

See {ref}`howto.solvers.write` for how to write and register a new solver.

## Future Plans

The following design-level topics are under active consideration.
Each links to the tracking issue for discussion and status.

**Multiple solvers per diffractometer** ({issue}`187`)
: Allow a single {class}`~hklpy2.diffract.DiffractometerBase` instance to switch between solvers at
  runtime without reconstruction, enabling side-by-side comparison of
  backends.

**Performance targets** ({issue}`221`, {issue}`223`)
: Minimum throughput of 2,000 {meth}`~hklpy2.diffract.DiffractometerBase.forward`
  and {meth}`~hklpy2.diffract.DiffractometerBase.inverse` operations per second.

**Coordinate system reconsideration** ({issue}`192`)
: Review whether the current {math}`UB` convention and axis ordering are the
  best defaults for all supported geometries.

**Analyzers and polarizers** ({issue}`222`)
: Support for additional optical elements as stand-alone ophyd objects that
  coordinate with the diffractometer.

**Fly scanning** ({issue}`11`)
: Built-in reciprocal-space fly-scan plans integrated with the Bluesky
  RunEngine.
