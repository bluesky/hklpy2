(v2_checklist)=
# Feature Checklist for v2

This is a first-cut checklist for the v2 release.
It could be re-organized.

* [ ] community wish list
  * [ ] user-requested changes
  * [x] move libhkl to be a replaceable back-end computation library
  * [x] easy to save/restore configuration
  * [x] easy to use different *engines*
    * [x] documentation
    * [x] example(s)
    * [x] tests
  * Axes
    * [x] Make it easy to provide additional axes
      * [x] rotation about arbitrary vector
      * [ ] scan along a crystallographic *zone*
      * [x] Solvers with different reciprocal-space axes
      * [x] extra parameters, as required by solver
    * [x] user can choose different names for any of the diffractometer axes
  * [x] Default diffractometer geometries
  * [-] Bragg Peak optimization tools
  * [x] Defining orientation matrix or matrices (set UB)
  * [x] Simulating diffraction and diffractometer modes
  * [x] Built in reciprocal space plans (or scans)
  * [x] Choice of calculation engines other than the hkl C package
  * Solver API
    * [x] a structure (dict or struct) describing a geometry (motors, reference positions, and constraints)
    * [x] observed mapping between real and reciprocal space to give you the "U" of the UB matrix
    * [ ] the crystallography to give you the "B"
    * [x] Solver: custom project
    * [x] Solver: python-wrapped components from `libhkl`
    * [ ] Solver: SPEC
    * [ ] easy to switch between solvers at run time so that new things can be validated
  * [ ] [analyzers and polarizers](https://github.com/bluesky/hklpy/issues/92)
  * Reflections
    * [x] [reflection is a Python class](https://github.com/bluesky/hklpy/issues/189)
    * [x] [`addReflection()`, when to use current positions](https://github.com/bluesky/hklpy/issues/219)
    * [x] [avoid duplications](https://github.com/bluesky/hklpy/issues/248)
    * [x] [label each reflection](https://github.com/bluesky/hklpy/issues/293)
    * [x] [write orientation reflections with scan](https://github.com/bluesky/hklpy/issues/158),
      [also](https://github.com/bluesky/hklpy/issues/247)
    * [x] [`cahkl()` should make nice report when reflection can't be reached](https://github.com/bluesky/hklpy/issues/178)
  * Other
    * [x] [modify existing sample](https://github.com/bluesky/hklpy/issues/157)
    * [x] [control display precision in `wh()` and `pa()`](https://github.com/bluesky/hklpy/issues/179)
    * [ ] [crystallographic *zones*](https://github.com/bluesky/hklpy/issues/291)
* [ ] Solvers
  * [x] `libhkl`
  * [ ] [*ad-hoc* geometries](https://github.com/bluesky/hklpy/issues/244)
  * [ ] [diffcalc](https://github.com/bluesky/hklpy/issues/163)
  * [ ] [SPEC server](https://certif.com/spec_help/server.html)
  * [x] [TwoC unknown](https://github.com/bluesky/hklpy/issues/165)
  * [ ] [xrayutilities](https://github.com/bluesky/hklpy/issues/162)
* [ ] Documentation
  * [ ] Choosing the default `forward()` solution.
  * [x] documentation from hklpy.
  * [x] How to calculate UB from 2 reflections.
  * [ ] How to hold axes fixed during `forward()` transformation.
  * [ ] How to migrate from hklpy v1.
  * [x] How to refine lattice from 3 reflections.
  * [x] Save & restore diffractometer configuration (includes orientation).
  * [x] How to set UB directly.
* [x] Diffractometer-Core API
  * [x] solver:
  * [x] geometry:
  * [x] samples:
  * [x] lattice:
  * [x] reflections:
  * [x] configuration:
    * [x] export
    * [x] restore
  * [x] axes
    * pseudos
      * [x] dict
      * [x] list
      * [x] tuple
      * [x] PseudoPosition
    * reals
      * [x] dict
      * [x] list
      * [x] tuple
      * [x] RealPosition
  * [x] transformations
    * [x] forward (pseudos -> reals)
    * [x] inverse (reals -> pseudos)
  * user friendliness
    * [x] features from hklpy
  * [x] position targets
    * [x] dicts
    * [x] lists
    * [-] numpy arrays
    * [x] scalars
    * [x] tuples
* [x] Core-Solver interface transactions API
  * [x] axes : assignment : automatic
  * [x] axes : assignment : named
  * [x] axes : convert names between diffractometer and solver
  * [x] axes : expected pseudos
  * [x] axes : expected reals
  * [x] constraints
  * [x] geometry : get
  * [x] geometry : set
  * [x] orientation : calculate UB from 2 reflections
  * [x] orientation : export
  * [x] orientation : restore
  * [x] reflection : add
  * [x] sample : add
  * [x] sample : get
  * [x] sample : list all
  * [x] sample : remove
  * [x] solver : get
  * [x] solver : set
  * [x] transformation : forward (pseudos -> reals)
  * [x] transformation : inverse (reals -> pseudos)
* [ ] Performance
  * [ ] minimum 2,000 `inverse()` operations/second
  * [ ] minimum 2,000 `forward()` operations/second
* [ ] Backends - feature support
  * [ ] *ad hoc* geometries
  * [ ] diffcalc_dls
  * [x] hkl_soleil (libhkl)
    * [x] axes : expected pseudos
    * [x] axes : expected reals
    * [x] extra parameters
    * [x] geometry : list all available geometries
    * [x] geometry : set
    * [x] list available solvers
    * [x] mode : list all available modes
    * [x] mode : set
    * [x] orientation : calculate UB from 2 reflections
    * [!] orientation : return B matrix (not provided by hkl_soleil)
    * [x] orientation : return U matrix
    * [x] orientation : return UB matrix
    * [!] orientation : set B matrix (not provided by hkl_soleil)
    * [x] orientation : set U matrix
    * [x] orientation : set UB matrix
    * [x] reflection : add
    * [x] reflection : remove all
    * [x] sample : add
    * [x] sample lattice : add
    * [x] sample lattice : refine from >2 reflections
    * [x] transformation : forward (pseudos -> reals)
    * [x] transformation : inverse (reals -> pseudos)
  * [x] no_op
  * [ ] SPEC server
  * [x] th_tth
    * axes : expected
      * [x] pseudos
      * [x] reals
    * geometry
      * [x] list all available geometries
      * [x] set
    * mode
      * [x] set
      * [x] list available
    * [x] orientation : calculate (does not apply, empty [] as result)
    * [x] sample : add
    * [x] sample lattice
      * [x] add
      * [!] refine (does not apply)
    * [x] reflection
      * [x] add
      * [x]  remove all
    * transformation
      * [x] forward (pseudos -> reals)
      * [x] inverse (reals -> pseudos)
    * [x] wavelength : set
  * [ ] xrayutilities
