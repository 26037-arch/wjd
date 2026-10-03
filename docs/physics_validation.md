# Local G-code physics experiment

`tools/run_physics_validation.py` adds an external `physics_cfd` adapter. It
does not replace the internal toolpath checks or VOLCO. The adapter accepts
the **generated** G-code, reads its `CONICAL_META`, and gives the XYZ/E moves
to [pyGCodeDecode](https://github.com/FAST-LB/pyGCodeDecode) 1.5.1. Its
planner models acceleration and junction behavior using a named printer
preset. The default `prusa_mini` preset is an **unverified surrogate** for
this repository's target machine. Set a measured preset before interpreting
times as real printer times. The adapter tracks B/C and temperatures as
metadata, but neither B/C nor thermal coupling is applied in OpenFOAM.

## Existing and new flow

```text
STL → conical_slice.py / conical.pipeline → G-code
                                        ├→ internal parser/toolpath check
                                        ├→ REP5X/Open5x machine checks
                                        ├→ VOLCO voxel/STL reconstruction → whole-model metrics
                                        └→ pyGCodeDecode → short window → OpenFOAM 14
                                             → alpha.polymer field → local STL and metrics
```

`conical.gcode` preserves raw non-movement commands; its parser is not a
motion planner. `conical.planar_slicer.FILAMENT_D` supplies the 1.75 mm
filament setting used by the built-in slicer. The generated `CONICAL_META`
supplies mode, nominal width, and layer height. The physics adapter does not
call VOLCO and shares no deposition algorithm with it.

## Scope and equations

The OpenFOAM Foundation **version 14** `incompressibleVoF` solver is used for
an incompressible polymer/air volume-of-fluid calculation. It solves phase
fraction advection, mass conservation, and momentum including gravity. The
installed `generalisedNewtonian` / `CrossPowerLaw` viscosity model uses
shear rate. The local G-code's positive extrusion increments become
`volumeSource` Function1 tables (m³/s) at successive spatial cells; the `U`
field sources carry lateral nozzle speed plus estimated nozzle-exit speed.
This is a *discrete fixed-grid moving source*: nozzle geometry and flow inside
the nozzle are not resolved. The first-layer case uses the build plate as a
no-slip solid wall. Later layers are rejected with `SUBSTRATE_UNAVAILABLE`
until a separately generated previous-material substrate is implemented.

OpenFOAM 14's installed model list includes `CrossPowerLaw` but not
`Cross-WLF`. The current case is isothermal. `M104/M109/M140/M190` setpoints
are recorded in the trajectory when present; they are not a solved
temperature field. Surface tension is disabled in the literature surrogate,
and solidification and viscoelasticity are absent. A completed run has
`CFD_COMPLETED_UNCALIBRATED` status, never `PASS`, when the filament and
machine are not calibrated. The default `materials/pla.example.yaml` has
missing values and returns `MATERIAL_PROFILE_INCOMPLETE`.

The numerical surrogate in
`materials/pla_4032d_literature_surrogate.yaml` illustrates execution; its
properties are literature references from different sources, not measured
for the user's filament. The zero surface tension entry explicitly removes
that force and is not presented as a PLA material property.

## Reproduce a bounded first-layer case

Install `requirements-physics.txt` in a Python environment and source
`/opt/openfoam14/etc/bashrc` inside WSL2. Run:

```text
python tools/run_physics_validation.py --gcode output.gcode --stl input.stl \
  --material materials/pla_4032d_literature_surrogate.yaml \
  --window layer:0 --max-path-mm 1 --output-dir physics --execute
```

`--window` also accepts `auto`, `slope`, `z`, `support`, `moves:A-B`, and
`lines:A-B`. `support` is only a geometric height/slope risk heuristic;
it does not import the internal support checker's verdict. `auto` may select
a later-layer segment and then report `SUBSTRATE_UNAVAILABLE`. The generated
`result.json` contains the exact `solver_command`, with a WSL path on
Windows. With `--execute`, the runner calls `blockMesh`, `foamRun`, and
`foamToVTK`, then `finalize_physics` reads the actual VTK field, extracts an
`alpha=0.5` surface, and writes calculated local metrics. A
failed or absent solver run never produces a predicted STL.

For REP5X, B/C angles are retained in the trajectory but CFD returns
`UNSUPPORTED_REP5X_ORIENTATION`; the result is not a 5-axis physical
validation. For Open5x, the moving bed is likewise not modeled.

## Metric meaning

`deposited_volume_mm3` integrates `alpha.polymer` over cells. The STL is an
iso-surface of a coarse grid and its enclosed mesh volume can differ from
the integrated field volume. `bead_center_deviation_mm` compares full-window
material and commanded-extrusion centroids. `cross_section_width_mm` and
`cross_section_height_mm` are extents of the local iso-surface, so the latter
is sensitive to its length and cell size. A bed-supported first layer has no
meaningful unsupported-span sag, so `max_sag_mm` is `null`. Comparing either
local reconstruction to the *whole* target STL would produce misleading
Chamfer/Hausdorff and relative-volume errors; those remain `null` in
`comparison.json`.

Sources: [pyGCodeDecode paper](https://joss.theoj.org/papers/10.21105/joss.06465.pdf),
[OpenFOAM Foundation](https://openfoam.org/download/),
[PLA rheology study](https://onlinelibrary.wiley.com/doi/10.1002/mame.202500204),
[PLA properties reference](https://www.mdpi.com/2305-7084/7/1/1).
