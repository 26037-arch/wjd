# REP5X deposition reconstruction

`tools/run_deposition_sim.py` creates a time ordered voxel reconstruction from
G-code, with XYZ interpreted as the RTCP nozzle tip and orientation from
`conical.rep5x.tool_direction(B, C)`. This is a spherical voxel deposition
model. It does not solve melt flow, heat transfer, cooling, or collisions.

```powershell
python tools/run_deposition_sim.py input.gcode --backend volco-oriented `
  --volco-dir ..\volco --voxel-size 0.025 `
  --output-dir results\deposition\input
```

The output contains `manifest.json`, `metrics.json`, `events.json`,
`frames.jsonl`, `final.stl`, and `simulation.log`. `frames.jsonl` stores only
new voxel indices per sample, so playback accumulates geometry without a full
mesh copy per frame. `events.json` retains every commanded XYZ/B/C move and the
recorded C trajectory, including rewind moves. The shared time interpolation
is linear in each G-code motion. It estimates motion duration from F and path
length, and treats pure rotation F as degrees/minute. It is not a firmware
acceleration planner. Delta E at or below zero adds no material.

The oriented extension follows VOLCO's sphere center offset and cumulative
voxel volume fitting. It rotates the sphere center and nozzle clipping plane
with B/C and uses the closest eligible empty grid centers to meet cumulative
commanded volume to within one voxel. The build plate remains horizontal at
Z=0. `manifest.json` records the external VOLCO commit when supplied, but the
external checkout is not changed. The old `validate_volco()` still runs
upstream VOLCO with B/C removed and marks orientation unmodeled. Its aligned
STL now corrects the upstream half voxel export shift in XYZ in addition to
removing XY padding.

The first layer convergence experiment found adjacent grid IoU of 89.5%,
94.0%, and 97.1% for 0.10→0.05, 0.05→0.025, and 0.025→0.0125 mm. Use 0.025 mm
for analysis and 0.0125 mm as a convergence check on a bounded segment. These
are numerical self comparisons, not accuracy against a printed object. Volume
agreement can mask simultaneous local excess and missing material. The
separate path step defaults to voxel size/4; fixed step comparisons should use
the same step value. A 0.0125 mm upstream run on just four segments took about
seven minutes, beyond the historical validator's 300 s timeout.
For the historical Cartesian validator, `tools/run_random_validation.py`
accepts `--volco-timeout` to raise that bound explicitly.
For two oriented runs using the same G-code and step size,
`python tools/compare_deposition_resolution.py coarse_dir fine_dir`
reports exact common grid overlap and separate coarse excess/missing volumes.
For a synthetic REP5X path with B=0→20° and C=0→90°, the oriented extension
at fixed 0.003125 mm sample step gave 0.025↔0.0125 mm IoU 0.9489, with
0.002273 mm³ coarse excess and 0.002270 mm³ coarse missing material. The
0.0125 mm run took 168 seconds and had 44,334 occupied voxels. This differs
from the original Cartesian VOLCO experiment, so neither convergence number
should be transferred to the other model or treated as physical accuracy.

## Browser playback and calibrated CAD

Open `tools/slicing_simulator.html`, select **VOLCO oriented 예측 + REP5X 3D**,
and choose `manifest.json`, `events.json`, and `frames.jsonl` together.
Playback uses the event time for nozzle XYZ, B/C, and cumulative voxel count.
The existing Plotly G-code line view remains selectable. Load the original STL
separately when needed.

The Three.js scene has hierarchy:

```text
static frame / fixed build plate
  carriage translated for RTCP XYZ
    C joint (local +Z yaw)
      B joint (local +Y tilt)
        hotend and nozzle tip
```

It shows only the nozzle axis until actual CAD meshes and measured rigid
transforms are supplied; the state is `UNCALIBRATED`. No generic cylinders are
called machine CAD. `profiles/rep5x_cad.example.json` is a template with
`null` transforms. It requires row-major 4×4 matrices `H_T_C`, `C_T_B`, and
`B_T_N`, each measured in mm from the corresponding home frame to the next
frame. All part meshes must be exported to STL in one common home coordinate
system. The renderer applies the calibrated transforms, rotates local C about
+Z and B about +Y, computes carriage translation to keep the transformed tip
at commanded XYZ, and reports the RTCP residual. The firmware defaults
recorded in `conical/rep5x.py` are LB=54.67 mm and LC=1.6 mm, but these do not
by themselves determine the signed 3D CAD pivot coordinates. The assembly
order follows the [public REP5X build guide](https://github.com/dennisklappe/Rep5x/blob/main/build-guide/assembly-instructions-universal.md),
and public CAD is in the [REP5X parts directory](https://github.com/dennisklappe/Rep5x/tree/main/build-guide/universal-parts/3d-printed-parts/current).
The CAD source and printer specific mount must be calibrated before the scene
can claim an actual machine assembly. Manual X/Y/Z/B/C input is available;
soft limit violations are shown as warnings.

## CFD boundary

The existing local OpenFOAM adapter in `tools/external/physics_cfd.py` accepts
Cartesian first layer input only. Its generated `volumeSource` now selects a
nozzle diameter area of cells and reports the count, following the controlled
source experiment. That experiment recovered commanded volume with distributed
injection, but its surface shape did not converge. The hybrid CLI mode returns
`NOT_AVAILABLE` and `cfd_applied=false`; it does not add an analytic bead or
label an unrun CFD case as corrected geometry. No OpenFOAM solver was run by
`run_deposition_sim.py`.

