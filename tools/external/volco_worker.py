"""Run upstream VOLCO in a bounded subprocess, using upstream run_simulation."""

import contextlib
import json
import math
import re
import sys
from pathlib import Path


def main():
    repo, gcode, folder, voxel_size = sys.argv[1:]
    sys.path.insert(0, repo)
    from volco import run_simulation
    folder = Path(folder).resolve()
    # Upstream rounds distance/step_size to an integer and divides by it.
    # Its default fails on short non-planar segments; choose a step below every
    # positive extrusion segment without modifying the upstream algorithm.
    pos = [0.0, 0.0, 0.0, 0.0]
    smallest = math.inf
    for line in Path(gcode).read_text(encoding="utf-8").splitlines():
        if not re.match(r"^G[01]\b", line, re.I):
            continue
        vals = {k.upper(): float(v) for k, v in
                re.findall(r"([XYZEF])([-+]?\d*\.?\d+)", line, re.I)}
        new = [vals.get(k, old) for k, old in zip("XYZE", pos, strict=False)]
        if new[3] > pos[3]:
            distance = math.dist(pos[:3], new[:3])
            if distance <= 0:
                raise ValueError("zero-length extrusion unsupported by VOLCO")
            smallest = min(smallest, distance)
        pos = new
    if not math.isfinite(smallest):
        raise ValueError("no extrusion segments for VOLCO")
    step_size = min(float(voxel_size) / 4, smallest * 0.4)
    printer = {"nozzle_jerk_speed": 8.0, "extruder_jerk_speed": 5.0,
               "nozzle_acceleration": 500.0, "extruder_acceleration": 1000.0,
               "feedstock_filament_diameter": 1.75, "nozzle_diameter": 0.4}
    sim = {"voxel_size": float(voxel_size), "step_size": step_size,
           "x_offset": 1.0, "y_offset": 1.0, "z_offset": 0.5,
           "sphere_z_offset": 0.2, "simulation_name": "reconstruction",
           "results_folder": str(folder), "radius_increment": 0.1,
           "solver_tolerance": 0.0001,
           "x_crop": ["all", "all"], "y_crop": ["all", "all"],
           "z_crop": ["all", "all"], "consider_acceleration": False,
           "stl_ascii": False}
    with (folder / "volco_upstream.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            output = run_simulation(gcode_path=gcode, printer_config=printer,
                                    sim_config=sim)
            stl = output.export_mesh_to_stl()
    print("VOLCO_RESULT=" + json.dumps({
        "stl": str(stl) if stl else None,
        "step_size_mm": step_size,
        "xy_translation_mm": output.voxel_space.filament_translations}))


if __name__ == "__main__":
    main()
