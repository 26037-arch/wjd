"""G-code driven, local OpenFOAM-14 deposition experiment.

This adapter deliberately reports an uncalibrated experiment as such.  A
successful CFD run is not evidence that a particular printer deposits PLA in
the predicted shape.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from conical.planar_slicer import FILAMENT_D

WORD = re.compile(r"([A-Za-z])([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
LAYER = re.compile(r";\s*layer\s+(\d+)\b", re.I)
META = re.compile(r"^;CONICAL_META\s+(.+)$")
TRAJECTORY_COLUMNS = (
    "time_s", "x_mm", "y_mm", "z_mm", "vx_mm_s", "vy_mm_s",
    "vz_mm_s", "E_mm", "extrusion_rate_mm_s", "volumetric_flow_mm3_s",
    "nozzle_temp_C", "bed_temp_C", "B_deg", "C_deg",
)


def _words(code):
    return {key.upper(): float(value) for key, value in WORD.findall(code)}


def parse_context(path):
    """Track modal coordinates independently of the repository's tiny parser."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    xyz_absolute, e_absolute = True, True
    pos = dict.fromkeys("XYZEB C".replace(" ", ""), 0.0)
    feed, nozzle, bed, layer = None, None, None, None
    meta, moves = {}, []
    for line_no, line in enumerate(lines, 1):
        match = META.match(line)
        if match:
            meta = json.loads(match.group(1))
        match = LAYER.search(line)
        if match:
            layer = int(match.group(1))
        code = line.split(";", 1)[0].strip().upper()
        if not code:
            continue
        command = code.split(maxsplit=1)[0]
        words = _words(code)
        if command in ("G2", "G3", "G02", "G03"):
            raise ValueError(f"UNSUPPORTED_GCODE: arc at line {line_no}")
        if command == "G20":
            raise ValueError(f"UNSUPPORTED_GCODE: inch units at line {line_no}")
        if command == "G90":
            xyz_absolute = True
        elif command == "G91":
            xyz_absolute = False
        elif command == "M82":
            e_absolute = True
        elif command == "M83":
            e_absolute = False
        elif command == "G92":
            for axis in "XYZEB C".replace(" ", ""):
                if axis in words:
                    pos[axis] = words[axis]
        elif command in ("M104", "M109"):
            nozzle = words.get("S", nozzle)
        elif command in ("M140", "M190"):
            bed = words.get("S", bed)
        elif command in ("G0", "G1", "G00", "G01"):
            before = pos.copy()
            for axis in "XYZBC":
                if axis in words:
                    pos[axis] = words[axis] if xyz_absolute else pos[axis] + words[axis]
            if "E" in words:
                pos["E"] = words["E"] if e_absolute else pos["E"] + words["E"]
            feed = words.get("F", feed)
            moves.append({"index": len(moves) + 1, "line": line_no,
                          "start": before, "end": pos.copy(),
                          "delta_e_mm": pos["E"] - before["E"],
                          "distance_mm": math.dist([before[k] for k in "XYZ"],
                                                   [pos[k] for k in "XYZ"]),
                          "feed_mm_min": feed, "layer": layer,
                          "nozzle_temp_C": nozzle, "bed_temp_C": bed})
    return lines, meta, moves


def select_window(moves, spec="auto", max_path_mm=2.0):
    extrusion = [m for m in moves if m["delta_e_mm"] > 0 and m["distance_mm"] > 0]
    if not extrusion:
        raise ValueError("UNSUPPORTED_GCODE: no spatial extrusion")
    if spec in ("auto", "slope", "z", "support"):
        if spec == "support":
            # Closest approach to an unsupported first-layer path is not known
            # from G-code alone.  Prefer the highest, steepest extrusion, and
            # label this a geometric risk heuristic in metadata.
            def key(m):
                return (m["end"]["Z"], abs(m["end"]["Z"] - m["start"]["Z"]))
        elif spec == "z":
            def key(m):
                return abs(m["end"]["Z"] - m["start"]["Z"])
        else:
            def key(m):
                return (abs(m["end"]["Z"] - m["start"]["Z"])
                        / m["distance_mm"], m["end"]["Z"])
        first = max(extrusion, key=key)
    elif spec.startswith("layer:"):
        layer = int(spec.partition(":")[2])
        first = next((m for m in extrusion if m["layer"] == layer), None)
    elif spec.startswith(("moves:", "lines:")):
        kind, _, range_text = spec.partition(":")
        a, sep, b = range_text.partition("-")
        lo, hi = int(a), int(b if sep else a)
        candidates = [m for m in extrusion if lo <= m["index" if kind == "moves" else "line"] <= hi]
        first = candidates[0] if candidates else None
    else:
        raise ValueError(f"UNSUPPORTED_WINDOW: {spec}")
    if first is None:
        raise ValueError(f"EMPTY_WINDOW: {spec}")
    selected = [first]
    length = first["distance_mm"]
    for move in moves[first["index"]:]:
        if move["delta_e_mm"] <= 0 or move["distance_mm"] <= 0:
            break
        if spec.startswith("layer:") and move["layer"] != first["layer"]:
            break
        if spec.startswith("moves:"):
            hi = int(spec.partition(":")[2].partition("-")[2] or first["index"])
            if move["index"] > hi:
                break
        if spec.startswith("lines:"):
            hi = int(spec.partition(":")[2].partition("-")[2] or first["line"])
            if move["line"] > hi:
                break
        if length + move["distance_mm"] > max_path_mm and selected:
            break
        selected.append(move)
        length += move["distance_mm"]
    return selected


def decode_trajectory(gcode_path, output_dir, spec="auto", max_path_mm=2.0,
                      machine_name="prusa_mini", filament_diameter_mm=FILAMENT_D):
    """Use pyGCodeDecode's planner blocks for acceleration and junctions."""
    from pyGCodeDecode import gcode_interpreter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines, meta, moves = parse_context(gcode_path)
    selected = select_window(moves, spec, max_path_mm)
    mode = meta.get("mode", "xyz")
    if mode not in ("xyz", "open5x", "rep5x"):
        raise ValueError(f"UNSUPPORTED_MODE: {mode}")
    # The planner has no B/C axes.  Keep original line numbers and record B/C
    # separately; do not claim a 5-axis CFD calculation.
    planner_input = output_dir / "planner_xyz.gcode"
    planner_input.write_text("\n".join(re.sub(r"\b[BC][-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                                               "", line) for line in lines) + "\n",
                             encoding="utf-8")
    sim = gcode_interpreter.simulation(planner_input, machine_name=machine_name,
                                       verbosity_level=0)
    blocks = {b.state_B.line_number: b for b in sim.blocklist}
    if any(m["line"] not in blocks for m in selected):
        raise ValueError("DECODER_MISMATCH: selected G-code lines missing from planner")
    area = math.pi * (filament_diameter_mm / 2) ** 2
    rows, segment_integral = [], 0.0
    first_time = float(blocks[selected[0]["line"]].get_segments()[0].t_begin)
    for move in selected:
        block = blocks[move["line"]]
        for segment in block.get_segments():
            t0, t1 = float(segment.t_begin), float(segment.t_end)
            v0 = segment.vel_begin.get_vec(withExtrusion=True)
            v1 = segment.vel_end.get_vec(withExtrusion=True)
            segment_integral += area * (float(v0[3]) + float(v1[3])) * (t1-t0) / 2
            for t, vel, pos in ((t0, v0, segment.pos_begin.get_vec(withExtrusion=True)),
                                (t1, v1, segment.pos_end.get_vec(withExtrusion=True))):
                row = {"time_s": t-first_time, "x_mm": float(pos[0]),
                       "y_mm": float(pos[1]), "z_mm": float(pos[2]),
                       "vx_mm_s": float(vel[0]), "vy_mm_s": float(vel[1]),
                       "vz_mm_s": float(vel[2]), "E_mm": float(pos[3]),
                       "extrusion_rate_mm_s": float(vel[3]),
                       "volumetric_flow_mm3_s": max(0.0, float(vel[3])) * area,
                       "nozzle_temp_C": move["nozzle_temp_C"],
                       "bed_temp_C": move["bed_temp_C"],
                       "B_deg": move["end"]["B"] if mode == "rep5x" else None,
                       "C_deg": move["end"]["C"] if mode == "rep5x" else None}
                if rows and t-first_time <= rows[-1]["time_s"] + 1e-12:
                    rows[-1] = row
                else:
                    rows.append(row)
    expected = area * sum(m["delta_e_mm"] for m in selected)
    if not math.isclose(segment_integral, expected, rel_tol=2e-5, abs_tol=1e-6):
        raise ValueError(f"VOLUME_MISMATCH: G-code={expected}, planner={segment_integral}")
    trajectory = output_dir / "trajectory.csv"
    with trajectory.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    window = output_dir / "selected_window.gcode"
    start = selected[0]["start"]
    initial_e = start["E"]
    window_lines = ["; Extracted from actual source G-code; absolute E rebased to zero",
                    "G21", "G90", "M82",
                    f"G0 X{start['X']:.6f} Y{start['Y']:.6f} Z{start['Z']:.6f} F3600",
                    "G92 E0"]
    if selected[0]["nozzle_temp_C"] is not None:
        window_lines.append(f"M104 S{selected[0]['nozzle_temp_C']:g}")
    if selected[0]["bed_temp_C"] is not None:
        window_lines.append(f"M140 S{selected[0]['bed_temp_C']:g}")
    for move in selected:
        end = move["end"]
        extra = (f" B{end['B']:.6f} C{end['C']:.6f}"
                 if mode == "rep5x" else "")
        window_lines.append(f"G1 X{end['X']:.6f} Y{end['Y']:.6f} Z{end['Z']:.6f} "
                            f"E{end['E']-initial_e:.6f} F{move['feed_mm_min']:.6f}{extra} "
                            f"; source_line={move['line']}")
    window.write_text("\n".join(window_lines) + "\n", encoding="utf-8")
    return {"mode": mode, "gcode_meta": meta, "window_spec": spec, "window": {
                "move_start": selected[0]["index"], "move_end": selected[-1]["index"],
                "line_start": selected[0]["line"], "line_end": selected[-1]["line"],
                "layer": selected[0]["layer"], "path_mm": sum(m["distance_mm"] for m in selected)},
            "selected_moves": selected, "rows": rows,
            "trajectory_csv": str(trajectory), "selected_window_gcode": str(window),
            "duration_s": rows[-1]["time_s"], "gcode_extrusion_volume_mm3": expected,
            "trajectory_integrated_volume_mm3": segment_integral,
            "machine_profile": machine_name, "machine_profile_verified_for_target": False,
            "gcode_motion_model": "pyGCodeDecode", "filament_diameter_mm": filament_diameter_mm,
            "nozzle_diameter_mm": float(sim.initial_machine_setup_dict["nozzle_diam"]),
            "orientation_modeled": mode == "xyz",
            "temperature_assumption": "Nozzle outlet temperature equals M104/M109 setpoint when supplied; no thermal field is solved",
            "missing_temperature_reason": "No M104/M109 or M140/M190 in source G-code" if
                all(m["nozzle_temp_C"] is None and m["bed_temp_C"] is None for m in selected)
                else None}


def load_material(path):
    material = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = ("density", "reference_temperature", "viscosity_parameters", "surface_tension",
                "thermal_conductivity", "specific_heat", "solidification")
    missing = [key for key in required if key not in material]
    for key in ("density", "surface_tension"):
        if key in material and not isinstance(material[key].get("value"), (int, float)):
            missing.append(key + ".value")
    for key in ("eta0_pa_s", "tau_pa", "high_shear_index"):
        if not isinstance(material.get("viscosity_parameters", {}).get(key), (int, float)):
            missing.append("viscosity_parameters." + key)
    return material, missing


def _foam_file(kind, name, body):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {kind};\n    object {name};\n}}\n\n{body}\n")


def _write(folder, relative, kind, body):
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_foam_file(kind, path.name, body), encoding="utf-8")


def _linux_case_path(path):
    path = Path(path).resolve()
    if path.drive:
        return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]
    return str(path)


def generate_openfoam14_case(decoded, material, case_dir, cell_size_mm=0.15):
    """Fixed mesh, nozzle-area moving volume sources; first-layer bed only."""
    if decoded["mode"] != "xyz":
        raise ValueError("UNSUPPORTED_REP5X_ORIENTATION: B/C or moving bed not modeled")
    if decoded["window"]["layer"] != 0:
        raise ValueError("SUBSTRATE_UNAVAILABLE: only the bare first-layer bed is modeled")
    rows = decoded["rows"]
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    minx, maxx = min(r["x_mm"] for r in rows)-0.8, max(r["x_mm"] for r in rows)+0.8
    miny, maxy = min(r["y_mm"] for r in rows)-0.8, max(r["y_mm"] for r in rows)+0.8
    maxz = max(r["z_mm"] for r in rows)+0.8
    nx = max(4, math.ceil((maxx-minx)/cell_size_mm))
    ny = max(4, math.ceil((maxy-miny)/cell_size_mm))
    nz = max(4, math.ceil(maxz/cell_size_mm))
    block = f"""convertToMeters 0.001;
vertices (({minx} {miny} 0) ({maxx} {miny} 0) ({maxx} {maxy} 0)
          ({minx} {maxy} 0) ({minx} {miny} {maxz}) ({maxx} {miny} {maxz})
          ({maxx} {maxy} {maxz}) ({minx} {maxy} {maxz}));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
    bed {{ type wall; faces ((0 3 2 1)); }}
    atmosphere {{ type patch; faces ((0 1 5 4) (1 2 6 5) (2 3 7 6)
                                     (3 0 4 7) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
    _write(case_dir, "system/blockMeshDict", "dictionary", block)
    duration = decoded["duration_s"]
    end_time = duration + min(0.02, max(0.005, duration * 0.2))
    control = f"""solver incompressibleVoF;
startFrom startTime; startTime 0; stopAt endTime; endTime {end_time:.9g};
deltaT 0.00002; writeControl adjustableRunTime; writeInterval {end_time:.9g};
writeFormat ascii; writePrecision 9; runTimeModifiable no;
adjustTimeStep yes; maxCo 0.25; maxAlphaCo 0.25; maxDeltaT 0.0001;
"""
    _write(case_dir, "system/controlDict", "dictionary", control)
    _write(case_dir, "system/fvSchemes", "dictionary", """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    div(phi,alpha) Gauss interfaceCompression vanLeer 1;
    div(rhoPhi,U) Gauss linearUpwind grad(U);
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }
""")
    _write(case_dir, "system/fvSolution", "dictionary", """solvers
{
    "alpha.polymer.*"
    {
        nCorrectors 2; nSubCycles 1; MULESCorr yes;
        MULES { nIter 10; tolerance 1e-2; }
        solver smoothSolver; smoother symGaussSeidel;
        tolerance 1e-8; relTol 0;
    }
    "pcorr.*" { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }
    p_rgh { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.05; }
    p_rghFinal { $p_rgh; relTol 0; }
    "(U|k|epsilon|omega).*"
    { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6;
      relTol 0; minIter 1; }
}
PIMPLE { momentumPredictor no; nOuterCorrectors 1; nCorrectors 3;
         nNonOrthogonalCorrectors 0; }
relaxationFactors { equations { ".*" 1; } }
""")
    _write(case_dir, "constant/g", "uniformDimensionedVectorField",
           "dimensions [acceleration]; value (0 0 -9.81);")
    rho = float(material["density"]["value"])
    eta0 = float(material["viscosity_parameters"]["eta0_pa_s"])
    tau = float(material["viscosity_parameters"]["tau_pa"])
    index = float(material["viscosity_parameters"]["high_shear_index"])
    _write(case_dir, "constant/phaseProperties", "dictionary",
           f"phases (polymer air); sigma {float(material['surface_tension']['value'])};")
    _write(case_dir, "constant/physicalProperties.polymer", "dictionary",
           f"viscosityModel constant; nu {eta0/rho:.9g}; rho {rho:.9g};")
    # Air values are the OpenFOAM 14 damBreakLaminar tutorial defaults, not
    # temperature-corrected properties for a real printer enclosure.
    _write(case_dir, "constant/physicalProperties.air", "dictionary",
           "viscosityModel constant; nu 1.48e-05; rho 1;")
    _write(case_dir, "constant/momentumTransport", "dictionary",
           "simulationType laminar;\nlaminar\n{\n model generalisedNewtonian;\n"
           f" viscosityModel CrossPowerLaw; nuInf 0; tauStar {tau/rho:.9g};"
           f" n {1-index:.9g};\n}}")
    _write(case_dir, "0/U", "volVectorField", """dimensions [velocity];
internalField uniform (0 0 0);
boundaryField
{
    bed { type noSlip; }
    atmosphere { type pressureInletOutletVelocity; value uniform (0 0 0); }
}
""")
    _write(case_dir, "0/alpha.polymer", "volScalarField", """dimensions [];
internalField uniform 0;
boundaryField
{
    bed { type zeroGradient; }
    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }
}
""")
    _write(case_dir, "0/p_rgh", "volScalarField", """dimensions [pressure];
internalField uniform 0;
boundaryField
{
    bed { type fixedFluxPressure; value uniform 0; }
    atmosphere { type prghTotalPressure; p0 uniform 0; value uniform 0; }
}
""")
    sources, u_sources, source_cell_counts = [], [], []
    dx, dy, dz = (maxx-minx)/nx, (maxy-miny)/ny, maxz/nz
    nozzle_radius = decoded["nozzle_diameter_mm"] / 2
    for i, (a, b) in enumerate(zip(rows, rows[1:], strict=False)):
        t0, t1 = a["time_s"], b["time_s"]
        if t1 <= t0:
            continue
        de = b["E_mm"] - a["E_mm"]
        if de <= 0:
            continue
        q_m3s = de * math.pi * (decoded["filament_diameter_mm"]/2)**2 * 1e-9 / (t1-t0)
        x_mm = (a["x_mm"]+b["x_mm"])*0.5
        y_mm = (a["y_mm"]+b["y_mm"])*0.5
        z_mm = (a["z_mm"]+b["z_mm"])*0.5
        kz = min(nz-1, max(0, math.floor(z_mm/dz)))
        cell_points = []
        for ix in range(nx):
            cx = minx + (ix+0.5)*dx
            if abs(cx-x_mm) > nozzle_radius:
                continue
            for iy in range(ny):
                cy = miny + (iy+0.5)*dy
                if (cx-x_mm)**2 + (cy-y_mm)**2 <= nozzle_radius**2:
                    cell_points.append((cx*0.001, cy*0.001,
                                        (kz+0.5)*dz*0.001))
        if not cell_points:
            # An under-resolved nozzle still has a defined nearest cell.  The
            # diagnostic count makes this plainly visible in the result.
            ix = min(nx-1, max(0, math.floor((x_mm-minx)/dx)))
            iy = min(ny-1, max(0, math.floor((y_mm-miny)/dy)))
            cell_points = [((minx+(ix+0.5)*dx)*0.001,
                            (miny+(iy+0.5)*dy)*0.001,
                            (kz+0.5)*dz*0.001)]
        source_cell_counts.append(len(cell_points))
        points_text = " ".join(
            f"({px:.9g} {py:.9g} {pz:.9g})" for px, py, pz in cell_points)
        eps = min(1e-7, (t1-t0)/10)
        zero_until_start = f"(0 0) ({t0:.9g} 0)" if t0 > 0 else "(0 0)"
        source = f"""source_{i}
{{
    type volumeSource;
    phase polymer;
    cellZone {{ type containsPoints; points ({points_text}); }}
    volumetricFlowRate table
    (
        {zero_until_start} ({t0+eps:.9g} {q_m3s:.9g})
        ({t1-eps:.9g} {q_m3s:.9g}) ({t1:.9g} 0)
        ({end_time:.9g} 0)
    );
}}
"""
        sources.append(source)
        nozzle_area_mm2 = math.pi * (decoded["nozzle_diameter_mm"]/2)**2
        vx = (a["vx_mm_s"]+b["vx_mm_s"])*0.0005
        vy = (a["vy_mm_s"]+b["vy_mm_s"])*0.0005
        vz = (a["vz_mm_s"]+b["vz_mm_s"])*0.0005 - q_m3s*1e6/nozzle_area_mm2
        u_sources.append(f"source_{i} {{ type uniformFixedValue; "
                         f"uniformValue ({vx:.9g} {vy:.9g} {vz:.9g}); }}")
    if not sources:
        raise ValueError("NO_SOURCE_INTERVALS: trajectory has no extrusion")
    _write(case_dir, "constant/fvModels", "dictionary", "\n".join(sources))
    u_file = case_dir / "0/U"
    u_file.write_text(u_file.read_text(encoding="utf-8") +
                      "\nsources\n{\n" + "\n".join(u_sources) + "\n}\n",
                      encoding="utf-8")
    return {"case_dir": str(case_dir), "source_intervals": len(sources),
            "source_cells_min": min(source_cell_counts),
            "source_cells_max": max(source_cell_counts),
            "source_cells_mean": sum(source_cell_counts)/len(source_cell_counts),
            "cells": nx*ny*nz, "cell_size_mm": cell_size_mm,
            "simulation_physical_time_s": end_time,
            "solver_command": "source /opt/openfoam14/etc/bashrc && "
                              f"blockMesh -case '{_linux_case_path(case_dir)}' && "
                              f"foamRun -case '{_linux_case_path(case_dir)}'",
            "substrate": "bare build plate wall (first layer only)",
            "nozzle_geometry_resolved": False,
            "source_model": "fixed-grid volumeSource over nozzle-area cell centers",
            "gravity_model": True, "free_surface_model": "VOF",
            "non_newtonian_model": "CrossPowerLaw",
            "thermal_model": False, "solidification_model": False,
            "surface_tension_model": float(material["surface_tension"]["value"]) > 0}


def validate_physics(gcode_path, original_mesh, output_dir, material_profile,
                     physics_window="auto", max_path_mm=2.0, machine_name="prusa_mini"):
    """Return JSON-serialisable status; never return PASS for an unrun CFD case."""
    start = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"backend": "openfoam14", "available": None, "status": "PREPARING",
              "valid": None, "predicted_stl": None, "metrics": None,
              "material_profile": str(Path(material_profile).resolve()),
              "original_mesh": str(Path(original_mesh).resolve()),
              "gcode_path": str(Path(gcode_path).resolve())}
    try:
        decoded = decode_trajectory(gcode_path, output_dir, physics_window,
                                    max_path_mm, machine_name)
        result.update({k: v for k, v in decoded.items()
                       if k not in ("rows", "selected_moves")})
        material, missing = load_material(material_profile)
        result["material"] = material.get("name")
        result["material_calibrated_for_target"] = material.get("calibrated_for_target", False)
        if missing:
            result.update(status="MATERIAL_PROFILE_INCOMPLETE", missing_material_fields=missing)
        elif decoded["mode"] != "xyz":
            result.update(status="UNSUPPORTED_REP5X_ORIENTATION",
                          reason="B/C or moving bed is not a CFD boundary condition")
        else:
            case = generate_openfoam14_case(decoded, material, output_dir / "openfoam_case")
            result.update(case)
            result.update(status="CASE_READY", available=True if shutil.which("foamRun") else None)
    except ImportError as exc:
        result.update(status="PYGCODEDECODE_UNAVAILABLE", reason=str(exc))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        message = str(exc)
        result.update(status=message.split(":", 1)[0] if ":" in message else "ERROR",
                      reason=message)
    result["preparation_wall_time_s"] = time.monotonic()-start
    result_path = output_dir / "result.json"
    result["result_json"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def finalize_physics(output_dir, solver_log=None):
    """Measure only fields actually produced by a completed OpenFOAM run."""
    import numpy as np
    import pyvista as pv

    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    log_path = Path(solver_log) if solver_log else output_dir / "physics.log"
    result["physics_log"] = str(log_path) if log_path.exists() else None
    if not log_path.exists():
        result.update(status="SOLVER_NOT_RUN", reason="No OpenFOAM log exists")
    else:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        result["time_steps"] = len(re.findall(r"^Time = ", log, flags=re.M))
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)", log)
        peak = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log)
        result["solver_wall_time"] = wall.group(1).strip() if wall else None
        result["peak_memory_kb"] = int(peak.group(1)) if peak else None
        result["cpu_count"] = 1
        if not re.search(r"^End\s*$", log, flags=re.M):
            result.update(status="SOLVER_DIVERGED_OR_FAILED",
                          reason="OpenFOAM log has no successful End marker")
        else:
            vtk_files = sorted((output_dir / "openfoam_case" / "VTK").glob("*.vtk"),
                               key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
            if not vtk_files:
                result.update(status="SURFACE_EXTRACTION_FAILED",
                              reason="foamToVTK output missing")
            else:
                grid = pv.read(vtk_files[-1])
                alpha = np.asarray(grid.cell_data["alpha.polymer"], dtype=float)
                volumes = np.asarray(grid.compute_cell_sizes().cell_data["Volume"], dtype=float)
                occupied_mm3 = float(np.sum(alpha * volumes) * 1e9)
                surface = grid.cell_data_to_point_data().contour(
                    [0.5], scalars="alpha.polymer")
                if surface.n_points == 0:
                    result.update(status="SURFACE_EXTRACTION_FAILED",
                                  reason="alpha=0.5 iso-surface is empty")
                else:
                    surface.points *= 1000
                    stl = output_dir / "openfoam_case" / "final_surface.stl"
                    surface.save(stl)
                    with Path(result["trajectory_csv"]).open(newline="", encoding="utf-8") as file:
                        trajectory = list(csv.DictReader(file))
                    commanded = []
                    weights = []
                    for a, b in zip(trajectory, trajectory[1:], strict=False):
                        de = float(b["E_mm"]) - float(a["E_mm"])
                        if de > 0:
                            commanded.append([(float(a[k])+float(b[k]))/2
                                              for k in ("x_mm", "y_mm", "z_mm")])
                            weights.append(de)
                    command_center = np.average(commanded, axis=0, weights=weights)
                    cell_center = grid.cell_centers().points * 1000
                    final_center = np.average(cell_center, axis=0, weights=alpha*volumes)
                    first = np.array([float(trajectory[0][k]) for k in ("x_mm", "y_mm")])
                    last = np.array([float(trajectory[-1][k]) for k in ("x_mm", "y_mm")])
                    direction = last-first
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        side = np.array([-direction[1], direction[0]]) / norm
                        lateral = surface.points[:, :2] @ side
                        width = float(np.ptp(lateral))
                    else:
                        width = None
                    result.update(
                        status="CFD_COMPLETED_UNCALIBRATED" if not result.get(
                            "material_calibrated_for_target") else "CFD_COMPLETED",
                        available=True, predicted_stl=str(stl),
                        vtk_field=str(vtk_files[-1]),
                        metrics={"deposited_volume_mm3": occupied_mm3,
                                 "volume_difference_fraction":
                                     (occupied_mm3-result["gcode_extrusion_volume_mm3"])
                                     / result["gcode_extrusion_volume_mm3"],
                                 "bead_center_deviation_mm": float(np.linalg.norm(
                                     final_center-command_center)),
                                 "max_sag_mm": None,
                                 "cross_section_width_mm": width,
                                 "cross_section_height_mm": float(np.ptp(surface.points[:, 2])),
                                 "commanded_path_center_mm": command_center.tolist(),
                                 "final_material_centroid_mm": final_center.tolist()},
                        metric_limitations=[
                            "Center deviation compares full-window volume centroids, not a local centerline.",
                            "Sag is undefined for this bed-supported first-layer case.",
                            "Width/height are iso-surface extents of a coarse, uncalibrated numerical demo.",
                        ])
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def execute_openfoam14_case(output_dir, timeout_s=120, distro="Ubuntu-24.04"):
    """Run an already generated case with bounded external processes."""
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "CASE_READY":
        return result
    case = shlex.quote(_linux_case_path(result["case_dir"]))
    base = (["wsl", "-d", distro, "--", "bash", "-lc"]
            if os.name == "nt" else ["bash", "-lc"])
    steps = (
        ("blockMesh", f"blockMesh -case {case}", "MESH_GENERATION_FAILED"),
        ("physics", f"if [ -x /usr/bin/time ]; then /usr/bin/time -v "
                    f"foamRun -case {case}; else foamRun -case {case}; fi",
         "SOLVER_DIVERGED_OR_FAILED"),
        ("foamToVTK", f"foamToVTK -case {case} -latestTime "
                       "-fields '(alpha.polymer)'", "SURFACE_EXTRACTION_FAILED"),
    )
    step_wall_times = {}
    for log_name, command, failure_status in steps:
        log_path = output_dir / (log_name + ".log")
        step_start = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(base + ["source /opt/openfoam14/etc/bashrc && " + command],
                                      stdout=log, stderr=subprocess.STDOUT,
                                      timeout=timeout_s, check=False)
            if proc.returncode:
                result.update(status="SOLVER_NOT_INSTALLED" if proc.returncode == 127
                              else failure_status,
                              reason=f"{command} exited {proc.returncode}",
                              failed_log=str(log_path))
                break
        except subprocess.TimeoutExpired:
            result.update(status="TIMEOUT", reason=f"{command} exceeded {timeout_s}s",
                          failed_log=str(log_path))
            break
        except OSError as exc:
            result.update(status="SOLVER_NOT_ACCESSIBLE", reason=str(exc),
                          failed_log=str(log_path))
            break
        finally:
            step_wall_times[log_name] = time.monotonic()-step_start
    else:
        # /usr/bin/time is optional in a normal user run; finalizer leaves
        # peak memory and wall time null when no such summary exists.
        result = finalize_physics(output_dir)
        result["step_wall_times_s"] = step_wall_times
        result["execution_wall_time_s"] = sum(step_wall_times.values())
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return result
    result["step_wall_times_s"] = step_wall_times
    result["execution_wall_time_s"] = sum(step_wall_times.values())
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def compare_local_validators(physics_result_path, volco_result_path, target_stl,
                             output_path):
    """Compare the same local window without treating it as a whole-part check."""
    import numpy as np
    import trimesh

    physics = json.loads(Path(physics_result_path).read_text(encoding="utf-8"))
    volco = json.loads(Path(volco_result_path).read_text(encoding="utf-8"))
    with Path(physics["trajectory_csv"]).open(newline="", encoding="utf-8") as file:
        trajectory = list(csv.DictReader(file))
    start = np.array([float(trajectory[0][k]) for k in ("x_mm", "y_mm")])
    end = np.array([float(trajectory[-1][k]) for k in ("x_mm", "y_mm")])
    direction = end-start
    norm = np.linalg.norm(direction)
    side = np.array([-direction[1], direction[0]]) / norm if norm else None
    volco_metrics = None
    if volco.get("status") == "OK" and Path(volco["aligned_reconstructed_stl"]).exists():
        mesh = trimesh.load(volco["aligned_reconstructed_stl"], force="mesh")
        lateral = mesh.vertices[:, :2] @ side if side is not None else None
        volco_metrics = {
            "volume_mm3": float(abs(mesh.volume)) if mesh.is_watertight else None,
            "width_mm": float(np.ptp(lateral)) if lateral is not None else None,
            "height_mm": float(np.ptp(mesh.vertices[:, 2])),
            "centroid_mm": mesh.center_mass.tolist() if mesh.is_watertight else None,
            "sag_mm": None,
            "stl": volco["aligned_reconstructed_stl"],
        }
    nominal_width = physics.get("gcode_meta", {}).get("extrusion_width")
    nominal_height = physics.get("gcode_meta", {}).get("layer_height")
    measured = physics.get("metrics") or {}
    comparison = {
        "scope": "same local G-code window; original whole-part STL is context only",
        "original_target": {"stl": str(target_stl),
                            "local_surface_metrics": None},
        "commanded_volume_mm3": physics["gcode_extrusion_volume_mm3"],
        "nominal_cross_section": {
            "extrusion_width_mm": nominal_width,
            "layer_height_mm": nominal_height,
            "cfd_width_minus_nominal_mm":
                measured["cross_section_width_mm"] - nominal_width
                if nominal_width is not None and measured.get("cross_section_width_mm") is not None
                else None,
            "cfd_height_extent_minus_nominal_mm":
                measured["cross_section_height_mm"] - nominal_height
                if nominal_height is not None and measured.get("cross_section_height_mm") is not None
                else None,
            "limitation": "CFD height is an iso-surface extent, not a local sliced bead height."
        },
        "volco": {"status": volco.get("status"), "metrics": volco_metrics,
                  "models": "volumetric voxel deposition without gravity, flow or thermal physics"},
        "physics_cfd": {"status": physics.get("status"),
                        "metrics": physics.get("metrics"),
                        "stl": physics.get("predicted_stl"),
                        "models": "OpenFOAM 14 isothermal VOF, gravity and CrossPowerLaw with an approximate moving source"},
        "whole_target_chamfer_mm": None,
        "whole_target_hausdorff_mm": None,
        "whole_target_volume_error": None,
        "note": "The local outputs cannot be compared to the entire STL with a meaningful global distance or volume error.",
    }
    Path(output_path).write_text(json.dumps(comparison, indent=2, allow_nan=False),
                                 encoding="utf-8")
    return comparison
