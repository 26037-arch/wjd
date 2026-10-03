"""Sparse, orientation-aware extension of VOLCO's spherical voxel deposition.

VOLCO's Cartesian kernel centers a volume-fitted sphere 0.2 mm below the
nozzle, clips it at the nozzle plane and the build plate, and adds empty
voxels.  This extension rotates both center offset and nozzle clipping plane
with REP5X B/C.  It uses nearest-center selection to fit cumulative commanded
volume to one voxel; the result is a numerical prediction, not a calibrated
print or CFD solution.  The external VOLCO checkout is never modified.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import trimesh

from conical.deposition_events import MotionEvent, parse_motion_events
from tools.external.validators import repository_commit


def mesh_from_voxels(voxels: set[tuple[int, int, int]], size: float):
    """Emit exposed cube faces, with voxel i occupying [i*h, (i+1)*h]."""
    corners = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
    sides = (
        ((0, 0, -1), ((0, 2, 1), (0, 3, 2))),
        ((0, 0, 1), ((4, 5, 6), (4, 6, 7))),
        ((0, -1, 0), ((0, 1, 5), (0, 5, 4))),
        ((0, 1, 0), ((2, 3, 7), (2, 7, 6))),
        ((-1, 0, 0), ((0, 4, 7), (0, 7, 3))),
        ((1, 0, 0), ((1, 2, 6), (1, 6, 5))),
    )
    indices, vertices, faces = {}, [], []
    for voxel in sorted(voxels):
        for offset, triangles in sides:
            neighbor = tuple(a + b for a, b in zip(voxel, offset, strict=True))
            if neighbor in voxels:
                continue
            for triangle in triangles:
                face = []
                for corner_id in triangle:
                    grid = tuple(a + b for a, b in
                                 zip(voxel, corners[corner_id], strict=True))
                    if grid not in indices:
                        indices[grid] = len(vertices)
                        vertices.append(tuple(axis * size for axis in grid))
                    face.append(indices[grid])
                faces.append(face)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _nearest_empty(voxels, tip, direction, size, offset, count,
                   max_candidates=2_000_000):
    """Choose the empty grid centers nearest the oriented sphere center."""
    if count <= 0:
        return []
    tip = np.asarray(tip, dtype=float)
    direction = np.asarray(direction, dtype=float)
    center = tip + direction * offset
    radius = max(size * 1.5, (3 * count * size**3 / (2 * math.pi))**(1 / 3)
                 + offset)
    for _ in range(24):
        low = np.floor((center - radius) / size).astype(int)
        high = np.ceil((center + radius) / size).astype(int)
        low[2] = max(low[2], 0)
        candidate_count = int(np.prod(np.maximum(high - low + 1, 0)))
        if candidate_count > max_candidates:
            raise ValueError("Local voxel search exceeds max_candidates; increase voxel size")
        candidates = []
        for i in range(low[0], high[0] + 1):
            for j in range(low[1], high[1] + 1):
                for k in range(low[2], high[2] + 1):
                    index = (i, j, k)
                    if index in voxels:
                        continue
                    point = (np.asarray(index, dtype=float) + 0.5) * size
                    relative = point - tip
                    if np.dot(relative, direction) < -size * 1e-9:
                        continue
                    distance2 = float(np.dot(point - center, point - center))
                    if distance2 <= radius * radius:
                        candidates.append((distance2, index))
        if len(candidates) >= count:
            candidates.sort()
            return [index for _, index in candidates[:count]]
        radius *= 1.5
    raise ValueError("Could not fit commanded volume in oriented local search")


def simulate(events: list[MotionEvent], output_dir, *, voxel_size=0.025,
             step_size=None, filament_diameter=1.75, nozzle_diameter=0.4,
             sphere_offset=0.2, volco_dir=None, max_voxels=2_000_000):
    if voxel_size <= 0 or filament_diameter <= 0 or nozzle_diameter <= 0:
        raise ValueError("Voxel, filament and nozzle dimensions must be positive")
    if step_size is None:
        step_size = voxel_size / 4
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "frames.jsonl"
    log_path = output_dir / "simulation.log"
    start = time.perf_counter()
    voxels = set()
    area = math.pi * (filament_diameter / 2)**2
    target = 0.0
    frame_count = 0
    b_values, c_values = [], []
    with frames_path.open("w", encoding="utf-8") as stream:
        for event_index, event in enumerate(events):
            b_values.extend((event.start_b, event.end_b))
            c_values.extend((event.start_c, event.end_c))
            if not event.extruding or event.segment_length <= 0:
                continue
            steps = max(1, math.ceil(event.segment_length / step_size))
            for step in range(steps):
                fraction = (step + 0.5) / steps
                pose = event.pose(fraction)
                target += area * event.delta_e / steps
                desired = round(target / voxel_size**3)
                needed = desired - len(voxels)
                if len(voxels) + needed > max_voxels:
                    raise ValueError("Simulation exceeds max_voxels; enlarge voxel size")
                added = _nearest_empty(voxels, pose["position"], pose["direction"],
                                       voxel_size, sphere_offset, needed)
                voxels.update(added)
                frame = {"event": event_index, "line": event.line,
                         "time_s": event.start_time + event.estimated_duration
                         * (step + 1) / steps,
                         "position": event.pose((step + 1) / steps)["position"],
                         "b": event.pose((step + 1) / steps)["b"],
                         "c": event.pose((step + 1) / steps)["c"],
                         "added": added, "voxel_count": len(voxels),
                         "target_volume_mm3": target,
                         "deposited_volume_mm3": len(voxels) * voxel_size**3}
                stream.write(json.dumps(frame) + "\n")
                frame_count += 1
    mesh_start = time.perf_counter()
    mesh = mesh_from_voxels(voxels, voxel_size)
    stl_path = output_dir / "final.stl"
    mesh.export(stl_path)
    deposited = len(voxels) * voxel_size**3
    metrics = {"target_volume_mm3": target, "deposited_volume_mm3": deposited,
               "relative_volume_error": (deposited - target) / target if target else None,
               "number_of_deposition_events": frame_count,
               "number_of_cfd_calls": 0, "number_of_cfd_cache_hits": 0,
               "occupied_voxels": len(voxels), "mesh_watertight": bool(mesh.is_watertight),
               "b_range_deg": [min(b_values), max(b_values)] if b_values else [0, 0],
               "c_range_deg": [min(c_values), max(c_values)] if c_values else [0, 0],
               "runtime_s": {"deposition": mesh_start - start,
                             "meshing": time.perf_counter() - mesh_start}}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {"source_gcode": None, "backend": "volco_oriented_extension",
                "volco_commit": repository_commit(volco_dir) if volco_dir else None,
                "voxel_size_mm": voxel_size, "step_size_mm": step_size,
                "filament_diameter_mm": filament_diameter,
                "nozzle_diameter_mm": nozzle_diameter,
                "sphere_offset_mm": sphere_offset,
                "frame_count": frame_count,
                "duration_s": events[-1].end_time if events else 0.0,
                "orientation_modeled": True, "cfd_applied": False,
                "cfd_status": "NOT_REQUESTED",
                "coordinate_convention": "XYZ is RTCP tip; d=Rz(C)Ry(B)(0,0,-1); voxel i occupies [i*h,(i+1)*h]",
                "model_limitations": ["Spherical voxel deposition, no melt flow or cooling",
                                      "Bare horizontal build plate only",
                                      "Linear motion timing estimate, not firmware planner"],
                "files": {"frames": frames_path.name, "final_mesh": stl_path.name,
                          "metrics": metrics_path.name, "log": log_path.name}}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_path.write_text(f"{frame_count} frames, {len(voxels)} occupied voxels\n",
                        encoding="utf-8")
    return manifest


def simulate_gcode(gcode_path, output_dir, **kwargs):
    events = parse_motion_events(Path(gcode_path).read_text(encoding="utf-8").splitlines())
    if not any(event.extruding and event.segment_length > 0 for event in events):
        raise ValueError("No spatial extrusion in G-code")
    manifest = simulate(events, output_dir, **kwargs)
    manifest["source_gcode"] = str(Path(gcode_path).resolve())
    manifest["files"]["events"] = "events.json"
    folder = Path(output_dir)
    (folder / "events.json").write_text(
        json.dumps([event.json() for event in events], indent=2), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
