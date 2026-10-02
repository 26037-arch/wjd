"""Narrow, evidence-based external validator adapters."""

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def unavailable(backend, reason, status="NOT_AVAILABLE"):
    return {"backend": backend, "available": False, "status": status,
            "valid": None, "reason": reason}


def repository_commit(path):
    if not path or not (Path(path) / ".git").exists():
        return None
    resolved = Path(path).resolve()
    p = subprocess.run(["git", "-c", f"safe.directory={resolved.as_posix()}",
                        "-C", str(resolved), "rev-parse", "HEAD"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=10)
    return p.stdout.strip() if p.returncode == 0 else None


def validate_pymeshlab(stl_path):
    try:
        import pymeshlab
    except ImportError as exc:
        return unavailable("pymeshlab", str(exc))
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(stl_path))
        top = ms.get_topological_measures()
        ms.compute_selection_by_self_intersections_per_face()
        self_intersecting_faces = ms.current_mesh().selected_face_number()
        out = {"backend": "pymeshlab", "available": True,
               "version": importlib.metadata.version("pymeshlab"),
               "status": "OK", "boundary_edges": int(top["boundary_edges"]),
               "components": int(top["connected_components_number"]),
               "non_manifold_edges": int(top["non_two_manifold_edges"]),
               "non_manifold_vertices": int(top["non_two_manifold_vertices"]),
               "self_intersecting_faces": int(self_intersecting_faces),
               "holes": int(top["number_holes"])}
        out["valid"] = (out["boundary_edges"] == 0 and out["components"] == 1
                        and out["non_manifold_edges"] == 0
                        and out["non_manifold_vertices"] == 0
                        and out["self_intersecting_faces"] == 0)
        return out
    except Exception as exc:
        return {"backend": "pymeshlab", "available": True,
                "status": "ERROR", "valid": None, "error": str(exc)}


def validate_admesh(stl_path, executable="admesh"):
    exe = shutil.which(executable) if not Path(executable).is_file() else executable
    if not exe:
        return unavailable("admesh", "ADMesh executable not found")
    try:
        proc = subprocess.run([str(exe), str(stl_path)], capture_output=True,
                              text=True, timeout=120, errors="replace")
        raw = proc.stdout + "\n" + proc.stderr
        def number(label):
            match = re.search(r"^" + re.escape(label) + r"\s*:\s*(\d+)", raw, re.M)
            return int(match.group(1)) if match else None
        result = {"backend": "admesh", "available": True,
                  "status": "OK" if proc.returncode == 0 else "ERROR",
                  "version": (re.search(r"ADMesh version\s+([\w.]+)", raw, re.I) or
                              [None, None])[1],
                  "command": [str(exe), str(stl_path)],
                  "exit_code": proc.returncode,
                  "original_disconnected_facets": None,
                  "parts": number("Number of parts"),
                  "degenerate_facets": number("Degenerate facets"),
                  "facets_removed": number("Facets removed"),
                  "facets_added": number("Facets added"),
                  "facets_reversed": number("Facets reversed"),
                  "backwards_edges": number("Backwards edges"),
                  "normals_fixed": number("Normals fixed"),
                  "stdout_tail": raw[-3500:]}
        match = re.search(r"^Total disconnected facets\s*:\s*(\d+)", raw, re.M)
        if match:
            result["original_disconnected_facets"] = int(match.group(1))
        required = ("original_disconnected_facets", "parts", "degenerate_facets")
        if proc.returncode == 0 and any(result[k] is None for k in required):
            result["status"] = "UNPARSEABLE_OUTPUT"
        result["valid"] = (all(result[k] == 0 for k in
                               ("original_disconnected_facets", "degenerate_facets",
                                "facets_removed", "facets_added", "facets_reversed",
                                "backwards_edges", "normals_fixed"))
                           and result["parts"] == 1
                           if result["status"] == "OK" else None)
        return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"backend": "admesh", "available": True,
                "status": "ERROR", "valid": None, "error": str(exc)}


def validate_gcode_toolkit(gcode_path, toolkit_dir, node="node"):
    repo = Path(toolkit_dir) if toolkit_dir else None
    entry = repo / "dist" / "src" / "index.js" if repo else None
    if not entry or not entry.is_file() or not shutil.which(node):
        return unavailable("gcode-toolkit", "Node or built dist/src/index.js missing")
    script = Path(__file__).with_name("gcode_toolkit_runner.mjs")
    try:
        p = subprocess.run([node, str(script), str(entry), str(gcode_path)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
        result = json.loads(p.stdout)
        result.update({"backend": "gcode-toolkit", "available": True,
                       "version": json.loads((repo / "package.json").read_text(encoding="utf-8"))["version"],
                       "git_commit": repository_commit(repo),
                       "exit_code": p.returncode})
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"backend": "gcode-toolkit", "available": True,
                "status": "ERROR", "parsed": None, "error": str(exc)}


def validate_mage(repo_path=None):
    # MAGE's CoreXY-BC rotates the bed; REP5X rotates the head and treats XYZ
    # as an RTCP nozzle-tip pose. Reusing MAGE's BC is not a valid comparison.
    out = unavailable("mage", "Published CoreXY-BC is bed-tilt/bed-rotation; "
                      "no verified REP5X head-head machine model or batch result API",
                      "UNSUPPORTED_MACHINE_MODEL")
    out["git_commit"] = repository_commit(repo_path)
    return out


def validate_multi_axis(repo_path=None):
    out = unavailable("multi-axis", "Qt/Visual Studio GUI workflow uses its own "
                      "waypoint/layer and machine model; no verified REP5X batch API",
                      "UNSUPPORTED_MACHINE_MODEL")
    out["git_commit"] = repository_commit(repo_path)
    return out


def builtin_mesh_check(mesh):
    finite = bool(np.isfinite(mesh.vertices).all())
    non_empty = bool(len(mesh.vertices) and len(mesh.faces))
    volume = float(mesh.volume) if non_empty else 0.0
    return {"finite": finite, "non_empty": non_empty,
            "positive_volume": volume > 0, "volume_mm3": volume,
            "watertight": bool(mesh.is_watertight),
            "valid": finite and non_empty and volume > 0 and mesh.is_watertight}


def validate_volco(gcode_path, original_mesh, output_dir, repo_path,
                   seed=0, voxel_size=0.5, timeout=300):
    repo = Path(repo_path) if repo_path else None
    if not repo or not (repo / "volco.py").is_file():
        return unavailable("volco", "VOLCO source path not configured")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # REP5X emits nozzle-tip XYZ with RTCP. This projection deliberately drops
    # head orientation, which VOLCO's Cartesian deposition model cannot use.
    projected = output_dir / "volco_cartesian.gcode"
    retained = []
    for line in Path(gcode_path).read_text(encoding="utf-8").splitlines():
        code = line.split(";", 1)[0].strip()
        if re.match(r"^G[01]\b", code, re.I):
            words = re.findall(r"([A-Za-z])([-+]?\d*\.?\d+)", code)
            retained.append(" ".join(f"{k}{v}" for k, v in words
                                     if k.upper() in "GXYZEF"))
        elif re.match(r"^(G90|G91|G92|M82|M83|G20|G21)\b", code, re.I):
            retained.append(code)
    projected.write_text("\n".join(retained) + "\n", encoding="utf-8")
    worker = Path(__file__).with_name("volco_worker.py")
    command = [sys.executable, str(worker), str(repo), str(projected),
               str(output_dir), str(voxel_size)]
    try:
        p = subprocess.run(command, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        (output_dir / "volco.log").write_text(p.stdout + "\n" + p.stderr,
                                                 encoding="utf-8")
        marker = next((line.removeprefix("VOLCO_RESULT=") for line in
                       reversed(p.stdout.splitlines())
                       if line.startswith("VOLCO_RESULT=")), None)
        if p.returncode or not marker:
            return {"backend": "volco", "available": True, "status": "ERROR",
                    "orientation_not_modeled": True, "git_commit": repository_commit(repo),
                    "error": p.stderr[-1200:] or p.stdout[-1200:]}
        result = json.loads(marker)
        if not result.get("stl"):
            raise ValueError("VOLCO produced no STL")
        import trimesh
        from scipy.spatial import cKDTree
        reconstruction = trimesh.load(result["stl"], force="mesh")
        if reconstruction.is_empty:
            raise ValueError("VOLCO STL is empty")
        translation = result.get("xy_translation_mm")
        if not translation:
            raise ValueError("VOLCO voxel translation not reported")
        reconstruction.apply_translation([-translation["x"],
                                          -translation["y"], 0.0])
        aligned_stl = output_dir / "reconstruction_aligned.stl"
        reconstruction.export(aligned_stl)
        a = trimesh.sample.sample_surface(original_mesh, 1500, seed=seed)[0]
        b = trimesh.sample.sample_surface(reconstruction, 1500, seed=seed + 1)[0]
        da = cKDTree(b).query(a)[0]
        db = cKDTree(a).query(b)[0]
        v0 = abs(float(original_mesh.volume))
        v1 = abs(float(reconstruction.volume))
        return {"backend": "volco", "available": True, "status": "OK",
                "git_commit": repository_commit(repo),
                "orientation_not_modeled": True,
                "sampled_bidirectional_chamfer_mm": float((da.mean() + db.mean()) / 2),
                "sampled_hausdorff_mm": float(max(da.max(), db.max())),
                "relative_volume_error":
                    abs(v0-v1)/v0 if v0 > 0 and reconstruction.is_watertight else None,
                "raw_volume_difference_ratio": abs(v0-v1)/v0 if v0 > 0 else None,
                "volume_metric_limitation":
                    None if reconstruction.is_watertight else
                    "Reconstructed surface is open; enclosed volume is not reliable",
                "bounding_box_extent_difference_mm":
                    (reconstruction.extents - original_mesh.extents).tolist(),
                "reconstruction_watertight": bool(reconstruction.is_watertight),
                "reconstructed_stl": str(result["stl"]),
                "aligned_reconstructed_stl": str(aligned_stl),
                "xy_translation_removed_mm": translation,
                "voxel_size_mm": voxel_size,
                "step_size_mm": result.get("step_size_mm")}
    except (OSError, ValueError, subprocess.TimeoutExpired, ImportError) as exc:
        return {"backend": "volco", "available": True, "status": "ERROR",
                "orientation_not_modeled": True, "git_commit": repository_commit(repo),
                "error": str(exc)}
