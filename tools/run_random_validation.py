"""Repeatable conical-angle study with independent checks where supported.

Run from any directory: python tools/run_random_validation.py --count 100 ...
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conical.config import DEFAULT_K, MAX_ANGLE_DEG
from conical.gcode import parse
from conical.pipeline import generate_conical_gcode
from conical.rep5x import REP5X, check_head_interference, check_rep5x
from conical.selector import select_cone
from conical.test_models import generate_random_model
from conical.toolpath import HotendProfile, check_support, sample_extrusions
from tools.external.validators import (
    builtin_mesh_check, unavailable, validate_admesh, validate_gcode_toolkit,
    validate_mage, validate_multi_axis, validate_pymeshlab, validate_volco,
)


def clean(value):
    """JSON-safe finite values; NaN and infinity must be null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def internal_gcode_check(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items = parse(lines)
    errors = []
    moves = 0
    for n, (kind, payload) in enumerate(items, 1):
        code = lines[n-1].split(";", 1)[0].strip()
        if re.match(r"^G[01](?:\.0+)?(?:\s|$)", code, re.I):
            remainder = re.sub(r"^G[01](?:\.0+)?", "", code, count=1, flags=re.I)
            remainder = re.sub(r"[A-Za-z][-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                               "", remainder)
            if remainder.strip():
                errors.append({"line": n, "reason": "malformed movement word"})
            if kind != "move":
                errors.append({"line": n, "reason": "movement not parsed"})
            else:
                moves += 1
                for key in ("x", "y", "z", "e", "f"):
                    v = getattr(payload, key)
                    if v is not None and not math.isfinite(v):
                        errors.append({"line": n, "reason": f"nonfinite {key}"})
    if not moves:
        errors.append({"reason": "no G0/G1 moves"})
    return {"valid": not errors, "moves": moves, "errors": errors[:30],
            "scope": "repository G0/G1 parser and finite numeric coordinates"}


def support_metrics(items, layer_height):
    pts, mid, weights = sample_extrusions(items)
    result = {}
    for name, chained in (("optimistic", False), ("chained", True)):
        _, stats = check_support(pts, mid, weights, layer_height=layer_height,
                                 require_supported_below=chained)
        result[name] = stats
    return result


def run_angle(mesh, model_id, angle, direction, args, output_dir):
    key = f"{angle:g}_{direction}"
    gcode_path = output_dir / "gcode" / f"model_{model_id:05d}_{key}.gcode"
    result = {"angle_deg": angle, "direction": direction,
              "gcode_path": str(gcode_path), "status": "OK"}
    try:
        pipeline = generate_conical_gcode(
            mesh, angle, direction, gcode_path,
            layer_height=args.layer_height, max_edge=args.max_edge,
            infill_spacing=args.infill_spacing)
        result["generation"] = {k: v for k, v in pipeline.items()
                                if k != "physical_items" and k != "rep5x_items"}
    except Exception as exc:
        result.update(status="GCODE_GENERATION_ERROR", error=str(exc))
        return result
    internal_gcode = internal_gcode_check(gcode_path)
    external_gcode = (validate_gcode_toolkit(gcode_path, args.toolkit_dir)
                      if args.gcode_validator == "gcode-toolkit" else
                      unavailable("gcode-toolkit", "not selected", "NOT_SELECTED"))
    result["gcode_validation"] = {"internal": internal_gcode,
                                   "external": external_gcode}
    if not internal_gcode["valid"]:
        result["status"] = "GCODE_INVALID"
        return result
    findings, kin_stats = check_rep5x(pipeline["rep5x_items"], REP5X)
    fatal = any(sev == "치명" for sev, _ in findings)
    result["kinematics"] = {
        "internal": {"valid": not fatal, "findings": findings, "stats": kin_stats},
        "external": {"mage": validate_mage(args.mage_dir) if args.collision_validator == "mage"
                     else unavailable("mage", "not selected", "NOT_SELECTED"),
                     "multi_axis_motion_planning":
                     validate_multi_axis(args.multi_axis_dir)
                     if args.collision_cross_validator == "multi-axis"
                     else unavailable("multi-axis", "not selected", "NOT_SELECTED")}}
    try:
        _, col_stats = check_head_interference(
            pipeline["rep5x_items"], REP5X, HotendProfile(), stride=args.collision_stride)
        result["collision"] = {"internal": col_stats,
                               "external": result["kinematics"]["external"]}
    except Exception as exc:
        result["collision"] = {"internal": {"status": "ERROR", "error": str(exc)},
                               "external": result["kinematics"]["external"]}
    try:
        result["support"] = {"internal": support_metrics(pipeline["physical_items"],
                                                            args.layer_height),
                             "external": {"status": "NO_SUITABLE_VALIDATOR"}}
    except Exception as exc:
        result["support"] = {"internal": {"status": "ERROR", "error": str(exc)},
                             "external": {"status": "NO_SUITABLE_VALIDATOR"}}
    result["geometry"] = {"volco":
        validate_volco(gcode_path, mesh, output_dir / "reconstructed" / f"model_{model_id:05d}_{key}",
                       args.volco_dir, seed=args.seed + model_id,
                       voxel_size=args.volco_voxel_size)
        if args.geometry_validator == "volco"
        else unavailable("volco", "not selected", "NOT_SELECTED")}
    if fatal:
        result["status"] = "KINEMATIC_FAIL"
    elif result["collision"]["internal"].get("collision_pct", 0) > 0:
        result["status"] = "COLLISION"
    elif args.geometry_validator == "volco" and result["geometry"]["volco"]["status"] == "ERROR":
        result["status"] = "GEOMETRY_RECONSTRUCTION_FAILED"
    return result


def mesh_checks(mesh, stl_path, args):
    checks = {"builtin": builtin_mesh_check(mesh),
              "pymeshlab": (validate_pymeshlab(stl_path)
                            if args.mesh_validator == "pymeshlab" else
                            unavailable("pymeshlab", "not selected", "NOT_SELECTED")),
              "admesh": (validate_admesh(stl_path, args.admesh_exe)
                         if args.mesh_cross_validator == "admesh" else
                         unavailable("admesh", "not selected", "NOT_SELECTED"))}
    # Only an actual independent invalid verdict rejects a mesh. Missing or
    # errored external checks are never counted as passes.
    accepted = checks["builtin"]["valid"] and all(
        checks[name]["valid"] is not False for name in ("pymeshlab", "admesh"))
    checks["mesh_accepted"] = accepted
    checks["mesh_validated_externally"] = bool(
        checks["pymeshlab"].get("valid") is True and
        (args.mesh_cross_validator != "admesh" or
         checks["admesh"].get("valid") is True))
    checks["validator_disagreement"] = (
        checks["pymeshlab"].get("valid") is not None and
        checks["admesh"].get("valid") is not None and
        checks["pymeshlab"]["valid"] != checks["admesh"]["valid"])
    return checks


def summarize_sweep(selector_angle, angles):
    eligible = [a for a in angles if a["status"] == "OK"]
    def best(metric):
        values = [(metric(a), a["angle_deg"]) for a in eligible]
        values = [(v, angle) for v, angle in values if v is not None]
        return min(values)[1] if values else None
    geometry = best(lambda a: a.get("geometry", {}).get("volco", {}).get(
        "sampled_bidirectional_chamfer_mm"))
    support = best(lambda a: a.get("support", {}).get("internal", {}).get(
        "chained", {}).get("unsupported_pct"))
    paired = []
    for a in eligible:
        geom = a.get("geometry", {}).get("volco", {}).get(
            "sampled_bidirectional_chamfer_mm")
        sup = a.get("support", {}).get("internal", {}).get(
            "chained", {}).get("unsupported_pct")
        if geom is not None and sup is not None:
            paired.append((a["angle_deg"], geom, sup))
    pareto = [angle for angle, geom, sup in paired
              if not any((other_geom <= geom and other_sup <= sup and
                          (other_geom < geom or other_sup < sup))
                         for other_angle, other_geom, other_sup in paired
                         if other_angle != angle)]
    return {"internally_eligible_angles": [a["angle_deg"] for a in eligible],
            "theta_best_geometry": geometry,
            "theta_best_support_internal": support,
            "delta_geometry_deg": abs(selector_angle-geometry) if geometry is not None else None,
            "delta_support_deg": abs(selector_angle-support) if support is not None else None,
            "pareto_geometry_vs_internal_support_angles": pareto,
            "note": "No weighted optimum; support metric is internal, not independent"}


def run_case(model_id, args, output_dir):
    case = {"model_id": model_id, "seed": args.seed, "status": "OK",
            "generation_attempts": []}
    mesh = None
    for attempt in range(args.max_generation_retries + 1):
        try:
            mesh, metadata = generate_random_model(args.seed, model_id, attempt)
            stl_path = output_dir / "models" / f"model_{model_id:05d}_attempt_{attempt}.stl"
            mesh.export(stl_path)
            checks = mesh_checks(mesh, stl_path, args)
            case["generation_attempts"].append({"mesh": metadata,
                                                  "stl_path": str(stl_path),
                                                  "mesh_validation": checks})
            if checks["mesh_accepted"]:
                case.update(mesh=metadata, stl_path=str(stl_path),
                            mesh_validation=checks)
                break
        except Exception as exc:
            case["generation_attempts"].append({"attempt": attempt,
                                                  "error": str(exc)})
    else:
        case["status"] = ("MESH_INVALID" if any(
            "mesh_validation" in a for a in case["generation_attempts"])
                          else "MODEL_GENERATION_FAILED")
        return case
    try:
        best, candidates = select_cone(mesh, args.selector_k, verbose=False)
        baseline = next((c["support"] for c in candidates
            if c["angle"] == 0), None)
        case["prediction"] = {"angle_deg": best["angle"],
                              "direction": best["direction"], "score": best["J"],
                              "support_baseline": baseline,
                              "support_predicted": best["support"],
                              "k": args.selector_k}
    except Exception as exc:
        case.update(status="SELECTOR_ERROR", error=str(exc))
        return case
    chosen = run_angle(mesh, model_id, best["angle"], best["direction"], args, output_dir)
    case["selected_angle_result"] = chosen
    case["status"] = chosen["status"]
    if args.angle_sweep:
        grid = set(range(0, args.angle_max + 1, args.angle_step)) | {int(best["angle"])}
        sweep = []
        for angle in sorted(grid):
            sweep.append(chosen if angle == best["angle"] else
                         run_angle(mesh, model_id, angle, best["direction"], args, output_dir))
        case["angle_sweep"] = sweep
        case["sweep_comparison"] = summarize_sweep(best["angle"], sweep)
    case["validator_disagreement"] = bool(
        case["mesh_validation"]["validator_disagreement"] or
        (chosen.get("gcode_validation", {}).get("external", {}).get("parsed") is False
         and chosen.get("gcode_validation", {}).get("internal", {}).get("valid") is True))
    requested = [
        case["mesh_validation"]["pymeshlab"] if args.mesh_validator else None,
        case["mesh_validation"]["admesh"] if args.mesh_cross_validator else None,
        chosen.get("gcode_validation", {}).get("external") if args.gcode_validator else None,
        chosen.get("kinematics", {}).get("external", {}).get("mage") if args.collision_validator else None,
        chosen.get("kinematics", {}).get("external", {}).get("multi_axis_motion_planning")
        if args.collision_cross_validator else None,
        chosen.get("geometry", {}).get("volco") if args.geometry_validator else None]
    case["requested_external_checks_complete"] = bool(
        requested and all(r and r.get("status") == "OK" for r in requested))
    # Neither candidate verifies REP5X head-head collision, and support still
    # requires a physical experiment. Full independent validation is impossible.
    case["external_validation_complete"] = False
    return case


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--output-dir", type=Path, default=Path("results/random_validation"))
    p.add_argument("--max-generation-retries", type=int, default=3)
    p.add_argument("--layer-height", type=float, default=0.3)
    p.add_argument("--selector-k", type=float, default=DEFAULT_K)
    p.add_argument("--max-edge", type=float, default=1.5)
    p.add_argument("--infill-spacing", type=float, default=2.5)
    p.add_argument("--collision-stride", type=int, default=8)
    p.add_argument("--mesh-validator", choices=["pymeshlab"])
    p.add_argument("--mesh-cross-validator", choices=["admesh"])
    p.add_argument("--admesh-exe", default="admesh")
    p.add_argument("--gcode-validator", choices=["gcode-toolkit"])
    p.add_argument("--toolkit-dir", default=os.environ.get("GCODE_TOOLKIT_DIR"))
    p.add_argument("--collision-validator", choices=["mage"])
    p.add_argument("--mage-dir", default=os.environ.get("MAGE_SIMULATOR_DIR"))
    p.add_argument("--collision-cross-validator", choices=["multi-axis"])
    p.add_argument("--multi-axis-dir", default=os.environ.get("MULTI_AXIS_DIR"))
    p.add_argument("--geometry-validator", choices=["volco"])
    p.add_argument("--volco-dir", default=os.environ.get("VOLCO_DIR"))
    p.add_argument("--volco-voxel-size", type=float, default=0.1)
    p.add_argument("--angle-sweep", action="store_true")
    p.add_argument("--angle-step", type=int, default=4)
    p.add_argument("--angle-max", type=int, default=MAX_ANGLE_DEG)
    args = p.parse_args(argv)
    if args.count < 1 or args.max_generation_retries < 0 or args.angle_step < 1 or not 0 <= args.angle_max <= MAX_ANGLE_DEG:
        p.error("count/angle-step must be positive; retries >= 0 and angle-max in configured range")
    output = args.output_dir.resolve()
    for name in ("models", "gcode", "reconstructed"):
        (output / name).mkdir(parents=True, exist_ok=True)
    statuses = Counter()
    mesh_external = Counter()
    gcode_external = Counter()
    geometry_external = Counter()
    disagreement = 0
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for model_id in range(args.count):
            case = clean(run_case(model_id, args, output))
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            statuses[case["status"]] += 1
            disagreement += bool(case.get("validator_disagreement"))
            mesh_external[case.get("mesh_validation", {}).get("pymeshlab", {}).get(
                "status", "NOT_REACHED")] += 1
            selected = case.get("selected_angle_result", {})
            gcode_external[selected.get("gcode_validation", {}).get("external", {}).get(
                "status", "NOT_REACHED")] += 1
            geometry_external[selected.get("geometry", {}).get("volco", {}).get(
                "status", "NOT_REACHED")] += 1
            print(f"[{model_id+1}/{args.count}] {case['status']}", flush=True)
    summary = {"seed": args.seed, "count": args.count,
               "statuses": dict(statuses), "validator_disagreement_cases": disagreement,
               "pymeshlab_statuses": dict(mesh_external),
               "gcode_toolkit_statuses": dict(gcode_external),
               "volco_statuses": dict(geometry_external),
               "external_validation_complete_cases": 0,
               "note": "No physical printing experiment was performed"}
    (output / "summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False,
                                                       indent=2, allow_nan=False),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
