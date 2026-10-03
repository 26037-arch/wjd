import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conical.deposition_events import parse_motion_events
from conical.rep5x import tool_direction
from conical.rep5x_machine import (
    carriage_for_rtcp,
    carriage_for_rtcp_transforms,
    rtcp_error,
    tip_from_carriage,
    tip_from_transforms,
)
from tools.compare_deposition_resolution import compare
from tools.external.cfd_backend import LocalDepositionInput, OpenFOAMLocalBackend
from tools.external.physics_cfd import generate_openfoam14_case, load_material
from tools.external.validators import (
    align_upstream_volco_mesh,
    validate_hybrid_deposition,
)
from tools.external.volco_oriented import mesh_from_voxels, simulate_gcode


def test_modal_events_and_synchronized_pose():
    events = parse_motion_events([
        "G21", "G90", "M82", "G0 X0 Y0 Z0.2 F600",
        "G1 X1 Y0 B20 C90 E0.1 F600", "G92 E0",
        "G91", "M83", "G1 X1 B10 C20 E0.1 F600",
        "G1 X1 E-0.2 F600",
    ])
    move = events[1]
    pose = move.pose(0.5)
    assert pose["position"] == pytest.approx((0.5, 0, 0.2))
    assert pose["b"] == pytest.approx(10)
    assert pose["c"] == pytest.approx(45)
    assert pose["direction"] == pytest.approx(tool_direction(10, 45))
    assert events[2].end_c == 110  # commanded trajectory, no shortcut wrap
    assert events[2].delta_e == pytest.approx(0.1)
    assert events[3].retracting and not events[3].extruding


def test_mesh_uses_voxel_cell_bounds():
    mesh = mesh_from_voxels({(0, 0, 0)}, 0.1)
    assert mesh.bounds[0] == pytest.approx([0, 0, 0])
    assert mesh.bounds[1] == pytest.approx([0.1, 0.1, 0.1])
    assert mesh.is_watertight
    assert abs(mesh.volume) == pytest.approx(0.001)


def test_upstream_half_voxel_and_xy_padding_are_removed():
    # Upstream exporter centered voxel zero at zero, despite evaluating its
    # occupancy at +h/2; it also adds an XY grid translation.
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh.apply_translation([1.625, 1.625, 0])
    correction = align_upstream_volco_mesh(mesh, {"x": 1.625, "y": 1.625}, 0.1)
    assert correction == [0.05, 0.05, 0.05]
    assert mesh.bounds[0] == pytest.approx([0, 0, 0])
    assert mesh.bounds[1] == pytest.approx([0.1, 0.1, 0.1])


def test_oriented_output_is_incremental_and_conserves_volume(tmp_path):
    gcode = tmp_path / "test.gcode"
    gcode.write_text("\n".join([
        "G21", "G90", "M82", "G0 X0 Y0 Z0.19 B0 C0 F600",
        "G1 X0.1 Y0 E0.01 F600",
        "G1 X0.2 Y0 B20 C0 E0.02 F600",
        "G1 X0.2 Y0.1 B20 C90 E0.03 F600",
        "G1 X0.3 Y0.1 E0.02 F600",  # retraction: no material
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    manifest = simulate_gcode(gcode, out, voxel_size=0.05)
    frames = [json.loads(line) for line in (out / "frames.jsonl").read_text(
        encoding="utf-8").splitlines()]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["orientation_modeled"] is True
    assert manifest["cfd_applied"] is False
    assert all(b["voxel_count"] >= a["voxel_count"] for a, b in
               zip(frames, frames[1:], strict=False))
    assert metrics["target_volume_mm3"] == pytest.approx(
        0.03 * math.pi * (1.75/2)**2)
    assert abs(metrics["deposited_volume_mm3"]-
               metrics["target_volume_mm3"]) <= 0.05**3/2 + 1e-12
    assert metrics["b_range_deg"] == [0, 20]
    assert metrics["c_range_deg"] == [0, 90]
    assert isinstance(metrics["mesh_watertight"], bool)


def test_openfoam_source_spans_nozzle_cross_section(tmp_path):
    # Use the repository material profile; no numerical values are invented.
    material_path = (Path(__file__).resolve().parents[1] / "materials" /
                     "pla_4032d_literature_surrogate.yaml")
    material, missing = load_material(material_path)
    assert not missing
    rows = [
        {"time_s": 0.0, "x_mm": 0, "y_mm": 0, "z_mm": 0.19,
         "E_mm": 0, "vx_mm_s": 10, "vy_mm_s": 0, "vz_mm_s": 0},
        {"time_s": 0.05, "x_mm": 0.5, "y_mm": 0, "z_mm": 0.19,
         "E_mm": 0.05, "vx_mm_s": 10, "vy_mm_s": 0, "vz_mm_s": 0},
    ]
    decoded = {"mode": "xyz", "window": {"layer": 0}, "rows": rows,
               "duration_s": 0.05, "filament_diameter_mm": 1.75,
               "nozzle_diameter_mm": 0.4}
    case = generate_openfoam14_case(decoded, material, tmp_path / "case",
                                    cell_size_mm=0.075)
    assert case["source_cells_min"] > 1
    source_text = (tmp_path / "case" / "constant" / "fvModels").read_text()
    assert "type containsPoints" in source_text
    assert "volumetricFlowRate table" in source_text


def test_calibrated_machine_rtcp_for_combined_xyz_b_c():
    cp, bp, tip_home = (1.2, -0.4, 52), (1.5, 0.2, 48), (1.5, 0.2, -6)
    h_t_c, c_t_b, b_t_n = (np.eye(4), np.eye(4), np.eye(4))
    h_t_c[:3, 3] = cp
    c_t_b[:3, 3] = np.subtract(bp, cp)
    b_t_n[:3, 3] = np.subtract(tip_home, bp)
    for xyz, b, c in [((0, 0, 0.2), 0, 0), ((12, -4, 6), 25, 0),
                      ((12, -4, 6), 0, 90), ((-3, 8, 2), 25, -170)]:
        carriage = carriage_for_rtcp(xyz, b, c, cp, bp, tip_home)
        assert tip_from_carriage(carriage, b, c, cp, bp,
                                 tip_home) == pytest.approx(xyz)
        assert rtcp_error(xyz, b, c, cp, bp, tip_home) < 1e-12
        carriage_matrix = carriage_for_rtcp_transforms(
            xyz, b, c, h_t_c, c_t_b, b_t_n)
        assert carriage_matrix == pytest.approx(carriage)
        assert tip_from_transforms(carriage_matrix, b, c, h_t_c,
                                   c_t_b, b_t_n) == pytest.approx(xyz)


def test_hybrid_does_not_claim_unrun_cfd(tmp_path):
    gcode = tmp_path / "one.gcode"
    gcode.write_text("G21\nG90\nM82\nG0 X0 Y0 Z0.19 F600\n"
                     "G1 X0.1 Y0 E0.01 F600\n", encoding="utf-8")
    result = validate_hybrid_deposition(gcode, tmp_path / "hybrid",
                                         voxel_size=0.05)
    assert result["status"] == "NOT_AVAILABLE"
    assert result["cfd_applied"] is False
    assert result["volco_oriented"]["status"] == "OK"
    assert (tmp_path / "hybrid" / "hybrid_status.json").is_file()


def test_local_cfd_boundary_carries_physical_inputs_without_claiming_correction():
    event = parse_motion_events(["G21", "G90", "M83",
                                 "G1 X1 Y0 Z0.2 B20 C90 E0.1 F600"])[0]
    request = LocalDepositionInput.from_event(
        event, 0.5, nozzle_diameter_mm=0.4, filament_diameter_mm=1.75)
    assert request.tip_xyz_mm == pytest.approx((0.5, 0, 0.1))
    assert request.tool_direction == pytest.approx(tool_direction(10, 45))
    assert request.volumetric_flow_mm3_s == pytest.approx(
        math.pi * (1.75 / 2)**2 * 0.1 / event.estimated_duration)
    assert request.nozzle_velocity_mm_s == pytest.approx(
        tuple((b-a) / event.estimated_duration for a, b in
              zip(event.start_position, event.end_position, strict=True)))
    backend = OpenFOAMLocalBackend()
    assert not backend.available()
    assert backend.simulate_local_deposition(request)["cfd_applied"] is False


def test_resolution_comparison_separates_excess_and_missing(tmp_path):
    coarse, fine = tmp_path / "coarse", tmp_path / "fine"
    for folder, h, additions in [
        (coarse, 0.1, [[0, 0, 0]]),
        (fine, 0.05, [[i, j, k] for i in range(2)
                      for j in range(2) for k in range(2)
                      if (i, j, k) != (1, 1, 1)] + [[3, 0, 0]]),
    ]:
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({
            "backend": "volco_oriented_extension", "source_gcode": "same.gcode",
            "step_size_mm": 0.01, "voxel_size_mm": h,
            "coordinate_convention": "common", "files": {"frames": "frames.jsonl"}}),
            encoding="utf-8")
        (folder / "frames.jsonl").write_text(
            json.dumps({"added": additions}) + "\n", encoding="utf-8")
    result = compare(coarse, fine)
    assert result["iou"] == pytest.approx(7/9)
    assert result["coarse_excess_mm3"] == pytest.approx(0.05**3)
    assert result["coarse_missing_mm3"] == pytest.approx(0.05**3)
