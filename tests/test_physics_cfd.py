import csv
import math
from pathlib import Path

import pytest

from tools.external.physics_cfd import (
    decode_trajectory,
    generate_openfoam14_case,
    load_material,
    parse_context,
    select_window,
)


def _fixture(tmp_path):
    path = tmp_path / "source.gcode"
    path.write_text("\n".join([
        ';CONICAL_META {"mode":"xyz","layer_height":0.3,"extrusion_width":0.45}',
        "G21", "G90", "M82", "M104 S210", "M140 S60",
        "; layer 0 z=0.3", "G0 X0 Y0 Z0.3 F3600",
        "G1 X1 Y0 E0.1 F1800", "G1 X2 Y0 E0.2 F900",
        "G92 E0", "G91", "M83", "G1 X1 Y0 E0.1 F1200",
    ]) + "\n", encoding="utf-8")
    return path


def test_modal_coordinates_temperature_and_g92(tmp_path):
    _, meta, moves = parse_context(_fixture(tmp_path))
    assert meta["mode"] == "xyz"
    assert [m["delta_e_mm"] for m in moves] == pytest.approx([0, 0.1, 0.1, 0.1])
    assert moves[-1]["end"]["X"] == 3
    assert moves[-1]["end"]["E"] == pytest.approx(0.1)
    assert moves[1]["feed_mm_min"] == 1800
    assert moves[2]["feed_mm_min"] == 900
    assert moves[1]["nozzle_temp_C"] == 210
    assert moves[1]["bed_temp_C"] == 60
    assert select_window(moves, "moves:2-3", 5)[-1]["index"] == 3
    assert select_window(moves, "lines:9-10", 5)[0]["line"] == 9


def test_unsupported_arcs_fail_explicitly(tmp_path):
    path = tmp_path / "arc.gcode"
    path.write_text("G2 X1 Y1 I0 J1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="UNSUPPORTED_GCODE"):
        parse_context(path)


def test_planner_volume_acceleration_and_case(tmp_path):
    pytest.importorskip("pyGCodeDecode")
    decoded = decode_trajectory(_fixture(tmp_path), tmp_path / "physics",
                                spec="moves:2-3", max_path_mm=5)
    rows = list(csv.DictReader(Path(decoded["trajectory_csv"]).open(newline="")))
    times = [float(row["time_s"]) for row in rows]
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    assert decoded["window"]["path_mm"] == pytest.approx(2.0)
    assert decoded["trajectory_integrated_volume_mm3"] == pytest.approx(
        0.2 * math.pi * (1.75/2)**2, rel=2e-5)
    assert decoded["gcode_extrusion_volume_mm3"] == pytest.approx(
        decoded["trajectory_integrated_volume_mm3"], rel=2e-5)
    # The first target F is 30 mm/s, but the planner starts below that speed.
    assert math.hypot(float(rows[0]["vx_mm_s"]),
                      float(rows[0]["vy_mm_s"])) < 30
    material, missing = load_material(
        Path(__file__).resolve().parents[1] / "materials" /
        "pla_4032d_literature_surrogate.yaml")
    assert not missing
    case = generate_openfoam14_case(decoded, material, tmp_path / "case")
    assert case["source_intervals"] > 0
    for name in ("system/blockMeshDict", "system/controlDict", "system/fvSchemes",
                 "system/fvSolution", "constant/fvModels", "0/alpha.polymer",
                 "0/U", "0/p_rgh"):
        assert (tmp_path / "case" / name).is_file()
    assert "volumetricFlowRate table" in (
        tmp_path / "case" / "constant" / "fvModels").read_text()
    assert "source_0" in (tmp_path / "case" / "0" / "U").read_text()


def test_unverified_pla_example_is_rejected():
    material, missing = load_material(
        Path(__file__).resolve().parents[1] / "materials" / "pla.example.yaml")
    assert material["name"] == "PLA_REQUIRES_CALIBRATION"
    assert "density.value" in missing
    assert "surface_tension.value" in missing


def test_rep5x_orientation_is_retained_but_not_sent_to_xyz_cfd(tmp_path):
    pytest.importorskip("pyGCodeDecode")
    path = tmp_path / "rep5x.gcode"
    path.write_text("\n".join([
        ';CONICAL_META {"mode":"rep5x"}', "G21", "G90", "M82",
        "; layer 0 z=0.3", "G0 X0 Y0 Z0.3 B0 C0 F3600",
        "G1 X1 Y0 Z0.3 B10 C20 E0.1 F1800",
    ]) + "\n", encoding="utf-8")
    decoded = decode_trajectory(path, tmp_path / "out", spec="layer:0")
    assert decoded["mode"] == "rep5x"
    assert decoded["orientation_modeled"] is False
    with Path(decoded["trajectory_csv"]).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert float(rows[-1]["B_deg"]) == 10
    assert float(rows[-1]["C_deg"]) == 20
    material, _ = load_material(
        Path(__file__).resolve().parents[1] / "materials" /
        "pla_4032d_literature_surrogate.yaml")
    with pytest.raises(ValueError, match="UNSUPPORTED_REP5X_ORIENTATION"):
        generate_openfoam14_case(decoded, material, tmp_path / "case")
