import os

import pytest

from tools.external.validators import (
    validate_admesh,
    validate_gcode_toolkit,
    validate_mage,
    validate_multi_axis,
)
from tools.run_random_validation import internal_gcode_check


def test_unavailable_admesh_is_not_a_pass(tmp_path):
    path = tmp_path / "box.stl"
    path.write_bytes(b"unused")
    result = validate_admesh(path, str(tmp_path / "missing-admesh.exe"))
    assert result["status"] == "NOT_AVAILABLE"
    assert result["valid"] is None


def test_invalid_gcode_rejected_by_internal_check(tmp_path):
    path = tmp_path / "bad.gcode"
    path.write_text("G1 Xnan Y0\n", encoding="utf-8")
    assert not internal_gcode_check(path)["valid"]


def test_unconfigured_toolkit_and_machine_mismatch(tmp_path):
    path = tmp_path / "small.gcode"
    path.write_text("G1 X1 Y1 E0.1\n", encoding="utf-8")
    assert validate_gcode_toolkit(path, None)["status"] == "NOT_AVAILABLE"
    assert validate_mage()["status"] == "UNSUPPORTED_MACHINE_MODEL"
    assert validate_multi_axis()["status"] == "UNSUPPORTED_MACHINE_MODEL"


def test_built_toolkit_reports_b_c_limitation(tmp_path):
    toolkit_dir = os.environ.get("GCODE_TOOLKIT_DIR")
    if not toolkit_dir:
        pytest.skip("Set GCODE_TOOLKIT_DIR to a compiled gcode-toolkit clone")
    path = tmp_path / "five_axis.gcode"
    path.write_text("G1 X1 Y1 Z1 E0.1 B20 C30\n", encoding="utf-8")
    result = validate_gcode_toolkit(path, toolkit_dir)
    assert result["status"] == "OK"
    assert result["parsed"]
    assert result["unsupported_axes"] == ["B", "C"]
