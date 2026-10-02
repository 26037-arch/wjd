import trimesh
import pytest
from types import SimpleNamespace

from conical.test_models import box
from tools.external.validators import builtin_mesh_check, validate_pymeshlab
from tools import run_random_validation as runner


def test_closed_box_and_open_box(tmp_path):
    closed = box(4, 4, 4)
    good_path = tmp_path / "good.stl"
    closed.export(good_path)
    assert builtin_mesh_check(closed)["valid"]
    broken = trimesh.Trimesh(vertices=closed.vertices,
                             faces=closed.faces[:-1], process=False)
    broken_path = tmp_path / "broken.stl"
    broken.export(broken_path)
    assert not builtin_mesh_check(broken)["valid"]


def test_pymeshlab_distinguishes_closed_and_open(tmp_path):
    closed = box(4, 4, 4)
    good_path = tmp_path / "good.stl"
    closed.export(good_path)
    good = validate_pymeshlab(good_path)
    if not good["available"]:
        pytest.skip("PyMeshLab is not installed")
    assert good["valid"] is True
    broken = trimesh.Trimesh(vertices=closed.vertices,
                             faces=closed.faces[:-1], process=False)
    broken_path = tmp_path / "broken.stl"
    broken.export(broken_path)
    bad = validate_pymeshlab(broken_path)
    assert bad["valid"] is False
    assert bad["boundary_edges"] > 0


def test_disagreement_is_recorded(monkeypatch, tmp_path):
    mesh = box(4, 4, 4)
    path = tmp_path / "box.stl"
    mesh.export(path)
    monkeypatch.setattr(runner, "validate_pymeshlab", lambda _: {
        "backend": "pymeshlab", "status": "OK", "valid": True})
    monkeypatch.setattr(runner, "validate_admesh", lambda *_: {
        "backend": "admesh", "status": "OK", "valid": False})
    args = SimpleNamespace(mesh_validator="pymeshlab",
                           mesh_cross_validator="admesh", admesh_exe="admesh")
    result = runner.mesh_checks(mesh, path, args)
    assert result["validator_disagreement"]
    assert not result["mesh_accepted"]
