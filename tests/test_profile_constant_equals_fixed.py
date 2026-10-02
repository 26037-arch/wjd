"""T2: 상수 프로필 = 기존 고정각과 수치적으로 동일 (기존 동작 보호 앵커).

(a) 상수 프로필 30° 메시 워프 정점 == transform_cone 30° (allclose)
(b) 데모 구 파이프라인 G-code 좌표·E == 기존 고정각 30° 결과 (allclose)

    python3 tests/test_profile_constant_equals_fixed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.profile import AngleProfile
from conical.transform import transform_cone, transform_cone_profile


def test_mesh_warp_equal():
    m = center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))
    v_fixed = transform_cone(m.vertices, 30.0, "outward")
    v_prof = transform_cone_profile(m.vertices, AngleProfile.constant(30.0),
                                    "outward")
    assert np.allclose(v_fixed, v_prof, atol=1e-9), \
        f"max diff {np.abs(v_fixed - v_prof).max()}"


def _pipeline_coords(mesh, angle_or_profile):
    if isinstance(angle_or_profile, AngleProfile):
        v = transform_cone_profile(mesh.vertices, angle_or_profile, "outward")
    else:
        v = transform_cone(mesh.vertices, angle_or_profile, "outward")
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    items = slice_mesh(warped, layer_height=0.4)
    out, _ = backtransform(items, angle_or_profile, "outward")
    rows = []
    for k, p in out:
        if k == "move":
            rows.append([p.x if p.x is not None else np.nan,
                         p.y if p.y is not None else np.nan,
                         p.z if p.z is not None else np.nan,
                         p.e if p.e is not None else np.nan])
    return np.array(rows)


def test_pipeline_equal():
    m = center_on_axis(trimesh.creation.icosphere(subdivisions=2, radius=10.0))
    a = _pipeline_coords(m, 30.0)
    b = _pipeline_coords(m, AngleProfile.constant(30.0))
    assert a.shape == b.shape, f"이동 수 다름: {a.shape} vs {b.shape}"
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9), \
        "좌표/E 불일치"


if __name__ == "__main__":
    test_mesh_warp_equal()
    test_pipeline_equal()
    print("PASS: 상수 프로필 30° == 고정각 30° (메시 워프 + 전체 파이프라인 G-code)")
