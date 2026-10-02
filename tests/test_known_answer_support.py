"""T25: **정답을 미리 아는** 형상으로 지지 검사기를 검정한다.

왜 필요한가 (2026-09-23): 검사기의 수치를 계속 결론에 써 왔는데, 정답이 알려진
입력으로 검정한 적이 없었다. 그래서 검사기를 고치거나 판정 기준을 바꿀 때
**결과가 이상해도 그게 형상 탓인지 검사기 탓인지 가릴 수단이 없었다.**

실제로 이 테스트가 먼저 있었다면 오경보 하나를 막았다: 연쇄 판정을 붙이고 재 보니
수직 원기둥에서 미지지 24% 가 나와 "검사기가 깨졌다" 고 볼 뻔했는데, 그건
**인필**이었고 결론에 쓰는 지표(`overhang_pct`, 페리미터만)는 0.000%p 였다.

고정하는 것:
  ① 수직 원기둥 θ=0 → 진짜 오버행 0%p (완전 자기지지)
  ② 직육면체 θ=0 → 0%p
  ③ 위 둘은 **연쇄 판정을 켜도** 0%p 여야 한다 (연쇄가 멀쩡한 기하를 깨면 안 된다)
  ④ 아래로 벌어지는 원뿔(천장이 넓어짐) θ=0 → **0 보다 커야 한다** (검사기가
     오버행을 실제로 잡는지 — 전부 0 을 내는 '죽은 검사기' 를 구별한다)

⚠ ①②가 0 인 것은 '검사기가 옳다' 는 증명이 아니라 **필요조건**이다.
  ④를 같이 두는 이유가 그것이다 — 통과만 시키는 검사기도 ①②는 통과한다.

    python3 -m pytest tests/test_known_answer_support.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from compare_waist import LAYER_H
from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.profile import AngleProfile
from conical.toolpath import check_support, sample_extrusions, support_breakdown
from conical.transform import transform_cone_profile


def _overhang(mesh, theta=0.0, chained=False, vwin_factor=1.5):
    prof = AngleProfile.constant(theta)
    v = transform_cone_profile(mesh.vertices, prof, "outward")
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    real, _ = backtransform(slice_mesh(warped, layer_height=LAYER_H),
                            prof, "outward")
    pts, mid, w, kinds, lay = sample_extrusions(real, return_types=True,
                                                return_layers=True)
    sup, _st = check_support(pts, mid, w, layer_height=LAYER_H,
                             vwin_factor=vwin_factor,
                             require_supported_below=chained)
    return support_breakdown(pts, mid, kinds, lay, sup, w)["overhang_pct"]


def _cylinder():
    return center_on_axis(trimesh.creation.cylinder(radius=8.0, height=20.0,
                                                    sections=64))


def _box():
    return center_on_axis(trimesh.creation.box(extents=[12.0, 12.0, 20.0]))


def _flaring_cone():
    """위로 갈수록 넓어지는 원뿔대 — 벽 경사가 수직 기준 60° 라 확실한 오버행."""
    H = 12.0
    r_bot, r_top = 2.0, 2.0 + H * np.tan(np.radians(60.0))
    pts = [(r, z) for r, z in zip(np.linspace(r_bot, r_top, 30),
                                  np.linspace(0.0, H, 30), strict=True)]
    pts += [(0.0, H)]
    m = trimesh.creation.revolve(np.array(pts), sections=64)
    m.merge_vertices()
    m.fix_normals()
    return center_on_axis(m)


def test_vertical_cylinder_is_fully_supported():
    assert _overhang(_cylinder()) == 0.0


def test_box_is_fully_supported():
    assert _overhang(_box()) == 0.0


def test_chained_judgement_does_not_break_sound_geometry():
    """연쇄 판정(엄격)을 켜도 자기지지 형상은 0 이어야 한다.

    이게 깨지면 연쇄 구현이 멀쩡한 기하를 끊고 있다는 뜻이고, 연쇄로 잰
    어떤 수치도 믿을 수 없다.
    """
    assert _overhang(_cylinder(), chained=True) == 0.0
    assert _overhang(_box(), chained=True) == 0.0


def test_checker_actually_detects_overhang():
    """'전부 0 을 내는 검사기' 를 배제한다 — 진짜 오버행은 잡혀야 한다."""
    oh = _overhang(_flaring_cone())
    assert oh > 0.5, f"60° 벌어지는 원뿔에서 오버행 {oh:.3f}%p — 검사기가 죽어 있다"


if __name__ == "__main__":
    test_vertical_cylinder_is_fully_supported()
    test_box_is_fully_supported()
    test_chained_judgement_does_not_break_sound_geometry()
    test_checker_actually_detects_overhang()
    print("T25 OK")
