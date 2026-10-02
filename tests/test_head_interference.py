"""T21: 헤드 회전식 노즐 간섭 (REP5X).

왜 필요한가: 베드 틸트식(Open5x)에서는 원뿔 레이어가 기계공간에서 수평면이 되어
**간섭이 정의상 불가능**했다(T19). 헤드 회전식은 부품이 고정이라 그 논증이 그대로는
성립하지 않는다. 그런데 3축과 같지도 않다 — 노즐이 **원뿔면의 법선을 따라간다.**
어느 쪽인지는 재 봐야 알고, 그 답이 플랫폼 선택과 최대 각도를 좌우한다.

재 보니 **outward 원뿔에서는 여기서도 간섭이 구조적으로 불가능**했다. 이유가 있다:

    워프 함수 F(x,y,z) = z + r·tanθ 는 **볼록**이다 (r 볼록 + z 선형, tanθ ≥ 0).
    원뿔 레이어는 F 의 등위면이고, 출력 중 이미 놓인 것은 {F ≤ F(p)} 안에 있다.
    공구 축은 ∇F(p) 방향이다. 볼록성이 곧 **지지초평면**이므로

        ∇F(p)·(q − p) ≤ F(q) − F(p) ≤ 0        (이미 놓인 모든 q)

    즉 **놓인 것이 전부 노즐 뒤 반공간에 있다.**

**inward 는 F = z − r·tanθ 로 오목**이라 이 논증이 뒤집힌다 — 그래서 간섭이
살아난다. 이 대비가 이 테스트의 뼈대다. '0% 가 나왔다' 만으로는 검사가 그냥
아무것도 못 잡는 것과 구별이 안 되므로, **잡아야 할 때 잡는지**도 같이 고정한다.

고정하는 성질:
  ① `tool_up=(0,0,1)` 이면 3축 경로와 **결과가 정확히 같다** (같은 기하의 특수 경우)
  ② 공구 축의 Z 성분이 정확히 cos(B) 이고 XY 는 방위각을 따라 돈다
  ③ **outward: 3축은 각도가 커지면 간섭하는데 헤드 회전식은 0%**
  ④ **inward: 헤드 회전식이 3축보다 크게 나쁘다** ← 검사가 실제로 잡는다는 증거
  ⑤ 되감기 쓸림 검출기가 **잡아야 할 때 잡고, 안 잡아야 할 때 안 잡는다**
  ⑥ `stride` 는 팁만 솎고 장애물은 안 솎는다

    python3 -m pytest tests/test_head_interference.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import trimesh

from conical.backtransform import backtransform
from conical.gcode import Move
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.rep5x import REP5X, check_rewind_sweep, move_tool_frames, to_rep5x
from conical.toolpath import HotendProfile, check_nozzle, sample_extrusions
from conical.transform import transform_cone

ROOT = Path(__file__).resolve().parent.parent


def _run(name, angle, direction, layer_height=0.6, stride=6):
    """같은 슬라이싱을 3축으로도 5축으로도 판정해 돌려준다."""
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / name, force="mesh"))
    warped = transform_cone(mesh.vertices, angle, direction) if angle else \
        mesh.vertices
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False),
                       layer_height=layer_height)
    real, _ = backtransform(items, angle, direction)
    pts, mid, _w = sample_extrusions(real)
    _c3, s3 = check_nozzle(pts, mid, stride=stride)
    r5, _ = to_rep5x(real, angle, direction)
    ups = move_tool_frames(r5, REP5X)
    _c5, s5 = check_nozzle(pts, mid, tool_up=ups[mid], stride=stride)
    return s3["collision_pct"], s5["collision_pct"]


def test_vertical_tool_frame_reproduces_three_axis_exactly():
    """`tool_up=(0,0,1)` 이면 3축과 **비트 단위로 같아야 한다.**

    같지 않으면 3축 판정과 5축 판정을 비교할 수 없다 — 기하가 달라져 버리므로
    '5축이 유리하다' 는 결론이 기하 차이의 산물이 된다.
    """
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    warped = transform_cone(mesh.vertices, 30.0, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False), layer_height=0.6)
    real, _ = backtransform(items, 30.0, "outward")
    pts, mid, _w = sample_extrusions(real)
    c_a, s_a = check_nozzle(pts, mid, stride=6)
    up = np.zeros((len(pts), 3))
    up[:, 2] = 1.0
    c_b, s_b = check_nozzle(pts, mid, tool_up=up, stride=6)
    assert np.array_equal(c_a, c_b), "공구 프레임 경로가 3축과 다른 답을 냈다"
    assert s_a["collision_pct"] == s_b["collision_pct"]


def test_tool_frames_follow_the_cone_normal():
    """공구 축의 Z 성분 = cos(B) 이고 XY 는 방위각을 따라 돈다."""
    items = [("move", Move(g=1, x=x, y=y, z=1.0, e=i * 0.1, f=1800.0))
             for i, (x, y) in enumerate([(10, 0), (0, 10), (-10, 0), (7, -7)])]
    for angle in (0.0, 20.0, 45.0):
        out, _ = to_rep5x(items, angle, "outward")
        ups = move_tool_frames(out, REP5X)
        # 헤더의 'G1 B.. F600' 도 이동이므로 마지막 4개만 본다
        ups = ups[-len(items):]
        assert np.allclose(ups[:, 2], math.cos(math.radians(angle)), atol=1e-9)
        assert np.allclose(np.linalg.norm(ups, axis=1), 1.0, atol=1e-12)
        if angle > 0:
            assert np.ptp(ups[:, 0]) > 0.1, "방위각을 안 따라간다"


def test_outward_head_tilt_has_no_interference_where_3axis_does():
    """**핵심 결과.** outward 에서 3축은 간섭하는데 헤드 회전식은 0% 다.

    워프 함수가 볼록이라 이미 놓인 것이 전부 노즐 뒤 반공간에 있기 때문이다
    (모듈 docstring 의 지지초평면 논증). 3축이 멀쩡하면 비교가 성립하지 않으므로
    3축이 실제로 걸리는 각도에서 잰다.
    """
    seen_three_axis_fail = False
    for angle in (24.0, 32.0, 40.0):
        s3, s5 = _run("funnel.stl", angle, "outward")
        if s3 > 0.5:
            seen_three_axis_fail = True
        assert s5 == pytest.approx(0.0, abs=1e-12), \
            f"outward {angle}°: 헤드 회전식에서 간섭 {s5:.2f}% 가 나왔다 " \
            f"(볼록성 논증이 깨졌거나 구현이 틀렸다)"
    assert seen_three_axis_fail, \
        "3축이 한 번도 안 걸렸다 — 비교가 성립하지 않는다 (모델/각도를 올릴 것)"


def test_inward_head_tilt_does_interfere():
    """**inward 에서는 살아난다** — 검사가 그냥 0 만 내는 게 아니라는 증거.

    F = z − r·tanθ 가 오목이라 지지초평면 논증이 뒤집힌다. 그리고 3축보다
    **더 나쁘다** — 헤드가 기울어 출력물 쪽으로 파고들기 때문이다.
    """
    s3, s5 = _run("funnel.stl", 32.0, "inward")
    assert s5 > 10.0, f"inward 인데 헤드 회전식 간섭이 {s5:.2f}% 뿐이다"
    assert s5 > s3, f"inward 에서 헤드({s5:.1f}%)가 3축({s3:.1f}%)보다 나아졌다"


def test_rewind_sweep_detector_fires_when_it_should():
    """되감기 쓸림 검출기가 **잡아야 할 때 잡는다.**

    낮은 자리에서 돌면 옆의 벽을 친다. 이걸 못 잡으면 위 '0/171' 이 '안전' 이
    아니라 '검사가 죽어 있음' 이라는 뜻이 된다.
    """
    hot = HotendProfile()
    wall = []
    e = 0.0
    # 팁 자리 옆 6mm 에 높이 12mm 벽을 세운다 (히트블록 반경 12mm 안)
    for k in range(60):
        e += 0.05
        wall.append(("move", Move(g=1, x=6.0, y=-3.0 + k * 0.1, z=0.2 + k * 0.2,
                                  e=e, f=1800.0, extra="B20.000 C0.000")))
    spin = [("raw", "; V_REWIND BEGIN +1 turn(s)"),
            ("move", Move(g=0, x=0.0, y=0.0, z=0.4, f=3000.0,
                          extra="B20.000 C0.000")),
            ("move", Move(g=0, x=0.0, y=0.0, z=0.4, f=3600.0,
                          extra="B20.000 C360.000")),
            ("raw", "; V_REWIND END")]
    findings, st = check_rewind_sweep(wall + spin, REP5X, hot)
    assert st["checked"] > 0, "되감기 이동을 하나도 못 봤다"
    assert st["swept_hits"] > 0, "벽 옆 낮은 자리에서 돌았는데 못 잡았다"
    assert any(sev == "치명" for sev, _ in findings)


def test_rewind_sweep_is_quiet_when_lifted_clear():
    """충분히 들어올린 뒤 돌면 안 잡아야 한다 (거짓 양성 방지)."""
    hot = HotendProfile()
    wall, e = [], 0.0
    for k in range(60):
        e += 0.05
        wall.append(("move", Move(g=1, x=6.0, y=-3.0 + k * 0.1, z=0.2 + k * 0.2,
                                  e=e, f=1800.0, extra="B20.000 C0.000")))
    top = 0.2 + 59 * 0.2
    spin = [("raw", "; V_REWIND BEGIN +1 turn(s)"),
            ("move", Move(g=0, x=0.0, y=0.0, z=top + 40.0, f=3000.0,
                          extra="B20.000 C0.000")),
            ("move", Move(g=0, x=0.0, y=0.0, z=top + 40.0, f=3600.0,
                          extra="B20.000 C360.000")),
            ("raw", "; V_REWIND END")]
    _f, st = check_rewind_sweep(wall + spin, REP5X, hot)
    assert st["checked"] > 0
    assert st["swept_hits"] == 0, "멀리 들어올렸는데 간섭이라고 했다"


def test_stride_thins_tips_not_obstacles():
    """`stride` 는 **팁만** 솎는다 — 장애물을 솎으면 간섭을 놓친다."""
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    warped = transform_cone(mesh.vertices, 32.0, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False), layer_height=0.8)
    real, _ = backtransform(items, 32.0, "outward")
    pts, mid, _w = sample_extrusions(real)
    _c1, s1 = check_nozzle(pts, mid, stride=1)
    _c8, s8 = check_nozzle(pts, mid, stride=8)
    assert s1["evaluated"] == len(pts)
    assert s8["samples"] == len(pts), "장애물 수가 줄었다"
    assert s8["evaluated"] < s1["evaluated"] / 4
    # 솎아도 같은 자릿수의 답이 나와야 쓸모가 있다
    assert abs(s8["collision_pct"] - s1["collision_pct"]) < \
        max(1.0, 0.3 * s1["collision_pct"]), \
        f"솎은 결과가 너무 다르다: {s1['collision_pct']:.2f} vs {s8['collision_pct']:.2f}"


def test_arm_is_off_unless_radius_given():
    """B_arm 반경은 **확인된 값이 아니다** — 주지 않으면 모델에 안 넣는다.

    모르는 치수를 조용히 가정해 넣으면 '통과' 가 근거 없는 안심이 된다.
    """
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 30.0]])
    mid = np.array([0, 1])
    _c, s_off = check_nozzle(pts, mid)
    assert s_off["arm_modeled"] is False
    _c2, s_on = check_nozzle(pts, mid, arm_length=REP5X.lb, arm_radius=10.0)
    assert s_on["arm_modeled"] is True
    assert s_on["collision_pct"] >= s_off["collision_pct"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
