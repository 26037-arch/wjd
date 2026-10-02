"""T19: 5축 도달 가능 영역과 최대 각도 (기계를 만들기 전에 정해야 하는 것).

왜 필요한가: 3축 원뿔의 최대각은 **노즐 간섭**이 정했다(검사기 B). 5축에서는
그 한계가 사라지고 **기계 도달**이 대신 정한다고 주장한다. 그 주장 위에 기계
설계 결정(X/Z 이동거리, 피벗 스탠드오프, Z 영점)을 얹을 것이므로, 해석식이
조용히 틀리면 **기계를 잘못 만든다.** 되돌릴 수 없는 종류의 오류다.

고정하는 성질:
  ① 해석식이 `open5x._map_point` 과 **정확히** 일치한다 (같은 변환의 두 표현)
  ② 원뿔 레이어 위에서 기계 Z 가 **r 과 무관한 상수**다 ← 간섭 불가능의 근거
  ③ 실제 G-code 에서 기계 Z 가 단조증가하고 층간격이 h·cosθ 다
  ④ 필요 이동거리(스팬)에 **피벗 깊이 d 가 안 들어간다** (d 는 위치만 정한다)
  ⑤ **대각선 규칙**: 0~90° 전 구간 가능 ⟺ √(R²+H²) ≤ min(X이동, Z이동)
  ⑥ 데이텀 고정(Z 소프트리밋 0)이면 **어떤 각도도 안 된다** — 베드가 음수로 내려간다
  ⑦ 막는 것을 이름으로 정확히 지목한다 (베드 반경 / 틸트 한계)

    python3 -m pytest tests/test_envelope.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import trimesh

from conical.backtransform import backtransform
from conical.envelope import (
    MachineEnvelope,
    check_planar_stacking,
    constraints_at,
    layer_machine_z,
    machine_xz,
    max_angle,
    part_samples,
    requirement,
)
from conical.meshio import center_on_axis
from conical.open5x import PRUSA_UV, MachineProfile, _map_point, to_open5x
from conical.planar_slicer import slice_mesh
from conical.transform import transform_cone

ROOT = Path(__file__).resolve().parent.parent


def _box(radius, height, n=12):
    r = np.concatenate([np.full(n, 0.0), np.full(n, radius),
                        np.linspace(0, radius, n), np.linspace(0, radius, n)])
    z = np.concatenate([np.linspace(0, height, n), np.linspace(0, height, n),
                        np.full(n, 0.0), np.full(n, height)])
    return r, z


def test_analytic_matches_open5x_transform():
    """해석식과 실제 변환 코드가 같은 것이어야 한다 — 하나만 고쳐도 조용히 갈린다."""
    prof = PRUSA_UV
    for theta in (0.0, 7.5, 20.0, 45.0, 72.0):
        for r in (0.0, 1.0, 14.0, 60.0):
            for z in (0.0, 0.3, 25.0, 140.0):
                # 압출점은 방위각 0 으로 회전돼 온다 → 부품좌표 (r, 0, z)
                x_c, y_c, z_c, _v = _map_point(r, 0.0, z, theta, prof, 0.0)
                x_a, z_a = machine_xz(r, z, theta, prof)
                assert abs(float(x_a) - x_c) < 1e-9, \
                    f"θ={theta} r={r} z={z}: x {float(x_a)} vs {x_c}"
                assert abs(float(z_a) - z_c) < 1e-9, \
                    f"θ={theta} r={r} z={z}: z {float(z_a)} vs {z_c}"
                assert abs(y_c) < 1e-12, "기계 Y 는 0 이어야 한다 (방위각 축퇴)"


def test_cone_layer_machine_z_is_constant_in_r():
    """원뿔 레이어 `z = z₀ − r·tanθ` 위에서 기계 Z 가 r 과 무관하다.

    이게 '5축 원뿔 = 기계공간의 평면 적층' 의 전부이고, 거기서 '노즐-출력물
    간섭이 정의상 불가능' 이 나온다. 이 성질이 깨지면 그 주장 전체가 무너진다.
    """
    for theta in (5.0, 20.0, 45.0, 70.0):
        t = math.tan(math.radians(theta))
        for z0 in (0.0, 3.7, 55.0):
            rs = np.linspace(0.0, 80.0, 41)
            _x, zm = machine_xz(rs, z0 - rs * t, theta, PRUSA_UV)
            assert float(np.ptp(zm)) < 1e-9, \
                f"θ={theta} z0={z0}: 레이어 안에서 기계 Z 가 {np.ptp(zm):.2e} 만큼 변한다"
            assert abs(float(zm[0]) - layer_machine_z(z0, theta)) < 1e-9


def test_layer_machine_z_is_monotone_in_layer_index():
    """레이어 순서대로 기계 Z 가 **증가**해야 한다 (감소하면 노즐이 파고든다)."""
    for theta in (0.0, 20.0, 45.0, 80.0):
        zs = [layer_machine_z(z0, theta) for z0 in np.arange(0, 50, 0.3)]
        assert all(b > a for a, b in zip(zs, zs[1:], strict=False)), f"θ={theta} 에서 비단조"


def test_real_gcode_is_planar_stacking():
    """실제 파이프라인 G-code 에서 ②가 성립하는지 — 해석이 아니라 실측."""
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    for ang in (20.0, 45.0):
        warped = transform_cone(mesh.vertices, ang, "outward")
        items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                           process=False), layer_height=0.3)
        real, _ = backtransform(items, ang, "outward")
        machine, _ = to_open5x(real, ang, "outward", PRUSA_UV)
        st = check_planar_stacking(machine)
        assert st["monotone"], \
            f"θ={ang}: 압출 중 기계 Z 가 {st['max_drop']:.4f}mm 내려갔다"
        assert st["flat"] < 1e-9, \
            f"θ={ang}: 한 층 안에서 기계 Z 가 {st['flat']:.2e} 만큼 변한다"
        assert st["layer_dz"] == pytest.approx(
            0.3 * math.cos(math.radians(ang)), rel=1e-6), \
            f"θ={ang}: 층간격이 h·cosθ 가 아니다 ({st['layer_dz']:.5f})"


def test_spans_do_not_depend_on_pivot_depth():
    """필요 **이동거리**에는 피벗 깊이가 약분된다 — d 는 위치만 정한다.

    스탠드오프(30/50/70mm)를 고르는 근거가 여기서 갈린다: 이동거리를 사려고
    d 를 바꾸는 것은 의미가 없고, 베드를 어디에 달지가 바뀐다.
    """
    r, z = _box(40.0, 90.0)
    for theta in (15.0, 45.0, 70.0):
        spans = []
        for d in (30.0, 50.0, 70.0):
            prof = MachineProfile(pivot_depth=d)
            q = requirement(r, z, theta, prof)
            spans.append((q["x_span"], q["z_span"]))
            # 위치는 실제로 바뀐다 (약분되는 것이 스팬뿐임을 같이 고정)
            assert q["bed_center_x"] == pytest.approx(
                -math.sin(math.radians(theta)) * d, rel=1e-9)
        for a, b in zip(spans, spans[1:], strict=False):
            assert a[0] == pytest.approx(b[0], rel=1e-9), f"X 스팬이 d 에 의존: {spans}"
            assert a[1] == pytest.approx(b[1], rel=1e-9), f"Z 스팬이 d 에 의존: {spans}"


def test_diagonal_rule_is_exact():
    """**0~90° 전 구간 가능 ⟺ √(R²+H²) ≤ min(X이동, Z이동).**

    기계를 만들기 전에 이동거리를 정하는 규칙이다. 틸트가 부품 경계상자를
    기계공간에서 그냥 θ 만큼 회전시키므로, 어느 각도에선가 대각선이 X 축과
    나란해지고 또 어느 각도에선가 Z 축과 나란해진다.
    """
    xt, zt = 250.0, 210.0
    env = MachineEnvelope(x_travel=xt, z_travel=zt, bed_radius=1e9,
                          max_tilt_deg=90.0, datum="free")
    checked = 0
    for R in (10, 30, 50, 70, 90, 110, 130, 150, 170, 190):
        for H in (20, 60, 100, 140, 180, 220, 260):
            r, z = _box(float(R), float(H))
            got = max_angle(r, z, env, step=0.5)["max_angle"] == 90.0
            want = math.hypot(R, H) <= min(xt, zt) + 1e-9
            assert got == want, \
                f"R={R} H={H} 대각 {math.hypot(R, H):.1f}: 규칙 {want}, 실제 {got}"
            checked += 1
    assert checked == 70


def test_fixed_datum_blocks_every_angle():
    """Z 소프트리밋이 0 이고 Z 영점이 베드면이면 **어떤 각도도 안 된다.**

    틸트하면 베드면이 기계 Z 로 −d(1−cosθ) 만큼 내려간다. 실제 출력에서
    −2.45mm 로 걸렸던 것(검사기 C)의 일반형이고, 기계를 만들기 전에 Z 영점을
    정해야 하는 이유다.
    """
    r, z = _box(14.0, 5.0)
    env = MachineEnvelope(x_min=-125.0, x_max=125.0, z_min=0.0, z_max=210.0,
                          bed_radius=90.0, max_tilt_deg=90.0, datum="fixed")
    res = max_angle(r, z, env, PRUSA_UV, step=0.25)
    assert res["max_angle"] == 0.0, f"0° 초과가 통과했다: {res}"
    assert "기계 Z 범위" in res["blockers"], res["blockers"]
    # 침하량이 해석식과 맞는지 같이 본다
    for th in (10.0, 20.0, 45.0):
        sink = requirement(r, z, th, PRUSA_UV)["bed_sink"]
        assert sink == pytest.approx(
            -PRUSA_UV.pivot_depth * (1 - math.cos(math.radians(th))), rel=1e-9)


def test_blockers_are_named_correctly():
    """막는 것을 뭉뚱그리지 않고 지목해야 한다 — 고칠 곳이 달라진다."""
    r, z = _box(120.0, 60.0)
    env = MachineEnvelope(x_travel=400.0, z_travel=400.0, bed_radius=90.0,
                          max_tilt_deg=90.0, datum="free")
    _ok, b, _q = constraints_at(r, z, 10.0, env)
    assert b == ["베드 반경"], b

    r, z = _box(20.0, 20.0)
    env2 = MachineEnvelope(x_travel=400.0, z_travel=400.0, bed_radius=90.0,
                           max_tilt_deg=30.0, datum="free")
    assert max_angle(r, z, env2)["max_angle"] == 30.0
    _ok, b2, _q = constraints_at(r, z, 40.0, env2)
    assert b2 == ["틸트축 한계"], b2


def test_real_models_fit_assumed_envelope():
    """실제 예제 모델은 가정 엔벨로프에서 막히지 않는다 — 작아서다.

    '엔벨로프가 한계다' 를 과장하지 않기 위한 고정이다. 우리 예제 크기에서는
    기계가 전혀 문제가 아니고, 부품이 커져야 비로소 막힌다.
    """
    env = MachineEnvelope(x_travel=250.0, z_travel=210.0, bed_radius=90.0,
                          max_tilt_deg=90.0, datum="free")
    for name in ("funnel.stl", "lamp.stl"):
        mesh = center_on_axis(trimesh.load(ROOT / "examples" / name,
                                           force="mesh"))
        r, z = part_samples(mesh)
        assert math.hypot(float(r.max()), float(z.max() - z.min())) < 210.0
        assert max_angle(r, z, env, step=2.5)["max_angle"] == 90.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
