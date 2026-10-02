"""T17: 5축 회전 요구량과 동기화 예산 (플랫폼 비교의 근거).

왜 필요한가: 하드웨어 후보가 둘이다 — REP5X(개조 Marlin, 펌웨어가 역기구학)와
뱀부랩 개조(원본 펌웨어 유지 + **외부 ESP32 가 회전축만 제어**). 어느 쪽이 빠른가를
묻기 전에 회전축 요구량을 알아야 하고, 그 수치가 곧 플랫폼 선택의 근거가 된다.
수치가 조용히 틀리면 하드웨어를 잘못 고른다.

고정하는 성질:
  ① `ω = f/r` — 해석식과 일치한다 (합성 원 경로로 확인)
  ② 동기 오차에서 **r 이 약분된다** — 반경이 달라도 같은 피드면 같은 오차
  ③ 접선 경로는 수직 성분이 0, 반경 방향 경로는 1 (오차가 무해/유해로 갈린다)
  ④ 축 한계를 낮추면 출력 시간이 단조증가하고, 한계가 없으면 배수 1
  ⑤ 베드 회전식에서는 기계 **Y 가 0 으로 축퇴**한다
  ⑥ 부품/기계 이동 수가 안 맞으면 조용히 계산하지 않고 멈춘다

    python3 tests/test_sync_demand.py
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
from conical.open5x import PRUSA_UV, to_open5x
from conical.planar_slicer import slice_mesh
from conical.sync import (
    feed_cap_cost,
    machine_xy_span,
    rotary_demand,
    sync_error_budget,
)
from conical.transform import transform_cone

ROOT = Path(__file__).resolve().parent.parent


def _circle(radius, feed_mm_s, n=72, z=1.0):
    """반경 `radius` 원을 도는 부품 경로 + V = −φ 인 기계 경로."""
    real, mach = [], []
    e = 0.0
    for i in range(n + 1):
        phi = 2.0 * math.pi * i / n
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        e += 2.0 * math.pi * radius / n * 0.05
        real.append(("move", Move(g=1, x=x, y=y, z=z, e=e,
                                  f=feed_mm_s * 60.0)))
        # 베드가 압출점을 방위각 0 으로 돌려놓는다 → 기계 XY 는 (r, 0)
        mach.append(("move", Move(g=1, x=radius, y=0.0, z=z, e=e,
                                  f=feed_mm_s * 60.0,
                                  extra=f"V{-math.degrees(phi):.6f}")))
    return real, mach


def test_omega_matches_f_over_r():
    """ω = f/r. 이 식이 이 분석 전체의 뼈대다."""
    for radius, feed in ((10.0, 30.0), (2.0, 30.0), (10.0, 60.0)):
        d = rotary_demand(*_circle(radius, feed))
        expected = math.degrees(feed / radius)
        got = float(np.median(d["omega"]))
        assert abs(got - expected) / expected < 0.02, \
            f"r={radius} f={feed}: ω {got:.1f} vs 예측 {expected:.1f} deg/s"


def test_sync_error_is_radius_independent():
    """오차 = r·ω·Δt 이고 ω = f/r 이므로 **r 이 약분된다.**

    반경이 5배 달라도 같은 피드면 같은 오차여야 한다. 이 성질이 깨지면
    '오차 = 지각한 시간 동안 노즐이 간 거리' 라는 해석이 무너진다.
    """
    budgets = []
    for radius in (2.0, 10.0):
        d = rotary_demand(*_circle(radius, 30.0))
        # 원 경로는 접선 방향이라 수직 성분이 0 이다 — 여기서는 크기만 보려고
        # perp 를 1 로 두고 잰다.
        d = dict(d)
        d["perp"] = np.ones_like(d["perp"])
        budgets.append(sync_error_budget(d, (1.0,))[1.0]["median"])
    assert abs(budgets[0] - budgets[1]) / budgets[0] < 0.02, \
        f"반경에 따라 달라졌다: {budgets}"
    # 30 mm/s × 1ms = 30µm
    assert abs(budgets[0] - 30.0) < 1.0, f"예측 30µm 인데 {budgets[0]:.1f}"


def test_tangential_path_has_no_perpendicular_error():
    """원(접선 경로)은 수직 성분 0 — 오차가 경로 위에서 앞뒤로 밀릴 뿐이다."""
    d = rotary_demand(*_circle(10.0, 30.0))
    assert float(np.median(d["perp"])) < 0.05, \
        f"접선 경로인데 수직 성분이 {np.median(d['perp']):.3f}"


def test_radial_path_is_fully_perpendicular():
    """반경 방향으로 움직이면 회전 오차가 경로에 수직 = 그대로 치수 오차."""
    real, mach = [], []
    e = 0.0
    for i in range(1, 21):
        r = 2.0 + i * 0.5
        e += 0.05
        real.append(("move", Move(g=1, x=r, y=0.0, z=1.0, e=e, f=1800.0)))
        # 반경 방향으로 가는 동안에도 V 가 조금씩 변하는 경우
        mach.append(("move", Move(g=1, x=r, y=0.0, z=1.0, e=e, f=1800.0,
                                  extra=f"V{i * 2.0:.3f}")))
    d = rotary_demand(real, mach)
    assert float(np.median(d["perp"])) > 0.95, \
        f"반경 경로인데 수직 성분이 {np.median(d['perp']):.3f}"


def test_feed_cap_cost_is_monotone():
    d = rotary_demand(*_circle(2.0, 30.0))        # ω ≈ 859 deg/s
    cost = feed_cap_cost(d, (180, 360, 720, 1440, 3600))
    ratios = [r["ratio"] for r in cost["rows"]]
    assert all(a >= b - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False)), \
        f"한계를 올렸는데 시간이 늘었다: {ratios}"
    assert ratios[-1] == pytest.approx(1.0, abs=1e-6), \
        "요구보다 높은 한계인데 배수가 1 이 아니다"
    assert ratios[0] > 1.5, f"느린 축인데 배수가 {ratios[0]:.2f} 뿐이다"


def test_machine_y_degenerates_on_real_pipeline():
    """베드 회전식 원뿔 모드에서 기계 Y 가 0 이 된다 (압출점이 항상 방위각 0).

    이게 '기반 프린터의 XY 속도는 이 모드에서 거의 안 쓰인다' 의 근거다.
    """
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    warped = transform_cone(mesh.vertices, 20.0, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False), layer_height=0.3)
    real, _ = backtransform(items, 20.0, "outward")
    machine, _ = to_open5x(real, 20.0, "outward", PRUSA_UV)

    span = machine_xy_span(machine)
    assert span["n"] > 1000
    assert span["y_abs_max"] < 1e-9, f"Y 가 0 이 아니다: {span}"
    assert span["x_span"] > 1.0, "X 는 실제로 움직여야 한다"

    d = rotary_demand(real, machine)
    assert d["omega"][d["extruding"]].max() > 1000, \
        "실제 출력이면 축 근처에서 각속도가 크게 나와야 한다"


def test_mismatched_lengths_raise():
    """짝이 안 맞으면 조용히 어긋난 채로 계산하지 않는다."""
    real, mach = _circle(10.0, 30.0)
    with pytest.raises(ValueError):
        rotary_demand(real, mach[:-3])


if __name__ == "__main__":
    test_omega_matches_f_over_r()
    test_sync_error_is_radius_independent()
    test_tangential_path_has_no_perpendicular_error()
    test_radial_path_is_fully_perpendicular()
    test_feed_cap_cost_is_monotone()
    test_machine_y_degenerates_on_real_pipeline()
    test_mismatched_lengths_raise()
    print("PASS: ω=f/r, r 약분, 접선/반경 분해, 한계 단조성, Y 축퇴, 짝 불일치 검출")
