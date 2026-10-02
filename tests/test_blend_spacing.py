"""T7: 층간격 제약 (블렌드 팽창 결함 수정).

가역성 조건은 배율 m = 1 − c·r·s 가 0 이하로 가는 것만 막는다. 각도가 '감소'하는
안정 방향은 가역성엔 안 걸리면서 m 이 커져(레이어가 벌어져) 지지가 사라진다 —
구 밴드2에서 실측 미지지 25.5%로 잡힌 결함(docs/verification.md).

    (a) 폭 공식이 실제로 배율을 상한에 맞추는가
    (b) from_bands 가 구(허리 없음)에서 블렌드를 넓혀 15.5배 → 상한 이내로
    (c) 허리가 있는 모델은 경계를 허리로 옮겨 '좁은' 블렌드로 해결하는가
    (d) spacing_limit=None 이면 예전 동작 그대로인가 (회귀 방지)

    python3 tests/test_blend_spacing.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.meshio import RadiusProfile, center_on_axis  # noqa: E402
from conical.profile import AngleProfile, blend_width_for_spacing  # noqa: E402


def _sphere():
    return center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))


def _waisted():
    """구 + 가는 목 + 위로 벌어지는 형상 (허리 있음). examples/lamp.stl 과 동일 형상."""
    R, ZC = 8.0, 8.0
    t_end = math.pi - math.asin(2.0 / R)
    pts = [(R * math.sin(t), ZC - R * math.cos(t))
           for t in np.linspace(0, t_end, 60)]
    z_neck = pts[-1][1]
    pts += [(2.0, z) for z in np.linspace(z_neck, 20.0, 8)[1:]]
    pts += [(r, z) for r, z in zip(np.linspace(2.0, 7.0, 12)[1:],
                                   np.linspace(20.0, 30.0, 12)[1:], strict=True)]
    pts += [(0.0, 30.0)]
    m = trimesh.creation.revolve(np.array(pts), sections=64)
    m.merge_vertices()
    m.fix_normals()
    return center_on_axis(m)


def test_width_formula_hits_the_limit():
    """(a) 공식이 준 폭에서 배율이 정확히 상한이 된다."""
    r, limit = 9.8, 1.5
    dtan = math.tan(math.radians(0.0)) - math.tan(math.radians(36.0))   # 팽창 방향
    w = blend_width_for_spacing(r, dtan, "outward", limit)
    m = 1.0 - 1.0 * r * (dtan / w)                                      # 1 − c·r·s
    assert abs(m - limit) < 1e-9, f"배율 {m} ≠ 상한 {limit}"
    # 압축 방향은 하한 1/limit 에 맞아야 한다
    w2 = blend_width_for_spacing(r, -dtan, "outward", limit)
    m2 = 1.0 - 1.0 * r * (-dtan / w2)
    assert abs(m2 - 1.0 / limit) < 1e-9, f"압축 배율 {m2} ≠ {1/limit}"


def test_sphere_blend_widened():
    """(b) 허리 없는 모델: 블렌드를 넓혀서 상한을 지킨다 (15.5배 → ≤1.5배)."""
    m = _sphere()
    rp = RadiusProfile(m)
    r_max = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max())
    bands = [(0.0, 10.0, 36.0), (10.0, 20.0, 0.0)]

    old = AngleProfile.from_bands(bands, r_max)
    assert old.max_spacing_factor(r_max, "outward", rp) > 10.0, "결함 재현 실패"

    new = AngleProfile.from_bands(bands, r_max, radius_profile=rp,
                                  spacing_limit=1.5, max_shift=2.5)
    f = new.max_spacing_factor(r_max, "outward", rp)
    assert f <= 1.5 + 1e-6, f"상한 위반: {f}"
    assert not new.check_spacing(r_max, "outward", 1.5, rp)
    (a, b), = new.blend_intervals()
    assert b - a > 10.0, f"구는 블렌드가 넓어져야 함 (폭 {b-a:.2f})"


def test_waist_moves_boundary_and_keeps_blend_narrow():
    """(c) 허리 있는 모델: 경계를 목으로 옮겨 좁은 블렌드로 해결."""
    m = _waisted()
    rp = RadiusProfile(m)
    r_max = float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max())
    bands = [(0.0, 15.0, 30.0), (15.0, 30.0, 0.0)]
    prof = AngleProfile.from_bands(bands, r_max, radius_profile=rp,
                                   spacing_limit=1.5,
                                   max_shift=0.25 * 15.0)
    f = prof.max_spacing_factor(r_max, "outward", rp)
    assert f <= 1.5 + 1e-6, f"상한 위반: {f}"
    (a, b), = prof.blend_intervals()
    assert b - a < 4.0, f"허리에서는 블렌드가 좁아야 함 (폭 {b-a:.2f})"
    zc = 0.5 * (a + b)
    assert zc > 16.0, f"경계가 목(z≈15.8~20)으로 이동해야 함 (중심 {zc:.2f})"
    assert prof.notes, "경계를 옮겼으면 그 사실이 notes 에 남아야 함"


def test_default_is_unchanged():
    """(d) spacing_limit 을 안 주면 예전 동작 그대로 (회귀 방지)."""
    r_max = 20.0
    bands = [(0.0, 10.0, 10.0), (10.0, 40.0, 40.0)]
    prof = AngleProfile.from_bands(bands, r_max)
    w_min = r_max * (math.tan(math.radians(40.0)) - math.tan(math.radians(10.0)))
    (a, b), = prof.blend_intervals()
    assert abs((b - a) - 1.5 * w_min) < 1e-9, f"폭이 safety×w_min 이 아님: {b-a}"
    assert prof.notes == []


if __name__ == "__main__":
    test_width_formula_hits_the_limit()
    test_sphere_blend_widened()
    test_waist_moves_boundary_and_keeps_blend_narrow()
    test_default_is_unchanged()
    print("PASS: 폭 공식=상한 / 구는 넓히기 / 허리는 경계 이동 / 기본값 회귀 없음")
