"""T9: J에 블렌드 비용을 넣은 밴드 선택 (select_banded_j).

문제: 해석식 판정은 면을 '그 높이의 국소 원뿔각'과만 비교해서 블렌드의 층간격
팽창을 못 본다 — 구 밴드2를 서포트 0.5%로 예측하지만 툴패스로는 균일각에 진다.
그래서 J 에 블렌드 비용을 명시적으로 넣고, 밴드 각도를 '실제로 만들어질 프로필'
단위로 공동 선택한다.

    (a) 상수 프로필이면 블렌드 비용 0, J = 균일각 J (일관성)
    (b) 결과는 최선 균일각보다 J 가 나쁠 수 없다 (좌표하강이 균일해를 놓치던
        국소 최적 문제를 균일 후보 전수 평가로 막았다 — 회귀 방지)
    (c) 허리 없는 모델(구)은 균일로 수렴, 허리 있는 모델은 각도를 나눈다
    (d) k_blend=0 이면 비용이 꺼져 블렌드가 큰 해도 허용된다

    python3 tests/test_blend_cost_j.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.analytic import support_fraction  # noqa: E402
from conical.config import MAX_SPACING_FACTOR  # noqa: E402
from conical.meshio import RadiusProfile, center_on_axis  # noqa: E402
from conical.profile import AngleProfile  # noqa: E402
from conical.varangle import (  # noqa: E402
    blend_penalty,
    profile_objective,
    select_banded_j,
)


def _sphere():
    return center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))


def _waisted():
    R, ZC = 8.0, 8.0
    t_end = math.pi - math.asin(2.0 / R)
    pts = [(R * math.sin(t), ZC - R * math.cos(t))
           for t in np.linspace(0, t_end, 40)]
    pts += [(2.0, z) for z in np.linspace(pts[-1][1], 20.0, 6)[1:]]
    pts += [(r, z) for r, z in zip(np.linspace(2.0, 7.0, 8)[1:],
                                   np.linspace(20.0, 30.0, 8)[1:], strict=True)]
    pts += [(0.0, 30.0)]
    m = trimesh.creation.revolve(np.array(pts), sections=48)
    m.merge_vertices()
    m.fix_normals()
    return center_on_axis(m)


def _setup(mesh):
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    h = mesh.bounds[1][2] - mesh.bounds[0][2]
    return rp, r_max, h


def test_constant_profile_matches_uniform_J():
    """(a) 상수 프로필 → 블렌드 비용 0, J 가 균일각 J 와 정확히 같다."""
    m = _sphere()
    rp, r_max, _ = _setup(m)
    k, angle = 0.2, 26.0
    prof = AngleProfile.constant(angle)
    assert blend_penalty(m, prof, rp) == 0.0

    base = support_fraction(m, 0.0, "outward")
    met = profile_objective(m, prof, base, k, 0.5, rp)
    expected = (base - support_fraction(m, angle, "outward")) - k * angle
    assert abs(met["J"] - expected) < 1e-9, f"{met['J']} vs {expected}"
    assert abs(met["avg_angle"] - angle) < 1e-9


def test_never_worse_than_best_uniform():
    """(b) 선택 결과의 J 는 최선 균일각의 J 이상이어야 한다."""
    for mesh in (_sphere(), _waisted()):
        rp, r_max, h = _setup(mesh)
        r = select_banded_j(mesh, 0.2, 2, r_max, rp, MAX_SPACING_FACTOR, 0.25 * h / 2)
        assert r["J"] >= r["uniform_J"] - 1e-9, \
            f"균일보다 나쁜 해를 골랐다: {r['J']} < {r['uniform_J']}"


def test_waist_decides_banding():
    """(c) 허리 없으면 균일로 수렴, 있으면 각도를 나눈다."""
    sph = _sphere()
    rp, r_max, h = _setup(sph)
    rs = select_banded_j(sph, 0.2, 2, r_max, rp, MAX_SPACING_FACTOR, 0.25 * h / 2)
    assert len(set(rs["thetas"])) == 1, f"구는 균일로 가야 함: {rs['thetas']}"
    assert rs["blend_penalty"] == 0.0

    lam = _waisted()
    rp2, r_max2, h2 = _setup(lam)
    rl = select_banded_j(lam, 0.2, 2, r_max2, rp2, MAX_SPACING_FACTOR, 0.25 * h2 / 2)
    assert len(set(rl["thetas"])) > 1, f"허리 있으면 나눠야 함: {rl['thetas']}"
    assert rl["J"] > rl["uniform_J"] + 1e-9, "부위별이 균일보다 나아야 함"
    (a, b), = rl["profile_obj"].blend_intervals()
    assert b - a < 4.0, f"블렌드가 목에 들어가 좁아야 함 (폭 {b-a:.2f})"


def test_k_blend_zero_disables_cost():
    """(d) k_blend=0 이면 비용이 꺼진다 (구에서 균일 수렴이 풀린다)."""
    m = _sphere()
    rp, r_max, h = _setup(m)
    r = select_banded_j(m, 0.2, 2, r_max, rp, MAX_SPACING_FACTOR, 0.25 * h / 2,
                        k_blend=0.0)
    assert len(set(r["thetas"])) > 1, \
        f"비용을 끄면 (해석식이 못 보는) 밴드 해를 고른다: {r['thetas']}"


if __name__ == "__main__":
    test_constant_profile_matches_uniform_J()
    test_never_worse_than_best_uniform()
    test_waist_decides_banding()
    test_k_blend_zero_disables_cost()
    print("PASS: 상수=균일 J / 균일보다 나쁘지 않음 / 허리가 밴딩을 결정 / k_blend=0 해제")
