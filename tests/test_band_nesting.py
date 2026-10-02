"""T12: 밴드 경계가 겹칠 때만 '흉내내기'가 성립한다.

`compare_bands.py` 의 단조성 점검은 이 레포에서 계획기 결함 셋을 잡아낸 도구다.
그 점검의 논거는 "N개 밴드는 이웃을 같은 각도로 두면 N−1개를 흉내낼 수 있다"
였는데, 밴드 경계가 `linspace` 라 **그 논거는 N 이 N−1 의 배수일 때만 성립한다.**
N=3 의 경계(33.3%, 66.7%)는 N=4 의 경계(25%, 50%, 75%)에 하나도 없다.

이 성질이 깨지면 점검이 헛경보를 내고(없는 계획기 버그를 쫓게 된다) 동시에
진짜 결함을 놓친다. 그래서 고정한다:

  ① N=3 의 내부 경계는 N=6 에 포함되고, N=4 에는 포함되지 않는다
  ② 경계가 겹치면 흉내낸 프로필의 J 가 **정확히** 같다 (차이 0)
  ③ 경계가 안 겹치면 N 을 늘려도 J 가 낮아질 수 있다 (허리 r=7 실측)

    python3 tests/test_band_nesting.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from compare_waist import waisted_model
from conical.analytic import support_fraction
from conical.config import (
    BLEND_COST_K,
    BLEND_SHIFT_RATIO,
    DEFAULT_K,
    MAX_SPACING_FACTOR,
    THRESHOLD_DEG,
)
from conical.meshio import RadiusProfile
from conical.profile import AngleProfile
from conical.varangle import _merge_bands, assign_height_bands, profile_objective


def _interior_fractions(mesh, n):
    _, e = assign_height_bands(mesh, n)
    return [(z - e[0]) / (e[-1] - e[0]) for z in e[1:-1]]


def _J(mesh, rp, r_max, max_shift, n_bands, thetas):
    _, edges = assign_height_bands(mesh, n_bands)
    prof = AngleProfile.from_bands(
        _merge_bands(edges, list(thetas)), r_max, radius_profile=rp,
        spacing_limit=MAX_SPACING_FACTOR, max_shift=max_shift)
    base = support_fraction(mesh, 0.0, "outward", THRESHOLD_DEG)
    return profile_objective(mesh, prof, base, DEFAULT_K, BLEND_COST_K, rp,
                             MAX_SPACING_FACTOR, THRESHOLD_DEG)["J"]


def test_boundaries_nest_only_on_multiples():
    mesh = waisted_model(7.0)
    f3 = _interior_fractions(mesh, 3)
    f4 = _interior_fractions(mesh, 4)
    f6 = _interior_fractions(mesh, 6)

    def covered(fine, coarse):
        return all(any(abs(a - b) < 1e-9 for b in fine) for a in coarse)

    assert covered(f6, f3), f"N=3 경계 {f3} 가 N=6 {f6} 에 없다"
    assert not covered(f4, f3), f"N=3 경계 {f3} 가 N=4 {f4} 에 있으면 안 된다"


def test_nested_imitation_is_exact():
    """N=6 으로 N=3 의 해를 흉내내면 J 가 정확히 같아야 한다 (차이 0)."""
    mesh = waisted_model(7.0)
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    ms = BLEND_SHIFT_RATIO * float(mesh.bounds[1][2] - mesh.bounds[0][2])

    for thetas in ([18, 18, 16], [24, 0, 0], [20, 20, 10]):
        j3 = _J(mesh, rp, r_max, ms, 3, thetas)
        j6 = _J(mesh, rp, r_max, ms, 6,
                [t for t in thetas for _ in range(2)])
        assert abs(j6 - j3) < 1e-12, \
            f"{thetas}: N=3 {j3:.6f} vs 흉내낸 N=6 {j6:.6f}"


def test_non_nested_step_may_lose():
    """N=4 는 N=3 의 전이 높이(66.7%)를 못 만든다 — 그래서 J 가 낮아질 수 있다.

    이 감소는 계획기 결함이 아니라 격자 비정합이다. 점검이 이걸 결함으로 부르면
    없는 버그를 쫓게 되므로, '이웃 N 감소 = 결함' 이 아님을 여기에 못 박는다.
    """
    mesh = waisted_model(7.0)
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    ms = BLEND_SHIFT_RATIO * float(mesh.bounds[1][2] - mesh.bounds[0][2])

    # 두 격자에서 '한 번만 꺾는' 해를 각각 전수 조사한다. 특정 해를 박아두면
    # k_blend 같은 기본값이 바뀔 때 증인이 낡아 테스트가 깨진다 — 성질은
    # 그대로인데도. 그래서 최적해를 그때그때 구해서 비교한다.
    def best_single_bend(n):
        return max(_J(mesh, rp, r_max, ms, n, [a] * cut + [b] * (n - cut))
                   for cut in range(1, n)
                   for a in range(0, 41, 4) for b in range(0, 41, 4))

    j3, j4 = best_single_bend(3), best_single_bend(4)
    assert j4 < j3 - 1e-6, \
        (f"N=4 최선 {j4:.4f} 이 N=3 {j3:.4f} 에 도달하면 안 된다 — "
         "N=4 는 66.7% 에 전이를 놓을 수 없다 (경계 비정합)")


if __name__ == "__main__":
    test_boundaries_nest_only_on_multiples()
    test_nested_imitation_is_exact()
    test_non_nested_step_may_lose()
    print("PASS: 경계는 배수에서만 겹치고, 겹치면 J 가 정확히 같고, "
          "안 겹치면 N 을 늘려도 질 수 있다")
