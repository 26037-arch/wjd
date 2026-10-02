"""T24: 블렌드 비용의 가중 `(m−1)/(limit−1)` 이 **항등**이라는 것을 고정한다.

왜 고정하나: 이 가중은 오랫동안 "유도된 물리가 아닌 휴리스틱"으로 표시돼 있었고,
그래서 '선형이 맞나' 를 닫아야 할 구멍으로 들고 있었다. 2026-09-22 에 재 보니
**계획기가 블렌드 폭을 층간격 제약의 최소값으로 잡기 때문에 `m = limit` 이 정확히
물리고, 따라서 risk ≡ 1** 이었다. 가중이 작동한 적이 없다는 뜻이다.

이 사실이 깨지면(예: 폭을 여유 있게 잡도록 계획기를 바꾸면) J 의 성격이 조용히
바뀌므로, 테스트로 붙잡아 둔다. 깨지는 것 자체는 결함이 아니지만 **모르고 깨지면
결함**이다.

같이 고정하는 것:
  · 축상 근사판과 정확 Z′ 판이 선택된 프로필에서 같은 답을 낸다
    (`analytic.support_fraction_profile` vs `..._exact`)
  · 상수 프로필에서는 둘이 자명하게 같다

    python3 -m pytest tests/test_blend_cost_identity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from compare_waist import waisted_model
from conical import analytic
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.profile import AngleProfile
from conical.varangle import select_banded_j

# 대표 표본 3 개 — 전체 스윕은 analyze_blend_cost.py 가 한다 (느려서 테스트엔 부적합)
NECKS = (1.0, 3.0, 5.0)


def _selected(r_neck):
    mesh = waisted_model(r_neck)
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    res = select_banded_j(mesh, DEFAULT_K, 2, r_max, rp, MAX_SPACING_FACTOR)
    return mesh, rp, res["profile_obj"]


def test_spacing_risk_is_saturated():
    """선택된 프로필의 모든 블렌드 구간에서 m == limit (risk == 1)."""
    seen = 0
    for r_neck in NECKS:
        mesh, rp, prof = _selected(r_neck)
        for i in range(len(prof.zs) - 1):
            dt = prof.tans[i + 1] - prof.tans[i]
            if abs(dt) < 1e-12:
                continue
            a, b = float(prof.zs[i]), float(prof.zs[i + 1])
            s = dt / (b - a)
            m = abs(1.0 - rp.max_between(a, b) * s)
            risk = min(1.0, max(0.0, (m - 1.0) / (MAX_SPACING_FACTOR - 1.0)))
            assert m == pytest_approx(MAX_SPACING_FACTOR), (
                f"r_neck={r_neck} 구간 [{a:.2f},{b:.2f}] 에서 m={m:.4f} — "
                f"계획기가 더 이상 최소 폭을 쓰지 않는다. J 의 성격이 바뀌었다.")
            assert risk == pytest_approx(1.0)
            seen += 1
    assert seen > 0, "블렌드 구간이 하나도 없었다 — 표본이 전부 균일로 수렴했다"


def test_axis_approximation_is_harmless_on_selected_profiles():
    """축상 근사판과 정확 Z′ 판이 선택된 프로필에서 같은 답."""
    for r_neck in NECKS:
        mesh, _rp, prof = _selected(r_neck)
        ap = analytic.support_fraction_profile(mesh, prof)
        ex = analytic.support_fraction_profile_exact(mesh, prof)
        assert abs(ex - ap) < 1e-9, (
            f"r_neck={r_neck}: 근사 {ap:.6f} vs 정확 {ex:.6f} — 계획기가 블렌드를 "
            f"반경이 큰 높이에 두기 시작했다면 근사를 쓴 결론을 다시 볼 것")


def test_constant_profile_exact_equals_approx():
    """상수 프로필이면 각도가 z 와 무관하므로 둘이 자명하게 같다."""
    mesh = waisted_model(3.0)
    for theta in (0.0, 15.0, 30.0):
        prof = AngleProfile.constant(theta)
        ap = analytic.support_fraction_profile(mesh, prof)
        ex = analytic.support_fraction_profile_exact(mesh, prof)
        assert abs(ex - ap) < 1e-12


def pytest_approx(x, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - x) <= tol
        def __repr__(self):
            return f"~{x}"
    return _A()


if __name__ == "__main__":
    test_spacing_risk_is_saturated()
    test_axis_approximation_is_harmless_on_selected_profiles()
    test_constant_profile_exact_equals_approx()
    print("T24 OK")
