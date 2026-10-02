"""T22: 표면 품질·치수 정확도 중 **기하가 강제하는 부분** (R5).

왜 필요한가: 표면 품질은 대부분 물리(브리징·수축·접착)라 실물 없이는 못 잰다.
그래서 '못 잰다' 로 끝내기 쉬운데, **기하가 강제하는 하한**은 지금 잴 수 있고
그게 실물이 나왔을 때의 **검증 대상**이 된다. 수치가 조용히 틀리면 실험 설계가
틀린 것을 확인하러 간다.

고정하는 성질:
  ① cusp = d·|n̂·b̂| 의 양 끝: 수직 벽 → 0, 평평한 윗면 → 층 간격 그대로
  ② 원뿔 레이어의 수직 간격이 `t·cosθ` 다
  ③ 원뿔 법선이 단위벡터이고 축 위에서 ẑ 로 간다
  ④ **서포트 감소와 상충한다** — 레이어를 표면과 나란히 놓으면 계단이 심해진다
     (lamp 에서 실제로 정렬 배수가 1 을 넘는다)
  ⑤ 회전축 분해능 오차가 **반경에 비례**한다 (R2 의 동기 오차는 반경 무관이었다)
  ⑥ 면적 가중이다 — 잘게 쪼개진 면이 과대평가되지 않는다

    python3 -m pytest tests/test_quality.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import trimesh

from conical.analytic import support_fraction
from conical.meshio import center_on_axis
from conical.quality import (
    cone_normal,
    cusp_compare,
    cusp_heights,
    rotary_resolution_error,
)

ROOT = Path(__file__).resolve().parent.parent


def _box(w=10.0, h=10.0):
    """옆면이 수직이고 윗면이 수평인 상자 — cusp 의 양 끝을 다 갖는다."""
    return trimesh.creation.box(extents=(w, w, h))


def test_cusp_is_zero_on_vertical_walls_and_full_on_flat_tops():
    """cusp = d·|n̂·b̂| 의 양 끝. 이게 틀리면 나머지 숫자가 전부 의미 없다."""
    t = 0.3
    q = cusp_heights(_box(), 0.0, "outward", t)
    n = np.asarray(_box().face_normals)
    vertical = np.abs(n[:, 2]) < 1e-9
    flat = np.abs(np.abs(n[:, 2]) - 1.0) < 1e-9
    assert vertical.any() and flat.any()
    assert np.allclose(q["cusp"][vertical], 0.0, atol=1e-12), "수직 벽에 계단이 있다"
    assert np.allclose(q["cusp"][flat], t, atol=1e-12), "평평한 윗면이 층고가 아니다"


def test_conical_layer_spacing_is_t_cos_theta():
    """이웃 원뿔면 사이 수직거리 = `t·cosθ`. 원뿔은 같은 반각이라 평행하다."""
    t = 0.3
    for angle in (0.0, 20.0, 45.0, 60.0):
        q = cusp_heights(_box(), angle, "outward", t)
        assert q["layer_spacing"] == pytest.approx(
            t * math.cos(math.radians(angle)), rel=1e-12)


def test_cone_normal_is_unit_and_axial_on_axis():
    """축 위에서는 방위각이 없다 → ẑ. 그 외에는 단위벡터."""
    for angle in (0.0, 25.0, 55.0):
        xs = np.array([0.0, 1e-15, 3.0, -4.0, 0.0])
        ys = np.array([0.0, 0.0, 4.0, 3.0, 7.0])
        b = cone_normal(xs, ys, angle, "outward")
        assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-12)
        assert np.allclose(b[0], [0.0, 0.0, 1.0], atol=1e-12)
        if angle > 0:
            # 반경 5 인 두 점은 같은 z 성분, 다른 방향
            assert b[2, 2] == pytest.approx(b[3, 2], rel=1e-12)
            assert not np.allclose(b[2, :2], b[3, :2])


def test_inward_flips_the_radial_component():
    """inward 는 법선의 반경 성분 부호가 뒤집힌다 (레이어가 반대로 기운다)."""
    o = cone_normal([5.0], [0.0], 30.0, "outward")[0]
    i = cone_normal([5.0], [0.0], 30.0, "inward")[0]
    assert o[0] == pytest.approx(-i[0], rel=1e-12)
    assert o[2] == pytest.approx(i[2], rel=1e-12)


def test_alignment_conflicts_with_support_reduction():
    """**핵심 결과.** 서포트를 없애는 각도에서 계단 정렬이 나빠진다.

    서포트를 줄이려면 레이어를 표면과 **나란히** 놓아야 하는데, 계단 자국은
    바로 그때 **최대**가 된다. 같은 층고로만 비교하면 `cosθ` 가 전면적에 깔려
    이 상충이 가려지므로, 정렬 항(`|n̂·b̂|`)을 떼어서 본다.
    """
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "lamp.stl",
                                       force="mesh"))
    s0 = support_fraction(mesh, 0.0, "outward")
    assert s0 > 1.0, "평면에서 서포트가 없으면 상충을 볼 수 없다"
    worst = None
    for angle in (10.0, 20.0, 30.0, 45.0, 60.0):
        cc = cusp_compare(mesh, angle, "outward", 0.3)
        sup = support_fraction(mesh, angle, "outward")
        if sup < 0.05 and (worst is None or cc["align_ratio"] > worst[1]):
            worst = (angle, cc["align_ratio"], cc["worse_area_pct"])
        # 같은 층고 기준은 cosθ 이득 때문에 항상 좋아 보인다
        assert cc["cusp_ratio"] < 1.0
    assert worst is not None, "서포트가 0 이 되는 각도가 없었다"
    assert worst[1] > 1.0, \
        f"서포트 0 인 θ={worst[0]}° 에서 정렬이 {worst[1]:.2f}배 — 상충이 안 보인다"
    assert worst[2] > 20.0, f"나빠진 면적이 {worst[2]:.0f}% 뿐이다"


def test_shallow_model_has_no_conflict():
    """얕은 형상에서는 상충이 없다 — '항상 상충' 이 아님을 같이 고정한다."""
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    for angle in (20.0, 45.0, 60.0):
        cc = cusp_compare(mesh, angle, "outward", 0.3)
        assert cc["align_ratio"] < 1.0
        assert cc["worse_area_pct"] == pytest.approx(0.0, abs=1e-9)


def test_rotary_resolution_error_scales_with_radius():
    """오차 = r·Δφ — **반경에 비례**한다.

    R2 의 동기 오차(`f·Δt`)는 반경이 약분돼 무관했다. 성질이 다른 오차원이고,
    둘을 섞으면 큰 부품의 바깥쪽 오차를 과소평가한다.
    """
    spd = 26.666                              # REP5X C축 실측 제원
    r = np.array([5.0, 10.0, 20.0, 50.0])
    out = rotary_resolution_error(r, spd)
    assert out["step_deg"] == pytest.approx(1.0 / spd, rel=1e-12)
    # 선형성: 반경 2배 → 오차 2배
    assert out["err_um"][1] == pytest.approx(2.0 * out["err_um"][0], rel=1e-12)
    assert out["err_um"][3] == pytest.approx(10.0 * out["err_um"][0], rel=1e-12)
    # 해석값과 대조: 20mm 에서 20 × (1/26.666)° = 13.1µm
    want = 20.0 * math.radians(1.0 / spd) * 1000.0
    assert out["err_um"][2] == pytest.approx(want, rel=1e-12)
    assert 12.0 < out["err_um"][2] < 14.0


def test_stats_are_area_weighted():
    """면 개수가 아니라 **면적**으로 가중한다.

    한 면을 잘게 쪼개도 통계가 변하면 메시 해상도가 결과를 바꿔 버린다.
    평면(θ=0)은 `b̂ = ẑ` 로 위치와 무관하므로 세분화에 **정확히** 불변이어야
    한다 — 가중만 떼어서 보는 경우다.
    """
    box = _box()
    fine = box.subdivide().subdivide()
    assert len(fine.faces) > 4 * len(box.faces)
    q1 = cusp_heights(box, 0.0, "outward", 0.3)
    q2 = cusp_heights(fine, 0.0, "outward", 0.3)
    assert q2["mean"] == pytest.approx(q1["mean"], rel=1e-12), \
        f"평면인데 세분화로 평균이 바뀌었다: {q1['mean']} → {q2['mean']}"

    # 원뿔은 법선이 면 중심에 따라 달라 세분화로 **표본**이 바뀐다 (가중 문제가
    # 아니라 수렴 문제다). 몇 % 안쪽이면 된다.
    c1 = cusp_heights(box, 30.0, "outward", 0.3)["mean"]
    c2 = cusp_heights(fine, 30.0, "outward", 0.3)["mean"]
    assert c2 == pytest.approx(c1, rel=0.05), \
        f"세분화로 5% 넘게 바뀌었다: {c1:.5f} → {c2:.5f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
