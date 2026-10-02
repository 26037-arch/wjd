"""T13: '페리미터 미지지'에서 진짜 오버행과 희소 인필 허상을 가른다.

왜 이게 회귀 테스트가 되어야 하나: 이 연구의 대표 수치가 '페리미터 미지지 %'였는데,
거기에는 오버행이 아닌 몫이 섞여 있었다. 위로 좁아지는 형상은 층마다 페리미터가
안쪽으로 들어가 아랫층 페리미터(수평 창 0.45mm)를 벗어나고, 그 자리 아래에는
희소 인필(간격 2.5mm)밖에 없다 — 물리적으로는 아랫층 단면 안이라 평범한 브리징인데
검사기는 미지지로 센다.

그 몫이 모델·전략마다 0~100% 로 달라져서 **순위를 뒤집었다.** 갈라내는 이 기능이
조용히 망가지면 같은 착오가 되돌아온다.

⚠ '아랫층 단면 안'은 오버행이 아니라는 뜻이지 무죄라는 뜻이 아니다. 그 안에서
브리징(정상)과 층간격 팽창(결함)을 더 가르는 것은 아직 못 했다 —
`conical.toolpath.classify_unsupported` 의 ⚠ 참고.

지키는 성질:
  ① 진짜 오버행은 **인필 간격과 무관**하다 (기하량이므로)
  ② 페리미터 미지지는 인필을 촘촘히 할수록 진짜 오버행으로 수렴한다
  ③ 두 몫(오버행 + 아랫층 단면 안)의 합은 페리미터 미지지와 같다
  ④ 위로만 좁아지는 형상(원뿔)은 진짜 오버행이 0 이다

    python3 tests/test_support_breakdown.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.toolpath import check_support, sample_extrusions, support_breakdown

LH = 0.4


def measure(mesh, infill_spacing=2.5):
    items = slice_mesh(
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False),
        layer_height=LH, infill_spacing=infill_spacing)
    real, _ = backtransform(items, 0.0, "outward")
    pts, mid, w, kinds, lay = sample_extrusions(real, return_types=True,
                                                return_layers=True)
    sup, _ = check_support(pts, mid, w, layer_height=LH)
    return support_breakdown(pts, mid, kinds, lay, sup, w)


def _cone_narrowing_up():
    """위로만 좁아지는 원뿔 — 아래를 보는 면이 바닥뿐이라 진짜 오버행이 **정의상 0**.

    납작하게(r=20, h=10) 잡는 이유: 층당 반경 감소가 판정 창(0.45mm)을 넘어야
    허상이 생긴다. r=10,h=20 이면 층당 0.20mm 라 창 안에 들어와 허상이 0 이고,
    그러면 이 테스트가 아무것도 검사하지 않는다. r=20,h=10 은 층당 0.80mm 라
    기존 지표가 **미지지 31%** 를 보고한다 — 오버행이 하나도 없는 형상에서.
    """
    return center_on_axis(trimesh.creation.cone(radius=20.0, height=10.0,
                                                sections=64))


def test_parts_sum_to_total():
    b = measure(_cone_narrowing_up())
    assert abs(b["overhang_pct"] + b["inside_pct"]
               - b["perimeter_unsupported_pct"]) < 1e-9


def test_true_overhang_is_infill_independent():
    """인필을 촘촘히 해도 진짜 오버행은 안 변한다 — 기하량이기 때문."""
    mesh = _cone_narrowing_up()
    vals = [measure(mesh, s)["overhang_pct"] for s in (2.5, 1.0)]
    assert abs(vals[0] - vals[1]) < 0.05, f"인필에 따라 변했다: {vals}"


def test_dense_infill_converges_to_true_overhang():
    """희소 인필이 원인이라면, 촘촘해질수록 허상 몫이 줄어야 한다."""
    mesh = _cone_narrowing_up()
    sparse, dense = measure(mesh, 2.5), measure(mesh, 0.6)
    assert dense["inside_pct"] < sparse["inside_pct"], \
        f"촘촘한 인필에서 '단면 안' 몫이 안 줄었다: {sparse} → {dense}"
    assert dense["perimeter_unsupported_pct"] < \
        sparse["perimeter_unsupported_pct"] + 1e-9


def test_narrowing_shape_has_no_true_overhang():
    """원뿔은 바닥면 말고 아래를 보는 면이 없다 — 진짜 오버행 0.

    이 성질이 깨지면 '단면 안/밖' 판정이 뒤집혔다는 뜻이다.
    """
    b = measure(_cone_narrowing_up())
    assert b["overhang_pct"] < 0.05, \
        f"위로 좁아지는 원뿔에 진짜 오버행이 {b['overhang_pct']:.2f}%p 나왔다"
    assert b["perimeter_unsupported_pct"] > 10.0, \
        ("허상이 작으면 이 테스트가 아무것도 검사하지 않는다 — "
         f"형상이 충분히 빨리 좁아지는지 확인: {b}")


def test_sphere_has_both_parts():
    """구(평면 슬라이싱)는 아랫면이 진짜 오버행이라 두 몫이 다 있어야 한다."""
    b = measure(center_on_axis(trimesh.creation.icosphere(subdivisions=3,
                                                          radius=10.0)))
    assert b["overhang_pct"] > 1.0, f"진짜 오버행이 너무 작다: {b}"
    assert b["inside_pct"] > 0.1, f"'단면 안' 몫이 너무 작다: {b}"


if __name__ == "__main__":
    test_parts_sum_to_total()
    test_true_overhang_is_infill_independent()
    test_dense_infill_converges_to_true_overhang()
    test_narrowing_shape_has_no_true_overhang()
    test_sphere_has_both_parts()
    print("PASS: 두 몫의 합 = 전체, 진짜 오버행은 인필 무관, "
          "촘촘하면 허상 감소, 원뿔=0, 구=둘 다 있음")
