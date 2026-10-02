"""
compare_overhang_methods.py — 두 오버행 판정 방식 비교 (검증/발표용).

  (A) 면 법선 방식 (overhang.py)      : 빠름. 아래보기 면을 전부 오버행으로 봄.
  (B) 레이어별 2D 방식 (overhang_layers.py) : 실제 슬라이서(Cura/Prusa) 방식.
      '아래에서 받쳐주는 면'과 '바닥에 닿는 면'을 올바르게 제외.

정육면체가 이 둘의 차이를 가장 잘 보여준다:
  면 법선은 밑면(바닥)을 서포트로 오판(16.7%)하지만, 레이어 방식은 0 (정답).

    python3 compare_overhang_methods.py            # 여러 기본 도형으로 비교
    python3 compare_overhang_methods.py model.stl  # 내 STL로 비교
"""

import trimesh

from conical.config import THRESHOLD_DEG
from conical.meshio import center_on_axis
from conical.overhang import analyze_overhangs, support_area_fraction
from conical.overhang_layers import layer_support_area


def compare_one(name, mesh, layer_height=0.4, threshold_deg=THRESHOLD_DEG):
    # 회전축(Z)에 XY 센터링 + 바닥 z=0 (off-axis 왜곡 방지)
    mesh = center_on_axis(mesh.copy())

    _, need = analyze_overhangs(mesh, threshold_deg)
    face_pct = support_area_fraction(mesh, need)      # 표면적 대비 %
    lay = layer_support_area(mesh, layer_height, threshold_deg)

    print(f"{name:18s} | (A) 면법선 {face_pct:5.1f}% (표면적) "
          f"| (B) 레이어 {lay['support_area']:8.1f} mm² "
          f"(바닥 {lay['footprint_area']:6.1f} mm²)")


def demo_shapes():
    shapes = {}
    shapes["cube (바닥밀착)"] = trimesh.creation.box(extents=(10, 10, 10))
    shapes["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=10)
    # 버섯: 갓 아랫면=진짜 오버행 / 기둥 위=받쳐짐(제외돼야)
    stem = trimesh.creation.cylinder(radius=2, height=14)
    stem.apply_translation([0, 0, 7])
    cap = trimesh.creation.cylinder(radius=8, height=3)
    cap.apply_translation([0, 0, 15.5])
    shapes["mushroom (받쳐짐)"] = trimesh.util.concatenate([stem, cap])
    return shapes


if __name__ == "__main__":
    import sys

    print(f"[임계각 {THRESHOLD_DEG:.0f}° · Cura 관례]  두 오버행 판정 방식 비교")
    print("=" * 78)
    if len(sys.argv) > 1:
        mesh = trimesh.load(sys.argv[1], force="mesh")
        compare_one(sys.argv[1], mesh)
    else:
        for name, mesh in demo_shapes().items():
            compare_one(name, mesh)
    print("=" * 78)
    print("해석: 면법선(A)은 아래보기 면을 전부 세므로 바닥·받쳐진 면을 과대평가한다.")
    print("      레이어(B)는 실제 슬라이서처럼 '아래층보다 튀어나온 부분'만 센다.")
