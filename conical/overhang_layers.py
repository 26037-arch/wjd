"""
overhang_layers.py — 레이어별 2D 오버행 판정 (실제 슬라이서 방식).

왜 만들었나 (조사 결과 반영):
    지금까지의 overhang.py 는 '면 법선'만 보고 아래를 보는 면을 전부 오버행으로
    판정했다. 이건 빠르지만, 실제 슬라이서(Cura/PrusaSlicer/OrcaSlicer)가 쓰는
    방식과 다르다. 실제 슬라이서는 각 레이어의 단면(2D 윤곽)을 '바로 아래 레이어'와
    비교해서 "아래층보다 튀어나온 부분"만 서포트가 필요하다고 본다.
    → 이러면 '밑에서 받쳐주는 면'(구멍 윗면, 계단 윗단)이나 '바닥에 닿는 면'을
      올바르게 서포트에서 제외한다. (면 법선 방식은 이걸 못 해서 과대평가한다.)

Cura 방식 그대로:
    max_dist = tan(임계각) × 레이어높이        # 한 층에서 벽이 옆으로 벌어져도 되는 거리
    이번층 오버행 = 이번층 윤곽  −  (아래층 윤곽을 max_dist 만큼 부풀린 것)
    (임계각은 '수직 기준' 각도. 우리 관례 = Cura 관례와 동일. config 참고)
    출처: CuraEngine src/support.cpp (computeBasicAndFullOverhang),
          PrusaSlicer src/libslic3r/Support/SupportMaterial.cpp (detect_overhangs)

주의(정직): 이건 여전히 근사다. 브릿지(두 기둥 사이 평평한 다리)는 실제 슬라이서가
'별도 단계'로 서포트에서 빼주는데(Prusa "Don't support bridges"), 여기선 그 단계는
넣지 않았다. 그래서 브릿지는 오버행으로 잡힌다. 필요하면 확장 지점으로 표시.
"""

import math

import numpy as np
from shapely.ops import unary_union

from .config import THRESHOLD_DEG


def _section_polygon(mesh, z):
    """높이 z에서 메시를 자른 단면을 shapely 폴리곤(구멍 포함)으로 돌려준다.
    단면이 없으면 None."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return None
    path_2d, _ = section.to_2D()
    polys = list(path_2d.polygons_full)   # 구멍(hole)까지 반영된 폴리곤들
    if not polys:
        return None
    return unary_union(polys)             # 떨어진 섬들을 하나로 합침


def layer_support_area(mesh, layer_height=0.5, threshold_deg=THRESHOLD_DEG):
    """레이어별 2D 방식으로 '서포트가 필요한 투영 넓이(mm²)'를 추정한다.

    반환 dict:
      support_area     : 서포트 필요한 넓이 합 (mm², 수평 투영 기준)
      footprint_area   : 맨 아래(바닥) 층 넓이 (mm²) — 정규화용 참고값
      n_layers         : 사용한 레이어 수
      layer_height     : 레이어 높이
    """
    z0 = mesh.bounds[0][2]
    z1 = mesh.bounds[1][2]
    # 한 층에서 윤곽이 옆으로 벌어져도 되는 최대 거리 (임계각은 수직 기준)
    max_dist = math.tan(math.radians(threshold_deg)) * layer_height

    # 레이어 '중앙' 높이들. 간격 = 레이어높이 (그래야 아래층이 정확히 한 층 밑)
    zs = np.arange(z0 + layer_height * 0.5, z1, layer_height)

    support_area = 0.0
    footprint_area = 0.0
    prev = None
    for z in zs:
        poly = _section_polygon(mesh, z)
        if poly is None or poly.is_empty:
            prev = poly
            continue
        if prev is None or prev.is_empty:
            # 처음 나타난 층 = 바닥. 빌드플레이트에 닿으므로 서포트 불필요.
            footprint_area = poly.area
            prev = poly
            continue
        # 아래층을 max_dist 만큼 부풀린 것보다 더 튀어나온 부분만 오버행
        overhang = poly.difference(prev.buffer(max_dist))
        if not overhang.is_empty:
            support_area += overhang.area
        prev = poly

    return {
        "support_area": support_area,
        "footprint_area": footprint_area,
        "n_layers": len(zs),
        "layer_height": layer_height,
    }
