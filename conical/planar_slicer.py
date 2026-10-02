"""
planar_slicer.py — 내장 미니 평면 슬라이서 (연구용 최소 기능).

⚠ 포지션을 분명히: PrusaSlicer/Orca 를 대체하려는 게 아니다. 파이프라인의
'평면 슬라이싱' 단계는 원래 외부 슬라이서(CLI)를 꽂는 자리인데(pipeline.py 참고),
외부 슬라이서가 없는 환경에서도 파이프라인 전체를 돌려보고 검증할 수 있도록
최소한의 슬라이서를 내장한 것이다. 지원 기능:

  · 레이어별 단면(trimesh.section) → 다각형(구멍 포함)
  · 페리미터 n개 (shapely 음수 버퍼로 안쪽 오프셋)
  · 희소 rectilinear 인필 (각도 레이어마다 90도 교차)
  · 압출량 E = 단면적 모델 (layer_h × width / 필라멘트 단면적)

없는 것(정직): 솔리드 상/하면, 브릿지, 리트랙션 튜닝, 냉각, 시임 최적화 등.
연구 검증·소형 테스트 출력용이지 품질 출력용이 아니다.
"""

import math

import numpy as np
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import unary_union

from .gcode import Move

FILAMENT_D = 1.75


def _layer_polygons(mesh, z, simplify_tol):
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return []
    path2d, _ = sec.to_2D()
    polys = []
    for p in path2d.polygons_full:
        if not p or p.area <= 1e-6:
            continue
        # 세분화된 메시 윤곽은 정점이 촘촘함 → 현 오차 tol 안에서 단순화
        # (안 하면 G-code가 0.05mm급 세그먼트로 폭발: 우리 gcode_stats 발견 그대로)
        if simplify_tol > 0:
            p = p.simplify(simplify_tol, preserve_topology=True)
        if p and p.area > 1e-6:
            polys.append(p)
    return polys


def _ring_to_moves(coords, z, e_per_mm, feed, state):
    """닫힌 링 좌표를 이동으로. state=dict(x,y,e) 이어붙이기용."""
    out = []
    x0, y0 = coords[0]
    out.append(Move(g=0, x=x0, y=y0, z=z, f=feed * 2))          # 이동(비압출)
    px, py = x0, y0
    for x, y in coords[1:]:
        d = math.hypot(x - px, y - py)
        if d < 1e-6:                       # 중복점(0길이 세그먼트) 제거
            continue
        state["e"] += d * e_per_mm
        out.append(Move(g=1, x=x, y=y, e=state["e"], f=feed))
        px, py = x, y
    return out


def slice_mesh(mesh, layer_height=0.3, extrusion_width=0.45, perimeters=2,
               infill_spacing=2.5, feed=1800.0, simplify_tol=0.05):
    """메시를 평면 슬라이싱해 (kind, payload) 리스트(G-code 아이템)를 돌려준다."""
    e_per_mm = (layer_height * extrusion_width) / (math.pi * (FILAMENT_D / 2) ** 2)
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    n_layers = max(1, int(round((z1 - z0) / layer_height)))

    items = [("raw", "; conical built-in planar slicer"),
             ("raw", f"; layers={n_layers} layer_h={layer_height} width={extrusion_width}"),
             ("raw", "G21"), ("raw", "G90"), ("raw", "M82")]  # mm, 절대좌표, 절대 E(M82)
    state = {"e": 0.0}

    for i in range(n_layers):
        z_cut = z0 + (i + 0.5) * layer_height          # 층 '중앙'에서 자름
        z_out = z0 + (i + 1) * layer_height            # 노즐 높이
        polys = _layer_polygons(mesh, z_cut, simplify_tol)
        if not polys:
            continue
        items.append(("raw", f"; layer {i} z={z_out:.3f}"))

        infill_region = []
        for poly in polys:
            # 페리미터: 바깥에서 안으로 (w/2, 3w/2, ...) 오프셋
            items.append(("raw", ";TYPE:PERIMETER"))   # Slic3r/PrusaSlicer 관례
            for k in range(perimeters):
                off = poly.buffer(-(extrusion_width * (0.5 + k)))
                if off.is_empty:
                    break
                geoms = off.geoms if isinstance(off, MultiPolygon) else [off]
                for g in geoms:
                    for ring in [g.exterior, *g.interiors]:
                        coords = list(ring.coords)
                        if len(coords) >= 3:
                            for it in _ring_to_moves(coords, z_out, e_per_mm, feed, state):
                                items.append(("move", it))
            inner = poly.buffer(-(extrusion_width * (perimeters + 0.2)))
            if not inner.is_empty:
                infill_region.append(inner)

        # 희소 인필: 레이어마다 0/90도 교차 직선
        if infill_region and infill_spacing > 0:
            items.append(("raw", ";TYPE:FILL"))        # Slic3r/PrusaSlicer 관례
            region = unary_union(infill_region)
            minx, miny, maxx, maxy = region.bounds
            vertical = i % 2 == 0
            coords_iter = (np.arange(minx, maxx, infill_spacing) if vertical
                           else np.arange(miny, maxy, infill_spacing))
            for c in coords_iter:
                line = (LineString([(c, miny - 1), (c, maxy + 1)]) if vertical
                        else LineString([(minx - 1, c), (maxx + 1, c)]))
                seg = region.intersection(line)
                geoms = getattr(seg, "geoms", [seg])
                for g in geoms:
                    if g.is_empty or g.length < extrusion_width:
                        continue
                    (xa, ya), (xb, yb) = g.coords[0], g.coords[-1]
                    items.append(("move", Move(g=0, x=xa, y=ya, z=z_out, f=feed * 2)))
                    state["e"] += g.length * e_per_mm
                    items.append(("move", Move(g=1, x=xb, y=yb, e=state["e"], f=feed)))

    items.append(("raw", "; end"))
    return items
