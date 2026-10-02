"""
bandplan.py — 실현 가능한 '밴드 계획' (팀메 bands.py 방식의 경량 미러) + 정직한 평가.

팀메(find_conical_angle/conical_slicing/bands.py)의 계획 로직을 우리 analytic 위에
재구현한 것. 크레딧: 계획 알고리즘(클러스터 → 변환높이 h구간 → 겹침 병합 → 수렴)은
팀메 설계를 따른다.

원리 (물리적 실현성):
    원뿔 출력은 변환높이 h = z + d·r·tanθ 순서로 진행되고, 각도 θ는 h에 따라
    서서히 바꿀 수 있다(변수각). 따라서 h구간이 '겹치지 않는' 클러스터들은 서로
    다른 각도를 받을 수 있고, 겹치는 클러스터들은 하나의 타협 각도를 공유해야 한다.

평가 (정직):
    밴드 각도는 클러스터(원래 오버행) 면만 보고 고르지만, 그 각도는 같은 높이의
    '멀쩡하던 면'에도 적용된다 → 멀쩡하던 면이 새로 오버행이 될 수 있다.
    그래서 최종 서포트%는 '모든 면'에 각자 배정된 각도를 적용해 계산한다.
    (이 부수피해까지 포함해야 실현 가능한 계획의 진짜 성능이다.)
"""

import numpy as np

from . import analytic
from .clusters import h_interval, overhang_clusters
from .config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG


def _best_joint(mesh, faces, threshold_deg, max_angle, step):
    """faces 합집합에 대해 남는 서포트 넓이 최소인 (각도, 방향). 동률→작은 각."""
    areas = mesh.area_faces
    best = None
    for direction in ("outward", "inward"):
        for angle in np.arange(0.0, max_angle + 1e-9, step):
            need = analytic.needs_support(mesh, angle, direction, threshold_deg)
            left = float(areas[faces][need[faces]].sum())
            if best is None or left < best[0] - 1e-12 or \
               (abs(left - best[0]) <= 1e-12 and angle < best[1]):
                best = (left, float(angle), direction)
    return best[1], best[2]


def plan_bands(mesh, threshold_deg=THRESHOLD_DEG,
               max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP):
    """클러스터 → 각도 → h구간 → 겹침 병합을 수렴까지 반복 (팀메 plan_slice_bands 미러).

    반환: bands = [{"faces", "angle", "direction", "h_lo", "h_hi"}, ...] (h 순)
    """
    clusters = overhang_clusters(mesh, threshold_deg)
    if not clusters:
        return []
    groups = [list(c) for c in clusters]

    for _ in range(len(groups) or 1):
        # 각 그룹의 타협 각도와 h구간 계산
        infos = []
        for g in groups:
            faces = np.array(g)
            ang, dirn = _best_joint(mesh, faces, threshold_deg, max_angle, step)
            lo, hi = h_interval(mesh, faces, ang, dirn)
            infos.append({"faces": faces, "angle": ang, "direction": dirn,
                          "h_lo": lo, "h_hi": hi})
        # h_lo 순으로 정렬 후 겹치는 이웃 병합
        infos.sort(key=lambda d: d["h_lo"])
        merged = []
        changed = False
        for info in infos:
            if merged and info["h_lo"] < merged[-1]["h_hi"] - 1e-9:
                merged[-1] = {"faces": np.concatenate([merged[-1]["faces"], info["faces"]]),
                              "angle": None, "direction": None,
                              "h_lo": min(merged[-1]["h_lo"], info["h_lo"]),
                              "h_hi": max(merged[-1]["h_hi"], info["h_hi"])}
                changed = True
            else:
                merged.append(dict(info))
        groups = [list(m["faces"]) for m in merged]
        if not changed:
            return infos
    # 수렴 못 하면(이론상 없음) 마지막 상태 재계산
    result = []
    for g in groups:
        faces = np.array(g)
        ang, dirn = _best_joint(mesh, faces, threshold_deg, max_angle, step)
        lo, hi = h_interval(mesh, faces, ang, dirn)
        result.append({"faces": faces, "angle": ang, "direction": dirn,
                       "h_lo": lo, "h_hi": hi})
    result.sort(key=lambda d: d["h_lo"])
    return result


def evaluate_band_plan(mesh, bands, threshold_deg=THRESHOLD_DEG):
    """밴드 계획의 '전체 모델' 서포트% (부수피해 포함 — 위 docstring 참고).

    면별 각도 배정: 클러스터 면 → 자기 밴드 각도.
    나머지 면 → centroid z가 밴드 면들의 원래 z구간 안이면 그 밴드 각도, 아니면 0°.
    """
    areas = mesh.area_faces
    total = float(areas.sum())
    F = len(mesh.faces)
    angle_of = np.zeros(F)
    dir_of = np.full(F, "outward", dtype=object)

    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    for b in bands:
        # 이 밴드 면들의 원래 z구간
        vz = mesh.vertices[np.unique(mesh.faces[b["faces"]].ravel())][:, 2]
        z_lo, z_hi = float(vz.min()), float(vz.max())
        in_z = (fz >= z_lo - 1e-9) & (fz <= z_hi + 1e-9)
        angle_of[in_z] = b["angle"]
        dir_of[in_z] = b["direction"]
        angle_of[b["faces"]] = b["angle"]          # 클러스터 면은 확정
        dir_of[b["faces"]] = b["direction"]

    left = 0.0
    for direction in ("outward", "inward"):
        sel = dir_of == direction
        if not sel.any():
            continue
        for ang in np.unique(angle_of[sel]):
            m = sel & (angle_of == ang)
            need = analytic.needs_support(mesh, float(ang), direction, threshold_deg)
            left += float(areas[m & need].sum())
    return left / total * 100.0
