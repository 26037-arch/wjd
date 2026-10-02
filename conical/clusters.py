"""
clusters.py — 오버행 클러스터 진단 + '적응 천장(ceiling)' 계산.

팀메(find_conical_angle)의 밴드 계획(bands.py)과 맞물리는 우리 쪽 반쪽:
  팀메: 성분별 각도 → 변환높이 h=z+d·r·tanα 구간 → 겹침 병합 → 실현 가능한 밴드
  우리: 같은 클러스터 분해에서 두 가지를 계산한다.
    (1) 천장(ceiling): 각 클러스터가 '자기 최적 (방향,각도)'를 아무 제약 없이
        받을 수 있다면 서포트가 얼마까지 줄까? — 물리적으로 한 번에 못 찍는
        이상적 상한. 부위별 자유 축(다방향 분해, Wu2020/Han2025)이 노리는
        성능의 conical 근사 상한으로, 가성비 곡선의 오른쪽 끝점이 된다.
    (2) 진단(diagnostic): 클러스터들의 변환높이 구간이 겹치는가?
        겹치지 않으면 → 단일축 θ(z) 밴드로 천장을 그대로 실현 가능.
        겹치면 → 그 높이에선 각도 하나로 타협해야 함(천장과의 격차 = 단일축
        원뿔의 구조적 한계). 이 격차를 정직하게 수치로 보고한다.

클러스터 = '각도 0에서 서포트가 필요한 면'들의 연결성분 (BFS, 팀메 방식과 동일).
크레딧: 클러스터 분해·h구간 겹침 아이디어는 팀메 저장소 26037-arch/find_conical_angle
        (conical_slicing/mesh.py 의 BFS, bands.py 의 h-구간 계획)와 공동.
"""

import numpy as np

from . import analytic
from .config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG


# ─────────────────────────────────────────────────────────────
# 1) 오버행 클러스터 (연결성분, BFS)
# ─────────────────────────────────────────────────────────────
def overhang_clusters(mesh, threshold_deg=THRESHOLD_DEG):
    """각도 0에서 서포트 필요한 면들의 연결성분 리스트를 돌려준다.

    반환: [face_index_array, ...]  (면적 큰 클러스터부터)
    """
    need = analytic.needs_support(mesh, 0.0, "outward", threshold_deg)  # 각도0은 방향 무관
    need_idx = np.where(need)[0]
    if len(need_idx) == 0:
        return []

    # 면 인접(변 공유) 그래프에서 '서포트 필요 면'끼리만 연결
    in_need = np.zeros(len(mesh.faces), dtype=bool)
    in_need[need_idx] = True
    adj = mesh.face_adjacency          # (M,2) 변을 공유하는 면 쌍
    both = in_need[adj[:, 0]] & in_need[adj[:, 1]]
    pairs = adj[both]

    # 인접 리스트 → BFS
    nbr = {int(f): [] for f in need_idx}
    for a, b in pairs:
        nbr[int(a)].append(int(b))
        nbr[int(b)].append(int(a))

    seen = set()
    clusters = []
    for start in need_idx:
        start = int(start)
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        comp = []
        while queue:
            f = queue.pop()
            comp.append(f)
            for g in nbr[f]:
                if g not in seen:
                    seen.add(g)
                    queue.append(g)
        clusters.append(np.array(comp))

    areas = mesh.area_faces
    clusters.sort(key=lambda c: -areas[c].sum())
    return clusters


# ─────────────────────────────────────────────────────────────
# 2) 클러스터별 최적 (방향, 각도) — 제약 없는 '천장'
# ─────────────────────────────────────────────────────────────
def best_for_cluster(mesh, faces, threshold_deg=THRESHOLD_DEG,
                     max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP):
    """이 클러스터 면들만 놓고, 남는 서포트 넓이가 최소인 (방향, 각도).

    동률이면 작은 각도(왜곡 적음)를 고른다. 반환 dict에 남는 넓이도 포함.
    """
    areas = mesh.area_faces[faces]
    best = None
    for direction in ("outward", "inward"):
        for angle in np.arange(0.0, max_angle + 1e-9, step):
            need = analytic.needs_support(mesh, angle, direction, threshold_deg)
            left = float(areas[need[faces]].sum())
            cand = (left, angle, direction)
            if best is None or cand[0] < best[0] - 1e-12 or \
               (abs(cand[0] - best[0]) <= 1e-12 and angle < best[1]):
                best = cand
    left, angle, direction = best
    return {"angle": float(angle), "direction": direction,
            "support_area_left": left, "cluster_area": float(areas.sum())}


def adaptive_ceiling(mesh, threshold_deg=THRESHOLD_DEG,
                     max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP):
    """'클러스터마다 자기 최적 (방향,각도)'라는 이상적 가정 아래 남는 서포트(%).

    ⚠ 이건 실현 가능한 슬라이싱이 아니라 상한(천장)이다: 같은 변환높이에 있는
    클러스터들은 실제로는 각도 하나를 공유해야 한다. 천장과 실현(밴드)의 격차가
    곧 '단일축 원뿔의 구조적 비용'이다.
    반환: (ceiling_pct, cluster_results)
    """
    clusters = overhang_clusters(mesh, threshold_deg)
    total = float(mesh.area_faces.sum())
    left = 0.0
    results = []
    for faces in clusters:
        r = best_for_cluster(mesh, faces, threshold_deg, max_angle, step)
        r["faces"] = faces
        left += r["support_area_left"]
        results.append(r)
    return left / total * 100.0, results


# ─────────────────────────────────────────────────────────────
# 3) 진단: 클러스터 변환높이 구간이 겹치나? (팀메 bands.py 기준과 동일)
# ─────────────────────────────────────────────────────────────
def h_interval(mesh, faces, angle_deg, direction):
    """이 면들 정점의 변환높이 h = z + d·r·tanα 구간 [h_min, h_max]."""
    d = 1.0 if direction == "outward" else -1.0
    vidx = np.unique(mesh.faces[faces].ravel())
    v = mesh.vertices[vidx]
    r = np.hypot(v[:, 0], v[:, 1])
    h = v[:, 2] + d * r * np.tan(np.radians(angle_deg))
    return float(h.min()), float(h.max())


def diagnose(mesh, threshold_deg=THRESHOLD_DEG,
             max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP, verbose=True):
    """천장 + 실현성 진단을 한 번에 요약한다.

    출력 요약:
      · baseline(각도0) 서포트%
      · uniform 최적(전역 한 각도) 서포트%
      · ceiling(클러스터별 자유 각도) 서포트%  ← 상한
      · 클러스터별 (방향, 각도, h구간) + h구간 겹침 여부
    """
    areas = mesh.area_faces
    total = float(areas.sum())
    base = analytic.support_fraction(mesh, 0.0, "outward", threshold_deg)

    # 전역 균일 최적 (서포트만 최소화; k 없음 = 순수 성능 비교)
    uni = None
    for direction in ("outward", "inward"):
        for angle in np.arange(0.0, max_angle + 1e-9, step):
            s = analytic.support_fraction(mesh, angle, direction, threshold_deg)
            if uni is None or s < uni[0] - 1e-12:
                uni = (s, float(angle), direction)

    ceiling_pct, clusters = adaptive_ceiling(mesh, threshold_deg, max_angle, step)

    # h구간 겹침 검사 (각 클러스터의 자기 최적 각도 기준)
    intervals = [h_interval(mesh, c["faces"], c["angle"], c["direction"])
                 for c in clusters]
    order = np.argsort([iv[0] for iv in intervals]) if intervals else []
    overlaps = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if intervals[b][0] < intervals[a][1]:      # 겹침
            overlaps.append((int(a), int(b)))

    if verbose:
        print("=" * 66)
        print(f"[클러스터 진단]  임계 {threshold_deg:.0f}°, 탐색 0~{max_angle}° (해석식)")
        print("=" * 66)
        print(f"  baseline(0°)      : {base:5.1f}%")
        if uni:
            print(f"  균일 최적         : {uni[0]:5.1f}%  ({uni[2]} {uni[1]:.0f}°)")
        print(f"  천장(클러스터 자유): {ceiling_pct:5.1f}%  ← 이상적 상한(그대로는 못 찍음)")
        print("-" * 66)
        print(f"  {'클러스터':>6} | {'면수':>5} | {'모델비중':>7} | {'방향':>8} | {'각도':>5} | 변환높이 h구간")
        for i, c in enumerate(clusters):
            lo, hi = intervals[i]
            print(f"  {i:6d} | {len(c['faces']):5d} | {c['cluster_area']/total*100:6.1f}% | "
                  f"{c['direction']:>8} | {c['angle']:4.0f}° | [{lo:7.1f}, {hi:7.1f}]")
        if overlaps:
            print(f"  ⚠ h구간 겹침 {len(overlaps)}쌍 → 겹친 높이에선 각도 타협 필요"
                  f" (천장과의 격차 = 단일축 원뿔의 구조적 비용)")
        elif len(clusters) > 1:
            print("  ✓ 모든 클러스터 h구간 분리 → θ(z) 밴드로 천장 그대로 실현 가능")
    return {"baseline_pct": base, "uniform": uni, "ceiling_pct": ceiling_pct,
            "clusters": clusters, "intervals": intervals, "overlaps": overlaps}
