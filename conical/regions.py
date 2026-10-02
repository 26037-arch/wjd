"""
regions.py — 부위별(영역별) 원뿔 각도 결정 (Stage 4, 확장).

지금까지(selector.py)는 모델 '전체'에 각도 하나를 정했다. 하지만 실제 모델은
부위마다 오버행 정도가 다르다. 이 파일은 모델을 '오버행이 심한 정도'로 몇 개
영역으로 나누고, 각 영역에 각도를 따로 정한다.

솔직한 전제 (중요, 발표에서 꼭 밝힐 것):
    이건 시뮬레이션 상의 '이상적 추정'이다. 영역 경계에서 슬라이싱 각도가 갑자기
    바뀌는 실제 전이 비용은 무시하고, "각 영역을 그 영역 최적 각도로 처리했다고
    가정했을 때 남는 서포트"를 합산해 '경향'만 본다. 증명이 아니다.

넓이 가중치 규칙:
    영역마다 각도가 달라서 변환 후 넓이 스케일(1/cos²θ)이 제각각이다. 그래서
    공정 비교를 위해 넓이 가중치는 '원본 모델' 넓이로 통일한다.
    → 오버행 '판정'은 변환된 법선으로, 넓이 '가중치'는 원본으로.
"""

import numpy as np

from .config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG
from .overhang import analyze_overhangs
from .selector import evaluate_J
from .varangle import angle_candidates


# ─────────────────────────────────────────────────────────────
# 1) 영역 나누기: 오버행이 심한 정도로 면들을 그룹핑
# ─────────────────────────────────────────────────────────────
def assign_regions_by_overhang(mesh, n_regions, threshold_deg=THRESHOLD_DEG):
    """면들을 '오버행 심한 정도'로 n_regions개 영역으로 나눈다.

      · 영역 0                : 오버행이 임계각 이하 (서포트 걱정 없는 면)
      · 영역 1 .. n_regions-1 : 임계각을 넘는 면들을, 심한 정도(임계각~90°)를
                                같은 폭으로 나눠 배정. 번호가 클수록 더 심함.

    반환: labels[F]  (각 면이 몇 번 영역인지, 정수 배열)
    """
    overhang_angle, _ = analyze_overhangs(mesh, threshold_deg)
    labels = np.zeros(len(overhang_angle), dtype=int)

    severe = overhang_angle > threshold_deg
    if n_regions <= 1 or not severe.any():
        return labels  # 나눌 게 없으면 전부 영역 0

    # 임계각~90°를 (n_regions-1)개 구간으로 등분한 경계.
    #   예) n_regions=3 -> edges=[45, 67.5, 90] -> 내부경계 [67.5]
    edges = np.linspace(threshold_deg, 90.0, n_regions)
    # severe 면들을 내부 경계로 나눠 1..n_regions-1 번 영역에 배정
    bucket = np.digitize(overhang_angle[severe], edges[1:-1], right=False) + 1
    labels[severe] = bucket
    return labels


# ─────────────────────────────────────────────────────────────
# 2) 서포트 판정 도우미: 변환 후에도 서포트가 필요한 면들
# ─────────────────────────────────────────────────────────────
def still_needs_support(mesh, cone_angle_deg, cone_type, threshold_deg=THRESHOLD_DEG):
    """이 각도에서도 서포트가 필요한 면들의 boolean 배열(전체 면 기준).

    판정 기준 통일(2026-07 리뷰): 변환공간 근사(메시 재변환+법선 재계산) 대신
    해석식(실공간 국소 레이어 각)을 쓴다. α=0 에서 두 정의는 일치.
    """
    from . import analytic
    return analytic.needs_support(mesh, cone_angle_deg, cone_type, threshold_deg)


# ─────────────────────────────────────────────────────────────
# 3) 한 영역에 대한 최적 각도/방향 (그 영역 면들만 대상)
# ─────────────────────────────────────────────────────────────
def best_angle_for_region(mesh, face_mask, orig_areas, k,
                          max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                          threshold_deg=THRESHOLD_DEG):
    """face_mask 면들만 놓고, J가 가장 큰 (각도, 방향)을 고른다.

    서포트는 '이 영역 안에서의 원본넓이 기준 %'로 잰다.
    """
    region_area = orig_areas[face_mask].sum()

    def region_support_pct(angle, direction):
        need = still_needs_support(mesh, angle, direction, threshold_deg)
        return orig_areas[face_mask & need].sum() / region_area * 100.0

    baseline = region_support_pct(0, "outward")   # 각도 0 = 일반 슬라이싱

    best = None
    for direction in ["outward", "inward"]:
        for angle in angle_candidates(max_angle, step):
            sup = region_support_pct(angle, direction)
            J = evaluate_J(sup, baseline, angle, k)
            cand = {"angle": angle, "direction": direction,
                    "support": sup, "gain": baseline - sup,
                    "cost": k * angle, "J": J}
            if best is None or J > best["J"]:
                best = cand
    best["baseline"] = baseline
    best["region_area"] = region_area
    return best


# ─────────────────────────────────────────────────────────────
# 4) 전체 흐름: 영역별로 각도를 정하고, 남는 서포트를 합산
# ─────────────────────────────────────────────────────────────
def select_regions(mesh, k, n_regions,
                   max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                   threshold_deg=THRESHOLD_DEG, verbose=True):
    """모델을 n_regions개 영역으로 나눠 각 영역의 각도를 정하고 결과를 요약한다.

    반환: (region_results[list], total_remaining_pct[float])
      · total_remaining_pct: 모델 전체 대비 '이상적 추정' 남은 서포트 넓이(%)
    """
    labels = assign_regions_by_overhang(mesh, n_regions, threshold_deg)
    orig_areas = mesh.area_faces
    total_area = orig_areas.sum()

    # 비교 기준: 각도 0(일반 슬라이싱)일 때 전체 남은 서포트
    need0 = still_needs_support(mesh, 0, "outward", threshold_deg)
    baseline_pct = orig_areas[need0].sum() / total_area * 100.0

    region_results = []
    remaining_area = 0.0
    for r in range(n_regions):
        mask = labels == r
        if not mask.any():
            continue  # 이 영역에 속한 면이 없으면 건너뜀
        best = best_angle_for_region(mesh, mask, orig_areas, k,
                                     max_angle, step, threshold_deg)
        need = still_needs_support(mesh, best["angle"], best["direction"], threshold_deg)
        area_left = orig_areas[mask & need].sum()
        remaining_area += area_left

        best["region"] = r
        best["face_count"] = int(mask.sum())
        best["area_pct_of_model"] = orig_areas[mask].sum() / total_area * 100.0
        best["support_area_left_pct"] = area_left / total_area * 100.0
        region_results.append(best)

    total_remaining_pct = remaining_area / total_area * 100.0

    if verbose:
        _print_regions(region_results, baseline_pct, total_remaining_pct,
                       n_regions, k, max_angle)

    return region_results, total_remaining_pct


def _print_regions(region_results, baseline_pct, total_remaining_pct,
                   n_regions, k, max_angle):
    """영역별 결정과 그 이유를 사람이 읽기 좋게 출력한다 (투명성)."""
    print("=" * 66)
    print(f"[부위별 결정]  영역 {n_regions}개,  k={k},  탐색 0°~{max_angle}°")
    print("=" * 66)
    print("  ※ 이상적 추정: 각 영역을 그 영역 최적 각도로 처리했다고 가정.")
    print(f"  · 일반 슬라이싱(0°) 전체 서포트 : {baseline_pct:.1f}%  (원본넓이 기준)")
    print("-" * 66)
    print(f"{'영역':>4} | {'모델비중':>7} | {'방향':>8} | {'각도':>5} | "
          f"{'영역서포트':>9} | {'남김(전체%)':>10}")
    print("-" * 66)
    for c in region_results:
        print(f"{c['region']:>4} | {c['area_pct_of_model']:6.1f}% | "
              f"{c['direction']:>8} | {c['angle']:4d}° | "
              f"{c['support']:5.1f}%→ | {c['support_area_left_pct']:9.1f}%")
    print("-" * 66)
    print(f"  · 부위별 처리 후 전체 남은 서포트(추정) : {total_remaining_pct:.1f}%")
    print(f"  · 일반 대비 감소                        : "
          f"{baseline_pct - total_remaining_pct:+.1f}%p")
