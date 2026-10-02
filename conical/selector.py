"""
selector.py — 원뿔 각도/방향 자동 결정기.

지금까지의 코드는 "각도별 서포트 표"만 보여줬고, 각도는 사람이 눈으로 골랐다.
이 코드는 그 마지막 판단을 컴퓨터가 하게 만든다. = "자동 + 투명"의 최소 증명

입력:  메시(모델)
출력:  추천 각도 + 방향 + "왜 그렇게 정했는지" 이유

핵심 아이디어 (평가함수 J):
    J(θ, 방향) = (서포트 감소량)  −  k × θ
    - 서포트 감소량: 각도 0일 때 대비 서포트가 몇 %p 줄었나 (이득)
    - k × θ        : 각도가 클수록 커지는 비용                 (대가)
    - k            : "각도 1도를 서포트 몇 %p와 맞바꿀 것인가"

왜 비용이 필요한가:
    서포트만 최소화하면 곡선이 끝까지 내려가므로 항상 '최대 각도'가 선택된다.
    (sweep 실험에서 확인함) 큰 각도는 대가가 있으므로 J로 균형을 잡는다.

k를 어떻게 정하나:
    임의로 정하면 "왜 0.2인가?"에 답할 수 없다. 그래서 k를 하나로 못박지 않고,
    k를 훑으며 "k에 따라 선택이 어떻게 변하는가"(민감도 분석)를 함께 출력한다.
"""

# 판정 기준 통일(2026-07 리뷰): 변환공간 근사(sweep) 대신 해석식(실공간 국소
# 레이어 각)을 쓴다. 시그니처 동일. 근거: docs/warped_threshold_finding.md
from .analytic import support_fraction
from .config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG
from .varangle import angle_candidates


# ─────────────────────────────────────────────────────────────
# 평가함수 J  ← 이 연구의 핵심. 여기를 바꾸면 알고리즘 성격이 바뀐다.
# ─────────────────────────────────────────────────────────────
def evaluate_J(support_at_angle, support_baseline, angle_deg, k):
    """
    J = 이득 − 비용
      이득 = 서포트 감소량 (%p)
      비용 = k × 각도

    확장 아이디어 (향후):
      - 변환 오차 항 추가:      − k2 * distortion(angle)
      - U축 각도 변화율 항 추가: − k3 * u_axis_variation(path)
      - 짐벌락/특이점 회피 페널티
    """
    gain = support_baseline - support_at_angle
    cost = k * angle_deg
    return gain - cost


def select_cone(mesh, k, max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP, verbose=True):
    """모든 (각도, 방향) 후보에 대해 J를 계산하고 최적을 고른다."""
    baseline = support_fraction(mesh, 0, "outward", THRESHOLD_DEG)  # 각도 0 = 일반 슬라이싱

    candidates = []
    for direction in ["outward", "inward"]:
        for angle in angle_candidates(max_angle, step):
            sup = support_fraction(mesh, angle, direction, THRESHOLD_DEG)
            J = evaluate_J(sup, baseline, angle, k)
            candidates.append({
                "angle": angle, "direction": direction,
                "support": sup, "gain": baseline - sup, "cost": k * angle, "J": J,
            })

    best = max(candidates, key=lambda c: c["J"])

    if verbose:
        print("=" * 62)
        print(f"[결정]  방향 = {best['direction']} 원뿔,  각도 = {best['angle']}°")
        print("=" * 62)
        print("[이유]")
        print(f"  · 일반 슬라이싱(0°) 서포트 : {baseline:.1f}%")
        print(f"  · 선택 각도에서의 서포트   : {best['support']:.1f}%")
        print(f"  · 서포트 감소(이득)        : {best['gain']:.1f}%p")
        print(f"  · 각도 비용 (k={k} × {best['angle']}°) : {best['cost']:.1f}")
        print(f"  · 최종 점수 J              : {best['J']:.2f}")
        print(f"  · 탐색 범위                : 0° ~ {max_angle}° (하드웨어 제약)")

        # 왜 더 큰 각도를 안 골랐는지 설명 (투명성)
        bigger = [c for c in candidates
                  if c["direction"] == best["direction"] and c["angle"] > best["angle"]]
        if bigger:
            nxt = min(bigger, key=lambda c: c["angle"])
            print(f"  · 더 큰 각도({nxt['angle']}°)를 안 고른 이유: "
                  f"서포트는 {nxt['gain'] - best['gain']:+.1f}%p 더 줄지만 "
                  f"비용이 {nxt['cost'] - best['cost']:+.1f} 늘어 J가 {nxt['J']:.2f}로 낮아짐")
        elif best["angle"] == max_angle:
            print("  · 주의: 최대 각도가 선택됨 → k가 너무 작아 비용이 무시되고 있을 수 있음")

    return best, candidates


def k_sensitivity(mesh, k_values, max_angle=MAX_ANGLE_DEG):
    """k를 바꿔가며 선택이 어떻게 변하는지 (정직한 방법: k를 못박지 않는다)"""
    print("\n" + "=" * 62)
    print("[k 민감도 분석]  k를 바꾸면 선택이 어떻게 달라지는가")
    print("=" * 62)
    print(f"{'k':>6} | {'방향':>8} | {'각도':>5} | {'서포트':>7} | {'감소':>6}")
    print("-" * 62)
    rows = []
    for k in k_values:
        best, _ = select_cone(mesh, k, max_angle, verbose=False)
        print(f"{k:6.2f} | {best['direction']:>8} | {best['angle']:4d}° | "
              f"{best['support']:6.1f}% | {best['gain']:5.1f}%p")
        rows.append((k, best))
    return rows
