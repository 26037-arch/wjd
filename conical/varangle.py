"""
varangle.py — 높이 구간별 '변수각 원뿔' 전략 (부위별 각도의 실현 가능한 형태).

왜 '높이 구간'인가 (물리적 실현):
    부위마다 각도를 다르게 하려면, 실제로는 각도가 '높이에 따라 변하는 함수' θ(z)여야
    실제로 프린트할 수 있다. 그래서 영역을 '오버행 심한 정도'가 아니라 '높이 구간'으로
    나눈다. (높이 구간 = θ(z)로 실현 가능)

⚠ 정정 (2026-09-21, RotBot 원문 대조):
    이 주석은 앞서 "이것이 RotBot 의 변수각(var_angle) 방식이다" 라고 썼다. **틀렸다.**
    RotBot 의 `Scripts for Variable Angle` 에서 'variable' 은 **실행마다 사용자가 고르는
    상수각**이라는 뜻이지 높이의 함수가 아니다. 원문 확인:
      · README `### Scripts for variable angle`:
        "the cone angle **can be changed**. So it does not only work for 45° angle as used
         for RotBot, but can also be used with much smaller angles (e.g. 15°)"
      · `Transformation_STL_var_angle.py:13` `CONE_ANGLE = 16` — 스칼라 상수
      · 같은 파일 `:18,:35` `transformation_kegel(points, cone_angle_rad, ...)`
        → `np.tan(cone_angle_rad)`, z 의존성 없음
      · `Backtransformation_GCode_var_angle.py:215–228` 도 스칼라
      · 기본 `Backtransformation_GCode.py:40` "divided by **sqrt(2)**" = cos(45°) 하드코딩
    즉 기본 스크립트가 45° 고정이고, var_angle 스크립트는 그것을 **임의 상수로 일반화**
    한 것이다. **θ(z) 는 RotBot 에 없다 — 이 저장소의 확장이다.**
    따름: 블렌드(각도 전환 구간)와 층간격 배율 m 은 RotBot 의 문제공간에 존재하지
    않는다. 상수각이면 s = dT/dZ′ = 0 이라 m ≡ 1 이기 때문이다 (profile.py 참고).
    ⚠ 논문 본문(Appl. Sci. 11(18):8760)의 future work 절은 아직 미확인(망 차단).

평가 방식 (정직):
    각 구간은 '상수각'으로 독립 평가한다(그 구간 면들에 그 각도를 적용했다고 가정).
    이는 이상적 추정이다 — 구간 경계에서 각도가 변하며 생기는 왜곡은 무시한다. 실측용이
    아니라 '균일각 하나 vs 구간별 여러 각도'의 경향 비교용이다. (각도가 급격히 변하면
    실제로는 왜곡이 생기므로, 구간은 적게/각도는 완만하게 두는 것이 전제.)

핵심 논지:
    균일각은 모델 전체에 대한 '타협값' 하나라 손해다. 구간별은 '각도 예산'을 오버행이
    심한 구간에만 몰아써서, 같은(또는 더 적은) 총 왜곡으로 서포트를 더 줄인다.
"""

import math

import numpy as np

# 판정 기준 통일(2026-07 리뷰): metrics(변환공간 근사) → analytic(해석식).
# α=0 에서 두 정의는 일치, α>0 에서 해석식이 물리 기준이다.
from .analytic import (
    face_support_and_staircase,
    support_fraction,
    support_fraction_profile,
)
from .config import (
    ANGLE_STEP,
    BLEND_COST_K,
    BLEND_SHIFT_RATIO,
    MAX_ANGLE_DEG,
    MAX_SPACING_FACTOR,
    THRESHOLD_DEG,
)
from .profile import AngleProfile


def angle_candidates(max_angle, step):
    """0 ~ max_angle 을 step 간격으로. **실수 step 을 받는다.**

    ⚠ 예전에는 `range(0, max_angle + 1, step)` 이라 **정수 step 만** 됐다.
      각도 격자를 0.5° 로 줄여 보려다 TypeError 로 막혔고, 그때 알았다 —
      "격자를 촘촘히 해 보자" 는 실험 자체가 **코드 때문에 불가능**했던 것이다
      (analyze_angle_grid.py, 2026-09-28).
      정수 step 에서는 예전과 **같은 값을 같은 순서로** 낸다(회귀 테스트가 강제).
    """
    n = int(math.floor(float(max_angle) / float(step) + 1e-9))
    return [round(i * float(step), 9) for i in range(n + 1)]

# ─────────────────────────────────────────────────────────────
# 높이 구간 나누기
# ─────────────────────────────────────────────────────────────
def assign_height_bands(mesh, n_bands):
    """면을 centroid 높이(z)로 n_bands개 구간에 배정한다. (0=맨 아래)"""
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    edges = np.linspace(fz.min(), fz.max(), n_bands + 1)
    # digitize 로 각 면을 구간에 배정 (경계 clip)
    labels = np.clip(np.digitize(fz, edges[1:-1]), 0, n_bands - 1)
    return labels, edges


# ─────────────────────────────────────────────────────────────
# 한 구간(또는 전체)에 대한 최적 각도/방향 — J 기준
# ─────────────────────────────────────────────────────────────
def best_angle_for_mask(mesh, mask, orig_areas, k,
                        max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                        threshold_deg=THRESHOLD_DEG):
    """mask 면들만 놓고 J=(서포트 감소 %) − k×각도 가 최대인 (각도, 방향)."""
    band_area = orig_areas[mask].sum()
    need0, _ = face_support_and_staircase(mesh, 0, "outward", threshold_deg)
    base_pct = orig_areas[mask & need0].sum() / band_area * 100.0

    best = (-1e9, 0, "outward")   # (J, angle, direction)
    for c in ("outward", "inward"):
        for a in angle_candidates(max_angle, step):
            need, _ = face_support_and_staircase(mesh, a, c, threshold_deg)
            pct = orig_areas[mask & need].sum() / band_area * 100.0
            J = (base_pct - pct) - k * a
            if J > best[0]:
                best = (J, a, c)
    return best[1], best[2]


# ─────────────────────────────────────────────────────────────
# 배정(각 구간의 각도)을 받아 전체 3지표를 계산
# ─────────────────────────────────────────────────────────────
def evaluate_assignment(mesh, assignment, threshold_deg=THRESHOLD_DEG):
    """assignment: [(mask, angle, cone_type), ...]  → 지표 dict.

      support_pct : 전체 대비 남은 서포트 넓이(%)   (낮을수록 좋음, 속도와 연결)
      staircase   : 강도 proxy (낮을수록 좋음)
      avg_angle   : 면적가중 평균 각도 (복잡도/왜곡 비용 proxy)
    """
    areas = mesh.area_faces
    total = areas.sum()
    sup_area = 0.0
    stair = 0.0
    ang_area = 0.0
    for mask, ang, c in assignment:
        need, st = face_support_and_staircase(mesh, ang, c, threshold_deg)
        sup_area += areas[mask & need].sum()
        stair += (st[mask] * areas[mask]).sum()
        ang_area += ang * areas[mask].sum()
    return {
        "support_pct": sup_area / total * 100.0,
        "staircase": stair / total,
        "avg_angle": ang_area / total,
    }


# ─────────────────────────────────────────────────────────────
# 전략들: 균일 / 구간별 / 세밀(면마다)
# ─────────────────────────────────────────────────────────────
def select_uniform(mesh, k, max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                   threshold_deg=THRESHOLD_DEG):
    """전역 단일 각도 (RotBot식 균일 원뿔 = 비교 대상)."""
    allmask = np.ones(len(mesh.faces), dtype=bool)
    a, c = best_angle_for_mask(mesh, allmask, mesh.area_faces, k,
                               max_angle, step, threshold_deg)
    m = evaluate_assignment(mesh, [(allmask, a, c)], threshold_deg)
    return {"strategy": "uniform", "n_regions": 1, "profile": [(a, c)], **m}


def select_banded(mesh, k, n_bands, max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                  threshold_deg=THRESHOLD_DEG):
    """높이 n_bands 구간, 각 구간에 최적 각도 (변수각 θ(z)로 실현 가능)."""
    labels, edges = assign_height_bands(mesh, n_bands)
    areas = mesh.area_faces
    assignment = []
    profile = []
    for i in range(n_bands):
        mask = labels == i
        if not mask.any():
            profile.append(None)
            continue
        a, c = best_angle_for_mask(mesh, mask, areas, k, max_angle, step, threshold_deg)
        assignment.append((mask, a, c))
        profile.append((a, c))
    m = evaluate_assignment(mesh, assignment, threshold_deg)
    return {"strategy": f"banded-{n_bands}", "n_regions": n_bands,
            "profile": profile, "edges": edges, **m}


# ─────────────────────────────────────────────────────────────
# 프로필 단위 평가 — J 에 '블렌드 비용'을 넣는다
# ─────────────────────────────────────────────────────────────
def blend_penalty(mesh, profile, radius_profile=None,
                  spacing_limit=MAX_SPACING_FACTOR, direction="outward"):
    """블렌드 비용 = Σ (그 구간에 있는 표면적 %) × (m−1)/(limit−1).

    각도 변화가 없으면 블렌드 구간 자체가 없어 0 이 되고, J 는 균일각 J 와
    정확히 같아진다.

    ⚠⚠ **이 항이 무엇인지 2026-09-22 에 다시 쟀다. 세 가지가 전부 틀렸다**
        (`analyze_blend_cost.py`, docs/verification.md):

    ① **`(m−1)/(limit−1)` 은 사실상 항등이다.** 계획기가 폭을 층간격 제약의
       **최소값**으로 잡으므로 `m = limit` 이 정확히 물린다 → risk ≡ 1
       (선택된 프로필 16/16, 값 1.0000). 즉 **'휴리스틱 선형 가중' 은 한 번도
       작동한 적이 없다.** 남는 것은 `Σ(블렌드 구간 표면적 %)` 뿐이다.
       (risk < 1 은 `min_blend` 바닥값이 제약보다 클 때만 가능하고, 이 표본에서는
        한 번도 안 일어났다.)
    ② **'레이어가 벌어져서 생기는 손상' 을 재는 게 아니다.** 층간격 제약이 이미
       `m ≤ 1.5` 로 가두는데 그 1.5 가 **검사기 A 의 지지 창과 같은 값**이다
       (`toolpath.check_support`: `vwin = layer_height * 1.5`). 즉 제약을 지키는
       블렌드는 구조적으로 미지지를 못 만든다. 실측도 그렇다 — 블렌드 구간 **안**의
       미지지 밀도가 **밖의 0.48 배**(중앙값, 16 모델)로 오히려 **낮다.**
    ③ **해석식 예측의 오차를 메우는 보정항도 아니다.** 그 예측은 블렌드가 없는
       상수 프로필에서도 실측과 1.5~4.6 배로 어긋난다(눈금이 안 맞는 순위 대리물).
       축상 근사 때문도 아니다 — 정확 Z′ 판과 차이가 0 이다(analytic 참고).

    ✅ **그럼 무엇인가: 정규화항이다.** 블렌드 폭이 `r_b·|Δtanθ|/(limit−1)` 이므로
       이 항은 결국 **반경으로 가중한 각도 변화량**을 벌한다. 실제 기능은 탐색을
       병리적 해에서 밀어내는 것이고, 그건 실측으로 확인된다 — `k_blend=0` 이면
       구가 `[36°, −44°]`(방향 전환, 블렌드 19.9mm, 1.811%p)를 고른다.

    ⇒ 따름: **`k_blend` 는 물리 계수가 아니라 정규화 세기다.** "왜 선형인가" 는
       물을 필요가 없는 질문이었다(가중이 항등이므로). 물어야 할 것은 "정규화로서
       세기가 맞나" 이고, 거기 답은 **전역값 하나로는 안 맞는다** 이다
       (docs/verification.md 2026-09-22: 0.1→0.05 에서 허리 r=5 는 4.8 배 좋아지고
        구는 4.2 배 나빠진다).
    """
    if spacing_limit is None or spacing_limit <= 1.0:
        return 0.0
    c = 1.0 if direction == "outward" else -1.0
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    areas = mesh.area_faces
    total = areas.sum()
    cost = 0.0
    for i in range(len(profile.zs) - 1):
        a, b = float(profile.zs[i]), float(profile.zs[i + 1])
        dt = profile.tans[i + 1] - profile.tans[i]
        if abs(dt) < 1e-12:
            continue
        s = dt / (b - a)
        r = (radius_profile.max_between(a, b) if radius_profile is not None
             else float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max()))
        m = abs(1.0 - c * r * s)
        risk = min(1.0, max(0.0, (m - 1.0) / (spacing_limit - 1.0)))
        area_pct = areas[(fz >= a) & (fz <= b)].sum() / total * 100.0
        cost += area_pct * risk
    return float(cost)


def profile_objective(mesh, profile, base_pct, k, k_blend=BLEND_COST_K,
                      radius_profile=None, spacing_limit=MAX_SPACING_FACTOR,
                      threshold_deg=THRESHOLD_DEG):
    """프로필 하나의 J = (서포트 감소 %p) − k×평균|θ| − k_blend×블렌드 비용.

    상수 프로필이면 블렌드 비용 0, 평균|θ| = 그 각도 → 균일각 J 와 동일하다
    (tests/test_blend_cost_j.py 가 강제).
    """
    sup = support_fraction_profile(mesh, profile, threshold_deg)
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    areas = mesh.area_faces
    mean_ang = float((np.abs(profile.theta_at(fz)) * areas).sum() / areas.sum())
    pen = blend_penalty(mesh, profile, radius_profile, spacing_limit)
    return {"J": (base_pct - sup) - k * mean_ang - k_blend * pen,
            "support_pct": sup, "avg_angle": mean_ang, "blend_penalty": pen}


def _merge_bands(edges, thetas):
    """같은 각도인 이웃 밴드 병합 → 불필요한 블렌드를 안 만든다."""
    bands = [[float(edges[0]), float(edges[1]), float(thetas[0])]]
    for i in range(1, len(thetas)):
        if abs(thetas[i] - bands[-1][2]) < 1e-12:
            bands[-1][1] = float(edges[i + 1])
        else:
            bands.append([float(edges[i]), float(edges[i + 1]), float(thetas[i])])
    return [tuple(b) for b in bands]


def select_banded_j(mesh, k, n_bands, r_max, radius_profile=None,
                    spacing_limit=MAX_SPACING_FACTOR, max_shift=None,
                    k_blend=BLEND_COST_K, max_angle=MAX_ANGLE_DEG,
                    step=ANGLE_STEP, threshold_deg=THRESHOLD_DEG,
                    max_sweeps=4):
    """밴드별 각도를 '실제로 만들어질 프로필'의 J 로 고른다 (좌표하강).

    select_banded 와의 차이: 저기는 밴드마다 독립적으로 '그 구간만 놓고' 최적
    각도를 고른다 — 각도를 바꾸는 데 드는 블렌드 비용을 아예 모른다. 여기서는
    후보 각도를 넣어 프로필을 실제로 만들고(층간격 제약 포함) 그 프로필의 J 를
    잰다. 그래서 '블렌드가 비싼 모델(구처럼 어디나 뚱뚱한)'에서는 저절로 균일각
    쪽으로 수렴하고, '허리가 있는 모델'에서만 각도를 나눈다.

    탐색: 밴드 수 N 에 대해 전수 조사는 후보^N 이라 못 한다. '한 밴드씩 돌아가며
    다시 고르기'(좌표하강)를 값이 안 변할 때까지. 이웃은 아래 ⚠ 때문에 '연속한
    밴드 덩어리'라 블록이 N(N+1)/2 개다 — 비용은 N 에 **선형이 아니라 2차**다
    (실측 선택 시간 1, 4, 8, 14, 21, 43초 @ N=1..6, docs/verification.md).

    ⚠ 좌표하강만으로는 균일해를 놓친다(실측으로 발견): 구에서 [36°,0°] 로 출발하면
      균일 [36°,36°] 로 가려면 두 밴드를 '동시에' 바꿔야 하는데, 한 칸씩 움직이는
      중간 상태([36°,2°] 등)가 블렌드 비용을 다 물어서 [2°,2°] 같은 국소 최적에
      갇힌다. 그래서 균일 후보(모든 밴드 같은 각도)를 전부 따로 평가해 넣는다.
      부수 효과로 **결과가 균일 원뿔보다 나쁠 수 없다**(J 기준)는 성질이 생긴다.
    """
    labels, edges = assign_height_bands(mesh, n_bands)
    areas = mesh.area_faces
    base_pct = support_fraction(mesh, 0.0, "outward", threshold_deg)
    if max_shift is None:      # 모델 높이 기준 (밴드 수와 무관 — config 주석 참조)
        max_shift = BLEND_SHIFT_RATIO * float(mesh.bounds[1][2] - mesh.bounds[0][2])

    # 후보 각도(부호 있음: 음수=inward). 0 은 한 번만.
    cands = sorted({float(sgn * a)
                    for a in angle_candidates(max_angle, step)
                    for sgn in (1, -1)})

    def build(thetas):
        return AngleProfile.from_bands(
            _merge_bands(edges, thetas), r_max, radius_profile=radius_profile,
            spacing_limit=spacing_limit, max_shift=max_shift)

    def score(thetas):
        try:
            prof = build(thetas)
        except ValueError:
            return None, {"J": -1e9}
        # 안전망: 층간격 제약을 어긴 프로필은 J 가 아무리 좋아도 채택하지 않는다.
        #   계획기가 자리를 못 찾으면 각도를 깎아 넣게 돼 있지만, 그 경로에 버그가
        #   있으면 '예측은 최고인데 실제로는 못 찍는' 프로필이 뽑힌다 —
        #   실제로 그랬다(램프 N=4, 폭 0.14mm 에 42°→0°, 배율 53.7배, J=5.72인데
        #   툴패스 미지지 5.74%). 평가 단계에서 한 번 더 막는다.
        if spacing_limit is not None and \
                prof.check_spacing(r_max, "outward", spacing_limit, radius_profile):
            return None, {"J": -1e9}
        return prof, profile_objective(mesh, prof, base_pct, k, k_blend,
                                       radius_profile, spacing_limit,
                                       threshold_deg)

    # 이웃(neighborhood)은 '연속한 밴드 덩어리'다 — 한 밴드씩이 아니라.
    #   블렌드 비용은 '이웃한 밴드의 각도가 다른 자리'에 붙으므로, 한 칸씩 바꾸는
    #   이웃으로는 밴드 두 개를 함께 내리는 수를 못 둔다(중간 상태가 비용을 다 문다).
    #   실측: N=4 램프에서 [24,24,24,24] → [24,24,0,0] 이 단일 좌표로는 도달 불가라
    #   J가 N에 대해 단조가 아니었다(N=2: 3.30 > N=4: 2.33).
    #   덩어리 이동은 크기 1 블록을 포함하므로 기존 좌표하강의 상위집합이고,
    #   전체 범위 블록이 곧 균일해라 '균일보다 나쁠 수 없다'가 구조적으로 보장된다.
    blocks = [(i, j) for i in range(n_bands) for j in range(i, n_bands)]

    def descend(thetas):
        thetas = list(thetas)
        prof, met = score(thetas)
        for _ in range(max_sweeps):
            changed = False
            for i, j in blocks:
                best_tr, best_m, best_p = None, met, prof
                for t in cands:
                    if all(thetas[b] == t for b in range(i, j + 1)):
                        continue
                    trial = list(thetas)
                    for b in range(i, j + 1):
                        trial[b] = t
                    p, m = score(trial)
                    if m["J"] > best_m["J"] + 1e-9:
                        best_tr, best_m, best_p = trial, m, p
                if best_tr is not None:
                    thetas = best_tr
                    met, prof = best_m, best_p
                    changed = True
            if not changed:
                break
        return thetas, prof, met

    # 출발점 ①: 기존 방식(밴드 독립 선택) — 블렌드 비용을 모르는 선택
    start = []
    for i in range(n_bands):
        mask = labels == i
        if not mask.any():
            start.append(start[-1] if start else 0.0)
            continue
        a, c = best_angle_for_mask(mesh, mask, areas, k, max_angle, step,
                                   threshold_deg)
        start.append(float(a) if c == "outward" else -float(a))

    # 출발점 ②: 균일 후보 전수 (좌표하강이 못 가는 곳 — 위 ⚠ 참조)
    best_uni, best_uni_m = None, {"J": -1e9}
    for t in cands:
        _, m = score([t] * n_bands)
        if m["J"] > best_uni_m["J"]:
            best_uni, best_uni_m = [t] * n_bands, m

    results = [descend(start)]
    if best_uni is not None:
        results.append(descend(best_uni))
    thetas, prof, met = max(results, key=lambda r: r[2]["J"])

    return {"strategy": f"banded-{n_bands}-J", "n_regions": n_bands,
            "edges": edges, "thetas": thetas, "start_thetas": start,
            "uniform_best": best_uni[0] if best_uni else None,
            "uniform_J": best_uni_m["J"],
            "profile_obj": prof, "staircase": float("nan"), **met}


def select_fine(mesh, k, max_angle=MAX_ANGLE_DEG, step=ANGLE_STEP,
                threshold_deg=THRESHOLD_DEG):
    """세밀: 면마다 각자 최적 각도 (복잡도 최대 = 성능 하한선/이론적 바닥).

    각 (각도,방향)을 전체에 한 번씩만 적용해 면별 지표를 미리 구하고, 면마다 자기
    J가 가장 큰 각도를 고른다. '이보다 더 줄이긴 어렵다'는 기준선.
    """
    areas = mesh.area_faces
    F = len(mesh.faces)
    need0, _ = face_support_and_staircase(mesh, 0, "outward", threshold_deg)
    # 면별 baseline 서포트(0/1). J는 면 단위로 (감소) - k*각도.
    best_J = np.full(F, -1e9)
    best_ang = np.zeros(F, dtype=int)
    best_need = need0.copy()
    best_stair = np.zeros(F)
    _, stair0 = face_support_and_staircase(mesh, 0, "outward", threshold_deg)
    for c in ("outward", "inward"):
        for a in angle_candidates(max_angle, step):
            need, st = face_support_and_staircase(mesh, a, c, threshold_deg)
            # 면 단위 J: baseline에서 서포트가 사라지면 +1(=100%p*면), 각도비용 -k*a
            gain = (need0.astype(float) - need.astype(float)) * 100.0
            J = gain - k * a
            better = J > best_J
            best_J[better] = J[better]
            best_ang[better] = a
            best_need[better] = need[better]
            best_stair[better] = st[better]
    total = areas.sum()
    return {
        "strategy": "fine", "n_regions": F,
        "support_pct": areas[best_need].sum() / total * 100.0,
        "staircase": (best_stair * areas).sum() / total,
        "avg_angle": (best_ang * areas).sum() / total,
        "profile": None,
    }
