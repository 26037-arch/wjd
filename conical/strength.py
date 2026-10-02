"""
strength.py — 이론 강도 지표 (Hankinson 식). 실측 시편과 '랭킹 비교' 가능.

배경 (인용):
    FDM 출력물은 비등방성이다 — 레이어 '안'(road 방향, XY)은 강하고 레이어 '사이'(Z,
    층간 용착)는 약하다. 층간이 약한 이유는 이웃 레이어가 식기 전에 부분적으로만
    융착되어(불완전 weld + road 사이 공극) 균열의 시작점이 되기 때문이다.
      · Ahn et al. (2002), Rapid Prototyping Journal 8(4):248-257,
        DOI 10.1108/13552540210441166  (FDM 비등방성의 고전)
    하중 방향이 레이어 면과 이루는 각 α에 따라 강도가 변하는 표준 근사식 = Hankinson 식:
        σ(α) = σ0·σ90 / (σ0·sin^n α + σ90·cos^n α),   n≈2
      · σ0  = 레이어와 나란한 방향 강도(강함, XY)
      · σ90 = 레이어를 가로지르는 방향 강도(약함, Z)
      · α=0  → 하중이 레이어 면 안 → 가장 강함(σ0)
      · α=90 → 하중이 레이어를 수직으로 잡아뜯음 → 가장 약함(σ90)
    (Hankinson 1921; 지수 n은 Kollmann 1934: 인장 1.5~2.0. 기본 n=2.)
    비평면/원뿔 레이어가 강해지는 이유: 곡면 표면을 따라 레이어가 놓이면 하중이 층간을
    가로지르지 않고 레이어 안(강한 방향)에 머문다 + 계단(응력집중)이 사라진다.
      · Fang et al. (2020) "Reinforced FDM", ACM TOG 39(6),
        DOI 10.1145/3414685.3417834  (하중정렬 곡면레이어로 1.42~6.35배 강도↑)

정직: Hankinson은 원래 나무(목재)용 경험식이라 FDM엔 '유추 적용'이다. 절대값 예측이
    아니라 '전략 간 랭킹' 도구로 쓰고, 비율 r=σ90/σ0 와 지수 n은 우리 시편 시험으로
    보정(calibrate)하는 것이 정석 — 이게 곧 '이론 예측 ↔ 실측' 매치 루프다.
"""

import numpy as np

# PLA 층간/층내 강도비 (문헌 대략 0.3~0.5). 시편 시험으로 보정 대상.
DEFAULT_RATIO = 0.4      # r = σ90/σ0
HANKINSON_N = 2          # 지수 (기본 2)


def layer_normals(mesh, angle_deg, cone_type):
    """각 면 위치에서의 '레이어 면 법선'(실공간 단위벡터).

    유도: 정변환 z' = z + c·ρ·tanθ 의 등고면은 z = h − c·ρ·tanθ 이고, 그
    위쪽 법선은 ∝ (c·sinθ·x/ρ, c·sinθ·y/ρ, cosθ). 따라서 반경 성분 부호는 +c.
    (2026-07 리뷰 수정: 기존 −c 는 outward 가 inward 레이어를 계산하는 부호
    오류였다. 기본 load="z" 경로는 cosθ 성분만 쓰므로 기존 결과에는 영향 없음.)
    angle_deg 는 스칼라(균일) 또는 면마다 다른 배열(구간별) 모두 가능.
    """
    th = np.radians(np.asarray(angle_deg, dtype=float)) * np.ones(len(mesh.faces))
    c = 1.0 if cone_type == "outward" else -1.0
    fc = mesh.vertices[mesh.faces].mean(axis=1)   # 면 centroid (원본 좌표)
    x, y = fc[:, 0], fc[:, 1]
    rho = np.sqrt(x**2 + y**2)
    safe = np.where(rho > 1e-9, rho, 1.0)
    nlx = np.where(rho > 1e-9, c * np.sin(th) * x / safe, 0.0)
    nly = np.where(rho > 1e-9, c * np.sin(th) * y / safe, 0.0)
    nlz = np.cos(th)
    n = np.column_stack([nlx, nly, nlz])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def face_strength(mesh, angle_deg, cone_type, load="z",
                  r=DEFAULT_RATIO, n_exp=HANKINSON_N):
    """면별 무차원 강도 Ŝ ∈ [r, 1] (1=최강, r=최약). Hankinson 식.

    load:
      "z"       : 빌드축(수직) 인장 — 표준 시편 시험과 직접 대응. (기본, 가장 방어적)
      "surface" : 표면 법선 방향 하중 — 곡면/오버행 케이스용.
    """
    nl = layer_normals(mesh, angle_deg, cone_type)
    if load == "z":
        L = np.array([0.0, 0.0, 1.0])
        d = np.abs(nl @ L)                         # |cos(하중, 레이어법선)| = sin α
    else:  # surface
        sn = mesh.face_normals
        d = np.abs(np.sum(nl * sn, axis=1))
    # α = 하중과 레이어'면'이 이루는 각. sin²α = d², cos²α = 1-d².
    # Hankinson(무차원): Ŝ = r / (sin^n α + r cos^n α). n=2 이면 아래 형태.
    if n_exp == 2:
        return r / (d**2 + r * (1 - d**2))
    sin_a = np.clip(d, 0, 1)
    cos_a = np.sqrt(np.clip(1 - d**2, 0, 1))
    return r / (sin_a**n_exp + r * cos_a**n_exp)


def part_strength(mesh, angle_deg, cone_type, load="z",
                  r=DEFAULT_RATIO, reduce="mean"):
    """부품 전체의 강도 대표값.
      reduce="min"  : 최약부 (취성 파괴는 가장 약한 곳에서 시작 — 가장 물리적)
      reduce="mean" : 면적가중 평균 (랭킹 노이즈 적음)
    """
    S = face_strength(mesh, angle_deg, cone_type, load, r)
    if reduce == "min":
        return float(S.min())
    A = mesh.area_faces
    return float((S * A).sum() / A.sum())
