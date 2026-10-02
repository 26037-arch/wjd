"""
analytic.py — 해석적(닫힌형) 오버행 판정 + 면별 임계각. 팀메 방식을 채택·통합.

우리 기존 방식(sweep.py/metrics.py)은 각도마다 메시를 '실제로 변환'해 법선을 다시
계산했다(느리고, 스윕 필요). 팀메(find_conical_angle)는 변환 없이, 원본 면 법선을
'국소 원뿔 빌드 방향'에 투영한 값을 바로 쓴다:

    g(α) = n_z·cosα + d·n_r·sinα                (d=방향부호, n_r=법선의 반경성분)

  · 이는 실공간에서 면이 국소 원뿔 레이어와 이루는 각(변환 왜곡 없음).
  · 닫힌형이라 빠르고, 면마다 '자기지지가 되는 최소 각도 θ*'를 asin으로 즉시 푼다.

자기지지 조건(우리 45° 관례와 정렬): g(α) ≥ κ,  κ = −sin(threshold).
  · α=0에서 g=n_z 이므로 n_z ≥ −sin45  →  기존 판정과 정확히 일치.

출처/크레딧: 팀메 저장소 26037-arch/find_conical_angle
             (conical_slicing/evaluation.py 의 _score = n_z·cos + d·n_r·sin,
              phase=atan2(a,b), boundary=asin(κ/R) 최소각 로직).
⚠ 이 방식도 원뿔 축(원점) 기준 n_r 을 쓰므로, 분석 전 meshio.center_on_axis 필요.
"""

import numpy as np

from .config import MAX_ANGLE_DEG, THRESHOLD_DEG


def radial_normal(mesh):
    """각 면 법선의 '반경(축에서 바깥) 방향' 성분 n_r. 축 위(중심)면은 0."""
    fn = mesh.face_normals
    c = mesh.vertices[mesh.faces].mean(axis=1)
    x, y = c[:, 0], c[:, 1]
    r = np.sqrt(x**2 + y**2)
    safe = np.where(r > 1e-9, r, 1.0)
    return np.where(r > 1e-9, (x * fn[:, 0] + y * fn[:, 1]) / safe, 0.0)


def _dir_sign(direction):
    return 1.0 if direction == "outward" else -1.0


def overhang_score(mesh, angle_deg, direction, radial=None):
    """g(α) = n_z·cosα + d·n_r·sinα. 클수록 안전(위를 봄), 작을수록 오버행."""
    nz = mesh.face_normals[:, 2]
    nr = radial_normal(mesh) if radial is None else radial
    d = _dir_sign(direction)
    a = np.radians(angle_deg)
    return nz * np.cos(a) + d * nr * np.sin(a)


def needs_support(mesh, angle_deg, direction, threshold_deg=THRESHOLD_DEG):
    """해석적 자기지지 판정: g(α) < κ 이면 서포트 필요."""
    kappa = -np.sin(np.radians(threshold_deg))
    return overhang_score(mesh, angle_deg, direction) < kappa


def support_fraction(mesh, angle_deg, direction, threshold_deg=THRESHOLD_DEG):
    """해석적 방식으로 남은 서포트 넓이 비율(%)."""
    need = needs_support(mesh, angle_deg, direction, threshold_deg)
    areas = mesh.area_faces
    return areas[need].sum() / areas.sum() * 100.0


def support_fraction_profile(mesh, profile, threshold_deg=THRESHOLD_DEG):
    """가변각 프로필 θ(Z′)의 남은 서포트 넓이(%) 추정 (부호 있는 각도, c=+1).

    ⚠ 각 면의 각도는 centroid 의 '축상 Z′≈z' 근사로 조회한다 — 축에서 먼 면은
      실제 Z′가 r·tanθ 만큼 달라 경계 부근에서 어긋날 수 있다.

    ✅ 그 오차의 실제 크기를 쟀다 (2026-09-22): **17 모델 × {균일, 밴드2} 에서
      `support_fraction_profile_exact` 와 차이가 정확히 0 이다.** 근사가 무해한
      이유는 우연이 아니라 설계다 — 계획기가 블렌드를 **반경이 작은 높이로
      옮기고**(`BLEND_SHIFT_RATIO`), 오차 `r·tanθ` 가 작아지는 곳이 바로 거기다.
      ⚠ 그래도 보장은 아니다. 경계를 손으로 큰 반경에 두면(`--spacing-limit 0`
      등) 어긋날 수 있으므로, 의심되면 아래 exact 판을 써서 대조할 것.
    """
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    th = np.radians(profile.theta_at(fz))
    nz = mesh.face_normals[:, 2]
    nr = radial_normal(mesh)
    g = nz * np.cos(th) + nr * np.sin(th)
    need = g < -np.sin(np.radians(threshold_deg))
    areas = mesh.area_faces
    return areas[need].sum() / areas.sum() * 100.0


def support_fraction_profile_exact(mesh, profile, threshold_deg=THRESHOLD_DEG,
                                   direction="outward"):
    """위와 같되 **축상 근사를 쓰지 않는다** — 면마다 정확한 Z′ 를 푼다.

    `support_fraction_profile` 은 면의 각도를 centroid 의 실공간 z 로 조회한다.
    실제로 그 면이 놓이는 변환공간 높이는 `Z′ = z + c·r·tanθ(Z′)` 이고,
    `profile.solve_forward` 가 그 방정식의 **정확 해**를 준다(조각별 선형이라
    닫힌 형태, 이분법 없음). 그 Z′ 로 각도를 조회한다.

    용도는 **속도가 아니라 검증**이다 — 근사판이 그 모델·프로필에서 실제로
    무해한지 대조한다. 둘이 어긋나면 근사가 아니라 근사를 쓴 결론을 의심할 것.
    (`tests/test_profile_axis_approx.py` 가 표본에서 일치를 고정한다.)
    """
    cent = mesh.vertices[mesh.faces].mean(axis=1)
    fr = np.hypot(cent[:, 0], cent[:, 1])
    zw = profile.solve_forward(cent[:, 2], fr, direction)
    th = np.radians(profile.theta_at(zw))
    nz = mesh.face_normals[:, 2]
    nr = radial_normal(mesh)
    g = nz * np.cos(th) + nr * np.sin(th)
    need = g < -np.sin(np.radians(threshold_deg))
    areas = mesh.area_faces
    return areas[need].sum() / areas.sum() * 100.0


def sweep_table(mesh, angles, direction, threshold_deg=THRESHOLD_DEG):
    """각도 리스트를 훑어 각 각도의 남은 서포트(%) 리스트 (sweep.sweep_table 의
    해석식 대체 — 판정 기준 통일)."""
    return [support_fraction(mesh, float(a), direction, threshold_deg)
            for a in angles]


def face_support_and_staircase(mesh, angle_deg, direction,
                               threshold_deg=THRESHOLD_DEG):
    """해석식 버전 (metrics.face_support_and_staircase 의 대체):
      needs_support = g < κ,  staircase = max(0, −g).

    g = n_z·cosα + d·n_r·sinα 는 변환공간 n_z' 의 실공간 대응물이므로
    α=0 에서 기존 metrics 정의와 정확히 일치하고, α>0 에서는 왜곡 없는
    물리 기준이 된다 (docs/warped_threshold_finding.md — 판정 기준 통일).
    """
    g = overhang_score(mesh, angle_deg, direction)
    kappa = -np.sin(np.radians(threshold_deg))
    return g < kappa, np.maximum(0.0, -g)


def critical_angle(mesh, direction, threshold_deg=THRESHOLD_DEG,
                   angle_max=MAX_ANGLE_DEG):
    """면마다 '자기지지가 되는 최소 원뿔각 θ*'를 닫힌형으로 계산.

    반환: theta_star[F]  (도)
      ·  0             : 이미(각도 0에서) 자기지지
      ·  0 < θ* ≤ max  : 이 각도부터 자기지지
      ·  np.nan        : 이 방향·각도 범위 안에서는 못 고침 (Rank1 'irreducible')

    유도: g(α)=n_z cosα + d·n_r sinα = R·sin(α+δ),  R=hypot(n_z, d·n_r), δ=atan2(n_z, d·n_r).
          g(α) ≥ κ 의 최소 α ≥ 0  →  α* = asin(κ/R) − δ.
    """
    nz = mesh.face_normals[:, 2]
    b = _dir_sign(direction) * radial_normal(mesh)      # = d·n_r
    R = np.hypot(nz, b)
    kappa = -np.sin(np.radians(threshold_deg))

    theta = np.full(len(nz), np.nan)
    # 1) 이미 자기지지 (각도 0)
    already = nz >= kappa
    theta[already] = 0.0

    # 2) 나머지: 각도를 키워 도달 가능한가? (b>0 이라야 g가 증가, 그리고 κ/R≥−1)
    rest = ~already
    with np.errstate(invalid="ignore", divide="ignore"):
        reachable = rest & (b > 1e-12) & (R > 1e-12) & (kappa / np.where(R > 0, R, 1) >= -1.0)
        ratio = np.clip(kappa / np.where(R > 0, R, 1.0), -1.0, 1.0)
        delta = np.arctan2(nz, b)
        alpha = np.degrees(np.arcsin(ratio) - delta)     # 최소 α (도)
    ok = reachable & (alpha >= -1e-9) & (alpha <= angle_max + 1e-9)
    theta[ok] = np.clip(alpha[ok], 0.0, angle_max)
    return theta
