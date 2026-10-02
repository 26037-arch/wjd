"""
meshio.py — STL 파일을 읽거나, 없으면 데모용 구(sphere)를 만든다.

세 CLI 스크립트가 똑같이 반복하던 "인자로 STL 주면 로드, 아니면 데모 구" 부분을
한 곳으로 모은 것. 구(sphere)는 아랫면이 곡면 오버행이라 원뿔 슬라이싱 효과를
보기에 좋은 기본 테스트 모델이다.
"""

import numpy as np
import trimesh


class RadiusProfile:
    """높이 구간별 최대 반경 r_max(z). 블렌드 폭·전이 높이 결정에 쓴다.

    왜 필요한가: 블렌드(각도가 변하는 구간)에서 층간격 배율은 1 − c·r·s 라
    '그 높이의 반경'에 비례해 커진다. 지금까지는 전 모델 최대 반경(r_max)만 써서
    ⓐ 가는 부분에서 필요 이상으로 넓은 블렌드를 잡고 ⓑ '어느 높이에 경계를 두면
    유리한가'를 아예 못 봤다. 높이별 반경을 알면 둘 다 해결된다.

    ⚠ 정직: 정점 기준 구간별 최댓값이다(면 내부는 보간하지 않음). 정점이 없는
      구간은 양 이웃의 큰 값으로 채워 보수적으로 잡는다.
    """

    def __init__(self, mesh, n_bins=120):
        v = np.asarray(mesh.vertices, dtype=float)
        r = np.hypot(v[:, 0], v[:, 1])
        z = v[:, 2]
        self.z0, self.z1 = float(z.min()), float(z.max())
        self.global_max = float(r.max())
        n = max(1, int(n_bins))
        self.n = n
        self.h = max((self.z1 - self.z0) / n, 1e-9)
        idx = np.clip(((z - self.z0) / self.h).astype(int), 0, n - 1)
        rmax = np.full(n, -1.0)
        np.maximum.at(rmax, idx, r)
        # 빈 칸(정점 없는 높이) 채우기: 양쪽 최근접 유효값 중 큰 쪽 (보수적)
        left = rmax.copy()
        for i in range(1, n):
            if left[i] < 0:
                left[i] = left[i - 1]
        right = rmax.copy()
        for i in range(n - 2, -1, -1):
            if right[i] < 0:
                right[i] = right[i + 1]
        filled = np.maximum(left, right)
        filled[filled < 0] = self.global_max
        self.rmax = filled

    def max_between(self, lo, hi):
        """높이 구간 [lo, hi] 에서의 최대 반경 (구간 밖은 가장 가까운 끝값)."""
        if hi < lo:
            lo, hi = hi, lo
        i0 = int(np.clip(np.floor((lo - self.z0) / self.h), 0, self.n - 1))
        i1 = int(np.clip(np.floor((hi - self.z0) / self.h), 0, self.n - 1))
        return float(self.rmax[i0:i1 + 1].max())

    def at(self, z):
        """높이 z 의 최대 반경."""
        return self.max_between(z, z)


def center_on_axis(mesh):
    """모델을 원뿔 회전축(Z축, x=y=0)에 맞춘다: XY를 바운딩박스 중심으로, 바닥을 z=0으로.

    ⚠ 꼭 필요한 이유: 원뿔 변환은 '원점 기준 반경 r=√(x²+y²)'로 기울인다
    (z' = z + c·r·tanθ). 모델이 축에서 멀리 떨어져 있으면 모델 전체의 r이 거의
    같아서, 변환이 '모든 점을 똑같이 올리는 균일 이동'에 가까워진다 → 오버행이 거의
    안 변하고 서포트가 상수처럼 나온다. 데모 구는 원점 중심이라 이 문제를 안 만나지만,
    실제 STL은 보통 축에서 벗어난 좌표에 있으므로 분석 전에 반드시 센터링해야 한다.
    (원뿔 슬라이싱은 회전축 중심으로 출력하므로 축에 맞추는 것이 물리적으로도 맞다.)
    """
    lo, hi = mesh.bounds
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    mesh.apply_translation([-cx, -cy, -lo[2]])
    return mesh


def waist_prominence(mesh, n_samples=80, radius_profile=None):
    """'허리'의 두드러짐 (0 = 없음, 1 에 가까울수록 깊은 목).

    ⚠⚠ **이것은 '밴드가 이기는 조건' 의 판정 기준이 아니다** (2026-09-21 반증).
      이 값은 반경의 **비율**이라 반경 스케일에 눈이 먼다. 허리 깊이를 그대로 둔 채
      모델을 XY 로만 λ 배 하면 값은 0.571 로 고정인데 블렌드 폭은 `r_b ∝ λ` 로
      비싸지고, 실측하면 **같은 0.571 에서 승패가 갈린다**(승 3건·패 3건).
      판정 기준은 `analyze_blend_ratio.py` 의 **ρ = 블렌드비용/잠재이득** 을 쓸 것.
      이 함수는 '형상을 재는' 용도로는 그대로 유효하다 — 기준으로 쓰는 것이 틀렸다.

    왜 있었나: 각도 변경에 필요한 블렌드 폭이 `r_b·|Δtanθ|/(limit−1)` 로 '그 높이의
    반경'에 비례하므로, 가는 높이에서만 각도를 싸게 바꿀 수 있다. 그 직관을 구·램프
    두 모델의 일화에서 표본으로 넓히려고 '이 모델에 허리가 있나'를 재현 가능한 수로
    만든 것이다. 직관은 맞았지만 **비율만으로는 비용의 절대 크기를 못 본다.**

    정의: 내부 높이 z* 에 대해 아래쪽 최대 반경과 위쪽 최대 반경 중 **작은 쪽**을
    기준으로 얼마나 잘록한지를 재고, 그 최댓값을 취한다.

        prominence = max over interior z*  of  1 − r(z*) / min(max r below, max r above)

    위쪽 최대를 함께 보는 것이 핵심이다. 그냥 min(r)/max(r) 로 재면 위로 갈수록
    가늘어지기만 하는 형상(구·원뿔·깔때기)의 **꼭대기 테이퍼**가 허리로 잡힌다 —
    거기는 위에 아무것도 없어서 각도를 바꿔봐야 얻을 게 없는데도 그렇다.
    위쪽 최대를 기준에 넣으면 그런 단조 형상은 자동으로 0 이 된다.

    ⚠ 정직: 축대칭을 가정한 반경 프로필(`RadiusProfile`, 정점 기준 구간 최댓값)
      위에서 잰다. 축비대칭 모델(L 브래킷, 아치)에서는 '그 높이의 최대 반경'이
      단면의 실제 모양을 대표하지 못하므로 이 수치도 대표성이 떨어진다.
      블렌드 폭 공식이 같은 근사를 쓰므로 **계획기와 같은 기준**이라는 점은 맞다.

    Returns: (prominence, z_waist, r_waist). 허리가 없으면 (0.0, nan, nan).
    """
    rp = radius_profile if radius_profile is not None else RadiusProfile(mesh)
    z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    h = z1 - z0
    if h <= 0 or n_samples < 3:
        return 0.0, float("nan"), float("nan")
    # 양 끝 2% 는 뺀다 — 바닥/꼭대기의 한 점짜리 반경은 허리가 아니다.
    zs = np.linspace(z0 + 0.02 * h, z1 - 0.02 * h, int(n_samples))
    rs = np.array([float(rp.at(z)) for z in zs])

    # 누적 최대(아래쪽)/역누적 최대(위쪽) 로 O(n) 에 끝낸다.
    max_below = np.maximum.accumulate(rs)
    max_above = np.maximum.accumulate(rs[::-1])[::-1]

    best = (0.0, float("nan"), float("nan"))
    for i in range(1, len(zs) - 1):
        ref = min(max_below[i - 1], max_above[i + 1])
        if ref <= 1e-9:
            continue
        p = 1.0 - rs[i] / ref
        if p > best[0]:
            best = (float(p), float(zs[i]), float(rs[i]))
    return best


def load_mesh_or_demo(argv, subdivisions=4, radius=10.0):
    """명령줄 인자에 STL 경로가 있으면 로드, 없으면 데모 구를 반환한다.

    로드/생성한 메시는 항상 회전축(Z)에 센터링해서 돌려준다(center_on_axis).

    반환: (mesh, path)
      - path: 실제 STL 경로(문자열) 또는 None(데모 구를 만든 경우)
    """
    if len(argv) > 1:
        mesh = trimesh.load(argv[1], force="mesh")
        return center_on_axis(mesh), argv[1]
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return center_on_axis(mesh), None
