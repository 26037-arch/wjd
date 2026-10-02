"""
profile.py — θ(Z′) 가변각 프로필. 부위별 최적 각도를 '실제 G-code'로 만드는 코어.

설계 (변환공간 높이 Z′의 함수):
    프로필 = 브레이크포인트 [(Z′_1, θ_1), ..., (Z′_n, θ_n)] (Z′ 오름차순).
    내부적으로 T(Z′) = tanθ(Z′) 를 브레이크포인트 사이 '선형 보간', 양 끝 밖은
    상수 외삽. θ가 아니라 tanθ를 보간하는 이유: 변환에 실제로 들어가는 양이
    tanθ이고, 아래 가역성 조건·블렌드 최소 폭이 정확한 닫힌 식으로 나온다.

    각도는 '부호 있는' 값으로 통일한다: θ>0 = outward, θ<0 = inward (c=+1 고정).
    tanθ가 0을 지나며 연속이므로 방향 전환 없는 밴드 조합을 하나의 c로 표현 가능.

역변환 (닫힌 형태 — 반복 불필요):
    G-code 점은 Zw(=Z′)를 이미 아므로  θ = theta_at(Zw),
    x = X·cosθ,  y = Y·cosθ,  z = Zw − c·√(X²+Y²)·sinθ.

정변환 (정점별 1차원 방정식의 '정확 해' — 이분법 금지):
    정점 (x,y,z), r=√(x²+y²) 고정일 때 Z′는  z = Z′ − c·r·T(Z′) 를 만족.
    T가 조각별 선형이므로 구간 [a,b]에서 T(Z′)=t_a+s·(Z′−a) 라면
        Z′ = ( z + c·r·(t_a − s·a) ) / ( 1 − c·r·s )
    가 그 구간의 정확 해다. 구간을 차례로 검사해 [a,b]에 드는 해를 채택한다.

가역성 조건과 블렌드 최소 폭 (이 유도가 우리 분석 기여):
    해가 유일하려면 F(Z′) = Z′ − c·r·T(Z′) 가 순증가해야 한다:
        1 − c·r·(dT/dZ′) > 0   ⟺   c·r·s < 1   (모든 구간, 모든 정점 r)
    각도가 θ₁→θ₂로 바뀌는 블렌드 구간(폭 w)에서 s = (tanθ₂−tanθ₁)/w 이므로,
    c·(tanθ₂−tanθ₁) > 0 (불안정 방향)일 때만 제약이 걸리고
        w_min = c · r_max · (tanθ₂ − tanθ₁)
    ⚠ 정직: r_max는 '전 모델 최대 반경'을 쓰는 보수적 근사다 — 블렌드가 걸리는
      높이 구간의 실제 최대 반경보다 클 수 있다(필요보다 넓은 블렌드 허용).

층간격 제약 (검사기가 실측으로 찾아낸 두 번째 제약 — 안정 방향에도 해당):
    연속 레이어는 변환공간에서 ΔZ′=층고 간격이지만, 실공간 수직 간격은 반경 r에서
        Δz = ΔZ′ · m(r),   m(r) = 1 − c·r·s        ← '층간격 배율'
    m>1 이면 레이어가 벌어지고(팽창) m<1 이면 겹친다(압축). 가역성 조건은 m>0 만
    막으므로, 각도가 '감소'하는 안정 방향(c·s<0)은 가역성엔 안 걸리면서 간격이
    (1+r|s|)배로 팽창한다 — 구의 적도(r≈9.8)에서 0.5mm 블렌드로 36°→0° 전환 시
    레이어가 수직 ~6mm씩 벌어져 툴패스 검사기가 미지지 25.5%를 실측했다
    (docs/verification.md, 재현 확인 완료).

    배율을 [1/limit, limit] 안에 유지하는 최소 블렌드 폭 (s = Δtan/w 대입):
        (i)  팽창 (c·Δtan<0):  1 + r·|Δtan|/w ≤ limit    → w ≥ r·|Δtan|/(limit−1)
        (ii) 압축 (c·Δtan>0):  1 − r·|Δtan|/w ≥ 1/limit  → w ≥ r·|Δtan|/(1−1/limit)
    (ii)는 가역성 w_min = c·r·Δtan 보다 항상 강하다(limit>1 이면 1−1/limit<1).
    즉 층간격 제약이 가역성 제약을 '포함'한다.

    ⚠ 이 제약은 '이론적으로 그럴 것'이 아니라 툴패스 검사기가 먼저 실측으로
      잡아낸 결함이다(예측은 0.5%인데 검사기는 25.5%). 제약을 넣은 뒤 같은
      검사기로 재측정해 4.93%로 내려간 것까지 확인했다 — 예측기와 검사기가
      서로를 검사하는 구조가 실제로 작동한 사례.

    여기서 r 은 전 모델 최대 반경이 아니라 '블렌드가 걸리는 높이의 최대 반경'
    r_b 를 쓴다(meshio.RadiusProfile). 그래야 ⓐ 가는 부분에서 과하게 넓은 블렌드를
    피하고 ⓑ 경계를 반경이 작은 높이로 옮기는 것이 이득이 된다(전이 위치 최적화).
    폭을 아무리 넓혀도 자리가 없으면 각도 차 자체를 줄인다(각도 예산 삭감) —
    깨진 프로필을 내보내는 대신 '출력 가능한 최선'으로 내려앉고, 무엇을 깎았는지
    profile.notes 에 남겨 CLI 가 보고한다.
"""

import math

import numpy as np

from .config import MAX_SPACING_FACTOR


def blend_width_for_spacing(r_blend, dtan, direction="outward",
                            limit=MAX_SPACING_FACTOR):
    """층간격 배율을 [1/limit, limit] 안에 유지하는 최소 블렌드 폭 (위 유도식).

    r_blend: 블렌드가 걸리는 높이의 최대 반경, dtan: tanθ₂ − tanθ₁ (부호 있음).
    """
    c = 1.0 if direction == "outward" else -1.0
    if limit <= 1.0:
        raise ValueError(f"limit 는 1보다 커야 함 (받은 값 {limit})")
    a = abs(r_blend * dtan)
    if a < 1e-12:
        return 0.0
    if c * dtan > 0:                      # 압축 분기 (가역성 제약 포함)
        return a / (1.0 - 1.0 / limit)
    return a / (limit - 1.0)              # 팽창 분기


def max_dtan_for_width(r_blend, w, sign, direction="outward",
                       limit=MAX_SPACING_FACTOR):
    """폭 w 에 넣을 수 있는 최대 |Δtan| (위 식의 역). sign = Δtan 의 부호."""
    c = 1.0 if direction == "outward" else -1.0
    if r_blend < 1e-12 or w <= 0:
        return float("inf")
    factor = (1.0 - 1.0 / limit) if c * sign > 0 else (limit - 1.0)
    return w * factor / r_blend


def _required_width(zc, dtan, r_global, radius_profile, safety, min_blend,
                    spacing_limit):
    """중심 zc 에 놓을 블렌드의 '진짜 필요 폭'과 그때의 블렌드 반경 r_b.

    r_b 는 블렌드가 덮는 높이 구간의 최대 반경인데, 그 구간이 '폭에 따라' 달라진다
    (넓힐수록 더 뚱뚱한 높이까지 덮을 수 있다). 그래서 (폭 → 구간 → 반경 → 폭)을
    수렴할 때까지 반복한다. w 는 단조증가하고 반경이 r_global 에서 포화하므로 수렴한다.
    ⚠ 2회만 돌리면 구처럼 위로 갈수록 뚱뚱해지는 모델에서 과소평가된다(실측으로 발견).

    ⚠ 자리(가용 폭)로 잘라서 반환하면 안 된다(실측으로 발견): 자리가 거의 없는 중심이
      '폭이 작아서 싸다'고 잘못 평가되고, 게다가 w == avail 이 되어 호출부의
      '자리 부족 → 각도 삭감' 검사가 안 걸린다. 램프 N=4 에서 0.14mm 폭에 42°→0° 가
      들어가 층간격 배율 53.7배짜리 프로필이 나왔다. 자를지 말지는 호출부가 정한다.
    """
    w_inv = r_global * dtan                        # c=+1 규약, 불안정 방향만 >0
    w = max(min_blend, safety * w_inv if w_inv > 0 else min_blend)
    r_b = r_global
    if spacing_limit is None:
        return w, r_b
    for _ in range(40):
        r_b = (radius_profile.max_between(zc - w / 2.0, zc + w / 2.0)
               if radius_profile is not None else r_global)
        w_new = max(w, blend_width_for_spacing(r_b, dtan, "outward", spacing_limit))
        if w_new <= w * (1 + 1e-4):
            return w_new, r_b
        w = w_new
    return w, r_b


def _plan_blend(zb, th1, th2, room_lo, room_hi, r_global, radius_profile,
                safety, min_blend, spacing_limit, max_shift):
    """경계 zb 의 블렌드 계획 → (lo, hi, 실제 채택 th2, 메모 또는 None).

    ① 필요 폭을 구하고 ② (max_shift>0) 반경이 작은 높이로 경계를 옮겨보고
    ③ 그래도 자리가 모자라면 각도 차를 깎는다.
    """
    tan1 = math.tan(math.radians(th1))
    tan2 = math.tan(math.radians(th2))
    dtan = tan2 - tan1
    margin = 1e-3

    def available(zc):
        return 2.0 * min(zc - room_lo - margin, room_hi - margin - zc)

    # ① 후보 중심들 (원위치 우선, max_shift 안에서 반경이 작은 높이 탐색)
    centers = [zb]
    if max_shift > 0 and radius_profile is not None:
        # 이동 범위가 넓어져도 해상도(~0.5mm)를 유지한다 — 고정 21점이면 넓은 범위에서
        # 표본 간격이 벌어져 좁은 '허리'를 통째로 건너뛴다.
        n_c = int(np.clip(round(2 * max_shift / 0.5) + 1, 21, 81))
        centers += list(np.linspace(zb - max_shift, zb + max_shift, n_c))
    cands = []
    for zc in centers:
        if available(zc) <= 0:
            continue
        w, r_b = _required_width(zc, dtan, r_global, radius_profile,
                                 safety, min_blend, spacing_limit)
        cands.append((zc, w, r_b, available(zc)))
    if not cands:                                   # 자리가 아예 없음 → 원위치 강행
        w, r_b = _required_width(zb, dtan, r_global, radius_profile,
                                 safety, min_blend, spacing_limit)
        cands = [(zb, w, r_b, max(available(zb), min_blend))]

    # ② 후보 선택: 비용 = 블렌드의 겉넓이 proxy  w · r_b.
    #    평가함수 J 가 물리는 블렌드 비용은 '블렌드 구간의 표면적 × risk' 인데,
    #    회전체에서 높이 w·반경 r_b 구간의 옆넓이가 ∝ r_b·w 다. 즉 계획기가
    #    최소화하는 양을 J 가 재는 양과 맞춘 것이다.
    #    ⚠ 예전에는 (w + |경계 이동거리|) 였다(실측으로 교체): 폭만 보면 '같은 폭이라도
    #      뚱뚱한 높이에서는 표면적이 훨씬 크다'를 놓치고, 이동거리 항은 J 에 대응물이
    #      아예 없는 임의 항이었다. 그 탓에 램프 N=3 에서 계획기가 목(허리) 대신
    #      제자리를 지켜 J 가 N 에 대해 단조가 아니었다.
    #    이동거리는 동점 처리에만 쓴다 — 이득이 같으면 안 옮긴다.
    ok = [c for c in cands if c[1] <= c[3] + 1e-9]
    pool = ok if ok else cands
    best_area = min(c[1] * max(c[2], 1e-9) for c in pool)
    near = [c for c in pool if c[1] * max(c[2], 1e-9) <= best_area * 1.01 + 1e-12]
    zc, w, r_b, avail = min(near, key=lambda c: abs(c[0] - zb))

    note = None
    if abs(zc - zb) > 1e-6:
        note = (f"경계 {zb:.2f} → {zc:.2f} mm 로 이동 "
                f"(그 높이 반경 {r_b:.1f}mm, 블렌드 폭 {w:.2f}mm)")

    # ③ 폭이 모자라면 각도 차를 깎는다 (깨진 프로필 대신 출력 가능한 최선)
    th2_used = th2
    if w > avail + 1e-9 and spacing_limit is not None and abs(dtan) > 1e-12:
        w = max(avail, min_blend)
        dtan_max = max_dtan_for_width(r_b, w, math.copysign(1.0, dtan),
                                      "outward", spacing_limit)
        if abs(dtan) > dtan_max:
            dtan_new = math.copysign(dtan_max, dtan)
            th2_used = math.degrees(math.atan(tan1 + dtan_new))
            note = ((note + "; ") if note else "") + (
                f"각도 예산 삭감: θ {th1:.1f}°→{th2:.1f}° 를 {th2_used:.1f}° 로 "
                f"(높이 {zc:.2f}의 반경 {r_b:.1f}mm 에서 폭 {w:.2f}mm 로는 "
                f"층간격 {spacing_limit:.1f}배를 못 지킴)")
    return zc - w / 2.0, zc + w / 2.0, th2_used, note


class AngleProfile:
    """θ(Z′) 프로필. breakpoints = [(Zw, theta_deg_signed), ...] (Zw 오름차순)."""

    def __init__(self, breakpoints):
        if not breakpoints:
            raise ValueError("빈 프로필")
        zs = np.array([float(b[0]) for b in breakpoints])
        th = np.array([float(b[1]) for b in breakpoints])
        if len(zs) > 1 and not np.all(np.diff(zs) > 1e-12):
            raise ValueError(f"브레이크포인트 Z′는 순증가여야 함: {zs.tolist()}")
        self.zs = zs
        self.thetas_deg = th
        self.tans = np.tan(np.radians(th))
        self.notes = []        # from_bands 가 각도/경계를 조정했을 때의 설명

    # ── 조회 ──
    def tan_at(self, zw):
        return np.interp(zw, self.zs, self.tans)      # 양 끝 상수 외삽

    def theta_at(self, zw):
        """도(deg, 부호 있음). 스칼라/배열 모두."""
        return np.degrees(np.arctan(self.tan_at(zw)))

    def is_constant(self):
        return len(self.zs) == 1 or np.allclose(self.tans, self.tans[0])

    def blend_intervals(self):
        """기울기 s≠0 인 내부 구간 [(a,b), ...] — 각도가 변하는 블렌드 구간."""
        out = []
        for i in range(len(self.zs) - 1):
            if abs(self.tans[i + 1] - self.tans[i]) > 1e-12:
                out.append((float(self.zs[i]), float(self.zs[i + 1])))
        return out

    # ── 검증 ──
    def validate(self, r_max, direction="outward"):
        """가역성 검사: 모든 구간에서 c·r_max·s < 1. 위반 시 w_min 포함 에러."""
        c = 1.0 if direction == "outward" else -1.0
        for i in range(len(self.zs) - 1):
            a, b = self.zs[i], self.zs[i + 1]
            dt = self.tans[i + 1] - self.tans[i]
            s = dt / (b - a)
            if c * r_max * s >= 1.0 - 1e-12:
                w_min = c * r_max * dt
                raise ValueError(
                    f"프로필 가역성 위반: 구간 [{a:.2f},{b:.2f}] 에서 "
                    f"c·r_max·s = {c * r_max * s:.3f} ≥ 1. "
                    f"θ {self.thetas_deg[i]:.1f}°→{self.thetas_deg[i+1]:.1f}° 전환에는 "
                    f"최소 폭 w_min = c·r_max·(tanθ₂−tanθ₁) = {w_min:.2f} mm 가 필요 "
                    f"(현재 폭 {b - a:.2f} mm). 블렌드를 넓히거나 각도 차를 줄일 것.")
        return True

    # ── 층간격 배율 (두 번째 제약) ──
    def spacing_factor(self, zw, r, direction="outward"):
        """변환높이 Zw·반경 r 에서의 층간격 배율 m = 1 − c·r·s."""
        c = 1.0 if direction == "outward" else -1.0
        zw = np.asarray(zw, dtype=float)
        s = np.zeros(zw.shape) if zw.ndim else 0.0
        for i in range(len(self.zs) - 1):
            si = (self.tans[i + 1] - self.tans[i]) / (self.zs[i + 1] - self.zs[i])
            inside = (zw >= self.zs[i]) & (zw <= self.zs[i + 1])
            s = np.where(inside, si, s)
        return 1.0 - c * np.asarray(r, dtype=float) * s

    def check_spacing(self, r_max, direction="outward",
                      limit=MAX_SPACING_FACTOR, radius_profile=None):
        """층간격 배율이 [1/limit, limit] 를 벗어나는 구간 목록 (에러 안 냄).

        radius_profile 을 주면 구간별 실제 반경으로, 없으면 전 모델 r_max 로 본다.
        반환: [{"lo","hi","r","factor","theta1","theta2"}, ...] (배율 큰 순)
        """
        c = 1.0 if direction == "outward" else -1.0
        bad = []
        for i in range(len(self.zs) - 1):
            a, b = float(self.zs[i]), float(self.zs[i + 1])
            dt = self.tans[i + 1] - self.tans[i]
            if abs(dt) < 1e-12:
                continue
            s = dt / (b - a)
            r = radius_profile.max_between(a, b) if radius_profile else r_max
            m = 1.0 - c * r * s
            if m > limit + 1e-9 or m < 1.0 / limit - 1e-9:
                bad.append({"lo": a, "hi": b, "r": float(r), "factor": float(m),
                            "theta1": float(self.thetas_deg[i]),
                            "theta2": float(self.thetas_deg[i + 1])})
        bad.sort(key=lambda d: -abs(math.log(max(d["factor"], 1e-9))))
        return bad

    def max_spacing_factor(self, r_max, direction="outward", radius_profile=None):
        """모든 블렌드 구간에서의 최대 층간격 배율 (1.0 = 평면 슬라이싱과 동일)."""
        c = 1.0 if direction == "outward" else -1.0
        worst = 1.0
        for i in range(len(self.zs) - 1):
            a, b = float(self.zs[i]), float(self.zs[i + 1])
            dt = self.tans[i + 1] - self.tans[i]
            if abs(dt) < 1e-12:
                continue
            s = dt / (b - a)
            r = radius_profile.max_between(a, b) if radius_profile else r_max
            m = 1.0 - c * r * s
            if abs(math.log(max(m, 1e-9))) > abs(math.log(max(worst, 1e-9))):
                worst = m
        return float(worst)

    # ── 정변환 (정확 해, 벡터화) ──
    def solve_forward(self, z, r, direction="outward"):
        """z = Z′ − c·r·T(Z′) 의 정확 해 Z′ (배열). validate 통과 프로필 전제."""
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        c = 1.0 if direction == "outward" else -1.0
        zs, tans = self.zs, self.tans
        out = np.full(z.shape, np.nan)

        # 왼쪽 바깥 (T = tans[0] 상수)
        cand = z + c * r * tans[0]
        m = np.isnan(out) & (cand <= zs[0] + 1e-9)
        out[m] = cand[m]
        # 내부 구간들
        for i in range(len(zs) - 1):
            a, b = zs[i], zs[i + 1]
            s = (tans[i + 1] - tans[i]) / (b - a)
            denom = 1.0 - c * r * s
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = (z + c * r * (tans[i] - s * a)) / denom
            m = np.isnan(out) & (denom > 1e-12) & \
                (cand >= a - 1e-9) & (cand <= b + 1e-9)
            out[m] = cand[m]
        # 오른쪽 바깥 (T = tans[-1] 상수)
        cand = z + c * r * tans[-1]
        m = np.isnan(out) & (cand >= zs[-1] - 1e-9)
        out[m] = cand[m]

        if np.isnan(out).any():
            raise RuntimeError(
                f"정변환 해를 못 찾은 정점 {int(np.isnan(out).sum())}개 — "
                "프로필 validate() 를 먼저 통과시킬 것")
        return out

    # ── 생성기 ──
    @classmethod
    def constant(cls, theta_deg):
        return cls([(0.0, float(theta_deg))])

    @classmethod
    def from_bands(cls, bands, r_max, safety=1.5, min_blend=0.5,
                   radius_profile=None, spacing_limit=None, max_shift=0.0):
        """밴드 리스트 → 블렌드 자동 삽입 프로필.

        bands: [(z_lo, z_hi, theta_deg_signed), ...]  실공간 z 구간(오름차순).
        ⚠ 경계 해석(정직): 실공간 z 경계를 '축상(r=0, 그곳에서 Z′=z)' 기준 Z′ 값으로
          근사한다 — 축에서 멀수록 경계가 최대 r·tanθ 만큼 어긋날 수 있다.
          이 오차가 실제로 문제인지는 툴패스 검사기(P2)가 판정한다.

        블렌드 폭 (기본):  w = max(min_blend, safety × w_min),  w_min = r_max·Δtan.
        spacing_limit 을 주면 층간격 제약(모듈 docstring)이 추가로 걸린다:
          · 폭은 blend_width_for_spacing 으로 넓힌다 (반경은 radius_profile 이
            있으면 블렌드 높이의 실제 최대 반경, 없으면 전 모델 r_max).
          · max_shift>0 이면 경계를 ±max_shift 안에서 옮겨 '반경이 작은 높이'를
            찾는다(필요 폭이 줄어드는 쪽). 필요 없으면 원위치를 유지한다.
          · 그래도 자리가 없으면 각도 차를 깎아 넣는다 — 무엇을 깎았는지는
            반환 프로필의 .notes 에 기록된다.
        spacing_limit=None(기본) 이면 예전 동작 그대로다.
        """
        if not bands:
            raise ValueError("빈 밴드 리스트")
        notes = []
        bps = [(float(bands[0][0]), float(bands[0][2]))]
        th_prev = float(bands[0][2])                   # 앞 밴드의 '실제 채택' 각도
        for i in range(len(bands) - 1):
            th1 = th_prev
            th2 = float(bands[i + 1][2])
            zb = float(bands[i][1])                    # 경계 (축상 Z′≈z 근사)
            # 이 블렌드가 쓸 수 있는 높이 범위 (다음 블렌드 자리를 남겨둔다)
            room_lo = bps[-1][0]
            nxt = float(bands[i + 1][1])
            room_hi = nxt if i == len(bands) - 2 else 0.5 * (zb + nxt)
            lo, hi, th2_used, note = _plan_blend(
                zb, th1, th2, room_lo, room_hi, r_max, radius_profile,
                safety, min_blend, spacing_limit, max_shift)
            if lo <= bps[-1][0] + 1e-9:
                raise ValueError(
                    f"블렌드 구간이 겹침: 경계 {zb:.2f} 의 블렌드 폭 {hi - lo:.2f}mm 가 "
                    f"이전 브레이크포인트 {bps[-1][0]:.2f} 와 충돌. 밴드를 넓힐 것.")
            if note:
                notes.append(note)
            bps.append((lo, th1))
            bps.append((hi, th2_used))
            th_prev = th2_used
        bps.append((float(bands[-1][1]), th_prev))
        # 같은 각도 연속 등 중복 제거는 하지 않음(무해) — 단조성만 보장됨
        prof = cls(bps)
        prof.validate(r_max, "outward")
        prof.notes = notes
        return prof

    @classmethod
    def from_banded_result(cls, banded_result, r_max, safety=1.5, min_blend=0.5,
                           radius_profile=None, spacing_limit=None, max_shift=0.0):
        """varangle.select_banded 결과(실공간 z 밴드 + 각도/방향) → 프로필 어댑터.

        ⚠ 경계 해석(정직): select_banded 의 z 경계를 축상(r=0, Z′=z) 기준 Z′ 로
          근사 — 축에서 멀수록 경계가 최대 r·tanθ 만큼 어긋날 수 있다. 이 오차의
          실질 영향은 툴패스 검사기(toolpath_check.py)가 판정한다.
        방향은 부호로 접음: inward 밴드 = 음수 각도 (c=+1 규약).
        빈 밴드(None)는 이전 밴드 각도를 이어받아 불필요한 블렌드를 피한다.
        """
        edges = banded_result["edges"]
        prof_list = banded_result["profile"]
        bands = []
        prev_theta = 0.0
        for i, p in enumerate(prof_list):
            if p is None:
                theta = prev_theta
            else:
                ang, direction = p
                theta = float(ang) if direction == "outward" else -float(ang)
            bands.append((float(edges[i]), float(edges[i + 1]), theta))
            prev_theta = theta
        # 같은 각도 연속 밴드 병합 (블렌드 최소화)
        merged = [list(bands[0])]
        for lo, hi, th in bands[1:]:
            if abs(th - merged[-1][2]) < 1e-12:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi, th])
        return cls.from_bands([tuple(b) for b in merged], r_max, safety, min_blend,
                              radius_profile, spacing_limit, max_shift)

    @classmethod
    def parse(cls, text):
        """CLI 문자열 "Z1:deg1,Z2:deg2,..." → 프로필 (deg 는 부호 허용)."""
        bps = []
        for part in text.split(","):
            zs, ds = part.split(":")
            bps.append((float(zs), float(ds)))
        return cls(bps)

    def describe(self):
        """사람용 표: 구간·각도·블렌드 표시."""
        lines = []
        for _i, (z, th) in enumerate(zip(self.zs, self.thetas_deg, strict=True)):
            lines.append(f"    Z'={z:8.2f}  θ={th:6.1f}°")
        for a, b in self.blend_intervals():
            lines.append(f"    (블렌드 [{a:.2f}, {b:.2f}] 폭 {b - a:.2f} mm)")
        return "\n".join(lines)
