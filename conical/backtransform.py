"""
backtransform.py — 변환공간 G-code → 실공간 G-code 역변환 (RotBot 3단계의 마지막).

수학 (transform.py 의 역):
    정변환:  X = x/cosθ,  Y = y/cosθ,  Z' = z + c·r·tanθ   (r = √(x²+y²))
    역변환:  x = X·cosθ,  y = Y·cosθ,  z = Z' − c·√(X²+Y²)·sinθ
    (유도: r_real = cosθ·√(X²+Y²) 이므로 c·r_real·tanθ = c·√(X²+Y²)·sinθ)

재분할 (docs/gcode_size_notes.md 의 발견을 그대로 적용):
    변환공간의 직선은 실공간에서 곡선이 된다. 고정 간격으로 잘게 쪼개면 G-code가
    폭발하므로, 실공간 반경 r에서 허용 현 길이 L = 2·√(2·r·ε) 로 '적응' 분할한다.
    (ε = 허용 현 오차. 축에서 멀수록 긴 직선 허용 → 줄 수 수십 배 절약)

압출량 보정:
    슬라이서는 '변환공간 길이' 기준으로 E를 계산했지만 실제로 압출되는 경로는
    실공간 길이다. 각 이동의 ΔE 에 (실길이/변환길이)를 곱해 보정한다.
    ⚠ 정직한 한계: 변환이 비등방(XY만 1/cosθ)이라 방향에 따라 선폭이 약간
    달라지는 것까지는 보정하지 않는다(RotBot 동일). 각도가 클수록 오차 커짐.
"""

import math

from .gcode import Move
from .profile import AngleProfile


def _inv_point(X, Y, Zw, profile, c):
    """역변환 (가변각 일반화): θ = θ(Zw) 를 점별로 조회 — 닫힌 식, 반복 불필요."""
    th = math.radians(float(profile.theta_at(Zw)))
    ct = math.cos(th)
    x = X * ct
    y = Y * ct
    z = Zw - c * math.hypot(X, Y) * math.sin(th)
    return x, y, z


def _crosses_blend(blends, z0, z1):
    lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
    return any(a < hi and b > lo for a, b in blends)


def backtransform(items, cone_angle_deg, cone_type, chord_tol=0.05, max_sub=400):
    """파싱된 G-code 아이템들을 역변환한다. raw 줄은 그대로 통과.

    cone_angle_deg 자리에 float(고정각) 또는 AngleProfile(가변각 θ(Z′)) 허용 —
    float 는 상수 프로필로 감싸므로 기존 호출부는 무수정 호환.
    ⚠ 블렌드 구간(각도가 변하는 Z′ 구간)을 지나는 이동은 허용 현 길이를 절반으로
      (분할 2배): θ 변화가 z 곡률을 추가하므로 보수적으로 잡는다.

    반환: (새 아이템 리스트, 통계 dict)
    """
    profile = cone_angle_deg if isinstance(cone_angle_deg, AngleProfile) \
        else AngleProfile.constant(cone_angle_deg)
    c = 1.0 if cone_type == "outward" else -1.0
    blends = profile.blend_intervals()

    out = []
    X = Y = Zw = None            # 변환공간 현재 위치 (모달)
    e_prev = 0.0
    n_in = n_out = 0

    for kind, payload in items:
        if kind != "move":
            out.append((kind, payload))
            continue
        mv = payload
        nX = mv.x if mv.x is not None else X
        nY = mv.y if mv.y is not None else Y
        nZ = mv.z if mv.z is not None else Zw

        if nX is None or nY is None or nZ is None or X is None:
            # 아직 위치가 정해지기 전(첫 이동 등): 점만 역변환해 그대로 내보냄
            if nX is not None and nY is not None and nZ is not None:
                x, y, z = _inv_point(nX, nY, nZ, profile, c)
                out.append(("move", Move(g=mv.g, x=x, y=y, z=z, e=mv.e, f=mv.f)))
                if mv.e is not None:
                    e_prev = mv.e
            else:
                out.append(("move", mv))
            X, Y, Zw = nX, nY, nZ
            n_in += 1
            n_out += 1
            continue

        n_in += 1
        dE = (mv.e - e_prev) if mv.e is not None else None
        warp_len = math.dist((X, Y, Zw), (nX, nY, nZ))

        # 순수 E 이동(리트랙트/프라임: XYZ 변화 없음)은 분할·스케일 대상이 아니다.
        # 기존에는 warp_len≈0 → scale=0 으로 ΔE가 사라져 리트랙션이 전부 무력화됐다
        # (내장 슬라이서는 리트랙션이 없어 무해했지만 외부 슬라이서 G-code에선 치명적).
        if dE is not None and warp_len < 1e-9:
            e_prev = e_prev + dE
            out.append(("move", Move(g=mv.g, e=e_prev, f=mv.f, extra=mv.extra)))
            n_out += 1
            continue

        # 시작/끝 실공간 좌표와 허용 현 길이로 분할 수 결정
        x0, y0, z0 = _inv_point(X, Y, Zw, profile, c)
        x1, y1, z1 = _inv_point(nX, nY, nZ, profile, c)
        real_len = math.dist((x0, y0, z0), (x1, y1, z1))
        r_mid = 0.5 * (math.hypot(x0, y0) + math.hypot(x1, y1))
        allowed = 2.0 * math.sqrt(max(2.0 * r_mid * chord_tol, 1e-12))
        if blends and _crosses_blend(blends, Zw, nZ):
            allowed *= 0.5     # ⚠ 블렌드 통과 이동: θ 변화가 z 곡률 추가 → 분할 2배
        n = max(1, min(max_sub, math.ceil(real_len / max(allowed, 1e-9))))
        # 실공간 누적 길이로 E 배분 (총 ΔE × 실길이/변환길이 보정 포함)
        pts = []
        total_real = 0.0
        px, py, pz = x0, y0, z0
        for i in range(1, n + 1):
            t = i / n
            xi, yi, zi = _inv_point(X + (nX - X) * t, Y + (nY - Y) * t,
                                    Zw + (nZ - Zw) * t, profile, c)
            seg = math.dist((px, py, pz), (xi, yi, zi))
            total_real += seg
            pts.append((xi, yi, zi, seg))
            px, py, pz = xi, yi, zi

        scale = (total_real / warp_len) if (dE is not None and warp_len > 1e-12) else 0.0
        acc = e_prev
        for xi, yi, zi, seg in pts:
            e_val = None
            if dE is not None:
                acc += dE * (seg / total_real) * scale if total_real > 1e-12 else 0.0
                e_val = acc
            out.append(("move", Move(g=mv.g, x=xi, y=yi, z=zi, e=e_val,
                                     f=mv.f, extra=mv.extra)))
            n_out += 1
        if dE is not None:
            e_prev = acc

        X, Y, Zw = nX, nY, nZ

    stats = {"moves_in": n_in, "moves_out": n_out,
             "expansion": n_out / max(n_in, 1)}
    return out, stats
