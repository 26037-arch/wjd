"""
open5x.py — 원뿔 G-code → Open5x(베드 틸트+회전) 5축 기계좌표 변환. [실험적]

조사로 확인된 사실 (docs 참고, 출처: FreddieHong19/Open5x 저장소·CHI'22 논문):
  · Prusa i3 판: 축 U=베드 틸트(Y평행선 중심), V=베드 회전. Duet RRF3.
  · 펌웨어는 회전 기구학을 하지 않는다(M669 K0 항등) → **슬라이서가 기계좌표를
    직접 계산해야 한다** (논문의 Grasshopper도 역기구학을 슬라이서에서 수행).
  · XY 원점 = 회전 베드 중심 (M208 주석). 틸트축-베드면 거리(피벗 깊이)는
    스탠드오프(30/50/70mm)에 따라 다름 → 반드시 기계에서 보정할 파라미터.
  · 회전축은 모듈로 감지 말고 절대값을 계속 누적할 것(RRF 당시 랩 기능 없음).
  · Open5x에서 원뿔 모드 출력 사례는 문서화된 것이 없음 → 이 모듈이 첫 시도.
    부호(틸트 방향·회전 방향)는 문서에 없어 기계에서 실측 보정해야 한다.

실기 체크리스트 (2026-07 리뷰 보강):
  (a) V는 한 방향으로 계속 누적된다 — 구 데모 기준 약 149회전. 슬립링이 없는
      베드라면 배선 감김 대책(또는 트래블 중 되감기)이 반드시 필요하다.
  (b) 축 근처를 지나는 트래블에서 V가 한 이동에 최대 ±180° 급회전한다
      (방위각 반전). 트래블 중 되감기/레이어별 방향 교대는 향후 과제로 남김.

원뿔 모드 원리 (노즐 수직 고정, 베드가 움직임):
  outward 원뿔 레이어의 국소 법선은 n̂ = sinθ·r̂(φ) + cosθ·ẑ (φ=점의 방위각).
  베드를 틸트 U=θ 로 고정하고, 회전 V가 현재 압출점의 방위각을 따라가면
  압출점에서 레이어면이 수평이 되어 수직 노즐로 원뿔 레이어를 쌓을 수 있다.

수학 (부품좌표 p → 기계좌표):
  1) 베드 회전: 압출점이 항상 틸트 '올라간 쪽'(방위각 φ0=0, +X)에 오도록
       V = −φ(p)   (연속 누적, 랩 없음)
       p1 = Rz(V)·p
  2) 베드 틸트: Y축 평행선(베드 중심, 베드면 아래 pivot_depth d) 중심 회전
       p2 = Ry(s·θ)·(p1 + [0,0,d]) − [0,0,d]     (s=틸트 부호, 기계 보정)
  3) G1 X..Y..Z..U..V.. 로 출력 (U는 상수 θ)

피드레이트 (조사 확인: RRF에서 U/V는 '선형' 취급 → 도(deg)가 norm에 mm처럼
  들어감): 논문 방식대로 세그먼트별 재조정 F' = F × (기계경로길이/표면경로길이).
"""

import math

from .gcode import Move


class MachineProfile:
    """기계별 파라미터. 전부 실기 보정 대상."""
    def __init__(self, tilt_axis="U", rot_axis="V", tilt_sign=-1.0, rot_sign=1.0,
                 pivot_depth=50.0, max_tilt_deg=90.0, uv_linear_feed=True):
        # tilt_sign 기본 -1: 이 부호에서 outward 원뿔 레이어 법선이 기계좌표에서
        # 정확히 수직이 됨을 수치 검증함 (2000점 오차 0). 실기에서 축 방향이
        # 반대로 조립됐으면 +1로 바꿀 것.
        self.tilt_axis = tilt_axis
        self.rot_axis = rot_axis
        self.tilt_sign = tilt_sign          # 틸트 방향 부호 (실측 보정)
        self.rot_sign = rot_sign            # 회전 방향 부호 (실측 보정)
        self.pivot_depth = pivot_depth      # 베드면→틸트축 거리 mm (스탠드오프별)
        self.max_tilt_deg = max_tilt_deg    # Voron 펌웨어 한계 ±110°, 시연 90°
        self.uv_linear_feed = uv_linear_feed  # RRF: U/V가 F norm에 선형으로 포함


PRUSA_UV = MachineProfile("U", "V")          # Open5x Prusa i3 판
VORON_BC = MachineProfile("B", "C", uv_linear_feed=False)  # B/C판(회전은 norm 제외)


def _map_point(x, y, z, theta_deg, prof, v_prev):
    """부품좌표 → (기계 x,y,z, V각도[연속누적]). v_prev: 직전 V (unwrap 기준)."""
    r = math.hypot(x, y)
    if r < 1e-6:
        v = v_prev                          # 축 위: 방위각 정의 불가 → V 유지
    else:
        phi = math.degrees(math.atan2(y, x))
        v_raw = -phi * prof.rot_sign
        # 연속 누적(unwrap): 직전 값과 ±180° 이내가 되도록 360° 단위 이동
        k = round((v_prev - v_raw) / 360.0)
        v = v_raw + 360.0 * k
    # 1) 베드 회전 Rz(v_bed): 부품이 베드에 붙어 돌므로 점은 Rz(v_bed)·p
    a = math.radians(v * (1.0 if prof.rot_sign >= 0 else -1.0))
    ca, sa = math.cos(a), math.sin(a)
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    z1 = z
    # 2) 틸트 Ry(s·θ), 피벗 (0,0,-d)
    t = math.radians(prof.tilt_sign * theta_deg)
    ct, st = math.cos(t), math.sin(t)
    zd = z1 + prof.pivot_depth
    x2 = ct * x1 + st * zd
    z2 = -st * x1 + ct * zd - prof.pivot_depth
    return x2, y1, z2, v


def to_open5x(items, cone_angle_deg, cone_type, profile=PRUSA_UV):
    """실공간 원뿔 G-code 아이템 → Open5x 기계좌표 G-code 아이템.

    inward 원뿔은 틸트 부호가 반대(레이어가 안쪽으로 기움).
    반환: (아이템 리스트, 통계 dict)
    """
    if cone_angle_deg > profile.max_tilt_deg:
        raise ValueError(f"cone angle {cone_angle_deg} > machine tilt limit "
                         f"{profile.max_tilt_deg}")
    theta = cone_angle_deg * (1.0 if cone_type == "outward" else -1.0)

    out = [("raw", f"; Open5x conical mode [EXPERIMENTAL] "
                   f"tilt {profile.tilt_axis}={theta:.1f}deg fixed, "
                   f"{profile.rot_axis} tracks azimuth"),
           ("raw", f"; pivot_depth={profile.pivot_depth}mm  "
                   f"(calibrate on machine; signs too)"),
           ("raw", f"G1 {profile.tilt_axis}{theta:.3f} F600  ; tilt bed")]
    x = y = z = None
    v_prev = 0.0
    px = py = pz = None                     # 직전 부품좌표 (F 재조정용)
    mx = my = mz = None                     # 직전 기계좌표
    v_range = [0.0, 0.0]
    for kind, payload in items:
        if kind != "move":
            out.append((kind, payload))
            continue
        mv = payload
        x = mv.x if mv.x is not None else x
        y = mv.y if mv.y is not None else y
        z = mv.z if mv.z is not None else z
        if x is None or y is None or z is None:
            out.append((kind, payload))
            continue
        X, Y, Z, v = _map_point(x, y, z, abs(theta), profile, v_prev)
        # 논문 방식 피드 재조정: F' = F × 기계길이/부품길이
        f_out = mv.f
        if mv.f is not None and px is not None:
            part_len = math.dist((px, py, pz), (x, y, z))
            if profile.uv_linear_feed:
                mach_len = math.sqrt((X - mx) ** 2 + (Y - my) ** 2 +
                                     (Z - mz) ** 2 + (v - v_prev) ** 2)
            else:
                mach_len = math.dist((mx, my, mz), (X, Y, Z))
            if part_len > 1e-9:
                f_out = mv.f * max(mach_len / part_len, 1e-3)
        extra = f"{profile.rot_axis}{v:.3f}"
        out.append(("move", Move(g=mv.g, x=X, y=Y, z=Z, e=mv.e, f=f_out,
                                 extra=extra)))
        v_prev = v
        v_range[0] = min(v_range[0], v)
        v_range[1] = max(v_range[1], v)
        px, py, pz = x, y, z
        mx, my, mz = X, Y, Z

    stats = {"v_min": v_range[0], "v_max": v_range[1],
             "v_turns": (v_range[1] - v_range[0]) / 360.0}
    return out, stats


# ─────────────────────────────────────────────────────────────
# 되감기: 배선 감김 대책
# ─────────────────────────────────────────────────────────────
def _with_v(extra, rot_axis, v):
    """extra 문자열의 회전축 값을 v 로 바꾼다 (다른 워드는 보존)."""
    toks = [t for t in (extra or "").split()
            if not t.upper().startswith(rot_axis.upper())]
    toks.append(f"{rot_axis}{v:.3f}")
    return " ".join(toks)


def add_v_rewinds(items, profile=PRUSA_UV, max_turns=1.0, clearance=2.0,
                  rot_feed=1200.0, lift_feed=3000.0):
    """배선 감김을 막기 위해 **트래블 중에 V 를 360°의 배수만큼 되감는다.**

    왜 되는가: 베드를 정확히 360° 돌리면 부품은 제자리로 온다 — `Rz` 는 360°
    주기이므로 부품 점의 기계좌표가 **변하지 않는다.** 그래서 되감기 이동은
    X/Y/Z 를 건드리지 않고 V 만 바꾸면 된다.

    ⚠ 그런데 **도는 도중의 중간 각도**에서는 부품이 다른 방향을 향한다. 노즐이
      그 자리에 그대로 있으면 부딪힌다. 그래서 되감기 전에 **지금까지 쌓은 것보다
      위로** 들어올린다(`clearance` mm 여유). 올릴 높이는 '여태 압출한 기계 Z 의
      최댓값'이다 — 아직 안 찍은 위쪽은 부딪힐 것이 없다.

    언제 되감나: **압출하지 않는 이동(트래블)** 에서만, 그리고 시작 기준 이탈이
    `max_turns` 회전을 넘었을 때만. 매 트래블마다 돌리면 출력 시간이 늘어난다.

    보장되는 상한은 **`max_turns + 1` 회전**이다: 되감기는 트래블에서만 하므로
    한 번의 압출 구간(둘레 한 바퀴 ≈ 1회전) 동안은 더 자랄 수 있다.
    실측 트레이드오프 (funnel 20°, rot_feed=1200):

    | max_turns | 되감기 | 감김 | 추가 시간 |
    |---|---|---|---|
    | 0.5 | 148회 | 1.0회전 | 44분 |
    | **1.0** | 32회 | 2.0회전 | **15분** |
    | 2.0 | 16회 | 3.0회전 | 13분 |

    0.5 는 1.0 보다 시간이 3배 드는데 감김은 1회전밖에 못 줄인다. 기본값 1.0.
    ⚠ 진짜 한계는 **기계의 배선 여유**에서 와야 한다 — 서비스 루프를 재서 정할 것.
    `rot_feed` 를 올리면 추가 시간이 비례해서 준다 (축 속도 한계 확인 후).

    ⚠ 한계: 여기서 보장하는 것은 '들어올린 높이가 여태 퇴적물보다 높다' 까지다.
      노즐 몸체·히트블록이 옆으로 부딪히는 것은 보지 못한다(검사기 B 는 3축
      가정이라 5축에 못 쓴다). `clearance` 는 넉넉하게 줄 것.
    """
    out = []
    offset = 0.0                 # 이후 V 에 더할 값 (360 의 배수)
    v0 = None                    # 첫 V (감김의 기준점)
    v_prev = None
    mx = my = mz = None          # 직전 기계 좌표
    z_deposited = None           # 여태 '압출한' 기계 Z 의 최댓값
    e_prev = 0.0
    n_rewinds, rewind_deg = 0, 0.0

    for kind, payload in items:
        if kind != "move":
            out.append((kind, payload))
            continue
        mv = payload
        v_raw = None
        for tok in (mv.extra or "").split():
            if tok.upper().startswith(profile.rot_axis.upper()):
                try:
                    v_raw = float(tok[len(profile.rot_axis):])
                except ValueError:
                    pass
        extruding = mv.e is not None and mv.e > e_prev + 1e-9

        if v_raw is not None:
            v = v_raw + offset
            if v0 is None:
                v0 = v
            # 트래블이고, 감김이 한계를 넘었으면 여기서 푼다.
            if (not extruding and abs(v - v0) > max_turns * 360.0
                    and None not in (mx, my, mz) and v_prev is not None):
                k = round((v0 - v_prev) / 360.0)
                if k != 0:
                    safe_z = max(z_deposited if z_deposited is not None else mz,
                                 mz) + clearance
                    v_new = v_prev + 360.0 * k
                    out.append(("raw", f"; V_REWIND BEGIN {k:+d} turn(s): "
                                       f"{v_prev:.1f} -> {v_new:.1f} deg "
                                       f"(lift to Z{safe_z:.2f})"))
                    out.append(("move", Move(g=0, x=mx, y=my, z=safe_z,
                                             f=lift_feed,
                                             extra=_with_v(mv.extra,
                                                           profile.rot_axis,
                                                           v_prev))))
                    out.append(("move", Move(g=0, x=mx, y=my, z=safe_z,
                                             f=rot_feed,
                                             extra=_with_v(mv.extra,
                                                           profile.rot_axis,
                                                           v_new))))
                    out.append(("move", Move(g=0, x=mx, y=my, z=mz,
                                             f=lift_feed,
                                             extra=_with_v(mv.extra,
                                                           profile.rot_axis,
                                                           v_new))))
                    out.append(("raw", "; V_REWIND END"))
                    rewind_deg += abs(360.0 * k)
                    offset += 360.0 * k
                    v = v_raw + offset
                    v_prev = v_new
                    n_rewinds += 1
            mv = Move(g=mv.g, x=mv.x, y=mv.y, z=mv.z, e=mv.e, f=mv.f,
                      extra=_with_v(mv.extra, profile.rot_axis, v))
            v_prev = v

        out.append(("move", mv))
        if mv.x is not None:
            mx = mv.x
        if mv.y is not None:
            my = mv.y
        if mv.z is not None:
            mz = mv.z
        if extruding and mz is not None:
            z_deposited = mz if z_deposited is None else max(z_deposited, mz)
        if mv.e is not None:
            e_prev = mv.e

    return out, {"rewinds": n_rewinds, "rewind_deg": rewind_deg,
                 "rewind_minutes": rewind_deg / max(rot_feed, 1e-9)}


# ─────────────────────────────────────────────────────────────
# 검사기 C: 기계좌표 출력 검사 (실기 전 사전 점검)
# ─────────────────────────────────────────────────────────────
def check_open5x(items, profile=PRUSA_UV, bed_radius=None, max_dv_deg=30.0,
                 min_z=0.0, max_feed=None, max_turns=None):
    """Open5x 기계좌표 G-code 를 기계에 걸기 **전에** 검사한다.

    이 모듈 docstring 의 '실기 체크리스트' 를 자동으로 보는 것이다. 5축은 3축보다
    사고가 쉽게 나고, 사고가 나면 기계가 상한다. 프린터가 없는 동안에도 할 수 있는
    검사를 미리 걸어 둔다.

    보는 것:
      · 틸트 U 가 기계 한계 안인가, 한 번만 설정되는가
      · **배선 감김** — 시작 기준 최대 이탈 |V − V₀|. Σ|ΔV| 가 아니다 (왕복은
        감기지 않는다). 슬립링이 없으면 이 값이 배선 여유를 넘으면 안 된다
      · **V 급회전** — 축 근처 트래블에서 한 이동에 ±180° 가 뛴다 (방위각 반전)
      · 기계 Z 가 음수로 가는가 (틸트에서는 정상일 수 있으나 소프트리밋에 걸린다)
      · 베드 밖으로 나가는가 (bed_radius 를 주면)
      · 피드레이트가 기계 최대를 넘는가 (max_feed 를 주면)

    ⚠ 보지 **못하는** 것: 노즐-출력물 간섭(검사기 B 의 3축 가정은 5축에 안 맞는다),
      축 가속도 한계, 실제 기구 충돌. 이 검사를 통과해도 첫 출력은 사람이 지켜봐야
      한다.

    반환: (findings, stats). findings 는 (심각도, 메시지) 목록으로
    심각도는 "치명"(기계가 상할 수 있음) / "경고" / "정보".
    """
    findings, u_seen = [], []
    # X 와 Y 는 따로 모으면 안 된다. G-code 의 한 이동이 X 만 지정할 수 있어서
    # 길이가 어긋나고, 그러면 zip 이 엉뚱한 X-Y 를 짝지어 반경을 잰다.
    # 짝으로 모아서 애초에 어긋날 수 없게 한다.
    xys, zs, vs, feeds = [], [], [], []
    in_rewind, n_rewind, rewind_max = False, 0, 0.0
    dv_max, dv_at = 0.0, None
    v_prev, v_first = None, None
    v_travel = 0.0                       # Σ|ΔV| — 총 회전량 (마모·시간). 감김 아님
    wind_max = 0.0                       # max |V − V_start| — **배선 감김의 척도**

    for kind, payload in items:
        if kind != "move":
            # 되감기 구간은 **의도된** 큰 회전이다. 사고(축 근처 방위각 반전)와
            # 구별해야 하므로 G-code 에 표시를 박아두고 여기서 가른다.
            if isinstance(payload, str):
                up = payload.upper()
                if "V_REWIND BEGIN" in up:
                    in_rewind, n_rewind = True, n_rewind + 1
                elif "V_REWIND END" in up:
                    in_rewind = False
            continue
        mv = payload
        extra = (mv.extra or "").upper()
        for tok in extra.split():
            if tok.startswith(profile.tilt_axis):
                try:
                    u_seen.append(float(tok[len(profile.tilt_axis):]))
                except ValueError:
                    findings.append(("치명", f"틸트 축 값을 못 읽었다: {tok}"))
            elif tok.startswith(profile.rot_axis):
                try:
                    v = float(tok[len(profile.rot_axis):])
                except ValueError:
                    findings.append(("치명", f"회전 축 값을 못 읽었다: {tok}"))
                    continue
                vs.append(v)
                if v_first is None:
                    v_first = v
                wind_max = max(wind_max, abs(v - v_first))
                if v_prev is not None:
                    dv = abs(v - v_prev)
                    v_travel += dv
                    if in_rewind:
                        rewind_max = max(rewind_max, dv)
                    elif dv > dv_max:
                        dv_max, dv_at = dv, (mv.x, mv.y, mv.z)
                v_prev = v
        if mv.x is not None and mv.y is not None:
            xys.append((mv.x, mv.y))
        if mv.z is not None:
            zs.append(mv.z)
        if mv.f is not None:
            feeds.append(mv.f)

    # --- 틸트 ---
    if not u_seen:
        findings.append(("치명", f"틸트 축 {profile.tilt_axis} 설정이 없다 — "
                                 "원뿔 레이어가 수평이 되지 않는다"))
    else:
        if max(abs(u) for u in u_seen) > profile.max_tilt_deg + 1e-9:
            findings.append(("치명", f"틸트 {max(u_seen, key=abs):.1f}° 가 기계 한계 "
                                     f"±{profile.max_tilt_deg}° 를 넘는다"))
        if len(set(round(u, 6) for u in u_seen)) > 1:
            findings.append(("경고", f"틸트가 여러 값으로 바뀐다 ({len(set(u_seen))}종) "
                                     "— 원뿔 모드는 상수여야 한다"))

    # --- 배선 감김 ---
    #   감김은 **시작 기준 최대 이탈**이다. Σ|ΔV| 가 아니다 — +360 돌고 −360 돌면
    #   배선은 제자리다. 실측: funnel 20° 에서 Σ|ΔV|=175회전인데 실제 감김은 44.5.
    #   Σ|ΔV| 는 마모·시간의 척도로 따로 본다.
    total_turns = v_travel / 360.0
    wind_turns = wind_max / 360.0
    if vs:
        limit = max_turns if max_turns is not None else 2.0
        if wind_turns > limit:
            findings.append(("치명",
                             f"배선 감김 {wind_turns:.1f}회전 (한계 {limit:.1f}) — "
                             "슬립링이 없으면 배선이 감긴다. 트래블 중 되감기"
                             "(`--v-rewind`) 또는 레이어별 방향 교대가 필요하다"))
        elif wind_turns > limit * 0.6:
            findings.append(("경고", f"배선 감김 {wind_turns:.1f}회전 — 여유 확인"))
        if wind_turns > 1e-9 and total_turns > wind_turns * 3:
            findings.append(("정보",
                             f"V 가 많이 왕복한다 (총 회전량 {total_turns:.1f} vs "
                             f"감김 {wind_turns:.1f}회전) — 감김은 아니지만 "
                             "베어링 마모와 출력 시간에 들어간다"))

    # --- 급회전 ---
    if dv_max > max_dv_deg:
        where = (f" (부근 X{dv_at[0]:.1f} Y{dv_at[1]:.1f} Z{dv_at[2]:.1f})"
                 if dv_at and None not in dv_at else "")
        sev = "치명" if dv_max > 90.0 else "경고"
        findings.append((sev, f"한 이동에 V 가 {dv_max:.0f}° 뛴다{where} — "
                              "축 근처 방위각 반전. 베드가 급회전하며 출력물을 "
                              "흔든다"))

    # --- 기계 Z 가 음수로 간다 ---
    #   틸트로 베드가 '내려간 쪽'을 따라가려면 기계 Z 가 음수여야 하는 것이
    #   정상일 수 있다. 위험한 것은 그 자체가 아니라 **펌웨어가 잘라낼 때**다.
    if zs and min(zs) < min_z - 1e-6:
        findings.append(("치명",
                         f"기계 Z 최소 {min(zs):.2f}mm < {min_z} — 틸트로 베드가 "
                         "내려간 쪽을 따라가려면 Z 가 음수여야 한다. 펌웨어 "
                         "소프트리밋이 0 이면 잘려서 노즐이 출력물을 파고든다. "
                         "Z 오프셋(원점을 띄우기)을 잡고 소프트리밋을 확인할 것"))

    # --- 베드 밖 ---
    if bed_radius is not None and xys:
        r = max(math.hypot(a, b) for a, b in xys)
        if r > bed_radius:
            findings.append(("치명", f"기계좌표 반경 {r:.1f}mm > 베드 반경 "
                                     f"{bed_radius}mm — 베드 밖으로 나간다"))

    # --- 피드 ---
    if max_feed is not None and feeds and max(feeds) > max_feed:
        findings.append(("경고", f"피드 {max(feeds):.0f} > 기계 최대 {max_feed} "
                                 "— 펌웨어가 잘라내면 동기가 어긋날 수 있다"))

    if n_rewind:
        findings.append(("정보", f"되감기 {n_rewind}회 (최대 {rewind_max:.0f}°) — "
                                 "의도된 회전이라 급회전 판정에서 뺐다. 들어올린 "
                                 "높이가 퇴적물 위인지는 이 검사가 보지 못한다"))
    stats = {"tilt": u_seen[0] if u_seen else None,
             "rewinds": n_rewind, "rewind_max_deg": rewind_max,
             "v_wind_turns": wind_turns,
             "v_total_turns": total_turns,
             "v_span_turns": ((max(vs) - min(vs)) / 360.0) if vs else 0.0,
             "v_max_step_deg": dv_max,
             "z_min": min(zs) if zs else None,
             "xy_radius_max": (max(math.hypot(a, b) for a, b in xys)
                               if xys else None),
             "moves": len(vs)}
    return findings, stats
