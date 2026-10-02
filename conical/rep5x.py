"""
rep5x.py — 원뿔 G-code → REP5X(헤드 틸트+요) 5축 출력. [실험적]

Open5x 와 **출력 계약이 다르다.** Open5x 는 펌웨어가 회전 기구학을 하지 않아
슬라이서가 기계좌표를 직접 계산했다(`open5x.py`). REP5X 는 **펌웨어가 역기구학을
한다** — 그래서 슬라이서는 **노즐 팁 위치(부품좌표 그대로) + 공구 방향**만 내면
되고, 좌표 변환이 아예 **빠진다.** `to_open5x` 보다 하는 일이 적다.

## 저장소·펌웨어 원문으로 확인한 것 (2026-09-19)

축이 **헤드에 붙는다** (`build-guide/assembly-instructions-universal.md`):
  · Step 10 "Attach **carriage mount** to **X-carriage**"
  · Step 11 "Mount **C-axis (yaw) stepper motor** to carriage mount"
  · Step 12 "Route all wires and Bowden tube through **B_arm** center"
  · 슬립링이 나르는 것: hotend fan / heater / thermistor / B-axis endstop
  → **베드는 안 움직인다.** 부품이 고정이고 노즐이 기울고 돈다.

펌웨어 규약 (`Rep5x-Marlin`, 브랜치 `Marlin2ForPipetBot`, `Marlin/Configuration.h`):
  · `AXIS4_NAME 'C'` + `AXIS4_ROTATES` — 요(yaw), Z축 둘레. 내부 축은 I
  · `AXIS5_NAME 'B'` + `AXIS5_ROTATES` — 틸트, **0 에서 Y 에 평행**. 내부 축은 J
  · `PENTA_AXIS_HH` (head-head 기구학, Marlin2ForPipetBot 파생)
  · `DEFAULT_TOOL_CENTERPOINT_CONTROL true` — **RTCP 기본 ON.**
    즉 **G-code 의 X/Y/Z 는 노즐 팁 위치**이고 펌웨어가 캐리지 위치를 역산한다.
    런타임 전환: `G43.4` 켜기 / `G49` 끄기 → 시작 G-code 에 `G43.4` 를 박는다
  · `DEFAULT_ROTATIONAL_JOINT_OFFSET_Z 54.67` (LB: 틸트축→노즐 팁)
  · `DEFAULT_ROTATIONAL_JOINT_OFFSET_Y 1.6`   (LC: 요축 중심→툴)
  · `I_MIN_POS -360` / `I_MAX_POS 360`  ← **C 는 ±360° 다**
  · `J_MIN_POS -135` / `J_MAX_POS 135`
  · `MIN_SOFTWARE_ENDSTOP_I` / `MAX_SOFTWARE_ENDSTOP_I` 둘 다 켜져 있다
  · `PRINTABLE_RADIUS 100.0`

## ⚠⚠ C 가 ±360° 라 **되감기가 여전히 필요하다**

"Continuous yaw rotation" 은 **슬립링이 푸는 기계적 제약**이지 소프트웨어 한계가
아니다. 펌웨어 소프트 엔드스톱이 C 를 ±360° 로 묶고 랩 기능이 없다. 우리 원뿔
G-code 는 C 가 방위각을 따라 **계속 누적**되므로(funnel 20° 에서 44.5회전)
그대로 걸면 **소프트 엔드스톱에 걸려 멈춘다.**

되감기는 `open5x.add_v_rewinds` 를 그대로 쓴다 — 그 함수는 `profile.rot_axis` 만
보므로 축 문자만 'C' 로 주면 된다. 되는 이유도 같다: C 를 360° 배수만큼 돌리면
**공구 방향이 정확히 같은 값으로 돌아온다**(Rz 주기). RTCP 가 켜져 있으면 팁도
제자리다.

⚠ 다만 **도는 도중에 B_arm 이 팁 둘레를 쓸고 지나간다** (반각 θ, 팁에서 LB 만큼
  위, 수평 반경 LB·sinθ ≈ 18.7mm @ θ=20°). 베드 회전식과 달리 부품은 안 움직이지만
  팔이 움직인다. 그래서 들어올림을 그대로 유지한다.

## 수학 (부품좌표 p → B/C)

원뿔 레이어의 국소 법선 (outward, 레이어 `z = z₀ − r·tanθ`):

    n̂ = sinθ·r̂(φ) + cosθ·ẑ           (φ = 점의 방위각)

노즐은 그 반대를 향해야 한다. 펌웨어 규약에서 공구 방향은

    d̂ = Rz(C)·Ry(B)·(0,0,−1) = (−sinB·cosC, −sinB·sinC, −cosB)

둘을 맞추면 **B = c·θ (상수), C = φ(p)** 가 나온다 (c = +1 outward / −1 inward).
`tests/test_rep5x.py` 가 이 일치를 수치로 강제한다 — 부호 하나가 틀리면 노즐이
반대로 기울므로 말로 두면 안 되는 부분이다.

⚠⚠ **부호는 실기 보정 대상이다.** 위 유도는 펌웨어 **설정 파일의 규약**을 따른
  것이고, 실제 조립에서 모터 방향이 반대로 붙었으면 `INVERT_I_DIR`/`INVERT_J_DIR`
  로 펌웨어 쪽에서 뒤집힌다. `tilt_sign` / `rot_sign` 으로 여기서도 뒤집을 수
  있게 두었다. **첫 출력 전에 `G0 B10` 을 눈으로 확인할 것.**

⚠ 피드레이트는 **그대로 통과시킨다.** X/Y/Z 가 노즐 팁의 실제 경로이므로 F 가
  이미 맞는 값이다 (Open5x 에서 필요했던 `F × 기계길이/부품길이` 재조정이 없다).
  회전축이 못 따라가면 Marlin 플래너가 알아서 전체를 늦춘다 — RRF 와 같다.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .gcode import Move


@dataclass
class Rep5xProfile:
    """REP5X 기계 파라미터. 기본값은 저장소 `Configuration.h` 에서 가져왔다."""

    tilt_axis: str = "B"
    rot_axis: str = "C"
    tilt_sign: float = 1.0          # 실기 보정: 조립이 반대면 -1
    rot_sign: float = 1.0
    b_min: float = -135.0           # J_MIN_POS
    b_max: float = 135.0            # J_MAX_POS
    c_min: float = -360.0           # I_MIN_POS ← 되감기가 필요한 이유
    c_max: float = 360.0            # I_MAX_POS
    printable_radius: float = 100.0  # PRINTABLE_RADIUS
    lb: float = 54.67               # DEFAULT_ROTATIONAL_JOINT_OFFSET_Z
    lc: float = 1.6                 # DEFAULT_ROTATIONAL_JOINT_OFFSET_Y
    rtcp_on: str = "G43.4"          # RTCP 켜기 (G49 가 끄기)
    name: str = "REP5X (Configuration.h 기본값)"


REP5X = Rep5xProfile()


def tool_direction(b_deg, c_deg):
    """(B, C) → 공구 방향 단위벡터 `d̂ = Rz(C)·Ry(B)·(0,0,−1)`.

    검증용 역함수다. `to_rep5x` 가 낸 각도를 이걸로 되돌려 원뿔 법선과 맞춰본다.
    """
    b, c = math.radians(b_deg), math.radians(c_deg)
    return (-math.sin(b) * math.cos(c),
            -math.sin(b) * math.sin(c),
            -math.cos(b))


def cone_tool_direction(x, y, cone_angle_deg, cone_type):
    """원뿔 레이어 법선의 반대 = 노즐이 향해야 할 방향 (해석식)."""
    th = math.radians(cone_angle_deg) * (1.0 if cone_type == "outward" else -1.0)
    r = math.hypot(x, y)
    if r < 1e-12:
        return (0.0, 0.0, -1.0)
    return (-math.sin(th) * x / r, -math.sin(th) * y / r, -math.cos(th))


def to_rep5x(items, cone_angle_deg, cone_type, profile=REP5X):
    """원뿔 G-code(부품좌표) → REP5X G-code. **좌표는 그대로 두고 B/C 만 붙인다.**

    반환: (아이템 리스트, 통계 dict)
    """
    if abs(cone_angle_deg) > max(abs(profile.b_min), abs(profile.b_max)):
        raise ValueError(f"cone angle {cone_angle_deg} > B 축 한계 "
                         f"[{profile.b_min}, {profile.b_max}]")
    c_dir = 1.0 if cone_type == "outward" else -1.0
    b = profile.tilt_sign * c_dir * cone_angle_deg

    out = [("raw", "; REP5X 5-axis conical mode [EXPERIMENTAL]"),
           ("raw", "; head-head kinematics: firmware does the IK "
                   "(PENTA_AXIS_HH), X/Y/Z = nozzle tip"),
           ("raw", f"; {profile.tilt_axis}={b:.3f}deg fixed tilt, "
                   f"{profile.rot_axis} tracks azimuth (continuous, unwrapped)"),
           ("raw", f"; LB={profile.lb} LC={profile.lc} "
                   f"(verify on machine; axis signs too)"),
           ("raw", f"{profile.rtcp_on}  ; enable tool centerpoint control (RTCP)"),
           ("raw", f"G1 {profile.tilt_axis}{b:.3f} F600  ; tilt head")]

    x = y = z = None
    c_prev = 0.0
    c_range = [0.0, 0.0]
    n_on_axis = 0
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
        r = math.hypot(x, y)
        if r < 1e-6:
            c = c_prev                      # 축 위: 방위각 정의 불가 → 유지
            n_on_axis += 1
        else:
            c_raw = profile.rot_sign * math.degrees(math.atan2(y, x))
            # 연속 누적(unwrap): 직전 값과 ±180° 이내가 되도록 360° 단위 이동
            c = c_raw + 360.0 * round((c_prev - c_raw) / 360.0)
        # 좌표·피드는 **그대로**. 이것이 Open5x 와의 차이다.
        out.append(("move", Move(g=mv.g, x=mv.x, y=mv.y, z=mv.z, e=mv.e,
                                 f=mv.f,
                                 extra=f"{profile.tilt_axis}{b:.3f} "
                                       f"{profile.rot_axis}{c:.3f}")))
        c_prev = c
        c_range[0] = min(c_range[0], c)
        c_range[1] = max(c_range[1], c)

    return out, {"b": b, "c_min": c_range[0], "c_max": c_range[1],
                 "c_turns": (c_range[1] - c_range[0]) / 360.0,
                 "on_axis_moves": n_on_axis}


def check_rep5x(items, profile=REP5X, max_dc_deg=30.0, max_feed=None):
    """REP5X G-code 를 기계에 걸기 **전에** 검사한다 (검사기 C 의 REP5X 판).

    Open5x 판(`check_open5x`)과 **보는 것이 다르다.** 기계좌표가 아니라 부품좌표를
    내므로 기계 Z 음수·베드 이탈 같은 항목이 없고, 대신 **펌웨어 소프트 엔드스톱**이
    핵심이다.

    보는 것:
      · **C 가 ±360° 안인가** ← 가장 중요. 누적되므로 되감기 없이는 반드시 넘는다
      · B 가 축 한계 안인가, 한 번만 설정되는가
      · **C 급회전** — 축 근처 트래블에서 한 이동에 ±180° 가 뛴다 (방위각 반전)
      · `PRINTABLE_RADIUS` 를 넘는가
      · RTCP(`G43.4`) 가 켜져 있는가 — 없으면 X/Y/Z 해석이 통째로 달라진다

    ⚠ 보지 **못하는** 것: 노즐 몸체·B_arm 과 출력물의 간섭. **헤드 회전식에서는
      이 간섭이 살아 있다** — 베드 틸트식(`envelope.check_planar_stacking`)에서
      성립하던 '정의상 불가능' 논증이 여기서는 성립하지 않는다. 미해결이다.

    반환: (findings, stats). 심각도 "치명"/"경고"/"정보".
    """
    findings, b_seen, cs, feeds, radii = [], [], [], [], []
    in_rewind, n_rewind, rewind_max = False, 0, 0.0
    dc_max, dc_at = 0.0, None
    c_prev = None
    has_rtcp = False
    x = y = None

    for kind, payload in items:
        if kind != "move":
            if isinstance(payload, str):
                up = payload.upper()
                if "V_REWIND BEGIN" in up:
                    in_rewind, n_rewind = True, n_rewind + 1
                elif "V_REWIND END" in up:
                    in_rewind = False
                elif profile.rtcp_on.upper() in up and not up.lstrip().startswith(";"):
                    has_rtcp = True
            continue
        mv = payload
        for tok in (mv.extra or "").upper().split():
            if tok.startswith(profile.tilt_axis):
                try:
                    b_seen.append(float(tok[len(profile.tilt_axis):]))
                except ValueError:
                    findings.append(("치명", f"틸트 축 값을 못 읽었다: {tok}"))
            elif tok.startswith(profile.rot_axis):
                try:
                    c = float(tok[len(profile.rot_axis):])
                except ValueError:
                    findings.append(("치명", f"회전 축 값을 못 읽었다: {tok}"))
                    continue
                cs.append(c)
                if c_prev is not None:
                    dc = abs(c - c_prev)
                    if in_rewind:
                        rewind_max = max(rewind_max, dc)
                    elif dc > dc_max:
                        dc_max, dc_at = dc, (mv.x, mv.y, mv.z)
                c_prev = c
        if mv.x is not None:
            x = mv.x
        if mv.y is not None:
            y = mv.y
        if x is not None and y is not None:
            radii.append(math.hypot(x, y))
        if mv.f is not None:
            feeds.append(mv.f)

    if not has_rtcp:
        findings.append(("치명", f"RTCP({profile.rtcp_on})가 없다 — 없으면 X/Y/Z 가 "
                                f"노즐 팁이 아니라 캐리지로 해석된다"))
    if not b_seen:
        findings.append(("치명", "틸트 축 값이 하나도 없다"))
    else:
        if max(b_seen) - min(b_seen) > 1e-6:
            findings.append(("경고", f"틸트가 여러 값이다 "
                                     f"({min(b_seen):.2f}~{max(b_seen):.2f}) — "
                                     f"고정 틸트 원뿔 모드가 아니다"))
        if min(b_seen) < profile.b_min - 1e-9 or max(b_seen) > profile.b_max + 1e-9:
            findings.append(("치명", f"틸트 {min(b_seen):.1f}~{max(b_seen):.1f}° 가 "
                                     f"기계 한계 [{profile.b_min}, "
                                     f"{profile.b_max}] 밖이다"))

    c_lo, c_hi = (min(cs), max(cs)) if cs else (0.0, 0.0)
    if cs and (c_lo < profile.c_min - 1e-9 or c_hi > profile.c_max + 1e-9):
        findings.append(("치명", f"C 가 {c_lo:.0f}~{c_hi:.0f}° 로 펌웨어 소프트 "
                                 f"엔드스톱 [{profile.c_min:.0f}, "
                                 f"{profile.c_max:.0f}] 을 넘는다 "
                                 f"({(c_hi - c_lo) / 360.0:.1f}회전). "
                                 f"되감기(`add_c_rewinds`)가 필요하다"))
    if dc_max > max_dc_deg:
        findings.append(("경고", f"C 가 한 이동에 {dc_max:.0f}° 뛴다"
                                 + (f" (부품좌표 {dc_at})" if dc_at else "")
                                 + " — 축 근처 방위각 반전. 헤드가 급회전한다"))
    if radii and max(radii) > profile.printable_radius + 1e-9:
        findings.append(("치명", f"부품 반경 {max(radii):.1f}mm 가 "
                                 f"PRINTABLE_RADIUS {profile.printable_radius}mm "
                                 f"를 넘는다"))
    if max_feed is not None and feeds and max(feeds) > max_feed + 1e-9:
        findings.append(("경고", f"피드 최대 {max(feeds):.0f} > {max_feed:.0f}"))

    stats = {"tilt": b_seen[0] if b_seen else None,
             "c_min": c_lo, "c_max": c_hi,
             "c_wind_turns": (max(abs(c - cs[0]) for c in cs) / 360.0) if cs else 0.0,
             "c_span_turns": (c_hi - c_lo) / 360.0,
             "c_max_step_deg": dc_max,
             "radius_max": max(radii) if radii else 0.0,
             "moves": len(cs), "rewinds": n_rewind,
             "rewind_max_deg": rewind_max, "rtcp": has_rtcp}
    return findings, stats


# ─────────────────────────────────────────────────────────────
# C 되감기: **절대 창** 제약 — Open5x 의 되감기와 종류가 다르다
# ─────────────────────────────────────────────────────────────
def _with_axis(extra, axis, value):
    """extra 문자열에서 `axis` 워드만 갈아끼운다 (B 등 다른 워드는 보존)."""
    toks = [t for t in (extra or "").split()
            if not t.upper().startswith(axis.upper())]
    toks.append(f"{axis}{value:.3f}")
    return " ".join(toks)


def _c_of(extra, axis):
    for tok in (extra or "").split():
        if tok.upper().startswith(axis.upper()):
            try:
                return float(tok[len(axis):])
            except ValueError:
                return None
    return None


def add_c_rewinds(items, profile=REP5X, clearance=2.0, rot_feed=3600.0,
                  lift_feed=3000.0):
    """C 를 펌웨어 소프트 엔드스톱 **창 안에** 유지한다 (트래블 중 360° 배수 되감기).

    ## 왜 `open5x.add_v_rewinds` 를 그대로 못 쓰나 — 제약의 **종류가 다르다**

      · Open5x: 배선 감김 = **시작점 기준 이탈** `max|V − V₀|`. 기준점에서
        멀어지지만 않으면 되므로 '가까운 쪽으로 되감기' 가 맞다.
      · REP5X: 펌웨어 소프트 엔드스톱 = **절대 창** `c_min ≤ C ≤ c_max`.
        **창 밖으로 나가면 기계가 멈춘다.**

    창 가운데로 되감으면 안 된다. 360° 배수로만 옮길 수 있어 **나머지 각도가
    ±180° 까지 남고**, 거기서 한 바퀴(360°)를 더 감으면 ±540° 로 창을 넘는다.
    실측으로 확인했다: 가운데로 되감으니 funnel 이 −450~275° 로 나갔다.

    ## 하는 것: **다음 압출 구간을 미리 보고** 딱 맞는 k 를 고른다

    되감기는 트래블에서만 할 수 있으므로, 한 번 정하면 **다음 트래블까지** 그
    값으로 버텨야 한다. 그래서 추측하지 않고 앞을 본다:

      ① 이 트래블부터 다음 트래블까지 C 가 훑을 범위 `[lo, hi]` 를 미리 구한다
      ② `[lo + 360k, hi + 360k] ⊆ [c_min, c_max]` 인 정수 k 를 고른다
         (여러 개면 창 가운데에 가장 가까운 것)
      ③ 그런 k 가 없으면 **되감기로 못 고치는 구간**이다 — `fixable=False` 로
         알리고 넘어간다. 창보다 넓게 감기는 압출 구간이 있다는 뜻이고,
         그때는 슬라이싱 쪽에서 경로를 끊어야 한다

    앞을 보기 때문에 방향을 추측할 필요가 없다. 추측하면 요동친다 — 직전 이동의
    부호로 방향을 정했더니 되감기가 214회로 늘고 C 가 여전히 창을 벗어났다.

    ⚠ C 를 360° 배수만큼 돌리면 **공구 방향이 정확히 같은 값으로 돌아온다**
      (`d̂` 는 C 에 대해 360° 주기). RTCP 가 켜져 있으면 노즐 팁도 제자리다.
      그래서 툴패스가 보존된다 — 회귀 테스트가 이걸 강제한다.

    ⚠ 도는 도중 **B_arm 이 팁 둘레를 쓸고 지나간다** (팁에서 LB 만큼 위, 수평
      반경 LB·sinθ ≈ 18.7mm @ θ=20°). 부품은 안 움직이지만 팔이 움직이므로
      들어올림을 유지한다. ⚠ 들어올림은 '여태 퇴적한 것보다 위' 까지만 보장하고
      팔이 **옆으로** 부딪히는 것은 **보지 못한다.**
    """
    axis = profile.rot_axis
    lo_lim, hi_lim = profile.c_min, profile.c_max
    center = (lo_lim + hi_lim) / 2.0

    # ── 1패스: 이동만 뽑아 C 와 압출 여부를 정리한다
    moves = [(i, p) for i, (k, p) in enumerate(items) if k == "move"]
    cs, extr = [], []
    e_prev = 0.0
    for _i, mv in moves:
        cs.append(_c_of(mv.extra, axis))
        extr.append(mv.e is not None and mv.e > e_prev + 1e-9)
        if mv.e is not None:
            e_prev = mv.e

    # 각 이동에서 '다음 트래블까지' C 가 훑는 범위 (되감기 판단용 look-ahead)
    n = len(moves)
    seg_lo, seg_hi = [None] * n, [None] * n
    for start in range(n):
        if extr[start]:
            continue                     # 트래블에서만 되감으므로 거기서만 계산
        # 다음 트래블에서 또 되감을 수 있으므로, 이번 결정이 책임질 구간은
        # [start, 다음 트래블 직전] 이다.
        nxt = start + 1
        while nxt < n and extr[nxt]:
            nxt += 1
        vals = [cs[j] for j in range(start, nxt) if cs[j] is not None]
        if vals:
            seg_lo[start], seg_hi[start] = min(vals), max(vals)

    # ── 첫 트래블 **이전** 구간은 어떤 되감기 결정에도 안 잡힌다.
    # 거기서 이미 창을 넘을 수 있으므로(실측: funnel 이 −450° 로 시작), 시작
    # 오프셋으로 분기를 골라 둔다. G-code 를 더 내지는 않는다 — 헤드는 어차피
    # 시작 전에 그 각도로 가 있어야 하고, C 는 360° 주기라 공구 방향이 같다.
    # 첫 이동에서는 되감기를 낼 수 없다(직전 좌표가 없다). 그래서 **1번부터**
    # 다음 트래블을 찾는다 — 실측: funnel 은 이동 0 이 트래블이고 1~16 이
    # 한 번에 0.94회전 감기는 압출 구간이라, 여기가 통째로 안 덮이면 −450° 로
    # 시작해 버린다.
    t0 = 1
    while t0 < n and extr[t0]:
        t0 += 1
    head_vals = [cs[j] for j in range(0, min(t0, n)) if cs[j] is not None]
    start_offset, start_unfixable = 0.0, 0
    if head_vals:
        h_lo, h_hi = min(head_vals), max(head_vals)
        k_lo = math.ceil((lo_lim - h_lo) / 360.0 - 1e-9)
        k_hi = math.floor((hi_lim - h_hi) / 360.0 + 1e-9)
        if k_lo <= k_hi:
            start_offset = 360.0 * min(range(k_lo, k_hi + 1),
                                       key=lambda kk: abs((h_lo + h_hi) / 2
                                                          + 360.0 * kk - center))
        else:
            # 첫 구간부터 창보다 넓게 감긴다 — 되감기로 못 고친다. 조용히
            # 넘어가면 기계 앞에서 알게 되므로 여기서 센다.
            start_unfixable = 1

    # ── 2패스: 되감기를 끼워 넣는다
    out, mi = [], 0
    offset = start_offset
    c_prev = None
    px = py = pz = None
    z_dep = None
    e_prev = 0.0
    n_rewinds, rewind_deg, unfixable = 0, 0.0, start_unfixable

    for kind, payload in items:
        if kind != "move":
            out.append((kind, payload))
            continue
        mv = payload
        c_raw = cs[mi]
        extruding = extr[mi]
        lo, hi = seg_lo[mi], seg_hi[mi]
        mi += 1

        if c_raw is not None:
            c = c_raw + offset
            if (not extruding and c_prev is not None and lo is not None
                    and None not in (px, py, pz)):
                lo_o, hi_o = lo + offset, hi + offset
                if lo_o < lo_lim - 1e-9 or hi_o > hi_lim + 1e-9:
                    # [lo+360k, hi+360k] ⊆ 창 인 k 들 중 가운데에 가장 가까운 것
                    k_lo = math.ceil((lo_lim - lo_o) / 360.0 - 1e-9)
                    k_hi = math.floor((hi_lim - hi_o) / 360.0 + 1e-9)
                    if k_lo > k_hi:
                        unfixable += 1          # 창보다 넓게 감기는 구간
                    else:
                        k = min(range(k_lo, k_hi + 1),
                                key=lambda kk: abs((lo_o + hi_o) / 2
                                                   + 360.0 * kk - center))
                        if k != 0:
                            safe_z = max(z_dep if z_dep is not None else pz,
                                         pz) + clearance
                            c_new = c_prev + 360.0 * k
                            out.append(("raw",
                                        f"; V_REWIND BEGIN {k:+d} turn(s): "
                                        f"{c_prev:.1f} -> {c_new:.1f} deg "
                                        f"(lift to Z{safe_z:.2f}) [{axis} window "
                                        f"{lo_lim:.0f}..{hi_lim:.0f}]"))
                            for zz, ff, cc in ((safe_z, lift_feed, c_prev),
                                               (safe_z, rot_feed, c_new),
                                               (pz, lift_feed, c_new)):
                                out.append(("move", Move(
                                    g=0, x=px, y=py, z=zz, f=ff,
                                    extra=_with_axis(mv.extra, axis, cc))))
                            out.append(("raw", "; V_REWIND END"))
                            rewind_deg += abs(360.0 * k)
                            offset += 360.0 * k
                            c = c_raw + offset
                            n_rewinds += 1
            mv = Move(g=mv.g, x=mv.x, y=mv.y, z=mv.z, e=mv.e, f=mv.f,
                      extra=_with_axis(mv.extra, axis, c))
            c_prev = c

        out.append(("move", mv))
        if mv.x is not None:
            px = mv.x
        if mv.y is not None:
            py = mv.y
        if mv.z is not None:
            pz = mv.z
        if extruding and pz is not None:
            z_dep = pz if z_dep is None else max(z_dep, pz)
        if mv.e is not None:
            e_prev = mv.e

    return out, {"rewinds": n_rewinds, "rewind_deg": rewind_deg,
                 "unfixable_segments": unfixable, "fixable": unfixable == 0,
                 "start_offset": start_offset,
                 "window_turns": (hi_lim - lo_lim) / 360.0}


# ─────────────────────────────────────────────────────────────
# 헤드 회전식 노즐 간섭 — 베드 틸트식의 '정의상 불가능' 이 여기서는 성립 안 한다
# ─────────────────────────────────────────────────────────────
def move_tool_frames(items, profile=REP5X):
    """이동별 (B, C) → 공구 축 '위' 방향 단위벡터 배열.

    `sample_extrusions` 의 `move_id` 가 **이동 인덱스**이므로, 여기서 만든 배열을
    `up[move_id]` 로 색인하면 샘플점마다의 공구 자세가 된다.

    모달이다: B/C 가 안 적힌 이동은 직전 값을 잇는다.
    """
    b = c = 0.0
    ups = []
    for kind, p in items:
        if kind != "move":
            continue
        for tok in (p.extra or "").split():
            t = tok.upper()
            if t.startswith(profile.tilt_axis.upper()):
                try:
                    b = float(tok[len(profile.tilt_axis):])
                except ValueError:
                    pass
            elif t.startswith(profile.rot_axis.upper()):
                try:
                    c = float(tok[len(profile.rot_axis):])
                except ValueError:
                    pass
        d = tool_direction(b, c)
        ups.append((-d[0], -d[1], -d[2]))       # 팁에서 공구 축을 따라 '위'
    return np.asarray(ups, dtype=float)


def check_head_interference(items, profile=REP5X, hotend=None, width=0.45,
                            clearance=0.3, arm_radius=None,
                            batch_samples=2000, stride=1):
    """헤드 회전식에서 **노즐/B_arm 이 출력물을 치는가**.

    ## 왜 이게 따로 필요한가

    베드 틸트식(Open5x)에서는 원뿔 레이어가 기계공간에서 수평면이 되어 **간섭이
    정의상 불가능**했다(`envelope.check_planar_stacking`). **헤드 회전식에서는 그
    논증이 성립하지 않는다** — 부품이 고정이라 레이어가 기울어 있고, 노즐이 기운다.

    그렇다고 3축과 같지도 않다. 3축은 노즐이 **수직 고정**이라 기운 원뿔면 위에서
    핫엔드가 출력물을 파고들었다(funnel 의 3축 MAX_ANGLE 이 24° 였던 이유).
    헤드 회전식은 노즐이 **원뿔면의 법선을 따라간다** — 국소적으로는 평면 출력과
    같은 자세다. 그래서 유리할 것으로 **예상**되는데, 예상은 근거가 아니므로 잰다.

    ## 어떻게

    `toolpath.check_nozzle` 을 **공구 프레임**으로 일반화해 그대로 쓴다. 3축은
    `tool_up=(0,0,1)` 인 특수 경우이고 두 경로의 결과가 **정확히 같다**
    (테스트가 강제). 그래서 3축 판정과 5축 판정을 **같은 기하·같은 코드**로
    비교할 수 있다 — 기하가 다르면 비교가 안 된다.

    ⚠ 보지 못하는 것:
      · **B_arm 의 실제 치수** — 저장소에 3MF 뿐이라 반경을 모른다. `arm_radius`
        를 주면 팁에서 `LB`(54.67mm)까지 원기둥으로 근사해 넣고, 안 주면 **끈다**.
        (`arm_modeled` 로 어느 쪽인지 보고한다)
      · **트래블·되감기 도중**의 자세 → `check_rewind_sweep`
      · 캐리지·팬 덕트·보덴 튜브
    """
    from .toolpath import HotendProfile, check_nozzle, sample_extrusions

    pts, mid, _wt = sample_extrusions(items, width=width)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool), {"collision_pct": 0.0,
                                         "first_collision_z": None,
                                         "arm_modeled": False, "samples": 0}
    ups = move_tool_frames(items, profile)
    col, st = check_nozzle(pts, mid, hotend or HotendProfile(),
                           batch_samples=batch_samples, clearance=clearance,
                           tool_up=ups[mid],
                           arm_length=profile.lb if arm_radius else None,
                           arm_radius=arm_radius, stride=stride)
    return col, st


def check_rewind_sweep(items, profile=REP5X, hotend=None, width=0.45,
                       n_angles=24, clearance=0.3, arm_radius=None):
    """되감기 **도중** 핫엔드가 쓸고 지나가는 부피가 출력물을 치는가.

    되감기는 C 를 360° 배수만큼 돌린다. **끝난 뒤**의 자세는 시작과 같지만
    (Rz 주기), **도는 도중**에는 다르다. RTCP 가 팁을 붙잡아 두므로 핫엔드 몸체가
    팁 둘레를 **반각 B 의 원뿔로 쓸고 지나간다.** 베드 회전식에서는 부품이 돌아
    문제였는데, 여기서는 팔이 돈다 — 어느 쪽이든 도중 자세를 봐야 한다.

    들어올림(`add_c_rewinds` 의 `clearance`)이 그 대책인데, 그 들어올림은
    '여태 퇴적한 것보다 위' 까지만 보장한다. 옆으로 부딪히는 것은 **이 검사가**
    본다.

    방법: 되감기 블록 안의 이동마다 **그 이동이 실제로 훑는 C 구간**을
    `n_angles` 개로 나눠 각 자세에서 간섭을 보고 OR 한다. 되감기는
    들어올림 → 회전 → 하강 세 이동인데 **C 가 도는 것은 가운데 하나뿐**이고
    나머지는 C 가 고정이다. 전부 360° 훑는 것으로 잘못 세면 들어올림을 30mm 로
    키워도 검출 수가 안 변한다 — 실제로 그렇게 만들었다가 잡았다.

    반환: (findings, stats). 되감기가 없으면 빈 목록.
    """
    from .toolpath import HotendProfile, sample_extrusions

    h = hotend or HotendProfile()
    pts, _mid, _wt = sample_extrusions(items, width=width)
    if len(pts) == 0:
        return [], {"rewinds": 0, "swept_hits": 0, "checked": 0}

    # 되감기 구간의 '들어올린 뒤 회전하는' 이동을 모은다: 그 위치와 B, 그리고
    # 그 시점까지 퇴적된 점만 후보다 (아직 안 찍은 것은 부딪힐 것이 없다).
    tree = cKDTree(pts)
    tan_half = math.tan(math.radians(h.cone_half_deg))
    dz_max = h.block_z0 + h.block_height
    far = profile.lb if arm_radius else dz_max
    r_max = max(h.block_radius, h.tip_radius + h.cone_height * tan_half,
                arm_radius or 0.0)
    radius = math.sqrt(r_max ** 2 + far ** 2)

    def hit(dz, horiz):
        cone = (dz > clearance) & (dz <= h.cone_height) & \
               (horiz < h.tip_radius + dz * tan_half)
        block = (dz > max(h.block_z0, clearance)) & (dz <= dz_max) & \
                (horiz < h.block_radius)
        out = cone | block
        if arm_radius:
            out = out | ((dz > dz_max) & (dz <= profile.lb) &
                         (horiz < arm_radius))
        return out

    in_rw, n_rw, hits, checked = False, 0, 0, 0
    worst = None
    px = py = pz = None
    b_cur = 0.0
    # 여태 퇴적한 샘플 수. `_mid` 는 이동 인덱스이고 샘플은 G-code 순서이므로,
    # '이동 m 까지의 샘플 수' = searchsorted(_mid, m, "right") 로 정확히 나온다.
    # (직접 포인터를 미는 방식은 이동 하나가 샘플을 0개 내는 경우에 어긋난다)
    n_dep = 0
    move_i = -1
    c_from = c_cur = None
    for kind, payload in items:
        if kind != "move":
            if isinstance(payload, str):
                up = payload.upper()
                if "V_REWIND BEGIN" in up:
                    in_rw, n_rw = True, n_rw + 1
                elif "V_REWIND END" in up:
                    in_rw = False
            continue
        mv = payload
        move_i += 1
        n_dep = int(np.searchsorted(_mid, move_i, side="right"))
        for tok in (mv.extra or "").split():
            if tok.upper().startswith(profile.tilt_axis.upper()):
                try:
                    b_cur = float(tok[len(profile.tilt_axis):])
                except ValueError:
                    pass
        c_new = _c_of(mv.extra, profile.rot_axis)
        if c_new is not None:
            c_from, c_cur = (c_cur if c_cur is not None else c_new), c_new
        px = mv.x if mv.x is not None else px
        py = mv.y if mv.y is not None else py
        pz = mv.z if mv.z is not None else pz
        if not in_rw or None in (px, py, pz) or n_dep == 0:
            continue
        checked += 1
        p = np.array([px, py, pz], dtype=float)
        nb = [j for j in tree.query_ball_point(p, r=radius) if j < n_dep]
        if not nb:
            continue
        q = pts[np.array(nb)] - p
        d2 = np.einsum("ij,ij->i", q, q)
        # 이 이동이 실제로 훑는 C 구간만 본다. C 가 안 변하면 자세 하나다.
        lo, hi = (c_from, c_cur) if c_from <= c_cur else (c_cur, c_from)
        n_a = 1 if abs(hi - lo) < 1e-9 else max(2, n_angles)
        for a in range(n_a):
            cc = lo if n_a == 1 else lo + (hi - lo) * a / (n_a - 1)
            d = tool_direction(b_cur, cc)
            u = np.array([-d[0], -d[1], -d[2]])
            dz = q @ u
            horiz = np.sqrt(np.maximum(d2 - dz * dz, 0.0))
            if np.any(hit(dz, horiz)):
                hits += 1
                if worst is None:
                    worst = (float(px), float(py), float(pz), float(cc))
                break

    findings = []
    if hits:
        findings.append(("치명", f"되감기 도중 핫엔드가 출력물을 친다 "
                                 f"({hits}/{checked} 자세, 처음 {worst}). "
                                 f"들어올림(`--v-rewind-clearance`)을 키우거나 "
                                 f"되감기 위치를 옮겨야 한다"))
    return findings, {"rewinds": n_rw, "swept_hits": hits, "checked": checked,
                      "arm_modeled": bool(arm_radius)}
