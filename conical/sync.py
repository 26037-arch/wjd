"""
sync.py — 5축 회전 요구량과 동기화 예산. [플랫폼 비교용]

왜 필요한가: 하드웨어 후보가 둘이 됐다 — REP5X(개조 Marlin, 펌웨어가 역기구학)와
뱀부랩 개조(원본 펌웨어 유지 + **아두이노/ESP32 가 회전축만 따로 제어**). 둘의
'성능/속도'를 비교하려면 먼저 **회전축에 무엇이 요구되는지**를 알아야 한다.

이 모듈은 기계가 없어도 G-code 만으로 그 요구량을 잰다.

핵심 세 가지:

**① 회전 각속도는 축 근처에서 발산한다.**
    베드 회전식에서 V 는 압출점의 방위각을 따라간다: V = −φ(p).
    둘레를 속도 f 로 그리면 dφ/dt = f/r 이므로 **ω = f/r** 이고, 축에 가까울수록
    (r→0) 무한대로 간다. 실측: funnel 20° 최대 7,396 deg/s, lamp 24° 최대 18,211.

**② 그래서 슬라이서가 피드를 깎아야 한다 — 그 대가가 플랫폼 비교의 실체다.**
    Open5x(RRF)는 회전이 같은 보간 이동에 묶여 있어 펌웨어가 전체를 느리게 만들어
    맞춘다. 외부 컨트롤러(ESP32)에는 그 플래너가 없으므로 슬라이서가 미리 깎아야
    한다. 어느 쪽이든 **축의 최대 각속도가 출력 시간을 정한다.**

**③ 동기 오차는 '경로 수직 성분'만 치수 오차가 된다.**
    베드가 δ(rad) 지각하면 부품 반경 r 인 점의 퇴적 위치가 접선 방향으로 r·δ 만큼
    어긋난다. ω = f/r 이므로 지각 Δt 에 대해 **r·ω·Δt = f·Δt** — r 이 약분된다.
    즉 오차는 '지각한 시간 동안 노즐이 간 거리'다. 그중 경로와 나란한 성분은
    비교적 무해하고(경로 위에서 조금 앞뒤로 밀릴 뿐), **경로에 수직인 성분이
    그대로 치수 오차**가 된다. 그래서 분해해서 본다.

⚠ 여기서 재는 것은 **기구학 요구량**이지 실제 시간이 아니다. 가속도 한계·저크·
  입력 셰이핑을 무시하므로 출력 시간은 **하한**이다. 축 토크와 관성도 안 본다 —
  '이 각속도가 물리적으로 가능한가'는 기계 쪽에서 답할 문제다.
"""

import math

import numpy as np


def _axis_value(move, letter):
    for tok in (move.extra or "").split():
        if tok.upper().startswith(letter.upper()):
            try:
                return float(tok[len(letter):])
            except ValueError:
                return None
    return None


def rotary_demand(real_items, machine_items, rot_axis="V"):
    """부품좌표 경로 + 기계좌표 경로 → 구간별 회전 요구량.

    두 목록은 `to_open5x` 가 이동 하나당 이동 하나를 내므로 1:1 로 짝지어진다.
    (짝이 안 맞으면 ValueError — 조용히 어긋난 채로 계산하지 않는다.)

    반환 dict (모두 같은 길이의 배열):
        L        구간 길이 (부품좌표, mm)
        dt       구간 소요 시간 (s). 기계 피드 재조정이 시간을 보존하므로
                 부품좌표 길이 ÷ 부품 피드로 구한다
        dV       회전 변화량 (deg)
        omega    요구 각속도 (deg/s)
        r        구간 끝점의 부품 반경 (mm)
        perp     접선 오차 중 **경로에 수직인** 비율 (0~1)
        extruding 압출 구간인가
    """
    rm = [p for k, p in real_items if k == "move"]
    mm = [p for k, p in machine_items if k == "move"]
    if len(rm) != len(mm):
        raise ValueError(f"이동 수가 다르다: 부품 {len(rm)} vs 기계 {len(mm)} — "
                         "같은 파이프라인에서 나온 짝인지 확인할 것")

    px = py = pz = None
    v_prev = None
    e_prev = 0.0
    out = {k: [] for k in ("L", "dt", "dV", "omega", "r", "perp", "extruding")}

    for rp, mp in zip(rm, mm, strict=True):
        nx = rp.x if rp.x is not None else px
        ny = rp.y if rp.y is not None else py
        nz = rp.z if rp.z is not None else pz
        v = _axis_value(mp, rot_axis)
        extruding = rp.e is not None and rp.e > e_prev + 1e-9

        if (None not in (px, py, pz, nx, ny, nz) and v is not None
                and v_prev is not None and rp.f):
            L = math.dist((px, py, pz), (nx, ny, nz))
            if L > 1e-9:
                dt = L / (rp.f / 60.0)
                r = math.hypot(nx, ny)
                dV = abs(v - v_prev)
                # 접선 방향 t̂ = ẑ × p̂ ; 경로 방향과의 각으로 수직 성분을 뺀다.
                dx, dy = nx - px, ny - py
                dn = math.hypot(dx, dy)
                if dn > 1e-12 and r > 1e-9:
                    along = abs((-ny * dx + nx * dy) / (r * dn))
                    perp = math.sqrt(max(0.0, 1.0 - along * along))
                else:
                    perp = 1.0          # 방향을 모르면 최악으로 본다
                out["L"].append(L)
                out["dt"].append(dt)
                out["dV"].append(dV)
                out["omega"].append(dV / dt if dt > 0 else 0.0)
                out["r"].append(r)
                out["perp"].append(perp)
                out["extruding"].append(extruding)

        if rp.e is not None:
            e_prev = rp.e
        px, py, pz = nx, ny, nz
        if v is not None:
            v_prev = v

    arr = {k: np.asarray(v, dtype=float) for k, v in out.items()
           if k != "extruding"}
    arr["extruding"] = np.asarray(out["extruding"], dtype=bool)
    return arr


def sync_error_budget(demand, delays_ms=(1.0, 5.0, 10.0), extruding_only=True):
    """동기 지각 Δt 당 **경로 수직** 치수 오차 (µm).

    오차 = r·ω·Δt 의 수직 성분. r 이 약분되어 사실상 'Δt 동안 노즐이 간 거리'다.
    압출하지 않는 구간은 표면에 남지 않으므로 기본으로 뺀다.
    """
    m = demand["extruding"] if extruding_only else np.ones_like(demand["omega"],
                                                                dtype=bool)
    if not m.any():
        return {}
    # mm/s: r(mm) × ω(rad/s) × 수직비율
    rate = demand["r"][m] * np.radians(demand["omega"][m]) * demand["perp"][m]
    out = {}
    for d in delays_ms:
        e_um = rate * (d / 1000.0) * 1000.0        # mm → µm
        out[d] = {"median": float(np.median(e_um)),
                  "p99": float(np.percentile(e_um, 99)),
                  "max": float(e_um.max())}
    return out


def feed_cap_cost(demand, caps_deg_s=(180, 360, 720, 1440, 3600)):
    """회전 각속도를 한계 이하로 깎을 때의 출력 시간 배수.

    한계를 넘는 구간은 그 구간의 소요 시간을 `dV/cap` 으로 늘린다(= 피드를 깎는다).
    ⚠ 가속도·저크를 무시한 **하한**이다.
    """
    dt, dV, w = demand["dt"], demand["dV"], demand["omega"]
    t0 = float(dt.sum())
    rows = []
    for cap in caps_deg_s:
        slowed = w > cap
        t = float(np.where(slowed, np.divide(dV, cap, out=np.zeros_like(dV),
                                             where=cap > 0), dt).sum())
        rows.append({"cap": cap, "slowed_frac": float(slowed.mean()),
                     "seconds": t, "ratio": t / t0 if t0 > 0 else float("nan")})
    return {"base_seconds": t0, "rows": rows}


def machine_xy_span(machine_items):
    """기계 XY 가 얼마나 움직이는가 (압출 구간만).

    베드 회전식 원뿔 모드에서는 V = −φ(p) 라 압출점이 **항상 방위각 0** 에 온다 —
    즉 기계 Y 가 0 으로 축퇴하고 XY 경로가 X 축 직선이 된다. 실측으로 확인되면
    '기반 프린터의 XY 속도는 이 모드에서 거의 쓰이지 않는다'는 뜻이고, 그것이
    플랫폼 비교의 결론을 상당 부분 정한다.
    """
    xs, ys, e_prev = [], [], 0.0
    for kind, p in machine_items:
        if kind != "move":
            continue
        if (p.e is not None and p.e > e_prev + 1e-9
                and p.x is not None and p.y is not None):
            xs.append(p.x)
            ys.append(p.y)
        if p.e is not None:
            e_prev = p.e
    if not xs:
        return {"n": 0}
    xs, ys = np.asarray(xs), np.asarray(ys)
    return {"n": len(xs),
            "x_span": float(xs.max() - xs.min()),
            "y_span": float(ys.max() - ys.min()),
            "y_abs_max": float(np.abs(ys).max())}
