"""
motion.py — 축별 속도·가속도 한계를 넣은 출력 시간 모델. [플랫폼 비교]

왜 필요한가: `sync.py` 의 시간 추정은 **가속도를 무시한 하한**이다 ("이 피드면
이만큼 걸린다"). 실제 프린터는 구간마다 가속하고 감속하며, 5축에서는 회전축이
그 병목이 되기 쉽다. 하드웨어 후보를 '속도'로 비교하려면 그 부분을 넣어야 한다.

무엇을 하나: G-code 이동열에 **사다리꼴 운동 계획(look-ahead)** 을 돌린다.
표준적인 방식이고, 펌웨어(Marlin/RRF/Klipper)가 하는 것의 단순화다.

  ① 구간마다 축별 한계로 정상 속도 `v_nom` 을 깎는다
        v_nom ≤ max_vel[a] · L / |Δa|      (각 축 a 에 대해)
  ② 구간의 가속도도 같은 방식으로 깎는다
        a_seg = min_a( max_accel[a] · L / |Δa| )
  ③ 접합부 속도는 **축별 저크**(순간 속도 변화 허용치)로 정한다
        v_j = min_a( jerk[a] / |u₂[a] − u₁[a]| )
     방향이 뒤집히는 축이 있으면 |Δu| 가 커져 v_j 가 작아진다 — 그래서 축 근처
     방위각 반전처럼 V 가 확 꺾이는 자리에서 저절로 느려진다.
  ④ 역방향·정방향 패스로 진입/이탈 속도를 실현 가능하게 만든 뒤 구간 시간을 적분

⚠ **정직**: 이것은 펌웨어 플래너의 **근사**다. 실제 기계는 이것과 다르다 —
  입력 셰이핑, S-커브, 세그먼트 합치기(arc/smoothing), 선입 버퍼 길이, 스텝
  생성 한계가 전부 빠져 있다. 그래서 **절대 시간의 정확도를 주장하지 않는다.**
  쓰는 방법은 **같은 툴패스를 두 기계 프로파일로 돌려 비교하는 것**이고,
  거기서 나오는 '어느 축이 병목인가'가 이 모델의 쓸모다.

⚠ 축 토크·관성도 안 본다. "이 가속도가 물리적으로 가능한가"는 기계 쪽 질문이다.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AxisLimits:
    """축별 운동 한계. 단위는 선형축 mm, 회전축 deg (둘 다 초 기준).

    `feed_axes`: 명령 F 가 적용되는 축들. RRF 는 U/V 를 '선형'으로 취급해 노름에
    넣고(`MachineProfile.uv_linear_feed`), Marlin 은 보통 E 를 노름에서 뺀다.
    이 선택이 시간에 직접 영향을 주므로 프로파일마다 명시한다.
    """

    max_vel: dict = field(default_factory=dict)
    max_accel: dict = field(default_factory=dict)
    jerk: dict = field(default_factory=dict)
    feed_axes: tuple = ("X", "Y", "Z", "V")
    name: str = "unnamed"

    def vel(self, axis):
        return self.max_vel.get(axis, math.inf)

    def acc(self, axis):
        return self.max_accel.get(axis, math.inf)

    def jrk(self, axis):
        return self.jerk.get(axis, math.inf)


_REGION = re.compile(r";\s*([A-Z_][A-Z0-9_]*)\s+(BEGIN|END)\b", re.I)


def _displacements(items, rot_axis="V", tilt_axis="U"):
    """이동열 → 구간별 (축 변위 dict, 명령 피드 mm/s, Move, 구역 이름).

    모달 좌표(지정 안 된 축은 이전 값 유지)를 풀어 절대 위치를 추적한 뒤 차분한다.

    **구역**: `; NAME BEGIN` / `; NAME END` 주석 사이의 이동에 그 이름을 붙인다.
    `add_v_rewinds` 가 박는 `; V_REWIND BEGIN` 이 이것이다 — 되감기에 쓰는 시간을
    출력 시간과 **섞어서 보고하면 안 되기 때문**이다. 되감기는 재료를 놓지 않는
    부대비용이고, 기계(REP5X 처럼 연속 회전이면 0)에 따라 통째로 사라진다.
    """
    pos = {"X": None, "Y": None, "Z": None, "E": None,
           rot_axis: None, tilt_axis: None}
    feed = None
    region = ""
    out = []
    for kind, p in items:
        if kind != "move":
            m = _REGION.search(p) if isinstance(p, str) else None
            if m:
                region = m.group(1).upper() if m.group(2).upper() == "BEGIN" else ""
            continue
        new = dict(pos)
        for letter, val in (("X", p.x), ("Y", p.y), ("Z", p.z), ("E", p.e)):
            if val is not None:
                new[letter] = val
        for tok in (p.extra or "").split():
            for letter in (rot_axis, tilt_axis):
                if tok.upper().startswith(letter.upper()):
                    try:
                        new[letter] = float(tok[len(letter):])
                    except ValueError:
                        pass
        if p.f is not None:
            feed = p.f / 60.0                    # mm/min → mm/s
        disp = {}
        for letter, v in new.items():
            if v is not None and pos.get(letter) is not None:
                d = v - pos[letter]
                if abs(d) > 1e-12:
                    disp[letter] = d
        if disp and feed:
            out.append((disp, feed, p, region))
        pos = new
    return out


def plan_motion(items, limits, rot_axis="V", tilt_axis="U"):
    """사다리꼴 운동 계획 → 구간별 시간과 병목 축.

    반환 dict:
        seconds       총 시간 (s)
        n_segments    구간 수
        times         구간별 시간 (배열)
        binding       구간별 속도를 깎은 축 (없으면 "" = 명령 피드 그대로)
        binding_time  축별로 '그 축이 병목이었던' 시간 합 (dict)
        region_time   구역별 시간 합 (`""` = 보통 출력, `"V_REWIND"` = 되감기)
        print_seconds 되감기 등 표시된 구역을 **뺀** 시간 — 이게 '출력 시간'이다
        v_nom, v_peak 구간별 정상/실제 최고 속도
    """
    segs = _displacements(items, rot_axis, tilt_axis)
    if not segs:
        return {"seconds": 0.0, "print_seconds": 0.0, "n_segments": 0,
                "times": np.zeros(0), "binding": [], "binding_time": {},
                "region_time": {}, "regions": [], "v_nom": np.zeros(0),
                "v_peak": np.zeros(0)}

    L, v_nom, a_seg, units, binding = [], [], [], [], []
    regions = [seg[3] for seg in segs]
    for disp, feed, _, _region in segs:
        # 명령 F 가 적용되는 거리 (노름). 그 축이 하나도 안 움직이면(순수 E 이동
        # 등) 움직인 축의 최대 변위를 대신 쓴다 — 시간이 0 이 되지 않게.
        norm = math.sqrt(sum(d * d for a, d in disp.items()
                             if a in limits.feed_axes))
        Lf = norm if norm > 1e-12 else max(abs(d) for d in disp.values())
        u = {a: d / Lf for a, d in disp.items()}

        v = feed
        who = ""
        for a, ua in u.items():
            cap = limits.vel(a) / abs(ua) if abs(ua) > 1e-12 else math.inf
            if cap < v:
                v, who = cap, a
        acc = min((limits.acc(a) / abs(ua) for a, ua in u.items()
                   if abs(ua) > 1e-12), default=math.inf)
        L.append(Lf)
        v_nom.append(v)
        a_seg.append(acc)
        units.append(u)
        binding.append(who)

    L = np.asarray(L)
    v_nom = np.asarray(v_nom)
    a_seg = np.asarray(a_seg)
    n = len(L)

    # 접합부 속도: 축별 저크로 정한다 (Marlin 고전 저크의 단순형).
    v_junc = np.zeros(n + 1)                     # v_junc[i] = 구간 i 진입 속도 상한
    for i in range(1, n):
        u1, u2 = units[i - 1], units[i]
        lim = math.inf
        for a in set(u1) | set(u2):
            du = abs(u2.get(a, 0.0) - u1.get(a, 0.0))
            if du > 1e-12:
                lim = min(lim, limits.jrk(a) / du)
        v_junc[i] = min(lim, v_nom[i - 1], v_nom[i])
    # 시작과 끝은 정지
    v_junc[0] = 0.0
    v_junc[n] = 0.0

    # 역방향 패스: 감속으로 도달 가능한 진입 속도로 낮춘다
    for i in range(n - 1, -1, -1):
        reach = math.sqrt(v_junc[i + 1] ** 2 + 2.0 * a_seg[i] * L[i]) \
            if math.isfinite(a_seg[i]) else math.inf
        v_junc[i] = min(v_junc[i], reach)
    # 정방향 패스: 가속으로 도달 가능한 이탈 속도로 낮춘다
    for i in range(n):
        reach = math.sqrt(v_junc[i] ** 2 + 2.0 * a_seg[i] * L[i]) \
            if math.isfinite(a_seg[i]) else math.inf
        v_junc[i + 1] = min(v_junc[i + 1], reach)

    times = np.zeros(n)
    v_peak = np.zeros(n)
    for i in range(n):
        ve, vx, vmax, a, d = v_junc[i], v_junc[i + 1], v_nom[i], a_seg[i], L[i]
        if not math.isfinite(a):
            v_peak[i] = vmax
            times[i] = d / vmax if vmax > 0 else 0.0
            continue
        # 삼각/사다리꼴: 도달 가능한 최고 속도
        vp = math.sqrt(max(0.0, (2.0 * a * d + ve * ve + vx * vx) / 2.0))
        vp = min(vp, vmax)
        vp = max(vp, ve, vx)
        d_acc = max(0.0, (vp * vp - ve * ve) / (2.0 * a))
        d_dec = max(0.0, (vp * vp - vx * vx) / (2.0 * a))
        d_cru = max(0.0, d - d_acc - d_dec)
        t = (vp - ve) / a + (vp - vx) / a + (d_cru / vp if vp > 0 else 0.0)
        v_peak[i] = vp
        times[i] = t

    bt, rt = {}, {}
    for who, reg, t in zip(binding, regions, times, strict=True):
        bt[who or "(명령 피드)"] = bt.get(who or "(명령 피드)", 0.0) + float(t)
        rt[reg] = rt.get(reg, 0.0) + float(t)

    total = float(times.sum())
    return {"seconds": total, "print_seconds": rt.get("", 0.0),
            "n_segments": n, "times": times,
            "binding": binding, "binding_time": bt,
            "region_time": rt, "regions": regions,
            "v_nom": v_nom, "v_peak": v_peak}


def limits_from_spec(name, xy_vel, xy_acc, z_vel, z_acc, rot_vel, rot_acc,
                     e_vel=120.0, e_acc=2000.0, xy_jerk=8.0, z_jerk=0.4,
                     rot_jerk=5.0, e_jerk=5.0, rot_axis="V",
                     rot_in_feed_norm=True):
    """익숙한 항목으로 `AxisLimits` 를 만든다 (슬라이서 설정과 같은 모양).

    `rot_in_feed_norm`: RRF 처럼 회전축이 F 노름에 들어가나. Marlin/Klipper 처럼
    빠지면 False — 그 경우 회전만 하는 이동의 시간이 달라진다.
    """
    feed_axes = ("X", "Y", "Z") + ((rot_axis,) if rot_in_feed_norm else ())
    return AxisLimits(
        name=name,
        max_vel={"X": xy_vel, "Y": xy_vel, "Z": z_vel, rot_axis: rot_vel,
                 "E": e_vel},
        max_accel={"X": xy_acc, "Y": xy_acc, "Z": z_acc, rot_axis: rot_acc,
                   "E": e_acc},
        jerk={"X": xy_jerk, "Y": xy_jerk, "Z": z_jerk, rot_axis: rot_jerk,
              "E": e_jerk},
        feed_axes=feed_axes)
