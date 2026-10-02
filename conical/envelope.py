"""
envelope.py — 5축 원뿔 모드의 도달 가능 영역과 최대 각도. [플랫폼 비교]

왜 필요한가: 3축 원뿔의 최대 각도는 **노즐이 출력물을 치는 것**이 정했다
(`find_max_safe_angle.py`, 검사기 B). 5축에서는 그 한계가 **구조적으로 사라지고**
전혀 다른 것이 한계가 된다 — 기계가 그 자세에 **도달할 수 있느냐**다. 다음 달
기계를 만들기 전에, 무엇이 각도를 막을지 알아야 설계에 넣을 수 있다.

## 기계좌표 해석식 (`open5x._map_point` 과 수치 일치, `tests/test_envelope.py`)

압출점은 항상 방위각 0 으로 회전돼 오므로 부품점 (r, z) 하나로 결정된다:

    x_m = cosθ·r − sinθ·(z + d)
    z_m = sinθ·r + cosθ·(z + d) − d          (d = pivot_depth, 베드면→틸트축)

**둘 다 (r, z) 에 선형이다.** 그래서 부품 전체의 기계좌표 범위는 정점만 보면
정확히 나오고, 각도 스윕에 슬라이싱이 필요 없다.

## ① 원뿔 레이어 위에서는 r 이 약분된다

원뿔 레이어는 `z = z₀ − r·tanθ` 다 (워프 `z' = z + r·tanθ` 의 역, **outward**). 대입하면

    z_m = sinθ·r + cosθ·(z₀ − r·tanθ + d) − d = **cosθ·(z₀ + d) − d**

r 이 사라진다. **레이어 하나가 기계 Z 값 하나**이고, z₀ 가 커지면 z_m 도 커지므로
(θ<90°) 압출 순서대로 기계 Z 가 **단조증가**한다.

→ **기계공간에서 5축 원뿔 출력은 그냥 평면 적층이다** (층고 `h·cosθ`).
  이미 놓인 것이 전부 노즐 팁보다 아래거나 같은 높이다. **노즐-출력물 간섭이
  정의상 불가능**하다. 말로만 두면 안 되는 주장이라 실제 G-code 로 센다:
  `check_planar_stacking`.

⚠ 이 성질이 보장하지 **않는** 것:
  · 노즐 몸체가 **옆으로** 부딪히는 것 — 현재 레이어는 노즐 팁과 같은 높이라
    위로는 괜찮지만, 부품이 안쪽으로 파여 있으면 별개 문제다
  · **트래블 도중**의 자세 — V 가 슬루하는 중간 각도에서는 부품이 다른 방향을
    향한다. 되감기 들어올림이 그 대책이고, V 179° 급회전은 미해결
  · 베드·클램프·프레임과의 충돌 — 기계 CAD 가 있어야 본다

## ② 스팬에서는 피벗 깊이 d 가 약분된다 ← 설계에 쓸 결론

d 는 x_m, z_m 을 **평행이동만** 시킨다. 그래서 **필요한 이동 거리(스팬)** 에는
d 가 안 들어간다. 부품 (r, z) 볼록껍질에 대해

    X 스팬 = max(cosθ·r − sinθ·z) − min(…)      (상자 근사: cosθ·R + sinθ·H)
    Z 스팬 = max(sinθ·r + cosθ·z) − min(…)      (상자 근사: sinθ·R + cosθ·H)

**틸트는 높이를 X 이동거리로 바꾼다.** H=100mm 부품을 45° 로 기울이면 X 가
71mm 필요하다 — 3축에서는 0 이던 값이다.

d 가 정하는 것은 **위치**다: 베드 중심이 x_m = −sinθ·d 로 간다. 즉 **회전 베드를
X 이동 중앙보다 sinθ·d 만큼 +X 쪽에 달아야** 균형이 맞는다 (θ=45°, d=50 → 35mm).
스탠드오프(30/50/70mm)는 우리가 고르는 값이므로 이건 설계 결정이다.

## 그래서 5축의 최대 각도를 정하는 것들

  ① 틸트축 기계 한계
  ② **기계 X 이동** ← 새로 나온 것. 부품 높이 × sinθ
  ③ 기계 Z 이동 (+ 데이텀을 못 옮기면 `−d(1−cosθ)` 만큼 음수로 내려간다)
  ④ 베드 반경 ≥ 부품 반경 (부품이 한 바퀴 쓸고 지나간다)

⚠ 이 모듈은 기계 **제원을 모른다.** `MachineEnvelope` 값은 전부 사용자가 넣는
  것이고 기본 프로파일은 **가정**이다. 쓸모는 '이 각도에 이만큼이 필요하다'는
  **요구량**에 있다 — 실기 제원이 나오면 그때 판정이 된다.

⚠ 여기서 나온 각도는 '기계가 못 하는 곳'의 상한이지 **'써야 할 각도'가 아니다.**
  서포트가 줄어드는가(J), 층간격이 버티는가(`MAX_SPACING_FACTOR`), 표면 품질은
  전부 별개 질문이다.
"""

import math
from dataclasses import dataclass

import numpy as np

from .open5x import PRUSA_UV


@dataclass
class MachineEnvelope:
    """기계가 도달할 수 있는 범위. ⚠ 전부 실기에서 재야 하는 값이다.

    `datum`:
      "free"  — 회전 베드를 X 어디에 달지, 기계 Z 영점을 어디로 둘지 **아직
                고를 수 있다**. 그러면 **스팬만** 맞으면 된다 (d 가 약분된다).
                기계를 만들기 **전**인 지금이 이 경우다.
      "fixed" — Open5x 관례대로 원점 = 회전 베드 중심, 기계 Z=0 = 베드면.
                지금 `to_open5x` 가 내는 G-code 를 그대로 걸 때의 판정이다.
    """

    x_travel: float = 250.0             # X 이동 가능 거리 (스팬)
    z_travel: float = 210.0             # Z 이동 가능 거리 (스팬)
    bed_radius: float = 90.0
    max_tilt_deg: float = 90.0
    x_min: float = -125.0               # datum="fixed" 에서만 쓴다
    x_max: float = 125.0
    z_min: float = 0.0
    z_max: float = 210.0
    datum: str = "free"                 # "free" | "fixed" (위 설명 참고)
    name: str = "unnamed"


# ⚠ 가정값. Prusa i3 MK3S 조형공간(250×210×210)에 원점만 베드 중심으로 옮긴 것.
#   Open5x 의 회전 베드는 틸트 크레이들 위에 얹히므로 **실제 Z 는 이보다 작다.**
#   소프트리밋 z_min=0 은 펌웨어 기본값 가정이다.
PRUSA_I3_ENV = MachineEnvelope(name="Prusa i3 (가정)")


def machine_xz(r, z, theta_deg, prof=PRUSA_UV):
    """부품점 (r, z) → 기계 (x_m, z_m). 배열도 받는다."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    zd = np.asarray(z, dtype=float) + prof.pivot_depth
    return c * np.asarray(r, dtype=float) - s * zd, \
        s * np.asarray(r, dtype=float) + c * zd - prof.pivot_depth


def layer_machine_z(z0, theta_deg, prof=PRUSA_UV):
    """원뿔 레이어(축 높이 z₀) 전체의 기계 Z. **r 과 무관한 한 값**이다."""
    th = math.radians(theta_deg)
    return math.cos(th) * (z0 + prof.pivot_depth) - prof.pivot_depth


def part_samples(mesh):
    """메시 정점 → (r, z). 사상이 선형이라 정점만으로 범위가 정확히 나온다."""
    v = np.asarray(mesh.vertices, dtype=float)
    return np.hypot(v[:, 0], v[:, 1]), v[:, 2]


def requirement(r, z, theta_deg, prof=PRUSA_UV):
    """이 각도에서 **기계가 내줘야 하는 것**. 판정이 아니라 요구량이다.

    `bed_center_x` = 회전 베드 중심의 기계 X (= −sinθ·d). 베드를 X 이동 중앙에서
    이만큼 옮겨 달아야 한다는 뜻이다.
    """
    xm, zm = machine_xz(r, z, theta_deg, prof)
    th = math.radians(theta_deg)
    return {"x_min": float(np.min(xm)), "x_max": float(np.max(xm)),
            "z_min": float(np.min(zm)), "z_max": float(np.max(zm)),
            "x_span": float(np.ptp(xm)), "z_span": float(np.ptp(zm)),
            "radius_max": float(np.max(r)),
            "bed_center_x": -math.sin(th) * prof.pivot_depth,
            "bed_sink": -prof.pivot_depth * (1.0 - math.cos(th))}


def constraints_at(r, z, theta_deg, env, prof=PRUSA_UV):
    """이 각도에서 **무엇이 막는가**. 반환: (통과여부, 막는 것 목록, 요구량)."""
    req = requirement(r, z, theta_deg, prof)
    blockers = []
    if theta_deg > min(env.max_tilt_deg, prof.max_tilt_deg) + 1e-9:
        blockers.append("틸트축 한계")
    if req["radius_max"] > env.bed_radius + 1e-9:
        blockers.append("베드 반경")
    if env.datum == "free":
        if req["x_span"] > env.x_travel + 1e-9:
            blockers.append("기계 X 이동")
        if req["z_span"] > env.z_travel + 1e-9:
            blockers.append("기계 Z 이동")
    else:
        if req["x_min"] < env.x_min - 1e-9 or req["x_max"] > env.x_max + 1e-9:
            blockers.append("기계 X 도달")
        if req["z_min"] < env.z_min - 1e-9 or req["z_max"] > env.z_max + 1e-9:
            blockers.append("기계 Z 범위")
    return (not blockers), blockers, req


def max_angle(r, z, env, prof=PRUSA_UV, step=0.25, hi=None):
    """도달 가능한 최대 원뿔각과 **그 바로 위에서 무엇이 막는가**.

    제약이 각도에 단조가 아닐 수 있으므로(예: Z 스팬은 한 번 줄었다 는다) 0 부터
    올라가며 **처음 막히는 곳**을 찾는다 — 연속 구간의 상한이라는 뜻이고,
    그 위에 다시 가능한 구간이 있어도 실용적으로는 무의미하다.
    """
    hi = min(env.max_tilt_deg, prof.max_tilt_deg) if hi is None else hi
    best, best_req, first_block = None, None, None
    th = 0.0
    while th <= hi + 1e-9:
        ok, b, req = constraints_at(r, z, th, env, prof)
        if not ok:
            first_block = b
            break
        best, best_req, th = th, req, th + step
    return {"max_angle": best,
            "blockers": first_block or ["(스윕 상한에 도달)"],
            "req": best_req}


def check_planar_stacking(items, tol=1e-6):
    """실제 기계 G-code 가 '기계공간에서 평면 적층' 인지 확인한다.

    이게 참이면 **노즐-출력물 간섭이 정의상 불가능**하다 — 이미 놓인 것이 전부
    노즐 팁보다 아래거나 같은 높이다. 5축이 3축 원뿔보다 나은 이유의 핵심이라
    G-code 로 직접 센다.

    반환 dict: n_layers, layer_dz(중앙 층간격), monotone(압출 중 기계 Z 가 한 번도
               안 내려가는가), max_drop, flat(한 층 안의 기계 Z 편차 최대)
    """
    zs, cz = [], None
    for kind, p in items:
        if kind != "move":
            continue
        if p.z is not None:
            cz = p.z
        if p.e is not None and cz is not None:
            zs.append(cz)
    if not zs:
        return {"n_layers": 0, "layer_dz": 0.0, "monotone": True,
                "max_drop": 0.0, "flat": 0.0}
    zs = np.asarray(zs)
    drops = np.diff(zs)
    uz = np.unique(np.round(zs / tol) * tol)
    dz = np.diff(uz)
    flat = 0.0
    if len(uz) > 1:
        idx = np.searchsorted((uz[:-1] + uz[1:]) / 2.0, zs)
        for k in range(len(uz)):
            grp = zs[idx == k]
            if grp.size:
                flat = max(flat, float(np.ptp(grp)))
    return {"n_layers": int(len(uz)),
            "layer_dz": float(np.median(dz)) if len(dz) else 0.0,
            "monotone": bool(np.all(drops >= -tol)),
            "max_drop": float(-np.min(drops)) if drops.size else 0.0,
            "flat": flat}
