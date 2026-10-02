"""
quality.py — 표면 품질·치수 정확도 중 **기하로 예측되는 부분**. [플랫폼 비교 R5]

⚠⚠ **표면 품질의 상당 부분은 실물 없이는 못 잰다.** 브리징·수축·접착·온도·
  팬·재료는 전부 물리이고 이 프로젝트의 범위 밖이다. 여기서 하는 것은 **기하가
  강제하는 하한**이다 — "이 이상은 좋을 수 없다" 는 부분. 실물이 나오면 이 예측이
  **검증 대상**이 된다(그게 R5 의 실제 쓸모다).

## ① 계단 자국 (cusp height) — 층이 강제하는 표면 오차

레이어가 두께 `d` 로 방향 `b̂` 에 수직하게 쌓일 때, 법선 `n̂` 인 표면의 계단
자국 높이는

    cusp = d · |n̂ · b̂|

  · 수직 벽 (n̂ ⊥ b̂) → 0. 계단이 없다
  · 평평한 윗면 (n̂ ∥ b̂) → d. 있거나 없거나라 최대다

평면 슬라이싱: `b̂ = ẑ`, `d = t` (층고).
원뿔 슬라이싱(outward, 각도 θ): 레이어가 `z = z₀ − r·tanθ` 이므로

    b̂ = n̂_cone = sinθ·r̂(φ) + cosθ·ẑ          (단위벡터)
    d = t·cosθ                                  (이웃 원뿔면 사이 수직거리)

**`cosθ` 가 공짜로 붙는다** — 원뿔 슬라이싱은 같은 층고에서 수직 간격이 `cosθ` 로
좁아진다. 대신 같은 부피를 덮는 데 레이어가 더 든다(출력 시간 ↑, R3).

⚠ **이게 서포트 감소와 상충할 수 있다.** 서포트를 줄이려면 레이어를 표면과
  **나란히** 놓아야 하는데(`n̂ ∥ b̂`), 계단 자국은 그럴 때 **최대**가 된다.
  둘은 같은 방향이 아니다. `analyze_quality.py` 가 이 상충을 실측한다.

⚠ 보지 못하는 것: 압출 비드의 실제 단면, 코너 반경, 다림질, 표면 광택, 그리고
  **비드가 계단을 메우는 효과**. 그래서 여기 값은 '이보다 좋을 수 없다' 지
  '이만큼 나쁘다' 가 아니다.

## ② 회전축 각분해능 → 치수 오차 (5축 고유)

3축에는 없는 오차원이다. 회전축이 `Δφ` 단위로만 움직일 수 있으면, 축에서 `r`
떨어진 점은 접선 방향으로 `r·Δφ` 만큼 양자화된다:

    오차 = r · Δφ(rad) = r / steps_per_deg · π/180

**반경에 비례**하므로 큰 부품의 바깥쪽이 가장 나쁘다. R2 의 동기 오차(`f·Δt`,
반경 무관)와 **다른 성질**이고, 둘 다 있으면 더해진다.

REP5X 실측 제원(`Rep5x-Marlin/Marlin/Configuration.h`):
`DEFAULT_AXIS_STEPS_PER_UNIT { 80, 80, 400, 26.666, 26.68, 415 }` — C축(I) 이
**26.666 steps/deg** 이므로 한 스텝이 **0.0375°** 다.

⚠ 이것은 **명령 분해능**이다. 실제 위치 정확도는 마이크로스텝 선형성·백래시·
  벨트 탄성에 달렸고 그건 실기에서만 잰다. 여기 값은 하한이다.
"""

import math

import numpy as np


def cone_normal(x, y, cone_angle_deg, cone_type="outward"):
    """원뿔 레이어의 국소 법선 `n̂ = sinθ·r̂ + cosθ·ẑ` (배열도 받는다).

    축 위(r≈0)에서는 방위각이 정의되지 않아 `ẑ` 를 준다 — 그 점에서는 레이어가
    수평이므로 맞다.
    """
    th = math.radians(cone_angle_deg) * (1.0 if cone_type == "outward" else -1.0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    out = np.zeros(x.shape + (3,), dtype=float)
    ok = r > 1e-12
    s, c = math.sin(th), math.cos(th)
    out[..., 0] = np.where(ok, s * x / np.where(ok, r, 1.0), 0.0)
    out[..., 1] = np.where(ok, s * y / np.where(ok, r, 1.0), 0.0)
    # 축 위는 원뿔 꼭짓점이라 법선이 정의되지 않는다. 방위 평균을 취하면 반경
    # 성분이 0 이므로 ẑ 가 자연스러운 극한이고, 무엇보다 **단위벡터여야 한다**
    # — cosθ 를 그대로 두면 길이가 cosθ 인 벡터가 나온다 (T22 가 잡았다).
    out[..., 2] = np.where(ok, c, 1.0)
    return out


def cusp_heights(mesh, cone_angle_deg=0.0, cone_type="outward",
                 layer_height=0.3):
    """면마다 계단 자국 높이 `d·|n̂·b̂|` 와 면적. `cone_angle_deg=0` 이면 평면.

    반환 dict: cusp(면별 mm), area(면별 mm²), mean(면적 가중 평균),
               p90(면적 가중 90분위), max, layer_spacing(=d)
    ⚠ 면적 가중이다 — 면 개수 가중이면 잘게 쪼개진 곳이 과대평가된다.
    """
    n = np.asarray(mesh.face_normals, dtype=float)
    area = np.asarray(mesh.area_faces, dtype=float)
    cen = np.asarray(mesh.triangles_center, dtype=float)
    if cone_angle_deg == 0.0:
        d = layer_height
        dot = np.abs(n[:, 2])
    else:
        d = layer_height * math.cos(math.radians(cone_angle_deg))
        b = cone_normal(cen[:, 0], cen[:, 1], cone_angle_deg, cone_type)
        dot = np.abs(np.einsum("ij,ij->i", n, b))
    cusp = d * dot
    return {"cusp": cusp, "area": area, "align": dot, "layer_spacing": d,
            "mean": _wmean(cusp, area), "p90": _wquantile(cusp, area, 0.9),
            "max": float(cusp.max()) if len(cusp) else 0.0,
            "align_mean": _wmean(dot, area)}


def _wmean(v, w):
    tot = w.sum()
    return float((v * w).sum() / tot) if tot > 0 else 0.0


def _wquantile(v, w, q):
    if len(v) == 0 or w.sum() <= 0:
        return 0.0
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w[o].sum()
    return float(v[o][np.searchsorted(cw, q, side="left").clip(0, len(v) - 1)])


def rotary_resolution_error(radii, steps_per_deg=26.666):
    """회전축 한 스텝이 만드는 접선 방향 치수 오차 (µm). 반경에 비례한다.

    `steps_per_deg` 기본값은 REP5X 설정의 C축 값이다. Open5x(Duet)는 다르므로
    그 기계 값을 넣을 것.
    """
    step_deg = 1.0 / steps_per_deg
    err = np.asarray(radii, dtype=float) * math.radians(step_deg) * 1000.0
    return {"step_deg": step_deg, "err_um": err,
            "median": float(np.median(err)) if len(err) else 0.0,
            "max": float(err.max()) if len(err) else 0.0}


def cusp_compare(mesh, cone_angle_deg, cone_type="outward", layer_height=0.3):
    """평면 대 원뿔을 **두 가지 방식으로** 비교한다. 둘을 섞으면 안 된다.

      · **같은 층고**(`layer_height` 그대로): 원뿔이 수직 간격 `t·cosθ` 로 공짜
        이득을 본다. 하지만 같은 부피에 레이어가 더 들어 **출력 시간이 는다**(R3).
        설정값을 그대로 두고 각도만 바꿨을 때 실제로 보게 될 값이다.
      · **정렬만**(`align`, `|n̂·b̂|`): 간격 이득을 뺀 순수 기하. 레이어가 표면과
        나란해질수록 커진다 = 계단이 심해진다. **서포트 감소와 상충하는 항이
        이쪽이다** — 서포트를 줄이려면 레이어를 표면과 나란히 놓아야 하므로.

    `worse_area_pct` 는 **정렬 기준으로 나빠지는 면적 비율**이다. 같은 층고 기준은
    `cosθ` 가 전면적에 깔려 상충을 가려 버린다.
    """
    pl = cusp_heights(mesh, 0.0, cone_type, layer_height)
    co = cusp_heights(mesh, cone_angle_deg, cone_type, layer_height)
    worse = co["align"] > pl["align"] + 1e-12
    tot = pl["area"].sum()
    return {"planar": pl, "conical": co,
            "cusp_ratio": (co["mean"] / pl["mean"]) if pl["mean"] > 0 else 1.0,
            "align_ratio": ((co["align_mean"] / pl["align_mean"])
                            if pl["align_mean"] > 0 else 1.0),
            "worse_area_pct": float(co["area"][worse].sum() / tot * 100.0)
            if tot > 0 else 0.0}
