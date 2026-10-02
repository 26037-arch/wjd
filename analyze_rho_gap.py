"""analyze_rho_gap.py — ρ 의 승/패 간격을 **깨러 간다**.

## 왜

`analyze_blend_ratio.py` 에서 ρ 가 17 케이스를 예외 없이 갈랐지만 간격이 좁다:

    승 최대 7.457 (허리3 로브0.4)  <  패 최소 8.144 (허리 r=5)      간격비 1.09 배
    cost 단독은 더 좁다: 14.434 < 14.669                            간격비 1.02 배

**표본 하나가 그 사이에 들어오면 깨진다.** 지금은 "아직 안 깨졌다" 에 가깝다.
그래서 **그 창을 겨냥한 표본을 일부러 만들어** 넣는다.

## 문제 — ρ 는 형상 파라미터에 **불연속**이다

목 반경을 4.2 → 4.4 → 4.6 → 4.8 로 옮기면 ρ 가 8.88 → 10.46 → **8.44** → 9.96 로
튄다. 이유: 각도 탐색이 2° 이산이고(`ANGLE_STEP`), 고른 각도가 바뀌면 ΔS 가 점프하고,
블렌드 경계 이동(`BLEND_SHIFT_RATIO`)도 불연속이다.

→ **조준이 안 된다.** 그래서 ρ 를 **촘촘히 훑어(싸다) 창에 드는 것만 줍고**, 비싼
툴패스는 그 후보에만 돌린다.

## 판정 기준 (**결과 보기 전에** 적는다)

* 새 표본이 전부 예측대로(낮은 ρ = 승, 높은 ρ = 패) → **간격이 좁아진다.** ρ 강화
* 승과 패가 ρ 순서로 **뒤섞인다** → **ρ 반증.** 순위 예측기로서 죽는다
* 창에 아무것도 안 들어온다 → 그 자체가 결과다. ρ 가 형상 파라미터에 불연속이라
  **간격을 채울 표본을 만드는 것이 구조적으로 어렵다**는 뜻이고, 그러면 "간격이
  좁다" 는 걱정의 성격이 달라진다(표본을 못 만들면 반증도 어렵다)

⚠ `cost` 단독도 같이 본다 — 간격비가 더 좁으므로(1.02) 먼저 깨질 후보다.

실행:
    python3 analyze_rho_gap.py --scan-only     # ρ 만 훑는다 (빠름)
    python3 analyze_rho_gap.py                 # 창 안 후보를 툴패스로 측정
"""

import argparse
import json
import sys
import time

import numpy as np

from analyze_blend_ratio import lobe, predictors, widen
from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.varangle import select_banded_j

# 기존 17 케이스의 간격 (analyze_blend_ratio.py --measure --asym)
GAP_RHO = (7.457, 8.144)
GAP_COST = (14.434, 14.669)
# 창은 간격보다 넉넉히 — 경계 바로 밖도 간격을 좁히는 데 쓸모가 있다.
WINDOW_RHO = (6.8, 9.0)


def axes():
    """세 축을 촘촘히. 각각 '다른 방식으로' ρ 를 올리므로 한 축이 깨면 충분하다."""
    b3 = waisted_model(3.0)
    for r in np.arange(3.6, 7.01, 0.1):
        yield ("목반경", round(float(r), 2), lambda r=r: waisted_model(float(r)))
    for lam in np.arange(1.15, 1.351, 0.01):
        yield ("λ", round(float(lam), 3), lambda lam=lam: widen(b3, float(lam)))
    for a in np.arange(0.30, 0.601, 0.02):
        yield ("로브", round(float(a), 3), lambda a=a: lobe(b3, float(a)))


def scan(k):
    """ρ·cost 만 계산 (해석식, 툴패스 없음)."""
    out = []
    for axis, val, build in axes():
        mesh = build()
        p = predictors(mesh, k)
        p.pop("_rp"), p.pop("_r_max")        # 예측기 내부값 — 기록에서 뺀다
        rec = dict(axis=axis, value=val, rho=p["rho"], cost=p["cost"],
                   dS=p["dS_ideal"], blend_mm=p["blend_mm"])
        out.append(rec)
        flag = "★" if WINDOW_RHO[0] <= rec["rho"] <= WINDOW_RHO[1] else ""
        print(f"  {axis:<7}{val:>7}  ρ={rec['rho']:>8.2f}  cost={rec['cost']:>7.2f}"
              f"  ΔS={rec['dS']:>5.2f} {flag}", flush=True)
    return out


def measure(axis, val, k):
    """그 모델을 균일(n=1) vs 밴드2(n=2) 로 툴패스 측정 → 파레토 우세 판정."""
    build = next(b for a, v, b in axes() if a == axis and v == val)
    mesh = build()
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    got = {}
    for n in (1, 2):
        r = select_banded_j(mesh, k, n, r_max, rp, MAX_SPACING_FACTOR)
        prof = r["profile_obj"]
        _p, _t, _f, oh, _i, _g = run_pipeline(mesh, prof, breakdown=True)
        got[n] = (oh, mean_abs_angle(mesh, prof))
    oh1, a1 = got[1]
    oh2, a2 = got[2]
    return dict(uniform_oh=oh1, uniform_angle=a1, band_oh=oh2, band_angle=a2,
                pareto=bool(oh2 < oh1 and a2 <= a1 + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="rho_gap_results.json")
    args = ap.parse_args()

    print(f"[1] ρ 를 촘촘히 훑는다 (창 {WINDOW_RHO} 안이면 ★)")
    rows = scan(args.k)
    inw = [r for r in rows if WINDOW_RHO[0] <= r["rho"] <= WINDOW_RHO[1]]
    ingap = [r for r in rows if GAP_RHO[0] < r["rho"] < GAP_RHO[1]]
    print(f"\n  훑은 표본 {len(rows)}개 · 창 안 {len(inw)}개 · "
          f"**간격 {GAP_RHO} 안 {len(ingap)}개**")
    if not inw:
        print("  → 창에 아무것도 없다. ρ 가 형상 파라미터에 불연속이라 간격을 채울")
        print("    표본을 만드는 것이 구조적으로 어렵다 — 그 자체가 결과다.")

    if args.scan_only or not inw:
        if args.json:
            json.dump(dict(scan=rows, measured=[]), open(args.json, "w"),
                      ensure_ascii=False, indent=1)
        return

    print(f"\n[2] 창 안 {len(inw)}개를 툴패스로 측정")
    print(f"{'축':<7}{'값':>7}{'ρ':>8}{'cost':>8}{'균일':>8}{'밴드2':>8}  판정")
    print("-" * 56)
    for r in sorted(inw, key=lambda x: x["rho"]):
        t0 = time.time()
        m = measure(r["axis"], r["value"], args.k)
        r.update(m)
        print(f"{r['axis']:<7}{r['value']:>7}{r['rho']:>8.2f}{r['cost']:>8.2f}"
              f"{m['uniform_oh']:>8.3f}{m['band_oh']:>8.3f}  "
              f"{'승' if m['pareto'] else '패'}  ({time.time()-t0:.0f}s)", flush=True)

    # ── 판정 ────────────────────────────────────────────────
    print("\n[3] 간격이 좁아졌나, 아니면 뒤섞였나")
    for key, gap, label in (("rho", GAP_RHO, "ρ"), ("cost", GAP_COST, "cost 단독")):
        win = [r[key] for r in inw if r["pareto"]]
        lose = [r[key] for r in inw if not r["pareto"]]
        lo, hi = gap
        new_lo = max([lo] + win)          # 승은 하한을 올린다
        new_hi = min([hi] + lose)         # 패는 상한을 내린다
        broken = new_lo >= new_hi
        print(f"  {label:<10} 기존 간격 ({lo:.3f}, {hi:.3f})  "
              f"새 표본 승 {sorted(round(v,2) for v in win)} / "
              f"패 {sorted(round(v,2) for v in lose)}")
        if broken:
            print(f"    → ⚠⚠ **뒤섞였다. {label} 반증됨.** "
                  f"승 최대 {new_lo:.3f} ≥ 패 최소 {new_hi:.3f}")
        else:
            ratio = new_hi / new_lo if new_lo > 0 else float("nan")
            print(f"    → 유지. 새 간격 ({new_lo:.3f}, {new_hi:.3f})  간격비 {ratio:.3f} 배"
                  + ("  ← 더 좁아졌다" if ratio < hi / lo else ""))

    if args.json:
        json.dump(dict(scan=rows, measured=inw), open(args.json, "w"),
                  ensure_ascii=False, indent=1, default=str)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
