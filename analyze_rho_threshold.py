"""analyze_rho_threshold.py — ρ 문턱이 왜 1 이 아니라 7.5~8.1 인가.

## 문제

`analyze_blend_ratio.py` 가 `ρ = 블렌드비용/잠재이득` 으로 승패를 17 케이스에서
예외 없이 갈랐다. 그런데 **문턱이 (7.46, 8.14) 다.** "비용이 이득을 넘으면 진다"
라면 1 이어야 자연스러운데 아니다. ρ 는 **순서만 맞히고 그 값의 의미는 없었다.**

## 가설 — 문턱은 물리가 아니라 우리가 J 에 넣은 가중치다

출하 선택기 `select_banded_j` 는 이 J 를 최대화하고, 균일 후보를 전수 평가해
섞는다(그래서 '균일보다 나쁠 수 없다'). 밴드가 균일을 이기려면:

    J_band > J_uniform
    (S_uni − S_band) + k·(θ̄_uni − θ̄_band) > k_blend · cost
    ⟺  ρ = cost/ΔS  <  (1 + k·Δθ̄/ΔS) / k_blend

즉 **ρ* ≈ 1/k_blend** 이어야 한다. 기본값 `k_blend=0.1` → 예측 **10**.
실측 7.5~8.1 은 같은 크기다 (아래로 어긋나는 이유는 ⓐ ΔS 를 '블렌드를 무시한
이상적' 값으로 재서 실제 실현되는 이득보다 크고 ⓑ 각도항이 밴드 쪽에 유리해서다).

## 시험

ρ 예측기는 `k_blend` 를 **쓰지 않는다**(`select_banded` + `blend_penalty`).
`k_blend` 를 바꾸면 **실측 승패만** 바뀐다. 그래서 같은 모델 표본을 여러 `k_blend`
에서 다시 재고 문턱이 어떻게 움직이는지 본다.

    · ρ*·k_blend 가 대략 상수  → 가설 참. 문턱은 우리 하이퍼파라미터의 되읽기다.
      그러면 ρ' = k_blend·ρ 로 재정의했을 때 **문턱이 1 이 된다** — "왜 1 이 아닌가"
      가 닫힌다. (디플레이션적 결론이지만 닫힌 것은 닫힌 것이다.)
    · 문턱이 k_blend 와 무관하게 버티면 → 가설 거짓. 문턱에 다른 출처가 있다.

⚠ 이 스크립트는 '문턱의 값' 을 설명하려는 것이지 ρ 의 분리 능력을 다시 재는 게
  아니다. 분리 자체는 `analyze_blend_ratio.py` 가 담당한다.

실행:
    python3 analyze_rho_threshold.py                     # 기본 k_blend 4 개
    python3 analyze_rho_threshold.py --k-blend 0.1 0.2
    python3 analyze_rho_threshold.py --json out.json
"""

import argparse
import json
import sys
import time

import numpy as np

from analyze_blend_ratio import build_specs, predictors
from compare_waist import mean_abs_angle, run_pipeline
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.varangle import select_banded_j

DEFAULT_KB = (0.05, 0.1, 0.2, 0.4)


def outcome(mesh, rp, r_max, k, k_blend):
    """이 k_blend 에서 균일(n=1) vs 밴드2(n=2) 를 툴패스로 재고 파레토 우세 판정."""
    got = {}
    for n in (1, 2):
        r = select_banded_j(mesh, k, n, r_max, rp, MAX_SPACING_FACTOR,
                            k_blend=k_blend)
        prof = r["profile_obj"]
        _p, _t, _f, oh, _i, _g = run_pipeline(mesh, prof, breakdown=True)
        got[n] = (oh, mean_abs_angle(mesh, prof), [round(t, 1) for t in r["thetas"]])
    oh1, a1, th1 = got[1]
    oh2, a2, th2 = got[2]
    return dict(uniform_oh=oh1, uniform_angle=a1, uniform_thetas=th1,
                banded_oh=oh2, banded_angle=a2, banded_thetas=th2,
                pareto=bool(oh2 < oh1 and a2 <= a1 + 1e-9))


def threshold(rows, kb):
    """이 k_blend 에서 ρ 의 승/패 경계 구간. 겹치면 separates=False."""
    win = [r["rho"] for r in rows if r["outcomes"][kb]["pareto"]]
    lose = [r["rho"] for r in rows if not r["outcomes"][kb]["pareto"]]
    if not win or not lose:
        return None
    lo, hi = max(win), min(lose)
    mid = float(np.sqrt(lo * hi)) if (np.isfinite(hi) and lo > 0) else float(lo)
    return dict(separates=bool(lo < hi), low=float(lo),
                high=(float(hi) if np.isfinite(hi) else float("inf")),
                mid=mid, n_win=len(win), n_lose=len(lose))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-blend", type=float, nargs="+", default=list(DEFAULT_KB))
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="rho_threshold_results.json")
    args = ap.parse_args()

    kbs = list(args.k_blend)
    specs = build_specs(asym=True)
    rows = []

    for name, build in specs:
        mesh = build()
        p = predictors(mesh, args.k)
        rp, r_max = p.pop("_rp"), p.pop("_r_max")
        rec = dict(name=name, rho=p["rho"], cost=p["cost"], dS=p["dS_ideal"],
                   prominence=p["prominence"], spread=p["spread"], outcomes={})
        for kb in kbs:
            t0 = time.time()
            rec["outcomes"][kb] = outcome(mesh, rp, r_max, args.k, kb)
            rec["outcomes"][kb]["seconds"] = round(time.time() - t0, 1)
        marks = "".join("승" if rec["outcomes"][kb]["pareto"] else "패" for kb in kbs)
        print(f"{name:<14} ρ={rec['rho']:>8.2f}   k_blend별 판정 {marks}", flush=True)
        rows.append(rec)

    print()
    print(f"{'k_blend':>9}{'승':>4}{'패':>4}{'문턱 하한':>10}{'문턱 상한':>10}"
          f"{'기하중앙 ρ*':>12}{'ρ*·k_blend':>12}   분리")
    print("-" * 72)
    summary = []
    for kb in kbs:
        t = threshold(rows, kb)
        if t is None:
            print(f"{kb:>9.2f}   — 한쪽 그룹이 비어 판정 불가")
            continue
        prod = t["mid"] * kb
        summary.append((kb, t, prod))
        print(f"{kb:>9.2f}{t['n_win']:>4}{t['n_lose']:>4}{t['low']:>10.2f}"
              f"{t['high']:>10.2f}{t['mid']:>12.2f}{prod:>12.2f}"
              f"   {'O' if t['separates'] else 'X 겹침'}")

    if len(summary) >= 2:
        prods = [p for _, _, p in summary]
        kb_lo, kb_hi = summary[0][0], summary[-1][0]
        print("\n[가설 판정]  ρ* ≈ 1/k_blend 인가")
        print(f"  k_blend 를 {kb_lo:g} → {kb_hi:g} 로 {kb_hi/kb_lo:.0f}배 올렸을 때")
        print(f"  ρ* 는 {summary[0][1]['mid']:.2f} → {summary[-1][1]['mid']:.2f} "
              f"({summary[0][1]['mid']/summary[-1][1]['mid']:.1f}배 감소)")
        print(f"  ρ*·k_blend = {', '.join(f'{p:.2f}' for p in prods)}")
        print(f"    → 변동 {min(prods):.2f} ~ {max(prods):.2f} "
              f"({max(prods)/min(prods):.2f}배)")
        print("\n  이 곱이 대략 상수면 문턱은 물리가 아니라 **k_blend 의 되읽기**다.")
        print("  그러면 ρ' = k_blend·ρ 로 재정의했을 때 문턱이 1 근처가 된다.")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1,
                  default=str)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
