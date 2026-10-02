"""
analyze_quality.py — 표면 품질·치수 정확도 중 기하가 강제하는 부분. [R5]

    python3 analyze_quality.py                     # 기본 모델, 각도 스윕
    python3 analyze_quality.py model.stl --layer-height 0.2
    python3 analyze_quality.py --steps-per-deg 26.666 --max-radius 100

⚠⚠ **표면 품질의 상당 부분은 실물 없이는 못 잰다.** 브리징·수축·접착·온도·팬·
  재료는 물리이고 범위 밖이다. 여기서 내는 것은 **기하가 강제하는 하한**이고,
  실물이 나오면 이 예측이 **검증 대상**이 된다 (`docs/experiment_plan.md`).

답하는 것:
  ① 각도를 올리면 계단 자국이 어떻게 변하나 (같은 층고 기준)
  ② **서포트 감소와 상충하는가** — 정렬 항만 떼어서 본다
  ③ 회전축 각분해능이 만드는 치수 오차 (5축 고유, 반경에 비례)
"""

import argparse
import math

import matplotlib
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical.analytic import support_fraction
from conical.meshio import center_on_axis
from conical.plotstyle import L as _L
from conical.quality import cusp_compare, cusp_heights, rotary_resolution_error

ANGLES = (10, 20, 30, 45, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--layer-height", type=float, default=0.3)
    ap.add_argument("--direction", choices=["outward", "inward"],
                    default="outward")
    ap.add_argument("--steps-per-deg", type=float, default=26.666,
                    help="회전축 분해능 (REP5X C축 실측 제원 26.666)")
    ap.add_argument("--max-radius", type=float, default=100.0,
                    help="부품 최대 반경 (PRINTABLE_RADIUS)")
    ap.add_argument("--out", default="quality_tradeoff.png")
    args = ap.parse_args()

    jobs = args.models or ["examples/funnel.stl", "examples/lamp.stl"]
    t = args.layer_height
    results = {}

    for stl in jobs:
        mesh = center_on_axis(trimesh.load(stl, force="mesh"))
        name = stl.split("/")[-1]
        pl = cusp_heights(mesh, 0.0, args.direction, t)
        s0 = support_fraction(mesh, 0.0, args.direction)
        print("=" * 78)
        print(f"[{name}]  층고 {t}mm,  평면 기준: 계단 자국 평균 "
              f"{pl['mean'] * 1000:.0f}µm / p90 {pl['p90'] * 1000:.0f}µm, "
              f"서포트 {s0:.1f}%")
        print(f"\n  ①②  {'θ':>3} | {'서포트':>7} | {'계단 평균':>9} "
              f"{'(배수)':>7} | {'정렬 배수':>9} {'정렬 나빠진 면적':>15}")
        rows = []
        for a in ANGLES:
            cc = cusp_compare(mesh, float(a), args.direction, t)
            sup = support_fraction(mesh, float(a), args.direction)
            co = cc["conical"]
            flag = "  ← 상충" if cc["align_ratio"] > 1.0 else ""
            print(f"      {a:>3} | {sup:>6.1f}% | {co['mean'] * 1000:>8.0f}µm "
                  f"{cc['cusp_ratio']:>7.2f} | {cc['align_ratio']:>9.2f} "
                  f"{cc['worse_area_pct']:>14.1f}%{flag}")
            rows.append((a, sup, co["mean"] * 1000, cc["cusp_ratio"],
                         cc["align_ratio"], cc["worse_area_pct"]))
        results[name] = (pl, s0, rows)
        worst = max(rows, key=lambda r: r[4])
        if worst[4] > 1.0:
            print(f"\n  → **상충이 있다.** θ={worst[0]}° 에서 정렬이 "
                  f"{worst[4]:.2f}배 나빠지고 면적의 {worst[5]:.0f}% 가 "
                  f"평면보다 계단이 심해진다.")
            print(f"     같은 층고 기준으로는 `cosθ` 가 전면적에 깔려 이게 "
                  f"가려진다 (계단 평균 배수 {worst[3]:.2f}) — 하지만 그 이득은")
            print("     레이어 수가 늘어나는 대가다(출력 시간, R3). "
                  "층고를 `t·cosθ` 로 낮춘 평면과 비교하면 사라진다.")
        else:
            print("\n  → 이 모델에서는 **상충이 없다.** 어느 각도에서도 정렬이 "
                  "나빠지지 않는다 (얕은 형상이라 평면 슬라이싱이 특히 불리하다).")

    # ③ 회전축 분해능
    print("=" * 78)
    rr = rotary_resolution_error([1.0], args.steps_per_deg)
    print("③ 회전축 각분해능 → 치수 오차 (5축 고유, **반경에 비례**)")
    print(f"   {args.steps_per_deg:g} steps/deg → 한 스텝 {rr['step_deg']:.4f}°")
    radii = [5, 10, 20, 50, args.max_radius]
    errs = rotary_resolution_error(radii, args.steps_per_deg)["err_um"]
    print(f"   {'반경':>8} " + " ".join(f"{r:>7.0f}" for r in radii) + " mm")
    print(f"   {'오차':>8} " + " ".join(f"{e:>7.1f}" for e in errs) + " µm")
    print("   ⚠ R2 의 동기 오차(`f·Δt`)는 **반경과 무관**했다 — 성질이 다르고,")
    print(f"     둘 다 있으면 더해진다. 압출폭 450µm 기준으로 반경 "
          f"{450 / (math.radians(rr['step_deg']) * 1000):.0f}mm 에서 압출폭에 닿는다.")
    print("   ⚠ 이것은 **명령 분해능**이다. 마이크로스텝 선형성·백래시·벨트")
    print("     탄성은 실기에서만 잰다 — 여기 값은 하한이다.")

    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    for name, (_pl, _s0, rows) in results.items():
        a = [r[0] for r in rows]
        ax.plot(a, [r[4] for r in rows], "o-", lw=1.8, label=f"{name} 정렬")
        ax.plot(a, [r[3] for r in rows], "s--", lw=1.4, alpha=.7,
                label=f"{name} 같은 층고")
    ax.axhline(1.0, color="#c0392b", lw=1.2, ls=":")
    ax.set_xlabel(_L("cone angle (deg)", "원뿔각 (도)"))
    ax.set_ylabel(_L("cusp vs planar (x)", "평면 대비 계단 자국 (배)"))
    ax.set_title(_L("above 1.0 = worse than planar",
                    "1.0 위 = 평면보다 나쁨"), fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for name, (_pl, _s0, rows) in results.items():
        ax.plot([r[1] for r in rows], [r[4] for r in rows], "o-", lw=1.8,
                label=name)
        for r in rows:
            ax.annotate(f"{r[0]}°", (r[1], r[4]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(1.0, color="#c0392b", lw=1.2, ls=":")
    ax.set_xlabel(_L("remaining support (%)", "남은 서포트 (%)"))
    ax.set_ylabel(_L("alignment cusp vs planar (x)", "정렬 계단 자국 (배)"))
    ax.set_title(_L("the trade-off: less support, worse stair-stepping",
                    "상충: 서포트를 줄이면 계단이 심해진다"), fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    fig.suptitle(_L("geometry-forced surface quality (physics not modelled)",
                    "기하가 강제하는 표면 품질 (물리는 안 봄)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=130)
    print(f"\n그림 저장: {args.out}")
    print("\n⚠ 비드가 계단을 메우는 효과·코너 반경·표면 광택은 보지 못한다.")
    print("  '이보다 좋을 수 없다' 이지 '이만큼 나쁘다' 가 아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
