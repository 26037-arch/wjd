"""
analyze_sync.py — 5축 회전 요구량과 동기화 예산. [플랫폼 비교]

    python3 analyze_sync.py                          # 기본: funnel 20° + lamp 24°
    python3 analyze_sync.py model.stl --angle 25
    python3 analyze_sync.py --caps 360,720,1440

왜: 하드웨어 후보가 둘이다 — REP5X(개조 Marlin, **펌웨어가 역기구학**)와
뱀부랩 개조(원본 펌웨어 유지 + **외부 ESP32 가 회전축만 제어**). 어느 쪽이
빠른가를 묻기 전에 **회전축에 무엇이 요구되는지**를 알아야 한다. 기계가 없어도
G-code 만으로 잴 수 있다.

답하는 것 셋:
  ① 회전 각속도가 얼마나 필요한가 (축 근처에서 발산한다)
  ② 축의 최대 각속도를 정하면 출력 시간이 몇 배가 되나
  ③ 외부 컨트롤러의 동기 오차가 표면에서 몇 µm 이 되나

⚠ 기구학 요구량이지 실제 시간이 아니다 (가속도·저크·입력 셰이핑 무시 → 시간은
  하한). 축 토크·관성은 안 본다 — '이 각속도가 물리적으로 가능한가'는 기계 쪽 질문.
"""

import argparse

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.open5x import PRUSA_UV, to_open5x
from conical.planar_slicer import slice_mesh
from conical.plotstyle import L as _L
from conical.sync import (
    feed_cap_cost,
    machine_xy_span,
    rotary_demand,
    sync_error_budget,
)
from conical.transform import transform_cone

LAYER_H = 0.3
EXTRUSION_WIDTH = 0.45


def build(stl, angle, layer_height=LAYER_H):
    mesh = center_on_axis(trimesh.load(stl, force="mesh"))
    warped = transform_cone(mesh.vertices, angle, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False),
                       layer_height=layer_height)
    real, _ = backtransform(items, angle, "outward")
    machine, _ = to_open5x(real, angle, "outward", PRUSA_UV)
    return real, machine


def report(name, real, machine, caps, delays):
    d = rotary_demand(real, machine)
    ext = d["extruding"]
    w = d["omega"][ext]
    print("=" * 78)
    print(f"[{name}]  구간 {len(d['omega']):,} (압출 {int(ext.sum()):,})")

    print("\n  ① 요구 회전 각속도 (압출 구간)")
    print(f"     중앙 {np.median(w):,.0f}  90% {np.percentile(w, 90):,.0f}  "
          f"99% {np.percentile(w, 99):,.0f}  최대 {w.max():,.0f} deg/s")
    print(f"     부품 반경 최소 {d['r'][ext].min():.2f}mm "
          f"— ω = f/r 이라 축에 가까울수록 발산한다")

    print("\n  ② 축 최대 각속도를 정하면 출력 시간이 몇 배가 되나")
    cost = feed_cap_cost(d, caps)
    print(f"     {'한계':>9} | {'느려지는 구간':>11} | {'시간':>8} | {'배수':>6}")
    for r in cost["rows"]:
        print(f"     {r['cap']:>6} d/s | {r['slowed_frac'] * 100:10.1f}% | "
              f"{r['seconds'] / 60:7.1f}분 | {r['ratio']:5.2f}x")
    print(f"     (기준 {cost['base_seconds'] / 60:.1f}분 — 회전 한계 없음, "
          f"가속도 무시한 하한)")

    print("\n  ③ 외부 컨트롤러 동기 오차 → 경로 **수직** 치수 오차")
    bud = sync_error_budget(d, delays)
    print(f"     {'지각':>7} | {'중앙':>8} | {'99%':>8} | {'최대':>8}")
    for delay in sorted(bud):
        b = bud[delay]
        print(f"     {delay:5.0f}ms | {b['median']:7.1f}µm | {b['p99']:7.1f}µm "
              f"| {b['max']:7.1f}µm")
    print(f"     (층고 {LAYER_H * 1000:.0f}µm, 압출폭 {EXTRUSION_WIDTH * 1000:.0f}µm "
          f"— 최대가 압출폭을 넘으면 경로가 이웃 비드로 넘어간다)")

    span = machine_xy_span(machine)
    print("\n  ④ 기계 XY 가 실제로 쓰이나")
    print(f"     X 폭 {span['x_span']:.2f}mm,  Y 폭 {span['y_span']:.4f}mm "
          f"(|Y| 최대 {span['y_abs_max']:.4f})")
    if span["y_span"] < 1e-6:
        print("     → **Y 가 0 으로 축퇴한다.** V = −φ(p) 라 압출점이 항상 방위각 0 에")
        print("        오기 때문이다. 기계 XY 경로가 X 축 직선이 되므로 **기반")
        print("        프린터의 XY 속도는 이 모드에서 거의 쓰이지 않는다.**")
    return d, cost, bud, span


def plot(results, path="sync_demand.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for name, (d, _, _, _) in results.items():
        ext = d["extruding"]
        ax.plot(np.sort(d["omega"][ext]),
                np.linspace(0, 100, int(ext.sum())), lw=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(_L("required rotary rate (deg/s, log)",
                     "요구 회전 각속도 (deg/s, 로그)"))
    ax.set_ylabel(_L("percent of extrusion below", "이하인 압출 구간 (%)"))
    ax.set_title(_L("the rotary axis is asked for a lot near the part axis",
                    "축 근처에서 회전 요구가 발산한다"), fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, (_, cost, _, _) in results.items():
        caps = [r["cap"] for r in cost["rows"]]
        ax.plot(caps, [r["ratio"] for r in cost["rows"]], "o-", lw=1.8,
                label=name)
    ax.axhline(1.0, color="#999", lw=.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(_L("rotary axis limit (deg/s, log)", "축 최대 각속도 (deg/s, 로그)"))
    ax.set_ylabel(_L("print time multiplier", "출력 시간 배수"))
    ax.set_title(_L("what a slow rotary axis costs in print time",
                    "회전축이 느리면 출력 시간이 얼마나 늘어나나"), fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    fig.suptitle(_L("5-axis rotary demand — kinematic lower bound "
                    "(accel/jerk ignored)",
                    "5축 회전 요구량 — 기구학 하한 (가속도·저크 무시)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    print(f"\n그림 저장: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("models", nargs="*", help="STL 경로 (없으면 funnel + lamp)")
    ap.add_argument("--angle", type=float, default=None,
                    help="원뿔 각도 (모델을 직접 줄 때)")
    ap.add_argument("--caps", default="180,360,720,1440,3600",
                    help="비교할 축 최대 각속도 (deg/s), 쉼표 구분")
    ap.add_argument("--delays", default="1,5,10",
                    help="비교할 동기 지각 (ms), 쉼표 구분")
    ap.add_argument("--out", default="sync_demand.png")
    args = ap.parse_args()

    caps = [float(c) for c in args.caps.split(",")]
    delays = [float(d) for d in args.delays.split(",")]
    if args.models:
        jobs = [(m, args.angle if args.angle is not None else 20.0)
                for m in args.models]
    else:
        jobs = [("examples/funnel.stl", 20.0), ("examples/lamp.stl", 24.0)]

    results = {}
    for stl, ang in jobs:
        name = f"{stl.split('/')[-1]} {ang:g}°"
        real, machine = build(stl, ang)
        results[name] = report(name, real, machine, caps, delays)

    plot(results, args.out)
    print("\n⚠ 기구학 요구량이지 실제 시간이 아니다 — 가속도·저크·입력 셰이핑을")
    print("  무시하므로 출력 시간은 하한이고, 축 토크·관성은 보지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
