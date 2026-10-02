"""
analyze_motion.py — 축 한계를 넣은 출력 시간 모델 + 민감도. [플랫폼 비교]

    python3 analyze_motion.py                       # 기본 모델 둘, 기준 프로파일
    python3 analyze_motion.py --angle 25 model.stl
    python3 analyze_motion.py --sweep rot_vel       # 한 항목만 훑기
    python3 analyze_motion.py --v-rewind 0          # 되감기 없는 기계(REP5X)

`analyze_sync.py` 의 시간은 **가속도를 무시한 하한**이었다. 여기서는 축별 속도·
가속도·저크를 넣어 사다리꼴 운동 계획을 돌린다 (`conical/motion.py`).

답하는 것:
  ① **출력 시간과 되감기 시간을 갈라서** 얼마나 되나 (하한 대비 몇 배)
  ② **어느 축이 병목인가** — 시간 기준으로 어느 축이 속도를 깎고 있나
  ③ **어느 스펙에 돈을 써야 하나** — 항목 하나씩 2배로 올릴 때 시간이 얼마나 주나

⚠ 되감기(`--v-rewind`)는 **출력이 아니다.** 배선이 감기는 기계에서만 내는 부대
  비용이고, REP5X 처럼 회전축이 무제한이면 통째로 0 이다. 그래서 절대 합쳐서
  보고하지 않는다 — 합치면 기계 비교가 망가진다.

⚠⚠ **아래 축 한계값은 우리 기계에서 확인된 것이 아니다.** 일반적인 소비자 FDM
  프린터의 관례를 출발점으로 적은 것이고, 회전축 값은 **순전한 가정**이다
  (5축 개조의 회전축 스펙을 아직 모른다). 실기 제원이 나오면 반드시 갈아끼울 것.
  이 도구의 쓸모는 절대 시간이 아니라 **같은 툴패스를 두 프로파일로 비교**하는 데
  있다.
"""

import argparse

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.motion import _displacements, limits_from_spec, plan_motion
from conical.open5x import PRUSA_UV, add_v_rewinds, check_open5x, to_open5x
from conical.planar_slicer import slice_mesh
from conical.plotstyle import L as _L
from conical.transform import transform_cone

# ⚠ 전부 가정값이었다 — 회전축은 특히 근거가 없었다.
ASSUMED = dict(xy_vel=150.0, xy_acc=3000.0,       # 소비자 FDM 관례
               z_vel=12.0, z_acc=300.0,
               rot_vel=360.0, rot_acc=1800.0,     # ⚠ 순전한 가정
               e_vel=120.0, e_acc=2000.0)

# ✅ REP5X **실측 제원** (`Rep5x-Marlin`, 브랜치 `Marlin2ForPipetBot`,
#    `Marlin/Configuration.h`). 순서는 { X, Y, Z, I(C), J(B), E }:
#      DEFAULT_MAX_FEEDRATE      { 500, 500, 600, 3600, 3600, 45 }
#      DEFAULT_MAX_ACCELERATION  { 3000, 3000, 500, 500, 500, 10000 }
#      DEFAULT_XJERK 10  ZJERK 0.3  EJERK 5  IJERK 20  JJERK 20
#
# ⚠⚠ 내 가정이 **틀린 쪽으로** 틀렸다. 회전축 최대 속도는 가정(360)보다 10배
#    빠른데(3600), **가속도는 가정(1800)보다 3.6배 느리다(500).** R3 에서
#    '가속도가 최대 속도보다 더 먹는다' 고 했으므로, 실제 기계는 **중요한 쪽에서
#    가정보다 나쁘다.** 아래 표가 그 차이를 보여준다.
REP5X_SPEC = dict(xy_vel=500.0, xy_acc=3000.0,
                  z_vel=600.0, z_acc=500.0,
                  rot_vel=3600.0, rot_acc=500.0,
                  e_vel=45.0, e_acc=10000.0,
                  xy_jerk=10.0, z_jerk=0.3, rot_jerk=20.0, e_jerk=5.0)

BASELINE = ASSUMED

REWIND_FEED = 1200.0        # mm/min (RRF 는 V 를 '선형'으로 봐서 = 20 deg/s)

SWEEPS = {
    "xy_vel": "XY 최대 속도 (mm/s)",
    "xy_acc": "XY 가속도 (mm/s²)",
    "z_vel": "Z 최대 속도 (mm/s)",
    "rot_vel": "회전축 최대 속도 (deg/s)",
    "rot_acc": "회전축 가속도 (deg/s²)",
    "e_acc": "압출기 가속도 (mm/s²)",
}


def build(stl, angle, layer_height=0.3, rewind_turns=1.0,
          rewind_feed=REWIND_FEED):
    """STL → (부품좌표, 기계좌표). `rewind_turns=0` 이면 되감기를 안 넣는다."""
    mesh = center_on_axis(trimesh.load(stl, force="mesh"))
    warped = transform_cone(mesh.vertices, angle, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False),
                       layer_height=layer_height)
    real, _ = backtransform(items, angle, "outward")
    machine, _ = to_open5x(real, angle, "outward", PRUSA_UV)
    if rewind_turns > 0:
        machine, _ = add_v_rewinds(machine, PRUSA_UV, max_turns=rewind_turns,
                                   rot_feed=rewind_feed)
    return real, machine


def run(machine, spec):
    return plan_motion(machine, limits_from_spec("spec", **spec))


def rewind_rotation(machine, rot_axis="V"):
    """되감기 구간에서 실제로 돌린 총 각도(deg). 이것이 되감기 시간의 분자다."""
    return sum(abs(disp.get(rot_axis, 0.0))
               for disp, _f, _p, reg in _displacements(machine)
               if reg == "V_REWIND")


def feed_lower_bound(machine, region=""):
    """가속도를 무시한 하한 Σ L/F (기계공간). 모델이 이것보다 작으면 버그다.

    기본은 **출력 구간만** — 되감기를 섞으면 '가속 때문에 몇 배' 가 무의미해진다.
    """
    return sum(np.sqrt(sum(d * d for a, d in disp.items()
                           if a in ("X", "Y", "Z", "V"))) / feed
               for disp, feed, _, reg in _displacements(machine)
               if region is None or reg == region)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--angle", type=float, default=None)
    ap.add_argument("--sweep", choices=sorted(SWEEPS),
                    help="이 항목만 훑는다 (기본: 전부)")
    ap.add_argument("--factors", default="0.25,0.5,1,2,4",
                    help="기준값의 몇 배로 훑을지")
    ap.add_argument("--v-rewind", type=float, default=1.0,
                    help="배선 되감기 문턱 (회전). 0 이면 되감기 없음 "
                         "= 슬립링/무제한 회전축 기계")
    ap.add_argument("--v-rewind-feed", type=float, default=REWIND_FEED,
                    help="되감기 피드 (mm/min). 되감기 시간을 지배하는 값이다")
    ap.add_argument("--out", default="motion_sensitivity.png")
    args = ap.parse_args()

    jobs = ([(m, args.angle if args.angle is not None else 20.0)
             for m in args.models]
            or [("examples/funnel.stl", 20.0), ("examples/lamp.stl", 24.0)])
    factors = [float(f) for f in args.factors.split(",")]
    keys = [args.sweep] if args.sweep else list(SWEEPS)

    results = {}
    for stl, ang in jobs:
        name = f"{stl.split('/')[-1]} {ang:g}°"
        real, machine = build(stl, ang, rewind_turns=args.v_rewind,
                              rewind_feed=args.v_rewind_feed)
        base = run(machine, BASELINE)
        lb = feed_lower_bound(machine)
        _, st = check_open5x(machine, PRUSA_UV)
        t_print = base["print_seconds"]
        t_rw = base["region_time"].get("V_REWIND", 0.0)

        print("=" * 78)
        print(f"[{name}]  구간 {base['n_segments']:,}")
        print(f"\n  ① 시간 (축 한계는 가정이다 — 특히 회전축 "
              f"{BASELINE['rot_vel']:.0f} deg/s 는 근거가 없다)")
        print(f"     출력      : **{t_print / 60:6.2f}분**")
        if t_rw > 0:
            print(f"     배선 되감기: {t_rw / 60:6.2f}분  "
                  f"({t_rw / base['seconds'] * 100:.0f}%, {st['rewinds']}회)"
                  f"  ← 출력이 아니다")
            print(f"     합계      : {base['seconds'] / 60:6.2f}분")
            spin = rewind_rotation(machine)
            print("     ⚠ 되감기는 **무제한 회전축(REP5X C축)이면 0** 이다.")
            print(f"       되돌린 총 회전량 {spin / 360:.0f}회전 ÷ 되감기 속도 "
                  f"{args.v_rewind_feed / 60:.0f} deg/s ≒ {spin / (args.v_rewind_feed / 60) / 60:.1f}분")
            print("       → 되돌릴 각도는 감긴 만큼 **정해져 있다.** "
                  "이 시간을 줄이는 유일한 레버는 되감기 **속도**이고, "
                  "되감기 횟수는 거의 무관하다.")
        print(f"     출력의 가속 무시 하한 Σ L/F : {lb / 60:.2f}분 "
              f"→ 가속 때문에 {t_print / lb:.2f}배")

        print("\n  ② 어느 축이 속도를 깎고 있나 (출력 구간만, 되감기 제외)")
        bt = {}
        for who, reg, t in zip(base["binding"], base["regions"], base["times"], strict=True):
            if reg:
                continue
            bt[who or "(명령 피드)"] = bt.get(who or "(명령 피드)", 0.0) + float(t)
        for axis, t in sorted(bt.items(), key=lambda kv: -kv[1]):
            print(f"     {axis:>12} : {t / 60:6.2f}분  "
                  f"({t / t_print * 100:5.1f}%)")

        print("\n  ③ 어느 스펙에 돈을 써야 하나 (기준값 대비 배수 → 시간 배수)")
        print("     ※ 출력 시간만 본다 (되감기 제외)")
        header = "  ".join(f"{f:g}×" for f in factors)
        print(f"     {'항목':<22} {header}")
        sens = {}
        for key in keys:
            row = []
            for f in factors:
                spec = dict(BASELINE)
                spec[key] = BASELINE[key] * f
                row.append(run(machine, spec)["print_seconds"] / t_print)
            sens[key] = row
            cells = "  ".join(f"{r:.2f}" for r in row)
            print(f"     {SWEEPS[key]:<22} {cells}")
        if 2.0 in factors:
            i2 = factors.index(2.0)
            best = min(keys, key=lambda k: sens[k][i2])
            print(f"     → 2배로 올려 가장 이득인 항목: **{SWEEPS[best]}** "
                  f"({sens[best][i2]:.2f}×)")
        results[name] = (base, sens, t_print, t_rw)

    # 그림: 민감도 곡선
    if len(keys) > 1:
        fig, axes = plt.subplots(1, len(results),
                                 figsize=(6.4 * len(results), 4.6),
                                 squeeze=False)
        for ax, (name, (_base, sens, t_print, t_rw)) in zip(axes[0],
                                                           results.items(), strict=True):
            for key in keys:
                ax.plot(factors, sens[key], "o-", lw=1.7, label=SWEEPS[key])
            ax.axhline(1.0, color="#999", lw=.9, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel(_L("spec multiplier (log)", "스펙 배수 (로그)"))
            ax.set_ylabel(_L("print time multiplier", "출력 시간 배수"))
            ax.set_title(f"{name}  ({t_print / 60:.1f}분 출력"
                         + (f" + {t_rw / 60:.1f}분 되감기)" if t_rw else ")"),
                         fontsize=10)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7.5)
        fig.suptitle(_L("which axis spec actually buys print speed "
                        "(assumed limits — replace with measured)",
                        "어느 축 스펙이 실제로 속도를 사주나 "
                        "(가정값 — 실기 제원으로 갈아끼울 것)"), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(args.out, dpi=130)
        print(f"\n그림 저장: {args.out}")

    print("\n⚠ 펌웨어 플래너의 근사다 — 입력 셰이핑·S커브·세그먼트 합치기·버퍼")
    print("  길이가 빠져 있다. 절대 시간이 아니라 **프로파일 간 비교**로 쓸 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
