"""analyze_angle_grid.py — ρ 반증의 원인이 **각도 격자**인가.

## 왜

`analyze_rho_gap.py` 가 ρ 를 반증했다(2026-09-26). 그런데 반증 데이터에서 더 이상한
것이 나왔다 — **형상이 매끄럽게 변하는데 결과가 튄다**:

    λ = 1.18 패 · 1.19 패 · 1.20 승 · 1.21 승 · 1.30 패

λ 는 XY 스케일 하나만 연속으로 바꾸는 축이다. 그런데 승/패가 비단조다. 즉
**비단조성의 출처가 형상이 아닐 수 있다.** 후보는 선택기의 이산성이다:

* **각도 격자 `ANGLE_STEP = 2°`** ← 이 실험이 보는 것
* 밴드 경계 `linspace` (TODO: "등간격으로 자르는 것부터가 임의")
* 블렌드 경계 이동 `BLEND_SHIFT_RATIO`
* 평평한 J 지형 (램프 N=2 에서 0.005 차이로 갈린다, TODO 에 있음)

## 시험

같은 λ 축을 `step = 2° / 1° / 0.5°` 로 다시 훑는다. `select_banded_j` 가 `step` 을
파라미터로 받으므로 config 를 건드리지 않는다.

지표는 **부호 변화 횟수**다. λ 를 따라 승/패 수열을 만들고 그 안의 전환 수를 센다.
단조라면 전환이 **1 회**(승…승패…패) 또는 0 회다. 많을수록 비단조다.

## 판정 기준 (**결과 보기 전에** 적는다)

* 전환 수가 `2° → 1° → 0.5°` 로 **줄어든다** → 원인은 **각도 격자**다. ρ 반증은
  "지표가 틀렸다" 보다 **"선택기가 이산적이라 결과가 튄다"** 에 가깝다.
  그러면 격자를 줄이거나 J 를 매끄럽게 만드는 쪽이 다음 수다.
* 전환 수가 **안 줄어든다** → 격자가 아니다. 남은 후보(밴드 경계·블렌드 이동·평평한
  J)로 넘어간다. ρ 반증은 그대로 "예측 불가" 로 남는다.
* 전환 수가 **늘어난다** → 격자를 촘촘히 할수록 더 튄다는 뜻이고, J 지형 자체가
  평평해서 미세한 차이로 해가 갈린다는 신호다(TODO 의 '평평한 J 지형' 과 연결).

⚠ 각도가 더 촘촘하면 **선택기가 더 좋은 해를 찾을 수도** 있다. 그래서 전환 수와
  별도로 **승 비율**과 **밴드2 오버행 중앙값**도 같이 본다 — 셋을 섞으면 안 된다.

실행:
    python3 analyze_angle_grid.py              # λ 10점 × step 3종
    python3 analyze_angle_grid.py --quick      # λ 6점
"""

import argparse
import json
import sys
import time

import numpy as np

from analyze_blend_ratio import widen
from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.varangle import select_banded_j

LAMBDAS = (1.16, 1.18, 1.19, 1.20, 1.21, 1.22, 1.24, 1.26, 1.28, 1.30)
QUICK = (1.18, 1.19, 1.20, 1.21, 1.24, 1.30)
STEPS = (2.0, 1.0, 0.5)


def outcome(mesh, k, step):
    """이 각도 격자로 균일(n=1) vs 밴드2(n=2) 를 고르고 툴패스로 판정."""
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    got = {}
    for n in (1, 2):
        # step 은 정수 range 에 쓰이므로 select_banded_j 가 받는 형태로 맞춘다.
        r = select_banded_j(mesh, k, n, r_max, rp, MAX_SPACING_FACTOR, step=step)
        prof = r["profile_obj"]
        _p, _t, _f, oh, _i, _g = run_pipeline(mesh, prof, breakdown=True)
        got[n] = (oh, mean_abs_angle(mesh, prof), [round(t, 1) for t in r["thetas"]])
    oh1, a1, th1 = got[1]
    oh2, a2, th2 = got[2]
    return dict(uniform_oh=oh1, uniform_angle=a1, uniform_thetas=th1,
                band_oh=oh2, band_angle=a2, band_thetas=th2,
                pareto=bool(oh2 < oh1 and a2 <= a1 + 1e-9))


def transitions(seq):
    """승/패 수열의 부호 전환 횟수. 단조면 0 또는 1."""
    return sum(1 for a, b in zip(seq, seq[1:], strict=False) if a != b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="angle_grid_results.json")
    args = ap.parse_args()

    lams = QUICK if args.quick else LAMBDAS
    base = waisted_model(3.0)
    meshes = {lam: widen(base, lam) for lam in lams}

    rows = []
    for step in STEPS:
        print(f"\n[각도 격자 {step}°]")
        print(f"{'λ':>6}{'균일 θ':>10}{'밴드2 θ':>16}{'균일 oh':>9}"
              f"{'밴드2 oh':>10}  판정")
        print("-" * 60)
        for lam in lams:
            t0 = time.time()
            m = outcome(meshes[lam], args.k, step)
            rows.append(dict(step=step, lam=lam, seconds=round(time.time()-t0, 1), **m))
            print(f"{lam:>6.2f}{str(m['uniform_thetas']):>10}"
                  f"{str(m['band_thetas']):>16}{m['uniform_oh']:>9.3f}"
                  f"{m['band_oh']:>10.3f}  {'승' if m['pareto'] else '패'}"
                  f"  ({rows[-1]['seconds']:.0f}s)", flush=True)

    # ── 판정 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'격자':>6}{'수열':>{len(lams)+4}}{'전환':>6}{'승 비율':>9}{'밴드2 oh 중앙':>14}")
    print("-" * 60)
    summary = []
    for step in STEPS:
        sub = [r for r in rows if r["step"] == step]
        seq = [r["pareto"] for r in sub]
        t = transitions(seq)
        win = sum(seq) / len(seq) * 100
        med = float(np.median([r["band_oh"] for r in sub]))
        summary.append((step, t, win, med))
        txt = "".join("승" if x else "패" for x in seq)
        print(f"{step:>6.1f}{txt:>{len(lams)+4}}{t:>6}{win:>8.0f}%{med:>14.3f}")

    ts = [t for _s, t, _w, _m in summary]
    print(f"\n[판정]  전환 수 {ts[0]} → {ts[1]} → {ts[2]} (격자 2° → 1° → 0.5°)")
    # ⚠ 표본이 10 점이라 전환 수 ±1 은 노이즈다. 1 차이로 '늘었다/줄었다' 라고
    #   말하면 앞서 한 점으로 "층고 조건부" 라고 과장한 것과 같은 실수가 된다.
    if ts[-1] <= ts[0] - 2:
        print("  → **줄어든다. 원인은 각도 격자다.** ρ 반증은 '지표가 틀렸다' 보다")
        print("    '선택기가 이산적이라 결과가 튄다' 에 가깝다.")
    elif ts[-1] >= ts[0] + 2:
        print("  → **늘어난다.** 촘촘히 할수록 더 튄다 — J 지형이 평평해서 미세한 차이로")
        print("    해가 갈린다는 신호다.")
    else:
        print(f"  → **안 줄어든다 (차이 {ts[-1]-ts[0]:+d}, 표본 {len(lams)}점에서 노이즈 수준).**")
        print("    **격자가 원인이 아니다.** 남은 후보로 넘어간다 —")
        print("    밴드 경계 linspace, 블렌드 경계 이동, 그리고 J 자체.")
    print(f"  ⚠ 승 비율 {summary[0][2]:.0f}% → {summary[1][2]:.0f}% → {summary[2][2]:.0f}%,"
          f"  밴드2 오버행 중앙 {summary[0][3]:.3f} → {summary[1][3]:.3f} → {summary[2][3]:.3f}")
    print("    (격자를 촘촘히 하면 선택기가 더 좋은 해를 찾을 수도 있다 — 전환 수와")
    print("     섞지 말 것. 위 두 줄은 별개의 축이다.)")

    # **균일 baseline 자체가 λ 에 비단조인가** — 그렇다면 불안정성은 '밴드 vs 균일'
    # 비교가 아니라 J 의 argmax 에 있다. 이번 실험에서 데이터로 드러난 핵심이다.
    print("\n[균일 baseline 도 흔들리나]  흔들리면 문제는 비교가 아니라 J 의 argmax 다")
    for step in STEPS:
        sub = [r for r in rows if r["step"] == step]
        angs = [r["uniform_thetas"][0] for r in sub]
        ohs = [r["uniform_oh"] for r in sub]
        print(f"  격자 {step:>4.1f}°  균일 각도 {angs}")
        print(f"{'':11}오버행 {min(ohs):.3f}~{max(ohs):.3f} "
              f"(최대/최소 {max(ohs)/max(min(ohs),1e-9):.1f}배)")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
