"""analyze_j_vs_optimum.py — J 가 고른 해는 **진짜 최적해와 얼마나 먼가**.

## 왜

계통이 J 로 모였다 (docs/verification.md 2026-09-28):

| 축 | J 로 고르면 | 언제 |
|---|---|---|
| 밴드 수 N | 오버행이 달성 가능한 최소보다 나쁘다 (5/6 모델) | 09-19 |
| 각도 격자 | 격자를 촘촘히 하면 J↑ 인데 오버행↓ | 09-28 |
| 형상 파라미터 | argmax 가 튄다 (λ 0.02 에 각도 8° 점프) | 09-28 |

그래서 ρ 반증도 "지표를 잘못 골랐다" 가 아니라 **"J 가 오버행을 대표하지 못한다"**
쪽으로 읽힌다. 그런데 그건 아직 **간접 증거**다. 직접 재야 한다.

## 설계 — J 를 바꾸는 대신 **최적해를 전수로 찾는다**

"J 를 오버행 직접 최적화로 교체" 를 그대로 하면 툴패스를 좌표하강 루프 안에 넣어야
해서 모델당 1.5 시간이다. 대신:

1. 밴드2 각도 조합 `(θ₁, θ₂)` 를 **격자 전수**로 훑는다 (0~44°, 4° 간격)
2. 각 조합의 프로필을 만들고(층간격 제약 포함, 불가는 건너뜀) **툴패스로 오버행**을 잰다
3. 동시에 그 조합의 **J** 도 계산한다 (해석식이라 싸다)
4. → 지형 전체에서 **J vs 오버행**을 비교할 수 있다

이게 "J 가 고른 해 vs 최적해" 두 점 비교보다 강하다 — **J 가 오버행 순위를 따라가는지
자체**를 잰다.

## 판정 기준 (**결과 보기 전에** 적는다)

* **스피어만 ρ(J, 오버행) ≈ −1** → J 가 오버행 순위를 잘 따라간다. J 는 문제가 아니고
  비단조성의 원인은 다른 곳이다 (J 는 클수록 좋고 오버행은 작을수록 좋으니 음의 상관)
* **|ρ| 가 작거나 부호가 이상** → **J 가 오버행 순위를 안 따라간다.** 계통이 확정된다
* **J-argmax 의 오버행 vs 전수 최소의 격차가 크다** → 그 격차가 우리가 잃고 있는 양이다
* **전수 최적해가 균일(θ₁=θ₂)** → 그 모델에서는 밴드가 **애초에 이길 수 없다.**
  J 와 무관한 사실이고, 이게 나오면 '밴드 우세' 주장의 범위가 다시 좁아진다

⚠ 전수 격자는 4° 다. **그 사이는 모른다.** 그리고 밴드 수는 2 로 고정이다.

실행:
    python3 analyze_j_vs_optimum.py            # 대표 4 모델
    python3 analyze_j_vs_optimum.py --quick    # 2 모델, 격자 8°
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from analyze_blend_ratio import _sphere, widen
from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.analytic import support_fraction
from conical.config import (
    BLEND_COST_K,
    BLEND_SHIFT_RATIO,
    DEFAULT_K,
    MAX_SPACING_FACTOR,
    THRESHOLD_DEG,
)
from conical.meshio import RadiusProfile
from conical.profile import AngleProfile
from conical.varangle import (
    _merge_bands,
    assign_height_bands,
    blend_penalty,
    profile_objective,
    select_banded_j,
)


def models(quick):
    out = [("허리 r=3", lambda: waisted_model(3.0)),
           ("램프 (허리 r=2)", lambda: waisted_model(2.0))]
    if not quick:
        out += [("구", _sphere),
                # λ=1.24 는 격자에 따라 판정이 뒤집힌 자리다 (analyze_angle_grid.py)
                ("허리3 ×1.24", lambda: widen(waisted_model(3.0), 1.24))]
    return out


def enumerate_profiles(mesh, step, k, k_blend=BLEND_COST_K):
    """(θ₁, θ₂) 격자 전수 → 만들 수 있는 프로필과 그 J. 못 만드는 조합은 건너뛴다."""
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    H = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    _labels, edges = assign_height_bands(mesh, 2)
    base_pct = support_fraction(mesh, 0.0, "outward", THRESHOLD_DEG)
    grid = [float(a) for a in np.arange(0.0, 44.0 + 1e-9, step)]

    out = []
    for t1 in grid:
        for t2 in grid:
            try:
                prof = AngleProfile.from_bands(
                    _merge_bands(edges, [t1, t2]), r_max, radius_profile=rp,
                    spacing_limit=MAX_SPACING_FACTOR,
                    max_shift=BLEND_SHIFT_RATIO * H)
            except ValueError:
                continue
            if prof.check_spacing(r_max, "outward", MAX_SPACING_FACTOR, rp):
                continue          # 층간격 제약 위반 — 못 찍는 프로필
            m = profile_objective(mesh, prof, base_pct, k, k_blend, rp,
                                  MAX_SPACING_FACTOR, THRESHOLD_DEG)
            out.append(dict(t1=t1, t2=t2, prof=prof, J=m["J"],
                            pen=m["blend_penalty"],
                            uniform=bool(abs(t1 - t2) < 1e-9)))
    return out, rp, r_max


def blend_pen_of(mesh, thetas, rp, r_max):
    """밴드 각도 → 그 프로필의 블렌드 벌점 (해석식이라 싸다). 재분석용."""
    H = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    _labels, edges = assign_height_bands(mesh, 2)
    prof = AngleProfile.from_bands(
        _merge_bands(edges, [float(t) for t in thetas]), r_max, radius_profile=rp,
        spacing_limit=MAX_SPACING_FACTOR, max_shift=BLEND_SHIFT_RATIO * H)
    return float(blend_penalty(mesh, prof, rp, MAX_SPACING_FACTOR))


def _dominates(a, b, keys):
    """a 가 b 를 keys 전부에서 이기거나 비기고, 하나 이상에서 엄격히 이기나."""
    tol = dict(overhang=1e-12, angle=1e-9, pen=1e-9)
    return (all(a[x] <= b[x] + tol[x] for x in keys)
            and any(a[x] < b[x] - tol[x] for x in keys))


def judge(combos, sel_pt):
    """판정 셋. ⚠ 2026-09-28 첫 전체 실행에서 스크립트 판정이 **두 번 틀렸다**:

    ① '오버행 단독 최적해가 밴드인가' 가 **동점을 처리 못 했다.** 허리류에서
       [40,0] 과 균일 40 이 둘 다 0.000%p 인데 min() 이 균일을 먼저 집어
       "균일 고각이 최적" 이라고 했다. [40,0] 은 **왜곡이 25° vs 40°** 다 —
       밴드가 균일을 **지배**한다. → 오버행 → 왜곡 순의 사전식으로 고친다.
    ② 파레토를 **2 축**(오버행, 왜곡)으로만 봤다. J 는 **3 항**이다 — 블렌드
       벌점이 셋째 축이다. 구에서 J 해가 '전선 밖' 으로 나온 것은 그 해를
       지배하는 조합들이 **블렌드 면적을 70~96% 쓰기** 때문이고, 그걸 '순수한
       손해' 라고 부르면 J 가 세는 축 하나를 무료로 친 것이다. → 3 축 전선을
       같이 본다.
    """
    best = min(combos, key=lambda c: (c["overhang"], c["angle"]))
    best_uni = min((c for c in combos if c["uniform"]),
                   key=lambda c: (c["overhang"], c["angle"]))
    band_wins = (not best["uniform"]) and _dominates(best, best_uni,
                                                     ("overhang", "angle"))
    dom2 = [c for c in combos if _dominates(c, sel_pt, ("overhang", "angle"))]
    dom3 = [c for c in combos if _dominates(c, sel_pt, ("overhang", "angle", "pen"))]
    return dict(best=best, best_uni=best_uni, band_wins=bool(band_wins),
                dom2=dom2, dom3=dom3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="j_vs_optimum_results.json")
    ap.add_argument("--reanalyze", metavar="JSON",
                    help="저장된 격자로 판정만 다시 (툴패스 안 돌림)")
    args = ap.parse_args()
    if args.reanalyze:
        rows = reanalyze(args.reanalyze)
        json.dump(rows, open(args.reanalyze, "w"), ensure_ascii=False, indent=1)
        return
    step = args.step if args.step else (8.0 if args.quick else 4.0)

    allrows = []
    for name, build in models(args.quick):
        mesh = build()
        combos, rp, r_max = enumerate_profiles(mesh, step, args.k)
        print(f"\n[{name}]  격자 {step:g}°  만들 수 있는 조합 {len(combos)}개 "
              f"(균일 {sum(c['uniform'] for c in combos)}개)")

        t0 = time.time()
        for c in combos:
            _p, _t, _f, oh, _i, _g = run_pipeline(mesh, c["prof"], breakdown=True)
            c["overhang"] = oh
            c["angle"] = mean_abs_angle(mesh, c["prof"])
        print(f"  툴패스 {len(combos)}회 측정 ({time.time()-t0:.0f}s)")

        # J 가 고른 해 (출하 선택기)
        sel = select_banded_j(mesh, args.k, 2, r_max, rp, MAX_SPACING_FACTOR)
        _p, _t, _f, oh_sel, _i, _g = run_pipeline(mesh, sel["profile_obj"],
                                                  breakdown=True)
        th_sel = [round(t, 1) for t in sel["thetas"]]

        sel_pt = dict(overhang=oh_sel, pen=float(blend_penalty(
                          mesh, sel["profile_obj"], rp, MAX_SPACING_FACTOR)),
                      angle=mean_abs_angle(mesh, sel["profile_obj"]))
        allrows.append(dict(
            model=name, step=step, n_combos=len(combos),
            j_pick=th_sel, j_pick_oh=oh_sel, j_pick_angle=sel_pt["angle"],
            j_pick_pen=sel_pt["pen"],
            grid=[{kk: c[kk] for kk in ("t1", "t2", "J", "overhang", "angle",
                                        "pen", "uniform")} for c in combos]))

    report(allrows)
    if args.json:
        json.dump(allrows, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.json}")


def reanalyze(path):
    """저장된 격자(툴패스 576 회, 40 분)를 다시 안 돌리고 판정만 다시 한다.
    블렌드 벌점이 없던 옛 JSON 이면 해석식으로 채운다 (툴패스와 무관, 초 단위)."""
    rows = json.load(open(path, encoding="utf-8"))
    builds = dict(models(False))
    for r in rows:
        if "j_pick_pen" in r and all("pen" in c for c in r["grid"]):
            continue
        mesh = builds[r["model"]]()
        rp = RadiusProfile(mesh)
        r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
        for c in r["grid"]:
            c["pen"] = blend_pen_of(mesh, (c["t1"], c["t2"]), rp, r_max)
        r["j_pick_pen"] = blend_pen_of(mesh, r["j_pick"], rp, r_max)
    report(rows)
    return rows


def report(rows):
    """모델별 판정 + 요약. 측정 경로와 재분석 경로가 **같은 판정 코드**를 쓴다."""
    for r in rows:
        combos = r["grid"]
        sel_pt = dict(overhang=r["j_pick_oh"], angle=r["j_pick_angle"],
                      pen=r["j_pick_pen"])
        v = judge(combos, sel_pt)
        best, uni = v["best"], v["best_uni"]
        rho, pval = spearmanr([c["J"] for c in combos],
                              [c["overhang"] for c in combos])
        r.update(spearman=float(rho), band_beats_uniform=v["band_wins"],
                 best_oh=[best["t1"], best["t2"]], best_oh_val=best["overhang"],
                 best_oh_angle=best["angle"], best_uniform=uni["t1"],
                 best_uniform_oh=uni["overhang"], best_uniform_angle=uni["angle"],
                 j_on_front2=not v["dom2"], n_dominating2=len(v["dom2"]),
                 j_on_front3=not v["dom3"], n_dominating3=len(v["dom3"]))

        print(f"\n[{r['model']}]  격자 {r['step']:g}°  조합 {r['n_combos']}개")
        print(f"  J 가 고른 해   θ={r['j_pick']}  오버행 {r['j_pick_oh']:.3f}%p  "
              f"왜곡 {r['j_pick_angle']:.1f}°  블렌드 {r['j_pick_pen']:.1f}")
        print(f"  오버행 최소    θ=[{best['t1']:g}, {best['t2']:g}]  "
              f"오버행 {best['overhang']:.3f}%p  왜곡 {best['angle']:.1f}°  "
              f"블렌드 {best['pen']:.1f}   (동점이면 왜곡이 작은 쪽)")
        print(f"  최선 균일      θ={uni['t1']:g}  오버행 {uni['overhang']:.3f}%p  "
              f"왜곡 {uni['angle']:.1f}°")
        print(f"  → 스피어만 ρ(J, 오버행) = {rho:+.3f} (p={pval:.1e})")
        print(f"  → 오버행 최적해가 균일을 지배하는 밴드인가: "
              f"**{'예' if v['band_wins'] else '아니오'}**")
        print(f"  → J 해가 2 축 전선(오버행, 왜곡) 위: "
              f"{'예' if not v['dom2'] else '아니오'} (지배 {len(v['dom2'])}개)")
        print(f"  → J 해가 3 축 전선(+블렌드) 위:     "
              f"{'예' if not v['dom3'] else '아니오'} (지배 {len(v['dom3'])}개)")
        if v["dom2"]:
            w = min(v["dom2"], key=lambda c: (c["overhang"], c["angle"]))
            print(f"     2 축에서 가장 세게 지배: θ=[{w['t1']:g}, {w['t2']:g}] "
                  f"오버행 {w['overhang']:.3f} / 왜곡 {w['angle']:.1f}° / "
                  f"**블렌드 {w['pen']:.1f}** (J 해 {sel_pt['pen']:.1f})")
            pens = [c["pen"] for c in v["dom2"]]
            print(f"     지배 조합 {len(pens)}개의 블렌드 {min(pens):.1f}~{max(pens):.1f}"
                  "  ← J 는 이것을 비용으로 센다")

    print("\n" + "=" * 74)
    print(f"{'모델':<16}{'스피어만':>9}{'2축전선':>8}{'3축전선':>8}"
          f"{'오버행최적해':>14}{'밴드우세':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['model']:<16}{r['spearman']:>+9.3f}"
              f"{('예' if r['j_on_front2'] else '아니오'):>8}"
              f"{('예' if r['j_on_front3'] else '아니오'):>8}"
              f"{str([f'{x:g}' for x in r['best_oh']]):>14}"
              f"{('예' if r['band_beats_uniform'] else '아니오'):>9}")

    n2 = sum(1 for r in rows if not r["j_on_front2"])
    n3 = sum(1 for r in rows if not r["j_on_front3"])
    print(f"\n[파레토] J 해가 전선 밖: 2 축 {n2}/{len(rows)}, 3 축 {n3}/{len(rows)}")
    if n3 == 0 and n2 > 0:
        print("  → 2 축에서만 밖이다. **블렌드를 비용으로 세느냐** 에 판정이 달렸다 —")
        print("    즉 그 모델들의 결론은 k_blend(정규화 세기) 에 달렸다. 순수한 손해가 아니다.")
    elif n3 > 0:
        # ⚠ 첫 판정 문구는 여기서 "순수한 손해" 라고 단정했다. 구에서 3 축으로
        #   지배하는 [12,44] 는 오버행 0.035%p·왜곡 0.14° 차이였고, 블렌드 0 인
        #   이유가 **층 압축(m=0.667)을 blend_penalty 가 안 세기** 때문이었다.
        #   그래서 격차를 숫자로 보이고, 압축을 따로 알린다.
        print("  → 3 축에서도 밖인 모델이 있다. 격차부터 본다 (작으면 사실상 동점):")
        for r in rows:
            if r["j_on_front3"]:
                continue
            sel = dict(overhang=r["j_pick_oh"], angle=r["j_pick_angle"],
                       pen=r["j_pick_pen"])
            for c in r["grid"]:
                if _dominates(c, sel, ("overhang", "angle", "pen")):
                    print(f"     {r['model']}: θ=[{c['t1']:g}, {c['t2']:g}]  "
                          f"오버행 −{sel['overhang']-c['overhang']:.3f}%p  "
                          f"왜곡 −{sel['angle']-c['angle']:.2f}°  블렌드 {c['pen']:.1f}")
        print("    ⚠ blend_penalty 는 층간격 **팽창(m>1)만** 센다. 압축(m<1)은 0 이다 —")
        print("      블렌드 0 인 지배 조합이 **압축 블렌드**인지 확인할 것.")
    else:
        print("  → J 는 모든 모델에서 전선 위의 한 점을 고른다.")
    rs = [r["spearman"] for r in rows]
    print(f"[순위] 스피어만 ρ(J, 오버행) {min(rs):+.3f} ~ {max(rs):+.3f}")
    nb = sum(1 for r in rows if r["band_beats_uniform"])
    print(f"[밴드] 오버행 최적해가 균일을 지배하는 밴드인 모델 {nb}/{len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
