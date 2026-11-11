"""analyze_theta_quadratic.py — J 에 θ 의 **2차항**을 넣으면 출력이 좋아지나.

## 무엇을 바꾸나

지금 J (varangle.profile_objective):

    J  = (서포트 감소 %p) − k·E|θ| − k_blend·블렌드

여기에 면적가중 평균 θ² 항을 더한다:

    J₂ = (서포트 감소 %p) − k·E|θ| − k₂·E[θ²] − k_blend·블렌드      (θ 단위: 도)

선형 비용은 각도 1° 의 값이 0° 에서나 40° 에서나 같다. 2차 비용은 **큰 각도일수록
한계비용이 커진다** (dJ/dθ = −k − 2k₂θ). "고각은 비선형으로 비싸다"(노즐 간섭, 기계 Z
침하 −d(1−cosθ), 층간격 팽창)는 직관을 넣은 형태다. 여기서 재는 것은 그 직관이 아니라
**그렇게 고른 해가 툴패스에서 더 나은가** 다.

## '출력' 을 무엇으로 재나

선택된 프로필을 실제 파이프라인(왜곡 → 슬라이스 → 역변환 → 검사기 A)에 넣어 나온
**툴패스 진짜 오버행 %p** (`compare_waist.run_pipeline(breakdown=True)[3]`) 와
**면적가중 평균 |θ|** (왜곡) 두 축. J 값 자체는 출력 지표가 아니다.

## 공정성 — 2차항을 넣으면 각도가 '당연히' 줄어든다

k₂ 를 더하면 각도 비용이 늘어 해가 저각 쪽으로 간다. 그건 k 를 키워도 일어난다.
그래서 "2차항이 좋다" 는 **"선형 k 를 아무리 조절해도 못 가는 (오버행, 각도) 점에
간다"** 일 때만 성립한다고 정한다 (결과 보기 전에 적음):

* **실험 1 (항 추가)**: k=0.2 고정, k₂ 를 키워가며 고른 해의 오버행·각도.
  균일각 행을 같이 적는다 (CLAUDE.md §2 — 균일을 못 이기면 진 것)
* **실험 2 (공정 비교)**: 선형만(k 를 0.02~1.0 전수) 으로 갈 수 있는 해들의 전선
  L(a) = min{오버행 | 각도 ≤ a} 을 만들고, 2차 J 가 고른 해 (a_q, oh_q) 에 대해
  **Δ = L(a_q) − oh_q** 를 잰다. Δ>0 이면 같은 왜곡 예산에서 2차가 오버행을 더 줄였다.
  Δ≤0 이면 2차항은 **k 를 바꾼 것과 다를 바 없다**
* **실험 3 (출하 선택기)**: 실험 1·2 는 4° 격자 밴드2 전수 argmax 다. 실제 선택기
  `select_banded_j` (2° 격자, 음각 포함, 좌표하강) 에 2차항을 꽂아 같은 비교를 한다.
  격자 밖 해가 나오면 툴패스를 새로 돌린다

실험 1·2 의 오버행 값은 `j_vs_optimum_results.json` 에 저장된 실측을 쓴다
(현재 코드로 소수점까지 재현됨을 확인함 — `--verify` 로 다시 확인 가능).

⚠ 표본은 4 모델(허리 r=3, 램프, 구, 허리3×1.24), 밴드 2 개다. 결론의 범위도 거기까지다.

실행:
    python3 analyze_theta_quadratic.py            # 실험 1·2·3 (~3분)
    python3 analyze_theta_quadratic.py --no-sel   # 실험 1·2 만 (~20초)
    python3 analyze_theta_quadratic.py --verify   # 저장 실측 표본 재측정
"""

import argparse
import functools
import json
import time

import numpy as np

import conical.varangle as va
from analyze_blend_ratio import _sphere, widen
from analyze_j_vs_optimum import enumerate_profiles
from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile

STORED = "j_vs_optimum_results.json"
MODELS = [("허리 r=3", lambda: waisted_model(3.0)),
          ("램프 (허리 r=2)", lambda: waisted_model(2.0)),
          ("구", _sphere),
          ("허리3 ×1.24", lambda: widen(waisted_model(3.0), 1.24))]

# k₂ [1/deg²]. 참고: 2k₂θ = k 가 되는 θ — k₂=0.004 면 25° 에서 2차 한계비용이 선형과 같다
K2_ADD = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.02, 0.03]
K1_LIN = [round(x, 3) for x in np.arange(0.02, 1.0 + 1e-9, 0.02)]   # 선형만 전선
K2_PURE = [round(x, 4) for x in np.arange(0.0005, 0.03 + 1e-9, 0.0005)]  # k=0 순수 2차
K2_SEL = [0.0, 0.002, 0.004, 0.008]


def mean_sq_angle(mesh, prof):
    """면적가중 E[θ²] (도²)."""
    areas = mesh.area_faces
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    return float((prof.theta_at(fz) ** 2 * areas).sum() / areas.sum())


def objective_q(mesh, profile, base_pct, k, k_blend=va.BLEND_COST_K,
                radius_profile=None, spacing_limit=MAX_SPACING_FACTOR,
                threshold_deg=va.THRESHOLD_DEG, *, k2, _orig):
    """profile_objective + (−k₂·E[θ²]). k₂=0 이면 원래 J 와 비트 단위로 같다."""
    m = _orig(mesh, profile, base_pct, k, k_blend, radius_profile,
              spacing_limit, threshold_deg)
    sq = mean_sq_angle(mesh, profile)
    return {**m, "J": m["J"] - k2 * sq, "mean_sq_angle": sq}


# ─────────────────────────────────────────────────────────────
# 격자(저장 실측) 위에서
# ─────────────────────────────────────────────────────────────
def load_grid(name, mesh, stored):
    """저장 격자 + 각 조합의 E|θ|·E[θ²]·서포트 감소(=J+k·E|θ|+k_b·pen)."""
    rec = next(r for r in stored if r["model"] == name)
    meas = {(g["t1"], g["t2"]): g for g in rec["grid"]}
    combos, _rp, _r = enumerate_profiles(mesh, rec["step"], DEFAULT_K)
    out = []
    for c in combos:
        g = meas[(c["t1"], c["t2"])]
        ang = mean_abs_angle(mesh, c["prof"])
        # J(저장, k=DEFAULT_K) 에서 이득 항을 되살린다 → 아무 (k, k₂) 로 재계산 가능
        gain = c["J"] + DEFAULT_K * ang + va.BLEND_COST_K * c["pen"]
        out.append(dict(t1=c["t1"], t2=c["t2"], uniform=c["uniform"],
                        prof=c["prof"], gain=gain, ang=ang, pen=c["pen"],
                        sq=mean_sq_angle(mesh, c["prof"]), oh=g["overhang"]))
    return out


def jq(c, k, k2):
    return c["gain"] - k * c["ang"] - k2 * c["sq"] - va.BLEND_COST_K * c["pen"]


def pick(grid, k, k2, uniform_only=False):
    pool = [c for c in grid if c["uniform"]] if uniform_only else grid
    return max(pool, key=lambda c: (jq(c, k, k2), -c["ang"]))


def frontier(points):
    """L(a) = min{oh | ang ≤ a}. points: (ang, oh) 목록 → 함수."""
    pts = sorted(points)

    def L(a):
        cand = [oh for ang, oh in pts if ang <= a + 1e-9]
        return min(cand) if cand else float("inf")
    return L


def fmt_th(c):
    return f"[{c['t1']:g},{c['t2']:g}]"


def grid_experiments(stored):
    rows = []
    for name, build in MODELS:
        mesh = build()
        grid = load_grid(name, mesh, stored)
        best_oh = min(grid, key=lambda c: (c["oh"], c["ang"]))
        print("=" * 86)
        print(f"[{name}]  격자 {len(grid)}조합 · 오버행 전수 최소 {best_oh['oh']:.3f}%p "
              f"@ {fmt_th(best_oh)} (E|θ|={best_oh['ang']:.1f}°)")

        # ── 실험 1: k=0.2 고정, k₂ 추가
        print(f"  실험1 (k={DEFAULT_K} 고정)")
        print(f"    {'k₂':>6} | {'J₂ 선택':<9} {'오버행':>7} {'E|θ|':>6} | "
              f"{'균일 선택':<9} {'오버행':>7} {'E|θ|':>6} | 밴드−균일 오버행")
        exp1 = []
        for k2 in K2_ADD:
            p = pick(grid, DEFAULT_K, k2)
            u = pick(grid, DEFAULT_K, k2, uniform_only=True)
            print(f"    {k2:6.3f} | {fmt_th(p):<9} {p['oh']:7.3f} {p['ang']:6.1f} | "
                  f"{fmt_th(u):<9} {u['oh']:7.3f} {u['ang']:6.1f} | "
                  f"{p['oh'] - u['oh']:+.3f}")
            exp1.append(dict(k2=k2, pick=[p["t1"], p["t2"]], oh=p["oh"], ang=p["ang"],
                             uni=[u["t1"], u["t2"]], uni_oh=u["oh"], uni_ang=u["ang"]))

        # ── 실험 2: 선형만으로 갈 수 있는 해의 전선 vs 2차 J 의 해
        lin = {(c["t1"], c["t2"]): c for c in (pick(grid, k, 0.0) for k in K1_LIN)}
        L = frontier([(c["ang"], c["oh"]) for c in lin.values()])
        quad = {}
        for k2 in K2_ADD[1:]:
            c = pick(grid, DEFAULT_K, k2)
            quad[(c["t1"], c["t2"])] = c
        for k2 in K2_PURE:
            c = pick(grid, 0.0, k2)
            quad[(c["t1"], c["t2"])] = c
        new = {key: c for key, c in quad.items() if key not in lin}
        deltas = [(L(c["ang"]) - c["oh"], c) for c in quad.values()]
        better = [(d, c) for d, c in deltas if d > 1e-9]
        print(f"  실험2  선형만(k∈[0.02,1]) 도달 해 {len(lin)}개 · 2차 J 도달 해 "
              f"{len(quad)}개 · 그중 선형으로 못 가는 해 {len(new)}개")
        if better:
            d, c = max(better, key=lambda x: x[0])
            print(f"    ✔ 같은 왜곡 예산에서 2차가 이긴 해 {len(better)}개 — 최대 "
                  f"Δ={d:+.3f}%p @ {fmt_th(c)} (E|θ|={c['ang']:.1f}°, "
                  f"오버행 {c['oh']:.3f} vs 선형 전선 {L(c['ang']):.3f})")
        else:
            worst = min(deltas, key=lambda x: x[0])
            print(f"    ✘ 2차 해가 선형 전선을 넘은 경우 없음 (최악 Δ={worst[0]:+.3f}%p) "
                  f"→ 2차항 = k 조절과 같거나 나쁨")
        for _key, c in sorted(new.items()):
            print(f"      새 해 {fmt_th(c)}: 오버행 {c['oh']:.3f} · E|θ| {c['ang']:.1f}° · "
                  f"선형 전선 {L(c['ang']):.3f} → Δ={L(c['ang']) - c['oh']:+.3f}")

        # ── 실험 2b: 후회(regret) — Δ 는 선형 전선이 듬성하면(빈 구간) 부풀 수 있다.
        #   그래서 **격자 전체**의 진짜 전선 F(a)=min{oh | ang≤a} 대비 각 해가 잃는 양
        #   regret = oh − F(ang) 을 두 계열에 똑같이 잰다. 0 = 그 왜곡에서 최선.
        F = frontier([(c["ang"], c["oh"]) for c in grid])
        reg_lin = [c["oh"] - F(c["ang"]) for c in lin.values()]
        reg_q = [c["oh"] - F(c["ang"]) for c in quad.values()]
        print(f"  실험2b 진짜 전선 대비 후회 %p — 선형: 평균 {np.mean(reg_lin):.3f} "
              f"(최대 {max(reg_lin):.3f}, 전선 위 {sum(r < 1e-9 for r in reg_lin)}/{len(reg_lin)})"
              f" · 2차: 평균 {np.mean(reg_q):.3f} (최대 {max(reg_q):.3f}, "
              f"전선 위 {sum(r < 1e-9 for r in reg_q)}/{len(reg_q)})")
        rows.append(dict(model=name, best_oh=best_oh["oh"],
                         regret_lin=reg_lin, regret_quad=reg_q,
                         best_oh_pick=[best_oh["t1"], best_oh["t2"]],
                         exp1=exp1,
                         lin_front=sorted([(c["ang"], c["oh"], c["t1"], c["t2"])
                                           for c in lin.values()]),
                         quad_points=[dict(t1=c["t1"], t2=c["t2"], ang=c["ang"],
                                           oh=c["oh"], delta=L(c["ang"]) - c["oh"])
                                      for c in quad.values()],
                         n_better=len(better), n_new=len(new)))
    return rows


# ─────────────────────────────────────────────────────────────
# 출하 선택기에 꽂아서
# ─────────────────────────────────────────────────────────────
def selector_experiment():
    orig = va.profile_objective
    rows = []
    print("=" * 86)
    print(f"실험3  select_banded_j (2° 격자, 음각 포함), k={DEFAULT_K}, 밴드 2")
    print(f"  {'모델':<14} {'k₂':>6} | {'선택 각도':<14} {'툴패스 오버행':>11} "
          f"{'E|θ|':>6} {'E[θ²]':>7} | {'균일최선':>7}")
    try:
        for name, build in MODELS:
            mesh = build()
            rp = RadiusProfile(mesh)
            r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
            cache = {}
            for k2 in K2_SEL:
                va.profile_objective = functools.partial(objective_q, k2=k2,
                                                         _orig=orig)
                t0 = time.time()
                sel = va.select_banded_j(mesh, DEFAULT_K, 2, r_max, rp,
                                         MAX_SPACING_FACTOR)
                key = tuple(round(t, 6) for t in sel["thetas"])
                if key not in cache:
                    cache[key] = run_pipeline(mesh, sel["profile_obj"],
                                              breakdown=True)[3]
                oh = cache[key]
                ang = mean_abs_angle(mesh, sel["profile_obj"])
                sq = mean_sq_angle(mesh, sel["profile_obj"])
                print(f"  {name:<14} {k2:6.3f} | {str([round(t, 1) for t in sel['thetas']]):<14} "
                      f"{oh:11.3f} {ang:6.1f} {sq:7.0f} | "
                      f"{sel['uniform_best']:6.1f}°  ({time.time() - t0:.0f}s)")
                rows.append(dict(model=name, k2=k2, thetas=list(sel["thetas"]),
                                 overhang=oh, angle=ang, mean_sq=sq,
                                 uniform_best=sel["uniform_best"]))
    finally:
        va.profile_objective = orig
    return rows


def verify(stored, n=3):
    """저장 실측이 현재 코드로 재현되는지 모델당 n 조합 재측정."""
    rng = np.random.default_rng(0)
    for name, build in MODELS:
        mesh = build()
        grid = load_grid(name, mesh, stored)
        for i in rng.choice(len(grid), n, replace=False):
            c = grid[int(i)]
            oh = run_pipeline(mesh, c["prof"], breakdown=True)[3]
            print(f"  {name:<14} {fmt_th(c):<9} 저장 {c['oh']:.6f}  재측정 {oh:.6f}  "
                  f"{'✔' if abs(oh - c['oh']) < 1e-9 else '✘ 불일치'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sel", action="store_true", help="실험 3 생략")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", default="theta_quadratic_results.json")
    args = ap.parse_args()
    stored = json.load(open(STORED, encoding="utf-8"))
    if args.verify:
        verify(stored)
        return 0
    out = {"grid": grid_experiments(stored)}
    if not args.no_sel:
        out["selector"] = selector_experiment()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"결과 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
