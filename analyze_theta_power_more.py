"""analyze_theta_power_more.py — 각도 비용 1~4차 비교를 **모델 18개**로 넓힌다.

analyze_theta_power.py 는 4 모델(저장 실측)뿐이었다. 그 결론("1차가 가장 낫다, 차수가
오를수록 나쁘다") 이 형상에 따라 뒤집히는지 본다.

새 모델은 (θ₁, θ₂) 4° 격자 144 조합의 툴패스 진짜 오버행을 **직접 잰다**
(모델당 ~7분, 코어 수만큼 병렬). 측정은 theta_power_more_grids.json 에 모델별로
캐시돼 끊겨도 이어서 돈다. 기존 4 모델은 j_vs_optimum_results.json 을 그대로 쓴다.

지표 (analyze_theta_power.py 와 같음, 결과 보기 전에 정함):
* B 후회: 차수 p 항만(k=0) m 을 0.01~3 전수로 바꿔 고른 해들이, 같은 각도 이하에서
  가능한 최선 오버행(144 조합 전수)보다 평균 얼마나 나쁜가. 낮을수록 좋다
* C 출하 선택기: select_banded_j(k=0.2, 2° 격자, 음각 포함) + 차수 항(m=0.1/0.2/0.4)
  이 고른 해의 툴패스 오버행과, 같은 전선 대비 후회 (2° 해라 격자 전선보다 약간
  좋을 수 있다 → 음수 가능)
* 균일 행: 모델마다 k=0.2 균일 최선의 오버행을 같이 적는다 (CLAUDE.md §2)

⚠ 밴드 2, outward 격자만(음각은 C 에서만), 4° 격자. 결론 범위도 거기까지.

실행:
    python3 analyze_theta_power_more.py            # 측정(캐시) + 분석 + 그림
    python3 analyze_theta_power_more.py --jobs 4   # 병렬 수
    python3 analyze_theta_power_more.py --plot-only
그림: theta_power_more_regret.png, theta_power_more_selector.png
"""

import argparse
import functools
import json
import math
import os
import time
from multiprocessing import Pool

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import conical.varangle as va
from analyze_blend_ratio import _sphere, lobe, widen
from analyze_j_vs_optimum import enumerate_profiles
from analyze_theta_power import (
    COLOR,
    INK,
    M_PURE,
    M_SEL,
    MARK,
    MUTED,
    ORDERS,
    _style,
    k_of,
    moment,
    pick,
)
from analyze_theta_quadratic import STORED, frontier, load_grid
from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile, center_on_axis
from conical.plotstyle import L
from conical.test_models import cylinder, flaring_cone, generate_random_model

GRIDS = "theta_power_more_grids.json"
OUT = "theta_power_more_results.json"
STEP = 4.0


def revolve(pts, sections=64):
    m = trimesh.creation.revolve(np.array(pts, dtype=float), sections=sections)
    m.merge_vertices()
    m.fix_normals()
    return center_on_axis(m)


def mushroom():
    """줄기 r=3 → 갓이 r=9 로 급히 벌어짐 → 돔. 허리 위 큰 오버행."""
    pts = [(0, 0), (3, 0), (3, 10), (9, 13), (9, 14)]
    pts += [(9 * math.cos(t), 14 + 4 * math.sin(t))
            for t in np.linspace(0, math.pi / 2, 10)[1:]]
    return revolve(pts)


def wine_glass():
    """받침 r=5 → 가는 대 r=1.5 → 넓은 잔 r=12 (벽 53°, 임계각 넘음)."""
    return revolve([(0, 0), (5, 0), (5, 1), (1.5, 2), (1.5, 8), (12, 16), (0, 16)])


def funnel():
    m = trimesh.load("examples/funnel.stl")
    m = m.subdivide_to_size(1.5)
    m.fix_normals()
    return center_on_axis(m)


def rnd(i):
    return generate_random_model(2026, i)[0]


ORIGINAL = ["허리 r=3", "램프 (허리 r=2)", "구", "허리3 ×1.24"]
NEW = [
    ("허리 r=1.5", lambda: waisted_model(1.5)),
    ("허리 r=4", lambda: waisted_model(4.0)),
    ("허리 r=5.5", lambda: waisted_model(5.5)),
    ("램프 ×0.8", lambda: widen(waisted_model(2.0), 0.8)),
    ("허리3 로브0.15", lambda: lobe(waisted_model(3.0), 0.15)),
    ("구 r=6", lambda: _sphere(radius=6.0)),
    ("급한 원뿔 3→13", lambda: flaring_cone(3.0, 13.0, 8.0)),
    ("벌어진 원뿔 3.5→10.4", lambda: flaring_cone(3.52, 10.37, 15.67)),
    ("깔때기 (examples)", funnel),
    ("버섯", mushroom),
    ("와인잔", wine_glass),
    ("원통 (대조)", lambda: cylinder(6.0, 12.0)),
    ("랜덤0 허리로브", lambda: rnd(0)),
    ("랜덤10 허리로브", lambda: rnd(10)),
]
NEW_EN = {"허리 r=1.5": "waist r=1.5", "허리 r=4": "waist r=4",
          "허리 r=5.5": "waist r=5.5", "램프 ×0.8": "lamp ×0.8",
          "허리3 로브0.15": "waist3 lobe0.15", "구 r=6": "sphere r=6",
          "급한 원뿔 3→13": "steep cone 3→13",
          "벌어진 원뿔 3.5→10.4": "flare cone 3.5→10.4",
          "깔때기 (examples)": "funnel (examples)", "버섯": "mushroom",
          "와인잔": "wine glass", "원통 (대조)": "cylinder (control)",
          "랜덤0 허리로브": "random0 waist-lobe", "랜덤10 허리로브": "random10 waist-lobe",
          "허리 r=3": "waist r=3", "램프 (허리 r=2)": "lamp (waist r=2)",
          "구": "sphere", "허리3 ×1.24": "waist3 ×1.24"}
BUILD = dict(NEW)


def build(name):
    if name in BUILD:
        return BUILD[name]()
    from analyze_theta_quadratic import MODELS
    return dict(MODELS)[name]()


# ─────────────────────────────────────────────────────────────
# 측정 (병렬, 캐시)
# ─────────────────────────────────────────────────────────────
_W = {}


def _measure_one(args):
    name, t1, t2 = args
    if _W.get("name") != name:
        mesh = build(name)
        combos, _rp, _r = enumerate_profiles(mesh, STEP, DEFAULT_K)
        _W.update(name=name, mesh=mesh,
                  combos={(c["t1"], c["t2"]): c for c in combos})
    c = _W["combos"][(t1, t2)]
    oh = run_pipeline(_W["mesh"], c["prof"], breakdown=True)[3]
    return dict(t1=t1, t2=t2, overhang=float(oh))


def measure_all(jobs):
    grids = json.load(open(GRIDS, encoding="utf-8")) if os.path.exists(GRIDS) else []
    done = {g["model"] for g in grids}
    for name, b in NEW:
        if name in done:
            continue
        mesh = b()
        combos, _rp, _r = enumerate_profiles(mesh, STEP, DEFAULT_K)
        t0 = time.time()
        with Pool(jobs) as pool:
            meas = pool.map(_measure_one, [(name, c["t1"], c["t2"]) for c in combos],
                            chunksize=4)
        grids.append(dict(model=name, step=STEP, grid=meas))
        with open(GRIDS, "w", encoding="utf-8") as f:
            json.dump(grids, f, ensure_ascii=False)
        print(f"  측정 {name:<20} 면 {len(mesh.faces):5d} · 조합 {len(meas)} · "
              f"{time.time() - t0:.0f}s", flush=True)
    return grids


# ─────────────────────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────────────────────
def selector(mesh, F):
    orig = va.profile_objective

    def obj(mesh_, profile, base_pct, k, k_blend=va.BLEND_COST_K,
            radius_profile=None, spacing_limit=MAX_SPACING_FACTOR,
            threshold_deg=va.THRESHOLD_DEG, *, p, m):
        r = orig(mesh_, profile, base_pct, k, k_blend, radius_profile,
                 spacing_limit, threshold_deg)
        return {**r, "J": r["J"] - k_of(p, m) * moment(mesh_, profile, p)}

    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    cache, rows = {}, []
    try:
        for p, m in [(1, 0.0)] + [(p, m) for p in ORDERS for m in M_SEL]:
            va.profile_objective = functools.partial(obj, p=p, m=m)
            sel = va.select_banded_j(mesh, DEFAULT_K, 2, r_max, rp,
                                     MAX_SPACING_FACTOR)
            key = tuple(round(t, 6) for t in sel["thetas"])
            if key not in cache:
                cache[key] = run_pipeline(mesh, sel["profile_obj"], breakdown=True)[3]
            ang = mean_abs_angle(mesh, sel["profile_obj"])
            rows.append(dict(p=p, m=m, thetas=[round(t, 1) for t in sel["thetas"]],
                             oh=float(cache[key]), ang=ang,
                             regret=float(cache[key] - F(ang))))
    finally:
        va.profile_objective = orig
    return rows


def analyze(name, rec, do_sel):
    mesh = build(name)
    grid = load_grid(name, mesh, [rec])
    for c in grid:
        c["mom"] = {p: moment(mesh, c["prof"], p) for p in ORDERS}
    F = frontier([(c["ang"], c["oh"]) for c in grid])
    B = {}
    for p in ORDERS:
        seen = {}
        for m in M_PURE:
            c = pick(grid, 0.0, p, m)
            seen.setdefault((c["t1"], c["t2"]), c["oh"] - F(c["ang"]))
        reg = list(seen.values())
        B[p] = dict(mean_regret=float(np.mean(reg)), max_regret=float(max(reg)),
                    on_front=int(sum(r < 1e-9 for r in reg)), n=len(reg))
    uni = pick(grid, DEFAULT_K, 1, 0.0, uniform_only=True)
    j0 = pick(grid, DEFAULT_K, 1, 0.0)
    best = min(grid, key=lambda c: (c["oh"], c["ang"]))
    row = dict(model=name, faces=len(mesh.faces), B=B,
               base_oh=grid[0]["oh"] if grid[0]["t1"] == grid[0]["t2"] == 0 else None,
               uniform=dict(t=[uni["t1"], uni["t2"]], oh=uni["oh"], ang=uni["ang"]),
               j_pick=dict(t=[j0["t1"], j0["t2"]], oh=j0["oh"], ang=j0["ang"]),
               best=dict(t=[best["t1"], best["t2"]], oh=best["oh"], ang=best["ang"]))
    if do_sel:
        row["selector"] = selector(mesh, F)
    return row


def report(rows):
    print("=" * 100)
    print("B  진짜 전선 대비 평균 후회 %p (낮을수록 좋음, ★=그 모델 최저)   |  "
          "k=0.2 J 선택 vs 균일 최선")
    print(f"  {'모델':<20} " + " ".join(f"{p}차{'':>4}" for p in ORDERS)
          + f" | {'J 선택':<8} {'오버행':>6} | {'균일':<8} {'오버행':>6}")
    wins = dict.fromkeys(ORDERS, 0.0)
    for r in rows:
        regs = [r["B"][p]["mean_regret"] for p in ORDERS]
        lo = min(regs)
        best = [p for p, v in zip(ORDERS, regs, strict=True) if v - lo < 1e-6]
        for p in best:
            wins[p] += 1 / len(best)
        cells = " ".join(f"{v:5.3f}{'★' if p in best else ' '}"
                         for p, v in zip(ORDERS, regs, strict=True))
        jp_, u = r["j_pick"], r["uniform"]
        print(f"  {r['model']:<20} {cells} | {str(jp_['t']):<8} {jp_['oh']:6.3f} | "
              f"{str(u['t']):<8} {u['oh']:6.3f}")
    med = {p: float(np.median([r["B"][p]["mean_regret"] for r in rows])) for p in ORDERS}
    print("  중앙값: " + "  ".join(f"{p}차 {med[p]:.3f}" for p in ORDERS))
    print("  최저 횟수(동점은 나눔): " + "  ".join(f"{p}차 {wins[p]:.1f}" for p in ORDERS))

    if all("selector" in r for r in rows):
        print("=" * 100)
        print("C  출하 선택기 k=0.2 + 차수 항: 툴패스 오버행 %p (현재 J 대비 증감)")
        print(f"  {'모델':<20} {'현재':>6} | " + " | ".join(
            f"{p}차 m=" + "/".join(f"{m:g}" for m in M_SEL) for p in ORDERS))
        worse = better = same = 0
        for r in rows:
            s = r["selector"]
            base = s[0]["oh"]
            cells = []
            for p in ORDERS:
                vals = [x["oh"] for x in s if x["p"] == p and x["m"] > 0]
                for v in vals:
                    if v > base + 1e-6:
                        worse += 1
                    elif v < base - 1e-6:
                        better += 1
                    else:
                        same += 1
                cells.append("/".join(f"{v - base:+.2f}" for v in vals))
            print(f"  {r['model']:<20} {base:6.3f} | " + " | ".join(f"{c:<17}" for c in cells))
        print(f"  현재 J 대비 오버행: 나빠짐 {worse} · 같음 {same} · 좋아짐 {better} "
              f"(총 {worse + same + better} 실행)")
        reg = {p: float(np.mean([x["regret"] for r in rows for x in r["selector"]
                                 if x["p"] == p and x["m"] > 0])) for p in ORDERS}
        print("  선택기 해의 평균 후회: " + "  ".join(f"{p}차 {reg[p]:.3f}" for p in ORDERS)
              + f"   (현재 J: {np.mean([r['selector'][0]['regret'] for r in rows]):.3f})")


# ─────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────
def mname(n):
    return L(NEW_EN.get(n, n), n)


def plot_regret(rows, path):
    rows = sorted(rows, key=lambda r: r["B"][1]["mean_regret"])
    names = [mname(r["model"]) for r in rows]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 0.42 * len(rows) + 2.2),
                                 gridspec_kw=dict(width_ratios=[3, 1.1]))
    _style(ax)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y = np.arange(len(rows))
    for r_i, r in enumerate(rows):
        vals = [r["B"][p]["mean_regret"] for p in ORDERS]
        ax.plot([min(vals), max(vals)], [r_i, r_i], color="#d6d5d0", lw=2, zorder=1)
    for p in ORDERS:
        ax.scatter([r["B"][p]["mean_regret"] for r in rows], y + (p - 2.5) * 0.12,
                   color=COLOR[p], marker=MARK[p], s=60, ec="white", lw=1.2,
                   zorder=3, label=L(f"order {p}", f"{p}차"))
    ax.set_yticks(y, names, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(L("mean regret vs true front (%p) — lower is better",
                    "진짜 전선 대비 평균 후회 (%p) — 낮을수록 좋다"), color=INK)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower right")
    ax.set_title(L(f"{len(rows)} models, order-p cost only (k=0)",
                   f"모델 {len(rows)}개 · 차수 항만 (k=0)"),
                 fontsize=10, color=INK, loc="left")

    _style(bx)
    bx.grid(True, axis="y")
    bx.grid(False, axis="x")
    wins = dict.fromkeys(ORDERS, 0.0)
    for r in rows:
        regs = [r["B"][p]["mean_regret"] for p in ORDERS]
        best = [p for p, v in zip(ORDERS, regs, strict=True) if v - min(regs) < 1e-6]
        for p in best:
            wins[p] += 1 / len(best)
    bars = bx.bar([f"{p}" for p in ORDERS], [wins[p] for p in ORDERS],
                  color=[COLOR[p] for p in ORDERS], edgecolor="white", lw=1, width=0.7)
    for b, p in zip(bars, ORDERS, strict=True):
        bx.text(b.get_x() + b.get_width() / 2, wins[p] + 0.15, f"{wins[p]:.1f}",
                ha="center", fontsize=8, color=INK)
    bx.set_xlabel(L("cost order", "비용 차수"), color=INK)
    bx.set_ylabel(L("models where lowest regret (ties split)",
                    "후회 최저인 모델 수 (동점 나눔)"), color=INK)
    bx.set_title(L("who wins", "누가 이기나"), fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_selector(rows, path):
    rows = [r for r in rows if "selector" in r]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    _style(ax)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    xs = np.arange(len(M_SEL))
    for p in ORDERS:
        d = np.array([[x["oh"] - r["selector"][0]["oh"] for x in r["selector"]
                       if x["p"] == p and x["m"] > 0] for r in rows])
        off = (p - 2.5) * 0.17
        for j in range(len(M_SEL)):
            ax.scatter(np.full(len(rows), xs[j] + off)
                       + np.linspace(-0.05, 0.05, len(rows)),
                       d[:, j], color=COLOR[p], marker=MARK[p], s=16, alpha=0.45,
                       lw=0, zorder=2)
        ax.plot(xs + off, np.median(d, axis=0), color=COLOR[p], marker=MARK[p],
                ms=9, mec="white", mew=1.5, lw=2, zorder=4,
                label=L(f"order {p} (median)", f"{p}차 (중앙값)"))
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.text(len(M_SEL) - 0.55, 0.03, L("current J", "현재 J"), color=MUTED,
            fontsize=8, ha="right", va="bottom")
    ax.set_xticks(xs, [f"m={m:g}" for m in M_SEL], color=INK)
    ax.set_xlabel(L("added marginal cost at 25° (%p/°)", "추가한 25° 한계비용"),
                  color=INK)
    ax.set_ylabel(L("Δ toolpath overhang vs current J (%p)",
                    "현재 J 대비 툴패스 오버행 증감 (%p)"), color=INK)
    ax.set_title(L(f"Shipping selector, {len(rows)} models — above 0 = worse output. "
                     "Higher orders sit lower only because they cut the angle less",
                   f"출하 선택기 · 모델 {len(rows)}개 — 0 위 = 출력 나빠짐. "
                     "고차가 낮은 건 각도를 덜 줄여서다(같은 각도 비교는 후회 그림)"),
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--no-sel", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        rows = json.load(open(OUT, encoding="utf-8"))
        for r in rows:
            r["B"] = {int(p): v for p, v in r["B"].items()}
    else:
        grids = measure_all(args.jobs)
        stored = json.load(open(STORED, encoding="utf-8"))
        recs = [r for r in stored if r["model"] in ORIGINAL] + grids
        t0 = time.time()
        with Pool(args.jobs) as pool:
            rows = pool.starmap(analyze, [(rec["model"], rec, not args.no_sel)
                                          for rec in recs], chunksize=1)
        print(f"  분석 {len(rows)} 모델 {time.time() - t0:.0f}s", flush=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    report(rows)
    plot_regret(rows, "theta_power_more_regret.png")
    figs = ["theta_power_more_regret.png"]
    if all("selector" in r for r in rows):
        plot_selector(rows, "theta_power_more_selector.png")
        figs.append("theta_power_more_selector.png")
    print("그림 저장: " + ", ".join(figs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
