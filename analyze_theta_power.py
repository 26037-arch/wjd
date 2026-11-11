"""analyze_theta_power.py — J 의 각도 비용을 1·2·3·4 차로 바꾸면 출력이 어떻게 되나.

analyze_theta_quadratic.py (2차) 의 일반화다.

    J_p = (서포트 감소 %p) − k·E|θ| − k_p·E[|θ|^p] − k_blend·블렌드     (θ: 도)

## 차수끼리 공정하게 세기를 맞추는 법

k_p 의 단위가 차수마다 달라서(1/도^p) 같은 숫자로 비교할 수 없다. 그래서 **기준각
θ_ref=25° 에서의 한계비용** m 으로 매개한다:

    d/dθ (k_p·θ^p) |θ=25°  = m    ⇒   k_p = m / (p·25^(p−1))

m 이 같으면 25° 근처에서는 어느 차수든 1° 더 기울이는 값이 같다. 차수가 높을수록
25° 아래는 싸고 위는 비싸다(곡률만 다르다). 1차는 'k 를 m 만큼 키운 것' 이다.

## 실험 (출력 = 툴패스 진짜 오버행 %p, 왜곡 = 면적가중 E|θ|)

* A  (항 추가, 격자): k=0.2 고정 + 차수 p 항, m 을 키워가며 고른 해
* B  (순수, 격자): k=0, 차수 p 항만, m 전수 → 그 차수로 도달 가능한 해들의
     **진짜 전선 대비 후회** `oh − min{oh | 각도 ≤ 그 해의 각도}` (144 조합 전수 기준)
* C  (출하 선택기): `select_banded_j` (2° 격자, 음각 포함)에 각 차수 항을 꽂고
     툴패스를 새로 잰다. k=0.2 + m ∈ M_SEL

A·B 의 오버행은 `j_vs_optimum_results.json` 저장 실측(현재 코드로 재현 확인,
`analyze_theta_quadratic.py --verify`). 표본: 4 모델, 밴드 2, 4° 격자 — 결론 범위도 거기까지.

실행:
    python3 analyze_theta_power.py               # A·B·C + 그림 (~8분)
    python3 analyze_theta_power.py --no-sel      # A·B + 그림 (~10초)
    python3 analyze_theta_power.py --plot-only   # 저장 JSON 으로 그림만
그림: theta_power_tradeoff.png, theta_power_regret.png, theta_power_selector.png
"""

import argparse
import functools
import json
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import conical.varangle as va
from analyze_theta_quadratic import MODELS, STORED, frontier, load_grid
from compare_waist import mean_abs_angle, run_pipeline
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.plotstyle import L

ORDERS = [1, 2, 3, 4]
THETA_REF = 25.0
M_ADD = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5]
M_PURE = [round(x, 3) for x in np.arange(0.01, 3.0 + 1e-9, 0.01)]
M_SEL = [0.1, 0.2, 0.4]
JSON_OUT = "theta_power_results.json"

# 범주 색 (dataviz 기본 팔레트 1~4 슬롯, validate_palette.js 통과) + 차수별 마커
COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
MARK = {1: "o", 2: "s", 3: "^", 4: "D"}
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#e4e3df"
MODEL_EN = {"허리 r=3": "waist r=3", "램프 (허리 r=2)": "lamp (waist r=2)",
            "구": "sphere", "허리3 ×1.24": "waist3 ×1.24"}


def k_of(p, m):
    return m / (p * THETA_REF ** (p - 1))


def moment(mesh, prof, p):
    areas = mesh.area_faces
    fz = mesh.vertices[mesh.faces].mean(axis=1)[:, 2]
    return float((np.abs(prof.theta_at(fz)) ** p * areas).sum() / areas.sum())


def jp(c, k, p, m):
    return (c["gain"] - k * c["ang"] - k_of(p, m) * c["mom"][p]
            - va.BLEND_COST_K * c["pen"])


def pick(grid, k, p, m, uniform_only=False):
    pool = [c for c in grid if c["uniform"]] if uniform_only else grid
    return max(pool, key=lambda c: (jp(c, k, p, m), -c["ang"]))


def tag(c):
    return f"[{c['t1']:g},{c['t2']:g}]"


# ─────────────────────────────────────────────────────────────
def grid_experiments(stored):
    out = []
    for name, build in MODELS:
        mesh = build()
        grid = load_grid(name, mesh, stored)
        for c in grid:
            c["mom"] = {p: moment(mesh, c["prof"], p) for p in ORDERS}
        F = frontier([(c["ang"], c["oh"]) for c in grid])
        front = sorted({(c["ang"], c["oh"]) for c in grid
                        if c["oh"] - F(c["ang"]) < 1e-12})
        print("=" * 84)
        print(f"[{name}]")

        # A: k=0.2 + 차수 p 항
        print(f"  A (k={DEFAULT_K} + 차수 항)   각 칸 = 선택 [θ1,θ2] 오버행%p/E|θ|°")
        print("    " + f"{'m':>5} | " + " | ".join(f"{p}차{'':<17}" for p in ORDERS))
        A = {p: [] for p in ORDERS}
        for m in M_ADD:
            cells = []
            for p in ORDERS:
                c = pick(grid, DEFAULT_K, p, m)
                A[p].append(dict(m=m, t=[c["t1"], c["t2"]], oh=c["oh"], ang=c["ang"]))
                cells.append(f"{tag(c):<8}{c['oh']:5.2f}/{c['ang']:4.1f}")
            print(f"    {m:5.2f} | " + " | ".join(cells))
        uni = pick(grid, DEFAULT_K, 1, 0.0, uniform_only=True)
        print(f"    균일 최선(k={DEFAULT_K}): {tag(uni)} {uni['oh']:.2f}%p / {uni['ang']:.1f}°")

        # B: 순수 차수 항 — 도달 해와 후회
        B = {}
        for p in ORDERS:
            seen = {}
            for m in M_PURE:
                c = pick(grid, 0.0, p, m)
                seen.setdefault((c["t1"], c["t2"]), dict(
                    t=[c["t1"], c["t2"]], oh=c["oh"], ang=c["ang"],
                    regret=c["oh"] - F(c["ang"]), m_first=m))
            pts = sorted(seen.values(), key=lambda d: d["ang"])
            reg = [d["regret"] for d in pts]
            B[p] = dict(points=pts, mean_regret=float(np.mean(reg)),
                        max_regret=float(max(reg)),
                        on_front=int(sum(r < 1e-9 for r in reg)), n=len(pts))
        print("  B (k=0, 차수 항만)  도달 해 · 진짜 전선 대비 후회 %p")
        for p in ORDERS:
            b = B[p]
            print(f"    {p}차: 해 {b['n']:2d}개 · 전선 위 {b['on_front']:2d} · "
                  f"평균 후회 {b['mean_regret']:.3f} · 최대 {b['max_regret']:.3f}")
        out.append(dict(model=name, A=A, B=B, uniform=dict(t=[uni["t1"], uni["t2"]],
                        oh=uni["oh"], ang=uni["ang"]),
                        all=[(c["ang"], c["oh"]) for c in grid], front=front))
    return out


def selector_experiment():
    orig = va.profile_objective

    def obj(mesh, profile, base_pct, k, k_blend=va.BLEND_COST_K,
            radius_profile=None, spacing_limit=MAX_SPACING_FACTOR,
            threshold_deg=va.THRESHOLD_DEG, *, p, m):
        r = orig(mesh, profile, base_pct, k, k_blend, radius_profile,
                 spacing_limit, threshold_deg)
        return {**r, "J": r["J"] - k_of(p, m) * moment(mesh, profile, p)}

    rows = []
    print("=" * 84)
    print(f"C  select_banded_j (2° 격자, 음각 포함), k={DEFAULT_K}, 밴드 2")
    try:
        for name, build in MODELS:
            mesh = build()
            rp = RadiusProfile(mesh)
            r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
            cache = {}
            runs = [(1, 0.0)] + [(p, m) for p in ORDERS for m in M_SEL]
            for p, m in runs:
                va.profile_objective = functools.partial(obj, p=p, m=m)
                t0 = time.time()
                sel = va.select_banded_j(mesh, DEFAULT_K, 2, r_max, rp,
                                         MAX_SPACING_FACTOR)
                key = tuple(round(t, 6) for t in sel["thetas"])
                if key not in cache:
                    cache[key] = run_pipeline(mesh, sel["profile_obj"],
                                              breakdown=True)[3]
                ang = mean_abs_angle(mesh, sel["profile_obj"])
                th = [round(t, 1) for t in sel["thetas"]]
                print(f"  {name:<14} {p}차 m={m:4.2f} | {str(th):<14} "
                      f"오버행 {cache[key]:6.3f}%p  E|θ| {ang:5.1f}°  "
                      f"({time.time() - t0:.0f}s)")
                rows.append(dict(model=name, p=p, m=m, thetas=th,
                                 overhang=cache[key], angle=ang))
    finally:
        va.profile_objective = orig
    return rows


# ─────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mname(name):
    return L(MODEL_EN[name], name)


def plot_tradeoff(grid_rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    for ax, r in zip(axes.flat, grid_rows, strict=True):
        _style(ax)
        xs, ys = zip(*r["all"], strict=True)
        ax.scatter(xs, ys, s=9, color="#cfcec9", lw=0, zorder=1,
                   label=L("all 144 grid combos", "격자 144조합 전체"))
        fx, fy = zip(*r["front"], strict=True)
        ax.step(fx, fy, where="post", color=MUTED, lw=1.2, ls="--", zorder=2,
                label=L("true front (best at ≤ angle)", "진짜 전선 (그 각도 이하 최선)"))
        for p in ORDERS:
            pts = r["B"][p]["points"]
            ax.plot([d["ang"] for d in pts], [d["oh"] for d in pts],
                    color=COLOR[p], lw=2, marker=MARK[p], ms=8, mec="white",
                    mew=1.5, zorder=3 + p,
                    label=L(f"order {p} picks", f"{p}차 J 가 고른 해"))
        u = r["uniform"]
        ax.scatter([u["ang"]], [u["oh"]], marker="*", s=180, color=INK,
                   ec="white", lw=1, zorder=10,
                   label=L("best uniform (k=0.2)", "균일 최선 (k=0.2)"))
        ax.set_title(mname(r["model"]), fontsize=10, color=INK, loc="left")
        ax.set_ylim(bottom=-0.1)
    for ax in axes[1]:
        ax.set_xlabel(L("mean |θ| (°, distortion)", "평균 |θ| (°, 왜곡)"), color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(L("toolpath true overhang (%p)", "툴패스 진짜 오버행 (%p)"),
                      color=INK)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(L("Angle cost order vs output: lower-left is better",
                   "각도 비용 차수별로 고른 해 — 왼쪽 아래일수록 좋다"),
                 fontsize=12, color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regret(grid_rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _style(ax)
    ax.grid(True, axis="y", color=GRID)
    ax.grid(False, axis="x")
    n = len(ORDERS)
    w = 0.8 / n
    x = np.arange(len(grid_rows))
    for i, p in enumerate(ORDERS):
        vals = [r["B"][p]["mean_regret"] for r in grid_rows]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.88, color=COLOR[p],
                      edgecolor="white", lw=1, label=L(f"order {p}", f"{p}차"))
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [mname(r["model"]) for r in grid_rows], color=INK)
    ax.set_ylabel(L("mean regret vs true front (%p)", "진짜 전선 대비 평균 후회 (%p)"),
                  color=INK)
    ax.set_title(L("Lower = picks closer to the best achievable overhang at that angle",
                   "낮을수록 그 각도에서 가능한 최선 오버행에 가깝다"),
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_selector(sel_rows, path):
    names = list(dict.fromkeys(r["model"] for r in sel_rows))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    for ax, name in zip(axes.flat, names, strict=True):
        _style(ax)
        rows = [r for r in sel_rows if r["model"] == name]
        base = next(r for r in rows if r["m"] == 0.0)
        ax.axhline(base["overhang"], color=MUTED, ls="--", lw=1.2)
        ax.text(max(M_SEL), base["overhang"],
                L(f" current J ({base['overhang']:.2f})", f" 현재 J ({base['overhang']:.2f})"),
                va="bottom", ha="right", fontsize=8, color=MUTED)
        for p in ORDERS:
            rr = [base] + [r for r in rows if r["p"] == p and r["m"] > 0]
            ax.plot([r["m"] for r in rr], [r["overhang"] for r in rr],
                    color=COLOR[p], lw=2, marker=MARK[p], ms=8, mec="white",
                    mew=1.5, label=L(f"order {p}", f"{p}차"))
        ax.set_title(mname(name), fontsize=10, color=INK, loc="left")
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel(L("added marginal cost at 25°, m (%p/°)",
                        "추가한 25° 한계비용 m (%p/°)"), color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(L("toolpath true overhang (%p)", "툴패스 진짜 오버행 (%p)"),
                      color=INK)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(L("Shipping selector (k=0.2) + order-p term. Higher orders stay lower only "
                   "because they cut the angle less — compare at equal angle in the trade-off figure",
                   "출하 선택기 (k=0.2) + 차수 항 — 고차가 낮은 건 각도를 덜 줄여서다. "
                   "같은 각도 비교는 trade-off 그림"),
                 fontsize=12, color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sel", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        out = json.load(open(JSON_OUT, encoding="utf-8"))
        for r in out["grid"]:          # JSON 은 dict 키를 문자열로 바꾼다
            r["A"] = {int(p): v for p, v in r["A"].items()}
            r["B"] = {int(p): v for p, v in r["B"].items()}
    else:
        stored = json.load(open(STORED, encoding="utf-8"))
        out = {"grid": grid_experiments(stored)}
        if not args.no_sel:
            out["selector"] = selector_experiment()
        with open(JSON_OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"결과 저장: {JSON_OUT}")
    plot_tradeoff(out["grid"], "theta_power_tradeoff.png")
    plot_regret(out["grid"], "theta_power_regret.png")
    figs = ["theta_power_tradeoff.png", "theta_power_regret.png"]
    if out.get("selector"):
        plot_selector(out["selector"], "theta_power_selector.png")
        figs.append("theta_power_selector.png")
    print("그림 저장: " + ", ".join(figs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
