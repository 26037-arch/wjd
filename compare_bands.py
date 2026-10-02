"""
compare_bands.py — 복잡도(밴드 수) vs 성능 곡선, 툴패스 실측판.

기존 compare_complexity.py 는 '이상적 추정'(각 밴드를 상수각으로 독립 평가)으로
곡선을 그렸다. 그 추정은 각도를 바꾸는 값(블렌드)을 모른다. 여기서는

  · 밴드 각도를 프로필 단위 J 로 고르고 (블렌드 비용 포함, select_banded_j)
  · 실제로 파이프라인을 돌려 G-code 를 만들고
  · 툴패스 검사기의 **진짜 오버행 %p** 로 잰다 (페리미터 미지지에서 '아랫층
    단면 안'을 뺀 몫 — conical.toolpath.classify_unsupported 참고. 옛 지표는
    위로 좁아지는 형상에서 오버행 아닌 것을 세어 순위를 바꿨다)

두 축을 같이 본다 — 서포트만 보면 항상 큰 각도가 이기기 때문에, 면적가중
평균 왜곡각 ⟨|θ|⟩ 를 나란히 기록한다.

    python3 compare_bands.py                       # 기본: 구 + 램프, N=1..6
    python3 compare_bands.py --n-max 4             # 곡선을 N=4 까지만
    python3 compare_bands.py model.stl             # 임의 STL
    python3 compare_bands.py sphere waist:4 waist:1    # 절차 생성 모델 섞기
    python3 compare_bands.py --waist-sweep         # 허리 깊이 축으로 표본 확대
    python3 compare_bands.py --replay compare_bands_results.json   # 그림만 다시 그리기

모델 지정자:
    sphere          구 (허리 없음, icosphere r=10)
    waist:<r>       목 반경 r 인 램프형 (r=7 이면 허리 없음, 1 이면 아주 깊은 목)
    lamp            waist:2 의 별칭 (기존 '램프' 모델)
    <경로>.stl      STL 파일

⚠ 시뮬레이션 경향이며 실물 출력 검증 전 — '증명'이 아니다.
"""

import argparse
import json
import math
import os
import time

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_waist import mean_abs_angle, run_pipeline, waisted_model
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile, center_on_axis, waist_prominence
from conical.plotstyle import L
from conical.varangle import select_banded_j

DEFAULT_N_MAX = 6

# 허리 깊이 축을 훑을 때 쓰는 목 반경들 (7 = 허리 없음 → 1 = 아주 깊은 목).
WAIST_SWEEP = [7.0, 5.0, 3.0, 2.0, 1.0]


def build_model(spec):
    """모델 지정자 → (표시 이름, 메시). 지정자 문법은 모듈 docstring 참조."""
    if spec == "sphere":
        return (L("sphere", "구"),
                center_on_axis(trimesh.creation.icosphere(subdivisions=3,
                                                          radius=10.0)))
    if spec == "lamp":
        spec = "waist:2"
    if spec.startswith("waist:"):
        r_neck = float(spec.split(":", 1)[1])
        return (L(f"waist r={r_neck:g}", f"허리 r={r_neck:g}"),
                waisted_model(r_neck))
    mesh = center_on_axis(trimesh.load(spec, force="mesh"))
    return (os.path.basename(spec), mesh)


def sweep(mesh, n_max, k=DEFAULT_K):
    rp = RadiusProfile(mesh)
    r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
    rows = []
    for n in range(1, n_max + 1):
        t0 = time.time()
        r = select_banded_j(mesh, k, n, r_max, rp, MAX_SPACING_FACTOR)
        t_sel = time.time() - t0
        prof = r["profile_obj"]
        peri, total, fill, oh, ins, gapz = run_pipeline(mesh, prof,
                                                        breakdown=True)
        blends = prof.blend_intervals()
        rows.append({
            "n": n, "thetas": r["thetas"], "J": r["J"],
            "uniform_J": r["uniform_J"], "blend_penalty": r["blend_penalty"],
            "n_blends": len(blends),
            "blend_mm": sum(b - a for a, b in blends),
            "peri": peri, "total": total, "overhang": oh, "inside": ins,
            "angle": mean_abs_angle(mesh, prof), "t_sel": t_sel,
        })
    return rows


def print_monotonicity(rows, tol=0.01):
    """J 가 N 에 대해 비감소인지 — 단, **경계가 겹치는 짝에서만** 점검한다.

    원래 논거는 "N개 밴드는 이웃을 같은 각도로 두면 N−1개를 흉내낼 수 있으므로
    J 는 N 에 대해 비감소여야 한다" 였다. 그 논거는 **틀렸다**: 밴드 경계가
    `linspace` 라 N=3 의 경계(33.3%, 66.7%)는 N=4 의 경계(25%, 50%, 75%)에
    하나도 들어 있지 않다. N=4 는 66.7% 에 전이를 놓을 수 없으므로 N=3 의 해를
    **원리적으로 흉내낼 수 없다.**

    실측(허리 r=7 모델): N=3 이 J=1.2826, N=4 는 한 번 꺾는 해를 전수 조사해도
    최대 1.1356 이다(좌표하강이 찾은 값과 동일 — 탐색이 아니라 격자의 문제다).
    같은 해를 N=6 으로 흉내내면(경계가 겹친다) J 차이가 정확히 0.00e+00 이다.

    그래서 흉내내기가 보장되는 짝은 **n 이 m 을 나눌 때뿐**이다(1→2, 2→4, 3→6 …).
    그 짝에서 J 가 줄면 그때는 진짜 결함이다 — 이 점검이 지금까지 계획기 결함
    셋을 잡아낸 방식이 그것이다. 이웃한 N 사이의 감소는 결함이 아니라 격자
    비정합이므로 참고로만 찍는다.
    """
    by_n = {r["n"]: r["J"] for r in rows}
    violations = [(n, m, by_n[n] - by_n[m])
                  for n in by_n for m in by_n
                  if m > n and m % n == 0 and by_n[m] < by_n[n] - tol]
    if violations:
        print("  ⚠ 경계가 겹치는 짝에서 J 가 감소했다 — 흉내내기가 보장되는 자리라 "
              "진짜 결함이다:")
        for a, b, d in sorted(violations):
            print(f"      N={a} → N={b} (N={a} 의 경계가 N={b} 에 포함): J가 {d:.3f} 감소")
    else:
        pairs = sorted((n, m) for n in by_n for m in by_n if m > n and m % n == 0)
        print(f"  ✓ 경계가 겹치는 짝 {len(pairs)}개 전부 비감소 "
              f"({', '.join(f'{a}→{b}' for a, b in pairs)})")

    gaps = [(rows[i - 1]["n"], rows[i]["n"], rows[i]["J"] - rows[i - 1]["J"])
            for i in range(1, len(rows))]
    dips = [(a, b, g) for a, b, g in gaps if g < -tol]
    print(f"    이웃 N 증분: {', '.join(f'{g:+.3f}' for _, _, g in gaps)}")
    if dips:
        print("    (이웃 감소는 결함이 아니라 경계 비정합 — "
              + ", ".join(f"N={a}→{b} {g:+.3f}" for a, b, g in dips) + ")")


def report(name, mesh, rows):
    """한 모델의 표를 찍고, 그 모델의 요약 dict 를 돌려준다."""
    prom, z_w, r_w = waist_prominence(mesh)
    waist_txt = (f"허리 {prom:.2f} (z={z_w:.1f}, r={r_w:.2f})" if prom > 0.01
                 else "허리 없음")
    print("=" * 92)
    print(f"[{name}]  면 {len(mesh.faces):,}  높이 {mesh.bounds[1][2]:.1f}mm  {waist_txt}")
    print(f"  {'N':>2} | {'선택 각도':<26} | {'오버행':>7} | {'(페리미터)':>9} | "
          f"{'⟨|θ|⟩':>6} | {'J':>6} | {'블렌드':>10} | {'선택':>5}")
    for r in rows:
        th = ",".join(f"{t:.0f}" for t in r["thetas"])
        bl = f"{r['n_blends']}개 {r['blend_mm']:.1f}mm" if r["n_blends"] else "없음"
        print(f"  {r['n']:>2} | [{th:<24}] | {r['overhang']:6.2f}%p | "
              f"{r['peri']:8.2f}% | {r['angle']:5.1f}° | {r['J']:6.2f} | "
              f"{bl:>10} | {r['t_sel']:4.0f}s")

    best = min(rows, key=lambda r: r["overhang"])
    jbest = max(rows, key=lambda r: r["J"])
    uni = rows[0]
    print(f"  → 오버행 최소 N={best['n']} ({best['overhang']:.2f}%p), "
          f"J 최대 N={jbest['n']} (J={jbest['J']:.2f})")
    # 밴드가 균일각을 이겼는가 — J 기준(선택이 실제로 쓰는 축)과 툴패스 기준 둘 다.
    print(f"  → 균일(N=1) 대비: J {uni['J']:.2f}→{jbest['J']:.2f} "
          f"({jbest['J'] - uni['J']:+.2f}), "
          f"오버행 {uni['overhang']:.2f}%p→{best['overhang']:.2f}%p "
          f"({uni['overhang'] / best['overhang'] if best['overhang'] > 0 else float('inf'):.2f}배)")

    print_monotonicity(rows)

    return summarize(name, prom, rows)


def summarize(name, prominence, rows):
    """rows 에서 요약값을 **계산**한다 (저장하지 않는다).

    저장해 두면 `--replay` 로 옛 JSON 을 읽을 때 새로 추가한 항목이 없어서
    깨진다. 실제로 한 번 깨졌다 — 파생값은 원천에서 그때그때 뽑는다.
    """
    uni = rows[0]
    best = min(rows, key=lambda r: r["overhang"])
    jbest = max(rows, key=lambda r: r["J"])
    return {"name": name, "prominence": prominence, "rows": rows,
            "J_gain": jbest["J"] - uni["J"], "n_best": jbest["n"],
            "peri_uniform": uni["peri"], "peri_best": best["peri"],
            "oh_uniform": uni["overhang"], "oh_best": best["overhang"],
            "n_oh_best": best["n"], "oh_at_jbest": jbest["overhang"],
            "ang_at_jbest": jbest["angle"], "ang_oh_best": best["angle"]}


def plot(summaries, path="compare_bands.png"):
    # 축 범위는 모델 간 공유한다 — 값이 전부 같은 패널(구)에서 축이 확대되면
    # 평평한 선이 구조가 있는 것처럼 보인다.
    all_peri = [r["overhang"] for s in summaries for r in s["rows"]]
    all_J = [r["J"] for s in summaries for r in s["rows"]]
    ylim_p = (0, max(all_peri) * 1.30)
    ylim_J = (min(0, min(all_J)) * 1.1, max(all_J) * 1.15)

    ncols = min(3, len(summaries))
    nrows = math.ceil(len(summaries) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.4 * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(summaries):]:
        ax.axis("off")

    for ax, s in zip(flat, summaries, strict=False):
        rows = s["rows"]
        ns = [r["n"] for r in rows]
        l1, = ax.plot(ns, [r["overhang"] for r in rows], "o-", color="#2e7d32",
                      label=L("true overhang (%p)", "진짜 오버행 (%p)"))
        for r in rows:
            ax.annotate(f"{r['angle']:.0f}°", (r["n"], r["overhang"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)
        ax2 = ax.twinx()
        l2, = ax2.plot(ns, [r["J"] for r in rows], "s--", color="#4a7ebb",
                       alpha=.75,
                       label=L("J (objective, higher is better)",
                               "J (평가함수, 클수록 좋음)"))
        ax2.set_ylabel(L("J (objective)", "J (평가함수)"), color="#4a7ebb")
        ax2.set_ylim(*ylim_J)
        ax.set_xlabel(L("number of bands N", "밴드 수 N"))
        ax.set_ylabel(L("true overhang (%p)", "진짜 오버행 (%p)"),
                      color="#2e7d32")
        ax.set_xticks(ns)
        ax.set_ylim(*ylim_p)
        ax.set_title(f"{s['name']}  ·  {L('prominence', '두드러짐')} "
                     f"{s['prominence']:.2f}", fontsize=10)
        ax.legend(handles=[l1, l2], loc="lower left", fontsize=8, framealpha=.9)

    fig.suptitle(L("complexity (bands) vs performance — measured on the toolpath, "
                   "not the ideal estimate",
                   "복잡도(밴드 수) vs 성능 — 이상적 추정이 아니라 툴패스 실측"),
                 fontsize=11)
    # 전체 제목이 첫 줄 패널 제목을 덮지 않게 위쪽을 비워 둔다.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 / nrows))
    fig.savefig(path, dpi=130)
    print(f"\n그림 저장: {path}")


def plot_waist_axis(summaries, path="bands_vs_waist.png"):
    """허리 두드러짐 ↔ 밴드의 오버행 이득(배수). 표본이 3개 이상일 때만 의미가 있다.

    세로축은 **진짜 오버행의 배수**(균일 N=1 ÷ 밴드2)다. J 이득을 쓰지 않는 이유:
    `k_blend=0.1` 에서 J 이득은 허리와 거의 무관하게 1.7~2.1 로 평평해서 허리
    이야기가 전혀 안 보인다 — J 는 왜곡 감소도 같이 세기 때문이다. 이 실험이
    묻는 것은 '허리가 있어야 오버행을 줄일 수 있나' 이므로 축도 그것이어야 한다.

    밴드2 를 쓰는 이유: N 은 사람이 주는 값이고(`--auto-bands N`) 기본으로 쓰는
    것이 2 다. N 을 J 로 고르면 안 된다는 것은 본문 표에서 따로 경고한다.
    """
    pts = []
    for s in summaries:
        r2 = next((r for r in s["rows"] if r["n"] == 2), None)
        if r2 is None or r2["overhang"] <= 0 or s["oh_uniform"] <= 0:
            continue                      # 배수가 정의되지 않는 모델은 뺀다
        pts.append((s["prominence"], s["oh_uniform"] / r2["overhang"], s["name"]))
    if len(pts) < 3:
        return
    pts.sort()
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x, y, "o-", color="#2e7d32",
            label=L("overhang reduction, uniform ÷ 2-band",
                    "오버행 감소 배수 (균일 ÷ 밴드2)"))
    ax.axhline(1.0, color="#999", lw=.9, ls=":")
    ax.annotate(L("1.0 = no gain", "1.0 = 이득 없음"), (max(x) * .97, 1.0),
                textcoords="offset points", xytext=(0, 6), ha="right",
                fontsize=8, color="#666")

    # 문턱: 이득이 처음 1.5 배를 넘는 지점 앞뒤로 구간을 나눈다 (측정에서 나온 값).
    below = [p for p in pts if p[1] < 1.5]
    above = [p for p in pts if p[1] >= 1.5]
    if below and above:
        edge = (max(p[0] for p in below) + min(p[0] for p in above)) / 2
        ax.axvspan(min(x) - .05, edge, color="#bbb", alpha=.18, lw=0)
        ax.annotate(L(f"waist < {edge:.2f}: no gain",
                      f"허리 < {edge:.2f}: 이득 없음"),
                    ((min(x) - .05 + edge) / 2, max(y) * .93), ha="center",
                    fontsize=8.5, color="#666")

    for i, (px, py, name) in enumerate(pts):
        ax.annotate(name, (px, py), textcoords="offset points",
                    xytext=(0, 12 if i % 2 == 0 else -20), ha="center",
                    fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", lw=.5, color="#aaa",
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(min(x) - .06, max(x) + .06)
    ax.set_ylim(0, max(y) * 1.25)
    ax.set_xlabel(L("waist prominence  1 − r(z*)/min(max below, max above)",
                    "허리 두드러짐  1 − r(z*)/min(아래 최대, 위 최대)"))
    ax.set_ylabel(L("overhang reduction (×)", "오버행 감소 배수 (×)"))
    ax.set_title(L("bands cut overhang only where the model has a waist",
                   "밴드가 오버행을 줄이는 것은 허리가 있을 때뿐"), fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"그림 저장: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*",
                    help="모델 지정자 (sphere / waist:<r> / lamp / *.stl)")
    ap.add_argument("--n-max", type=int, default=DEFAULT_N_MAX,
                    help=f"밴드 수 곡선의 끝 (기본 {DEFAULT_N_MAX})")
    ap.add_argument("--waist-sweep", action="store_true",
                    help=f"허리 깊이 축으로 표본 확대 (목 반경 {WAIST_SWEEP})")
    ap.add_argument("--out", default="compare_bands.png")
    ap.add_argument("--results", default="compare_bands_results.json",
                    help="측정 결과를 남길 JSON (그림만 다시 그릴 때 --replay 로 읽는다)")
    ap.add_argument("--replay", metavar="JSON",
                    help="측정을 다시 하지 않고 저장된 결과로 그림만 다시 그린다. "
                         "스윕이 수십 분 걸리므로 라벨·축을 손볼 때 쓴다")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            saved = json.load(f)
        summaries = [summarize(d["name"], d["prominence"], d["rows"])
                     for d in saved]
        print(f"측정 없이 {args.replay} 로 다시 그린다 (모델 {len(summaries)}개)")
    else:
        specs = list(args.models)
        if args.waist_sweep:
            specs = ["sphere"] + [f"waist:{r:g}" for r in WAIST_SWEEP] + specs
        if not specs:
            specs = ["sphere", "lamp"]

        summaries = []
        for spec in specs:
            name, mesh = build_model(spec)
            summaries.append(report(name, mesh, sweep(mesh, args.n_max)))

        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1, default=float)
        print(f"\n측정 결과 저장: {args.results}")

    plot(summaries, args.out)
    if len(summaries) >= 3:
        # 두 번째 그림은 --out 과 같은 폴더에 둔다 (스크래치로 뽑을 때 레포를 안 건드리게).
        root, ext = os.path.splitext(args.out)
        plot_waist_axis(summaries, f"{os.path.dirname(root) or '.'}/bands_vs_waist{ext}")

    print("\n" + "=" * 92)
    # 세 값을 나란히 둔다 — J 가 고르는 것과 달성 가능한 최소가 갈리기 때문.
    # 알고리즘이 실제로 내놓는 것은 'J 최대' 쪽이다. 최소만 보고하면 과장이 된다.
    print(f"  {'모델':<22} | {'허리':>5} | {'균일 N=1':>15} | "
          f"{'밴드2 (실제 기본)':>16} | {'오버행 최소':>19}")
    for s in summaries:
        r2 = next((r for r in s["rows"] if r["n"] == 2), s["rows"][0])
        # 기준선이 0 이면 배수가 정의되지 않는다 (원뿔이 이길 여지가 없는 모델).
        gain = (f"×{s['oh_uniform'] / r2['overhang']:4.1f}"
                if r2["overhang"] > 0 and s["oh_uniform"] > 0 else "    —")
        print(f"  {s['name']:<22} | {s['prominence']:5.2f} | "
              f"{s['oh_uniform']:6.2f}%p {s['rows'][0]['angle']:5.1f}° | "
              f"{r2['overhang']:6.2f}%p {r2['angle']:5.1f}° {gain} | "
              f"N={s['n_oh_best']} {s['oh_best']:6.2f}%p {s['ang_oh_best']:5.1f}°")

    # N 을 J 로 고르면 안 된다는 것이 이 표에서 나온다 — J 는 N 에 대해 비감소라
    # (경계가 겹치는 짝에서) 항상 큰 N 으로 가는데, 진짜 오버행은 거기서 나빠진다.
    diverge = [s for s in summaries if s["oh_at_jbest"] > s["oh_best"] + 0.05]
    if diverge:
        print(f"\n  ⚠ **밴드 수 N 을 J 로 고르면 안 된다.** J 최대 N 의 오버행이 "
              f"달성 가능한 최소보다 나쁜 모델 {len(diverge)}/{len(summaries)}:")
        for s in diverge:
            print(f"      {s['name']:<22} J 최대 N={s['n_best']} → "
                  f"{s['oh_at_jbest']:.2f}%p  (최소는 N={s['n_oh_best']} "
                  f"{s['oh_best']:.2f}%p)")
        print("    J 는 각도 비용 k 때문에 왜곡이 적은 쪽(밴드가 많고 각도가 낮은 쪽)을"
              " 선호한다.\n    N 은 지금처럼 사람이 준다(--auto-bands N). 자동화하려면 "
              "N 에 대한 비용이 따로 필요하다.")
    print("⚠ 시뮬레이션 경향이며 실물 출력 검증 전 — '증명'이 아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
