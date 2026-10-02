"""
compare_ceiling.py — 결합 실험: 실현 가능한 밴드 계획 vs 이상적 천장.

같은 모델에 대해 네 단계를 한 번에 잰다 (전부 해석식·k 없음 = 순수 성능 비교):

    baseline(0°)  →  균일 최적(각도 1개)  →  밴드 계획(실현 가능)  →  천장(이상 상한)
                     RotBot식 고정 원뿔      팀메 bands 방식          클러스터별 자유 각도

읽는 법:
  · '균일→밴드' 격차 = 부위별 가변각의 실제 이득 (실현 가능)
  · '밴드→천장' 격차 = 단일축 원뿔의 구조적 비용 (h구간 겹침 때문에 포기하는 것)
  · '천장'이 높게 남아 있으면 = 원뿔 계열 전체의 근본 한계 (irreducible)

    python3 compare_ceiling.py             # 데모 모델 6종
    python3 compare_ceiling.py model.stl   # 내 STL
"""

import time

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical import analytic
from conical.bandplan import evaluate_band_plan, plan_bands
from conical.clusters import adaptive_ceiling
from conical.config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG
from conical.meshio import center_on_axis
from conical.plotstyle import L


def uniform_best(mesh):
    best = None
    for direction in ("outward", "inward"):
        for angle in np.arange(0.0, MAX_ANGLE_DEG + 1e-9, ANGLE_STEP):
            s = analytic.support_fraction(mesh, angle, direction, THRESHOLD_DEG)
            if best is None or s < best[0] - 1e-12:
                best = (s, float(angle), direction)
    return best


def run_model(name, mesh):
    mesh = center_on_axis(mesh.copy())
    base = analytic.support_fraction(mesh, 0.0, "outward", THRESHOLD_DEG)

    t0 = time.perf_counter()
    uni = uniform_best(mesh)
    t_uni = time.perf_counter() - t0
    t0 = time.perf_counter()
    bands = plan_bands(mesh)
    banded = evaluate_band_plan(mesh, bands) if bands else base
    t_band = time.perf_counter() - t0
    t0 = time.perf_counter()
    ceil_pct, _ = adaptive_ceiling(mesh)
    t_ceil = time.perf_counter() - t0

    prof = " ".join(f"{b['angle']:.0f}{b['direction'][:3]}" for b in bands) or "-"
    print(f"{name:16s} | {base:5.1f} | {uni[0]:5.1f} ({uni[2][:3]}{uni[1]:2.0f}°) "
          f"| {banded:5.1f} [{prof}] | {ceil_pct:5.1f} "
          f"| {t_uni*1000:4.0f}/{t_band*1000:4.0f}/{t_ceil*1000:4.0f}ms")
    return {"name": name, "baseline": base, "uniform": uni[0],
            "banded": banded, "ceiling": ceil_pct, "n_bands": len(bands)}


def demo_models():
    m = {}
    m["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=10)
    funnel = trimesh.creation.cone(radius=14, height=5)
    funnel.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    m["funnel"] = funnel
    s = trimesh.creation.icosphere(subdivisions=3, radius=8)
    cyl = trimesh.creation.cylinder(radius=3, height=20)
    cyl.apply_translation([0, 0, 18])
    m["sphere+stalk"] = trimesh.util.concatenate([s, cyl])
    stem = trimesh.creation.cylinder(radius=2, height=14)
    stem.apply_translation([0, 0, 7])
    cap = trimesh.creation.cylinder(radius=8, height=3)
    cap.apply_translation([0, 0, 15.5])
    m["mushroom"] = trimesh.util.concatenate([stem, cap])
    a = trimesh.creation.icosphere(subdivisions=3, radius=6)
    b = trimesh.creation.icosphere(subdivisions=3, radius=6)
    b.apply_translation([0, 0, 30])
    m["two-spheres"] = trimesh.util.concatenate([a, b])
    m["torus"] = trimesh.creation.torus(major_radius=10, minor_radius=4)
    return m


if __name__ == "__main__":
    import sys

    print(f"{'model':16s} | base% | uniform      | banded (plan) | ceiling | time u/b/c")
    print("-" * 100)
    if len(sys.argv) > 1:
        rows = [run_model(sys.argv[1], trimesh.load(sys.argv[1], force="mesh"))]
    else:
        rows = [run_model(n, mesh) for n, mesh in demo_models().items()]
    print("-" * 100)
    print("읽는 법: 균일→밴드 격차 = 부위별 가변각의 실현 이득 / 밴드→천장 격차 = 단일축의 구조적 비용")
    print("        천장에 남은 값 = 원뿔 계열의 근본 한계(irreducible)")

    # 발표용 그래프 (영문 라벨 — 한글 폰트 문제 회피)
    names = [r["name"] for r in rows]
    xpos = np.arange(len(rows))
    w = 0.2
    fig, ax = plt.subplots(figsize=(1.9 * len(rows) + 2, 4.2))
    for i, (key, label, color) in enumerate([
            ("baseline", "planar (0°)", "#b0b0b0"),
            ("uniform", "uniform cone", "#4c72b0"),
            ("banded", "banded θ(z) plan", "#55a868"),
            ("ceiling", "per-cluster ceiling", "#c44e52")]):
        ax.bar(xpos + (i - 1.5) * w, [r[key] for r in rows], w, label=label, color=color)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(L("remaining support area (%)", "남은 서포트 면적 (%)"))
    ax.set_title(L("Support vs strategy: realizable band plan vs idealized ceiling",
                   "전략별 서포트: 실현 가능한 밴드 계획 vs 이상적 천장"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("ceiling_comparison.png", dpi=130)
    print("\nsaved: ceiling_comparison.png")
