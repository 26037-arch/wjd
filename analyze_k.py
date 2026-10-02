"""
analyze_k.py — '구간별 각도의 이득'이 k에 어떻게 의존하나 (검토 대응 · 발표 핵심).

문제(정직하게):
    "구간별이 균일보다 낫다"는 결과는 k(각도 비용 가중치)에 의존한다.
      · k가 너무 작으면 → 균일이 이미 최대각으로 감 → 구간별과 동점 (이득 0)
      · k가 너무 크면   → 둘 다 각도 0으로 감           → 동점 (이득 0)
    이득은 '중간 k' 구간에서만 난다. 그럼 그 구간이 왜 하필 거기인가?

가설(물리적 근거):
    k = "각도 1도를 서포트 몇 %p와 맞바꾸나"(교환율)이다. 따라서 물리적으로 의미 있는
    k의 규모 = 서포트 곡선의 기울기 |dSupport/dAngle| (%p per °)다. 이득이 나는 k 구간이
    이 기울기와 겹치면, k는 임의값이 아니라 '모델이 정해주는 자연 스케일'이 된다.

출력: k별 (균일/구간3 서포트, 이득) 표 + 자연 k 스케일 + 그래프(analyze_k.png)
    python3 analyze_k.py            # 데모 모델들
    python3 analyze_k.py model.stl  # 내 STL
"""

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 판정 기준 통일(2026-07 리뷰): metrics → analytic (해석식)
from conical.analytic import face_support_and_staircase
from conical.config import ANGLE_STEP, MAX_ANGLE_DEG, THRESHOLD_DEG
from conical.meshio import center_on_axis
from conical.plotstyle import L
from conical.varangle import select_banded, select_uniform


def support_curve(mesh, cone_type="outward", threshold_deg=THRESHOLD_DEG):
    """전체 모델의 각도별 남은 서포트(%) 곡선 (자연 k 스케일 계산용)."""
    areas = mesh.area_faces
    total = areas.sum()
    angles = list(range(0, MAX_ANGLE_DEG + 1, ANGLE_STEP))
    sup = []
    for a in angles:
        need, _ = face_support_and_staircase(mesh, a, cone_type, threshold_deg)
        sup.append(areas[need].sum() / total * 100.0)
    return np.array(angles), np.array(sup)


def natural_k_scale(mesh):
    """서포트 곡선의 평균 하강 기울기 |dSupport/dAngle| (%p per °).
    outward/inward 중 더 많이 줄이는 쪽 기준."""
    best = 0.0
    for ct in ("outward", "inward"):
        ang, sup = support_curve(mesh, ct)
        drop = sup[0] - sup.min()
        a_at_min = ang[int(np.argmin(sup))]
        if a_at_min > 0 and drop > 0:
            best = max(best, drop / a_at_min)
    return best


def k_sweep(mesh, k_values):
    rows = []
    for k in k_values:
        u = select_uniform(mesh, k)
        b = select_banded(mesh, k, 3)
        rows.append((k, u["support_pct"], b["support_pct"],
                     u["support_pct"] - b["support_pct"]))
    return rows


def analyze(name, mesh, ax=None):
    mesh = mesh.copy()
    mesh.merge_vertices()
    center_on_axis(mesh)   # 회전축(Z)에 XY 센터링 + 바닥 z=0 (off-axis 왜곡 방지)

    kscale = natural_k_scale(mesh)
    ks = np.round(np.linspace(0.02, 1.0, 25), 3)
    rows = k_sweep(mesh, ks)

    print(f"\n{'='*60}\n[{name}]  면수 {len(mesh.faces):,}")
    print(f"  자연 k 스케일(서포트 곡선 기울기) ≈ {kscale:.2f} %p/°")
    print(f"{'k':>6} | {'균일서포트':>8} | {'구간3서포트':>9} | {'이득':>6}")
    print("-" * 60)
    peak = max(rows, key=lambda r: r[3])
    for i, (k, us, bs, g) in enumerate(rows):
        is_peak = (k, us, bs, g) == peak
        if i % 2 == 0 or is_peak:   # 너무 빽빽하지 않게 한 줄 걸러 + 피크는 항상
            mark = "  <= 이득 최대" if is_peak else ""
            print(f"{k:6.3f} | {us:7.1f}% | {bs:8.1f}% | {g:+5.1f}%p{mark}")
    print(f"  → 이득 최대: k={peak[0]:.3f} 에서 {peak[3]:+.1f}%p "
          f"(자연 스케일 {kscale:.2f}와 비교)")

    if ax is not None:
        ks_arr = [r[0] for r in rows]
        gain = [r[3] for r in rows]
        ax.plot(ks_arr, gain, marker="o", ms=3, label=L("banding gain", "밴딩 이득"))
        ax.axvline(kscale, color="tab:red", ls="--", lw=1,
                   label=L(f"natural k≈{kscale:.2f}", f"자연 스케일 k≈{kscale:.2f}"))
        ax.set_title(name)
        ax.set_xlabel(L("k (angle cost weight)", "k (각도 비용 가중치)"))
        ax.set_ylabel(L("banding gain (support %p)", "밴딩 이득 (서포트 %p)"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return rows, kscale


def demo_models():
    models = {}
    models["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=10)
    s = trimesh.creation.icosphere(subdivisions=3, radius=8)
    cyl = trimesh.creation.cylinder(radius=3, height=20)
    cyl.apply_translation([0, 0, 18])
    models["sphere+stalk"] = trimesh.util.concatenate([s, cyl])
    return models


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        models = {sys.argv[1]: trimesh.load(sys.argv[1], force="mesh")}
    else:
        print("[STL 없음 → 데모 모델들로 실행]")
        models = demo_models()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for ax, (name, mesh) in zip(axes[0], models.items(), strict=True):
        analyze(name, mesh, ax)
    fig.suptitle(L("Banding advantage vs angle-cost k  (gain>0 only in a middle window)",
                   "각도 비용 k에 따른 밴딩 이득  (이득>0 구간이 중간에만 있다)"))
    fig.tight_layout()
    fig.savefig("analyze_k.png", dpi=130)
    print("\nsaved: analyze_k.png")
    print("해석: 이득이 나는 k 구간이 '자연 k 스케일'(빨간 점선) 근처에 있으면,")
    print("      k는 임의값이 아니라 모델의 서포트 곡선이 정해주는 값이라는 근거가 된다.")
