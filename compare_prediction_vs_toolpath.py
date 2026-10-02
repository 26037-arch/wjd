"""
compare_prediction_vs_toolpath.py — 본편 실험: 메시 예측 vs 툴패스 검증.

각도 [0,10,20,30,36,44]에 대해 파이프라인(내장 슬라이서, 데모 구 subdiv3,
layer 0.4)을 돌리고 검사기 A(지지)를 실행해, '메시 기반 해석식 예측(%)'과
'툴패스 미지지(%)'를 나란히 비교한다. 스피어만 순위상관까지 —
**하드웨어 없이 예측 체계를 검증**하는 실험이다.
마지막에 --auto-bands 2 가변각 결과 1행을 추가해, 가변각이 균일각 최적보다
툴패스 기준으로도 나은지 1차 확인한다.

    python3 compare_prediction_vs_toolpath.py [model.stl]

⚠ 두 지표는 측정 대상이 다르다(메시 표면 vs 인필·시임 포함 툴패스).
  절대값 비교가 아니라 각도 간 '순위·경향' 일치가 검증 목표다.
"""

import sys
import time

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from conical import analytic
from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.plotstyle import L
from conical.profile import AngleProfile
from conical.toolpath import check_support, sample_extrusions
from conical.transform import transform_cone, transform_cone_profile
from conical.varangle import select_banded

ANGLES = [0, 10, 20, 30, 36, 44]
LAYER_H = 0.4


def toolpath_unsupported(mesh, angle_or_profile, direction="outward"):
    if isinstance(angle_or_profile, AngleProfile):
        v = transform_cone_profile(mesh.vertices, angle_or_profile, direction)
    elif angle_or_profile > 0:
        v = transform_cone(mesh.vertices, angle_or_profile, direction)
    else:
        v = mesh.vertices
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    items = slice_mesh(warped, layer_height=LAYER_H)
    real, _ = backtransform(items, angle_or_profile, direction)
    pts, mid, w = sample_extrusions(real)
    _, st = check_support(pts, mid, w, layer_height=LAYER_H)
    return st["unsupported_pct"]


def main():
    if len(sys.argv) > 1:
        base = center_on_axis(trimesh.load(sys.argv[1], force="mesh"))
        name = sys.argv[1]
    else:
        base = center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10))
        name = "demo sphere (subdiv3, r10)"
    # 파이프라인용 세분화 메시 (예측은 원본 메시로 — 각자의 자연스러운 입력)
    v, f = trimesh.remesh.subdivide_to_size(base.vertices, base.faces, max_edge=2.0)
    fine = trimesh.Trimesh(vertices=v, faces=f, process=False)

    print("=" * 66)
    print(f"[예측 vs 툴패스] {name}  (layer {LAYER_H}, 내장 슬라이서)")
    print("-" * 66)
    print(f"  {'각도':>5} | {'메시 예측(analytic %)':>20} | {'툴패스 미지지(%)':>16}")
    pred, meas = [], []
    for a in ANGLES:
        t0 = time.perf_counter()
        p = analytic.support_fraction(base, a, "outward")
        m = toolpath_unsupported(fine, float(a))
        pred.append(p)
        meas.append(m)
        print(f"  {a:4d}° | {p:19.2f}% | {m:15.2f}%   ({time.perf_counter()-t0:.0f}s)")

    rho, pval = spearmanr(pred, meas)
    print("-" * 66)
    print(f"  스피어만 순위상관 ρ = {rho:.3f} (p={pval:.3f})")

    # 가변각 1행 (--auto-bands 2 상당)
    banded = select_banded(base, 0.2, 2)
    r_max = float(np.hypot(base.vertices[:, 0], base.vertices[:, 1]).max())
    prof = AngleProfile.from_banded_result(banded, r_max)
    p_b = analytic.support_fraction_profile(base, prof)
    m_b = toolpath_unsupported(fine, prof)
    prof_txt = " ".join(f"{t:.0f}°" for t in prof.thetas_deg)
    print(f"  가변각(밴드2) [{prof_txt}] | 예측 {p_b:.2f}% | 툴패스 {m_b:.2f}%")

    # 균일 최적과 비교 코멘트
    best_uni = min(zip(meas, ANGLES, strict=True))
    print(f"  균일 최적(툴패스 기준): {best_uni[1]}° → {best_uni[0]:.2f}%  "
          f"{'/ 가변각이 더 낫다' if m_b < best_uni[0] else '/ 가변각이 이기지 못함'}")
    print("  ⚠ 절대값 아닌 순위·경향 비교 (기하 판정, 브리징·수축 무시)")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred, meas, c="#4c72b0")
    for a, x, y in zip(ANGLES, pred, meas, strict=True):
        ax.annotate(f"{a}°", (x, y), fontsize=8,
                    textcoords="offset points", xytext=(4, 4))
    ax.scatter([p_b], [m_b], c="#55a868", marker="s", label="banded-2")
    ax.set_xlabel(L("mesh prediction: analytic support (%)",
                    "메시 예측: 해석식 서포트 (%)"))
    ax.set_ylabel(L("toolpath unsupported (%)", "툴패스 미지지 (%)"))
    ax.set_title(L(f"prediction vs toolpath (Spearman ρ={rho:.2f})",
                   f"예측 vs 툴패스 (스피어만 ρ={rho:.2f})"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("prediction_vs_toolpath.png", dpi=130)
    print("  저장: prediction_vs_toolpath.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
