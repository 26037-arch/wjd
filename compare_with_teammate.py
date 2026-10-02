"""
compare_with_teammate.py — 팀메 독립 구현(find_conical_angle)과 점 단위 교차 검증.

왜 이게 검증 계층의 한 층인가:
    실물 출력이 불가능한 조건에서, "우리 계산이 맞다"의 마지막 근거는
    **같은 수식을 다른 사람이 따로 구현한 코드와 맞춰보는 것**이다.
    팀메 저장소(26037-arch/find_conical_angle)의 evaluation.py 는 우리
    conical/analytic.py 와 독립적으로 작성됐지만 같은 수식을 쓴다:

        g(α) = n_z·cosα + d·n_r·sinα          (면이 국소 원뿔 레이어와 이루는 각)
        α*   = asin(κ/R) − atan2(n_z, d·n_r)   (그 면이 자기지지가 되는 최소 각)

    구현은 완전히 다르다 — 우리는 numpy 벡터화, 팀메는 면 단위 파이썬에
    주기적 사인 구간 교차와 실패 시 대체각(fallback)까지 처리한다.
    같은 입력에 같은 답이 나오면, 두 사람이 같은 실수를 했을 확률만 남는다.

⚠ κ(임계) 정의는 서로 다르다 — 이 스크립트는 **같은 κ 를 넣어** 비교한다:
    · 우리   : 고정 κ = −sin(임계각) = −sin(45°) ≈ −0.7071  (Cura/PrusaSlicer 관례)
    · 팀메   : 성분마다 q = B/√A 로부터 κ = −κ_max·q/(q+q₀),  κ ∈ (−0.3, 0]
    즉 팀메 쪽은 형상에 따라 임계가 변하고 범위도 훨씬 엄격하다(0°~17.5° 상당).
    어느 쪽이 옳은지는 이 스크립트가 판단하지 않는다. 여기서 검증하는 것은
    **κ 가 주어졌을 때 두 구현이 같은 각도를 내는가** 뿐이다.

사용:
    python3 compare_with_teammate.py --teammate /path/to/find_conical_angle [model.stl]

팀메 저장소가 없으면 안내만 하고 종료한다(회귀 테스트를 깨지 않는다).
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

from conical import analytic
from conical.config import MAX_ANGLE_DEG, THRESHOLD_DEG
from conical.meshio import center_on_axis


def load_teammate(path):
    """팀메 패키지를 import 경로에 올린다. 실패하면 None."""
    p = Path(path).expanduser().resolve()
    if not (p / "conical_slicing" / "evaluation.py").exists():
        return None
    sys.path.insert(0, str(p))
    try:
        from conical_slicing import evaluation
        from conical_slicing import mesh as tm_mesh
        return evaluation, tm_mesh
    except Exception as exc:      # noqa: BLE001
        print(f"⚠ 팀메 패키지 import 실패: {exc}")
        return None


def compare(mesh, evaluation, tm_mesh, direction="outward",
            threshold_deg=THRESHOLD_DEG, angle_max=MAX_ANGLE_DEG):
    """면별 임계각을 두 구현으로 계산해 대조."""
    kappa = -math.sin(math.radians(threshold_deg))

    # 우리: 벡터화 (NaN = 이 방향으로는 도달 불가)
    ours = analytic.critical_angle(mesh, direction, threshold_deg, angle_max)

    # 팀메: 면 단위. 같은 κ·같은 각도범위를 넣는다.
    nz = mesh.face_normals[:, 2]
    nr = analytic.radial_normal(mesh)
    areas = mesh.area_faces
    cents = mesh.vertices[mesh.faces].mean(axis=1)

    theirs = np.full(len(nz), np.nan)
    feasible = np.zeros(len(nz), dtype=bool)
    for i in range(len(nz)):
        face = tm_mesh.FaceGeometry(
            face_index=int(i), area=float(areas[i]), nz=float(nz[i]),
            radial_normal=float(nr[i]), centroid=[float(v) for v in cents[i]])
        r = evaluation.minimum_feasible_angle(
            face, kappa, direction, 0.0, float(angle_max))
        theirs[i] = float(r["angle"])
        feasible[i] = bool(r["feasible"])

    # 우리가 해를 낸 면(= 도달 가능)만 정직하게 대조한다.
    both = ~np.isnan(ours) & feasible
    diff = np.abs(ours[both] - theirs[both]) if both.any() else np.array([])

    # 점수 함수 g 자체도 몇 각도에서 대조
    gmax = 0.0
    for a in (0.0, 10.0, 26.0, 44.0):
        g_ours = analytic.overhang_score(mesh, a, direction)
        rad, sgn = math.radians(a), (1.0 if direction == "outward" else -1.0)
        g_theirs = nz * math.cos(rad) + sgn * nr * math.sin(rad)
        gmax = max(gmax, float(np.abs(g_ours - g_theirs).max()))

    return {
        "faces": len(nz), "compared": int(both.sum()),
        "ours_nan": int(np.isnan(ours).sum()),
        "theirs_infeasible": int((~feasible).sum()),
        "max_diff": float(diff.max()) if diff.size else float("nan"),
        "mean_diff": float(diff.mean()) if diff.size else float("nan"),
        "over_0p01": int((diff > 0.01).sum()) if diff.size else 0,
        "score_max_diff": gmax,
    }


def main():
    ap = argparse.ArgumentParser(description="팀메 독립 구현과 교차 검증")
    ap.add_argument("stl", nargs="?", default=None)
    ap.add_argument("--teammate", default=os.environ.get("TEAMMATE_REPO"),
                    help="find_conical_angle 저장소 경로 (또는 TEAMMATE_REPO 환경변수)")
    args = ap.parse_args()

    if not args.teammate:
        print("팀메 저장소 경로가 필요하다:")
        print("  git clone https://github.com/26037-arch/find_conical_angle")
        print("  python3 compare_with_teammate.py --teammate ./find_conical_angle")
        return 0
    loaded = load_teammate(args.teammate)
    if loaded is None:
        print(f"⚠ {args.teammate} 에서 conical_slicing 패키지를 못 찾았다.")
        return 0
    evaluation, tm_mesh = loaded

    if args.stl:
        models = [(args.stl, center_on_axis(trimesh.load(args.stl, force="mesh")))]
    else:
        import compare_waist
        models = [
            ("구 (subdiv3)",
             center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))),
            ("램프", compare_waist.waisted_model()),
            ("funnel", center_on_axis(trimesh.load("examples/funnel.stl", force="mesh"))),
        ]

    print("=" * 78)
    print("팀메 독립 구현 대조 — 같은 κ 를 넣고 면별 임계각을 비교")
    print(f"  κ = −sin({THRESHOLD_DEG:.0f}°) = {-math.sin(math.radians(THRESHOLD_DEG)):.4f}, "
          f"각도범위 0~{MAX_ANGLE_DEG}°")
    worst = 0.0
    for name, m in models:
        for direction in ("outward", "inward"):
            r = compare(m, evaluation, tm_mesh, direction)
            worst = max(worst, r["score_max_diff"],
                        0.0 if math.isnan(r["max_diff"]) else r["max_diff"])
            print(f"\n[{name} · {direction}]  면 {r['faces']:,}")
            print(f"  g(α) 최대 차이     : {r['score_max_diff']:.2e}  (4개 각도에서)")
            print(f"  임계각 대조 대상   : {r['compared']:,}면 "
                  f"(우리 해 없음 {r['ours_nan']:,}, 팀메 불가 {r['theirs_infeasible']:,})")
            if r["compared"]:
                print(f"  임계각 최대 차이   : {r['max_diff']:.2e}°  "
                      f"평균 {r['mean_diff']:.2e}°  (0.01° 초과 {r['over_0p01']}면)")
    print("\n" + "=" * 78)
    if worst < 1e-6:
        print(f"판정: 두 독립 구현이 일치한다 (최대 차이 {worst:.2e}).")
    else:
        print(f"⚠ 판정: 최대 차이 {worst:.3e} — 원인 조사 필요.")
    print("⚠ 이 검사는 '같은 κ 에서 두 구현이 같은가'만 본다. κ 정의 차이는")
    print("  docs/teammate_comparison.md 참고 (우리=고정 45°, 팀메=형상 의존).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
