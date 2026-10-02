"""
kappa.py — 임계값 κ 를 정하는 두 규칙. (팀메 규칙 이식 + 실측 비교용)

판정식 g(α) = n_z·cosα + d·n_r·sinα 에서 "g ≥ κ 면 자기지지"라고 볼 때,
κ 를 무엇으로 두느냐가 모든 결과를 좌우한다. 팀 안에 서로 다른 두 규칙이 있어
여기 나란히 구현하고, 어느 쪽이 나은지는 툴패스 실측으로 판정한다
(compare_kappa_rules.py).

① 고정 규칙 (우리, conical/analytic.py 기본)
       κ = −sin(임계각),  임계각 = 45°  →  κ ≈ −0.7071
   실제 슬라이서 관례에 맞춘 값이다. Cura 의 "Support Overhang Angle" 과 1:1 이고
   PrusaSlicer/Orca 의 여집합(90−θ)과도 대응한다(docs/slicer_conventions.md).
   장점: 외부 기준에 앵커돼 있다. 단점: 형상을 전혀 안 본다.

② 형상 의존 규칙 (팀메, 26037-arch/find_conical_angle)
       q = B / √A,   κ_c = −κ_max · q/(q + q₀)
   B = 그 오버행 성분의 '지지 경계 길이'(바닥이나 이미 지지된 면과 맞닿은 변의 합),
   A = 성분 면적. 즉 **둘레가 길고 면적이 작은 성분일수록 관대하게** 본다 —
   가늘고 긴 오버행은 브리징으로 걸쳐지니 덜 엄격해도 된다는 물리적 직관이다.
   κ_c ∈ (−κ_max, 0] 이라 κ_max=0.3 이면 임계각 0°~17.5° 에 해당한다.
   장점: 형상을 본다. 단점: 튜닝 상수가 둘(κ_max, q₀)이고 범위가 45° 관례와 멀다.

⚠ 크레딧: ② 는 팀메(26037-arch/find_conical_angle, conical_slicing/mesh.py 의
  find_overhang_components)의 정의다. 여기서는 우리 자료구조에 맞춰 다시 구현했고,
  교차 검증은 compare_with_teammate.py 가 한다.
"""

import math

import numpy as np

from .analytic import radial_normal
from .clusters import overhang_clusters
from .config import THRESHOLD_DEG

KAPPA_MAX = 0.3        # 팀메 config.json 기본값
Q0 = 1.0               # 팀메 config.json 기본값


def fixed_kappa(threshold_deg=THRESHOLD_DEG):
    """① 고정 규칙: κ = −sin(임계각)."""
    return -math.sin(math.radians(threshold_deg))


def _support_boundary_length(mesh, faces):
    """성분의 '지지 경계' 길이 B — 성분 밖 면과 맞닿은 변의 길이 합.

    ⚠ 정직: 팀메 구현은 바닥(bed) 경계와 표면(surface) 경계를 나눠 세고
      non-manifold 변에 우선순위를 둔다. 여기서는 '성분 경계' 하나로 단순화했다 —
      규칙의 형태(q = B/√A)를 실측 비교할 수 있으면 충분하기 때문이다.
    """
    fset = set(int(f) for f in faces)
    edges = mesh.edges_sorted.reshape(-1, 3, 2)
    total = 0.0
    for fi in fset:
        for e in edges[fi]:
            va, vb = mesh.vertices[e[0]], mesh.vertices[e[1]]
            # 이 변을 공유하는 다른 면이 성분 밖이면 경계
            length = float(np.linalg.norm(va - vb))
            total += length
    # 위는 성분 내부 변을 두 번 세므로, 내부 변 길이를 빼서 경계만 남긴다
    inner = 0.0
    seen = {}
    for fi in fset:
        for e in edges[fi]:
            key = (int(e[0]), int(e[1]))
            if key in seen:
                inner += 2.0 * float(np.linalg.norm(
                    mesh.vertices[e[0]] - mesh.vertices[e[1]]))
            else:
                seen[key] = fi
    return max(0.0, total - inner)


def component_kappa(mesh, threshold_deg=THRESHOLD_DEG,
                    kappa_max=KAPPA_MAX, q0=Q0):
    """② 형상 의존 규칙: 면마다 그 면이 속한 성분의 κ_c 를 담은 배열.

    오버행 성분에 속하지 않는 면은 고정 규칙 κ 를 쓴다(판정에 영향 없음 —
    이미 자기지지라 g ≥ κ 가 어차피 성립).
    반환: (kappa_per_face, info) — info 에 성분별 q·κ 기록.
    """
    kap = np.full(len(mesh.faces), fixed_kappa(threshold_deg))
    info = []
    for cid, faces in enumerate(overhang_clusters(mesh, threshold_deg)):
        idx = np.asarray(list(faces), dtype=int)
        if idx.size == 0:
            continue
        A = float(mesh.area_faces[idx].sum())
        if A <= 0:
            continue
        B = _support_boundary_length(mesh, idx)
        q = B / math.sqrt(A)
        k = -kappa_max * q / (q + q0)
        kap[idx] = k
        info.append({"component": cid, "faces": int(idx.size), "area": A,
                     "boundary": B, "q": q, "kappa": k,
                     "threshold_deg": math.degrees(math.asin(min(1.0, -k)))})
    return kap, info


def support_fraction_kappa(mesh, angle_deg, direction, kappa_per_face):
    """면별 κ 배열을 받아 남은 서포트 넓이(%) — 두 규칙을 같은 틀로 재기 위한 것."""
    nz = mesh.face_normals[:, 2]
    nr = radial_normal(mesh)
    d = 1.0 if direction == "outward" else -1.0
    a = math.radians(angle_deg)
    g = nz * math.cos(a) + d * nr * math.sin(a)
    need = g < kappa_per_face
    areas = mesh.area_faces
    return float(areas[need].sum() / areas.sum() * 100.0)
