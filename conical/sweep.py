"""
sweep.py — Stage 2: 각도 vs 남은 서포트 곡선.

⚠ 레거시: 변환공간 근사. 판정 기준은 analytic 으로 통일됨(비교·재현용으로만 유지).
   변환공간의 45° 판정은 α>0 에서 물리와 다르다 — docs/warped_threshold_finding.md.

아이디어(재혁): 원뿔 각도를 이리저리 바꿔보면서, 오버행(서포트)이 가장 잘
없어지는 각도를 찾는다. 프린트할 필요 없이, 각 각도마다 '변환된 모델'에
Stage 1 오버행 분석(overhang.py)을 다시 돌리기만 하면 된다.
"""

import trimesh

from .config import THRESHOLD_DEG
from .overhang import analyze_overhangs, support_area_fraction
from .transform import transform_cone


def support_fraction(mesh, cone_angle_deg, cone_type, threshold_deg=THRESHOLD_DEG):
    """주어진 각도로 변환한 뒤, 남은 '서포트 필요 넓이' 비율(%)을 계산."""
    if cone_angle_deg == 0:
        # 각도 0 = 변환 없음 = 일반(평면) 슬라이싱. 원본 메시를 그대로 쓴다.
        tmesh = mesh
    else:
        v = transform_cone(mesh.vertices, cone_angle_deg, cone_type)
        tmesh = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    _, need = analyze_overhangs(tmesh, threshold_deg)
    return support_area_fraction(tmesh, need)


def sweep_table(mesh, angles, cone_type, threshold_deg=THRESHOLD_DEG):
    """각도 리스트를 훑어 각 각도의 남은 서포트(%) 리스트를 돌려준다."""
    return [support_fraction(mesh, a, cone_type, threshold_deg) for a in angles]
