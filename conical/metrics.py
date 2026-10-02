"""
metrics.py — 원뿔 각도 성능을 재는 '지표' 모음.

⚠ 레거시: 변환공간 근사. 판정 기준은 analytic 으로 통일됨(비교·재현용으로만 유지).
   analytic.face_support_and_staircase 가 같은 시그니처의 물리 기준 대체다.

세 가지 관점 (발표의 '복잡도 vs 성능' 표에 그대로 들어감):
  1) 서포트   : 변환 후에도 서포트가 필요한 면 (적을수록 좋음) → 재료·시간(속도)과 연결
  2) 강도proxy: '아래보기 면'이 변환공간에서 얼마나 수평인가(=계단/약함). 낮을수록 좋음.
                레이어가 표면에 나란히 정렬될수록(수직에 가까울수록) 계단이 줄고 강해진다는 가정.
  3) 평균각   : 쓴 각도의 면적가중 평균 → 왜곡·하드웨어 부담(복잡도 비용) proxy

주의(정직): 강도는 실측이 아니라 '레이어-표면 정렬' 기반의 가벼운 proxy다. '증명'이
아니라 '경향'으로만 쓴다. (실측하려면 하드웨어로 하중 시험 필요.)
"""

import numpy as np
import trimesh

from .config import THRESHOLD_DEG
from .transform import transform_cone


def transformed_mesh(mesh, angle_deg, cone_type):
    """각도 0이면 원본 그대로, 아니면 원뿔 변환된 메시를 돌려준다."""
    if angle_deg == 0:
        return mesh
    v = transform_cone(mesh.vertices, angle_deg, cone_type)
    return trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)


def face_support_and_staircase(mesh, angle_deg, cone_type, threshold_deg=THRESHOLD_DEG):
    """변환 후 각 면에 대해 (needs_support[bool], staircase[float])를 돌려준다.

      needs_support : 변환공간에서 오버행 각도가 임계각을 넘는 면
      staircase     : 아래보기 면의 |nz'| (변환공간 법선 z성분 절대값).
                      1에 가까우면 수평 천장(계단 심함/약함), 0에 가까우면 수직 벽(강함).
                      아래보기가 아닌 면은 0.
    """
    tm = transformed_mesh(mesh, angle_deg, cone_type)
    nz = tm.face_normals[:, 2]
    down = nz < 0
    over = np.zeros(len(nz))
    over[down] = np.degrees(np.arcsin(np.clip(-nz[down], 0, 1)))
    needs_support = over > threshold_deg
    staircase = np.where(down, np.abs(nz), 0.0)
    return needs_support, staircase
