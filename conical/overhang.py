"""
overhang.py — 오버행 분석 (Stage 1). 적응형 원뿔 슬라이싱의 첫 오리지널 코드.

하는 일:
  1) 메시(모델)를 받아서
  2) 각 삼각형 면이 '아래를 얼마나 보고 있나'(오버행 각도)를 계산하고
  3) 임계각보다 심한 면 = 서포트가 필요한 오버행으로 표시한다.

이게 왜 중요하냐면, 나중에 "이 부위는 오버행이 심하니 원뿔 각도를 크게" 하고
자동으로 정하려면, 먼저 '어디가 얼마나 오버행인지'를 알아야 하기 때문이다.
"""

import numpy as np
import trimesh

from .config import THRESHOLD_DEG

# --- 각도 정의 (이 부분은 꼭 이해하고 있어야 함) --------------------------
# 출력 방향(위)은 +Z 라고 하자.
# 각 면의 바깥쪽 법선 벡터 n 의 z 성분(n_z)을 보면:
#   n_z > 0  : 면이 위를 본다 (윗면)          -> 오버행 아님
#   n_z = 0  : 수직 벽                          -> 문제 없음
#   n_z < 0  : 면이 아래를 본다 (밑면/천장)    -> 오버행 후보
#
# 오버행 각도(수평 기준)를 이렇게 정의한다:
#   overhang_angle = arcsin(-n_z)   (아래를 보는 면에 대해)
#     - 수직에 가까운 밑면 -> 0°   (괜찮음)
#     - 45° 경사 밑면      -> 45°  (보통 여기까지는 서포트 없이 됨)
#     - 완전 수평 천장     -> 90°  (제일 심한 오버행)
# 임계각(threshold)보다 크면 "서포트 필요"로 본다. 기본 45°.
# ------------------------------------------------------------------------

def analyze_overhangs(mesh: trimesh.Trimesh, threshold_deg: float = THRESHOLD_DEG):
    """면마다 오버행 각도를 계산하고 서포트 필요 여부를 반환한다.

    반환: (overhang_angle[F], needs_support[F])  — 둘 다 면 개수 F 길이의 배열
    """
    normals = mesh.face_normals          # (F, 3) 각 면의 단위 법선
    nz = normals[:, 2]                   # z 성분만 뽑기

    # 아래를 보는 면(nz<0)에 대해서만 오버행 각도 계산, 나머지는 0
    downward = nz < 0
    overhang_angle = np.zeros(len(normals))
    overhang_angle[downward] = np.degrees(np.arcsin(np.clip(-nz[downward], 0, 1)))

    needs_support = overhang_angle > threshold_deg
    return overhang_angle, needs_support


def support_area_fraction(mesh, needs_support):
    """서포트가 필요한 면들의 '넓이'가 전체의 몇 %인지 (넓이 가중치, 0~100).

    면 '개수'가 아니라 '넓이'로 재는 이유: 큰 오버행 면 하나가 작은 면 여럿보다
    실제로 더 중요하기 때문. sweep/selector 모두 이 값 하나로 서포트를 비교한다.
    """
    areas = mesh.area_faces               # 면마다 실제 넓이 (크기 가중치)
    return areas[needs_support].sum() / areas.sum() * 100.0


def summarize(mesh, overhang_angle, needs_support, threshold_deg):
    """분석 결과를 사람이 읽기 좋게 요약 출력한다."""
    areas = mesh.area_faces
    total_area = areas.sum()
    support_area = areas[needs_support].sum()

    print(f"총 삼각형 수      : {len(mesh.faces):,}")
    print(f"임계각            : {threshold_deg:.0f}°")
    print(f"서포트 필요 면 수 : {needs_support.sum():,} "
          f"({needs_support.mean()*100:.1f}%)")
    print(f"서포트 필요 넓이  : {support_area/total_area*100:.1f}% (넓이 기준)")
    print(f"최대 오버행 각도  : {overhang_angle.max():.1f}°")
