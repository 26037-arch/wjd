"""
transform.py — 원뿔(conical) 좌표 변환식. 이 연구의 기하학적 핵심 한 줄.

RotBot 실제 코드(Transformation_STL_var_angle.py)의 변환식을 그대로 사용한다.
    f(x, y, z) = ( x/cosθ,  y/cosθ,  z + c·√(x²+y²)·tanθ )
    c = +1 (outward 원뿔) / -1 (inward 원뿔)

⚠ 파일 이름 주의: RotBot 의 'var_angle' 은 **θ(z) 가 아니라 실행마다 고르는 상수각**
   이다(원문 대조 2026-09-21 — varangle.py 주석 참고). 위 변환식 자체는 그대로 차용이
   맞지만, 아래 transform_cone_profile 의 θ(Z′) 일반화는 RotBot 에 없는 확장이다.

즉 평평한 레이어를 원뿔 모양으로 '기울여' 쌓는 효과를, 모델 좌표를 미리
왜곡시키는 방식으로 흉내 낸다. (선행연구: RotBot/ZHAW, slicer4rtn)
"""

import numpy as np


def transform_cone(vertices, cone_angle_deg, cone_type):
    """정점 배열(N,3)에 원뿔 변환을 적용해 '새 정점 배열(N,3)'을 돌려준다.

    입력 vertices 는 건드리지 않는다(원본 보존).
    """
    th = np.radians(cone_angle_deg)
    c = 1 if cone_type == "outward" else -1
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.sqrt(x**2 + y**2)
    xt = x / np.cos(th)
    yt = y / np.cos(th)
    zt = z + c * r * np.tan(th)
    return np.column_stack([xt, yt, zt])


def transform_cone_profile(vertices, profile, direction="outward"):
    """θ(Z′) 가변각 프로필로 정점 배열을 워프 (profile.AngleProfile 사용).

    정점별로 z = Z′ − c·r·T(Z′) 의 정확 해 Z′를 구한 뒤(profile.solve_forward),
    그 지점의 θ로 XY를 1/cosθ 스케일한다. 상수 프로필이면 transform_cone 과
    수치적으로 동일하다(테스트 T2로 강제).
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.sqrt(x**2 + y**2)
    zw = profile.solve_forward(z, r, direction)
    th = np.radians(profile.theta_at(zw))
    return np.column_stack([x / np.cos(th), y / np.cos(th), zw])
