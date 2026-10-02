"""
cone_selector.py — Stage 3 실행 스크립트 (CLI).

추천 각도/방향 + '이유' + k 민감도 분석을 출력한다. 실제 결정 로직은
conical/selector.py 에, 설정값은 conical/config.py 에 있다.
    python3 cone_selector.py            # 데모용 구로 실행
    python3 cone_selector.py model.stl  # STL 파일에 대해 결정
"""

from conical.meshio import load_mesh_or_demo
from conical.selector import k_sensitivity, select_cone

if __name__ == "__main__":
    import sys

    mesh, path = load_mesh_or_demo(sys.argv, subdivisions=4)
    if path is not None:
        print(f"[모델] {path}")
    else:
        print("[STL 없음 → 데모용 구(sphere)로 실행]")

    # 1) 특정 k에 대한 결정 + 이유
    select_cone(mesh, k=0.2)

    # 2) k 민감도 분석
    k_sensitivity(mesh, [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
