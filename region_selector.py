"""
region_selector.py — 부위별(영역별) 각도 결정 실행 스크립트 (CLI).

모델을 오버행 심한 정도로 2~3개 영역으로 나눠, 각 영역에 원뿔 각도를 따로
정하고 그 이유를 출력한다. 실제 로직은 conical/regions.py 에 있다.

    python3 region_selector.py             # 데모용 구, 영역 2개와 3개 둘 다 보여줌
    python3 region_selector.py model.stl   # STL 파일로 실행
"""

from conical.meshio import load_mesh_or_demo
from conical.regions import select_regions

if __name__ == "__main__":
    import sys

    mesh, path = load_mesh_or_demo(sys.argv, subdivisions=4)
    if path is not None:
        print(f"[모델] {path}\n")
    else:
        print("[STL 없음 → 데모용 구(sphere)로 실행]\n")

    # 영역 2개, 3개를 각각 보여준다 (부위를 더 잘게 나누면 어떻게 달라지나)
    for n in (2, 3):
        select_regions(mesh, k=0.2, n_regions=n)
        print()
