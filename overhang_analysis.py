"""
overhang_analysis.py — Stage 1 실행 스크립트 (CLI).

실제 로직은 conical/overhang.py 로 옮겨졌고, 이 파일은 '실행 방법'만 담당한다.
    python3 overhang_analysis.py            # 데모용 구로 실행
    python3 overhang_analysis.py model.stl  # STL 파일 분석

(기존처럼 `from overhang_analysis import analyze_overhangs` 도 계속 동작한다.)
"""

from conical.meshio import load_mesh_or_demo
from conical.overhang import analyze_overhangs, summarize

if __name__ == "__main__":
    import sys

    mesh, path = load_mesh_or_demo(sys.argv, subdivisions=4)  # 데모 구 통일 (14.7%)
    if path is None:
        print("[STL 없음 -> 데모용 구로 실행]")

    overhang_angle, needs_support = analyze_overhangs(mesh, threshold_deg=45.0)
    summarize(mesh, overhang_angle, needs_support, 45.0)

    # (확장) 결과를 색으로 칠해 STL로 저장하고 싶으면 아래 주석을 풀 것:
    # import numpy as np
    # colors = np.tile([180, 180, 180, 255], (len(mesh.faces), 1))
    # colors[needs_support] = [220, 60, 60, 255]   # 서포트 필요 = 빨강
    # mesh.visual.face_colors = colors
    # mesh.export("overhang_marked.ply")
