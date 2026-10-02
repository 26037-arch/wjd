"""T6: 노즐 간섭 검사기 합성 검증.

낮은 곳의 기퇴적 점 '위'로 원뿔 범위 안에 팁 배치 → 간섭 검출, 범위 밖 → 통과.

    python3 tests/test_nozzle_clearance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.gcode import parse
from conical.toolpath import HotendProfile, check_nozzle, sample_extrusions

H = HotendProfile(tip_radius=0.6, cone_half_deg=30.0, cone_height=3.0,
                  block_radius=12.0, block_z0=5.0, block_height=12.0)


def run(lines):
    items = parse(lines)
    pts, mid, w = sample_extrusions(items, width=0.45)
    return pts, *check_nozzle(pts, mid, H)


def test_collision_inside_cone():
    # 높은 벽(z=10)을 먼저 찍고, 그 '바로 옆 낮은 곳'(z=8)에 팁 배치:
    # dz=2 ≤ cone_h=3, 허용 반경 = 0.6 + 2·tan30° ≈ 1.75 → 수평 1.0 이격 → 간섭
    pts, col, st = run([
        "G1 X0 Y0 Z10 E0", "G1 X5 Y0 Z10 E1",       # 기퇴적 (높음)
        "G0 X0 Y1.0 Z8", "G1 X0 Y1.0 Z8 E1.0001",
        "G1 X5 Y1.0 Z8 E2",                          # 팁이 낮게 지나감
    ])
    low = pts[:, 2] < 9
    assert col[low].any(), "원뿔 범위 안인데 간섭 미검출"


def test_clear_outside_cone():
    # 같은 배치이되 수평 3.0 이격 → 허용 반경 1.75 밖 → 통과 (블록 dz=2 < block_z0=5)
    pts, col, st = run([
        "G1 X0 Y0 Z10 E0", "G1 X5 Y0 Z10 E1",
        "G0 X0 Y3.0 Z8", "G1 X0 Y3.0 Z8 E1.0001",
        "G1 X5 Y3.0 Z8 E2",
    ])
    low = pts[:, 2] < 9
    assert not col[low].any(), "범위 밖인데 간섭 오검출"


if __name__ == "__main__":
    test_collision_inside_cone()
    test_clear_outside_cone()
    print("PASS: 원뿔 범위 안 간섭 검출 / 범위 밖 통과")
