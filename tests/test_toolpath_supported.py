"""T5: 지지 검사기 합성 검증.

(a) 공중의 단일 수평선(z=5, 아래 없음) → 미지지 100%
(b) 그 선 위 layer_height 간격의 두 번째 선 → 두 번째 선 지지 100%
(c) 베드 위 첫 층(z=0.3) → 지지 100%

    python3 tests/test_toolpath_supported.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.gcode import parse
from conical.toolpath import check_support, sample_extrusions

LH = 0.3


def run(lines):
    items = parse(lines)
    pts, mid, w = sample_extrusions(items, width=0.45)
    return pts, *check_support(pts, mid, w, layer_height=LH, width=0.45)


def test_floating_line_unsupported():
    pts, sup, st = run(["G1 X0 Y0 Z5 E0", "G1 X20 Y0 Z5 E1"])
    assert st["unsupported_pct"] == 100.0, st


def test_second_line_supported():
    pts, sup, st = run([
        "G1 X0 Y0 Z5 E0", "G1 X20 Y0 Z5 E1",          # 1층 (공중)
        "G0 X0 Y0 Z5.3", "G1 X0 Y0 Z5.3 E1.0001",     # 위치 복귀
        "G1 X20 Y0 Z5.3 E2",                           # 2층 (1층 바로 위)
    ])
    second = pts[:, 2] > 5.15
    assert sup[second].all(), "2층이 미지지로 나옴"
    assert not sup[~second].any(), "공중 1층이 지지로 나옴"


def test_first_layer_on_bed():
    pts, sup, st = run(["G1 X0 Y0 Z0.3 E0", "G1 X20 Y0 Z0.3 E1"])
    assert st["unsupported_pct"] == 0.0, st


if __name__ == "__main__":
    test_floating_line_unsupported()
    test_second_line_supported()
    test_first_layer_on_bed()
    print("PASS: 공중선 100% 미지지 / 적층선 지지 / 첫층 베드 지지")
