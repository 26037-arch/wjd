"""역변환이 순수 E 이동(리트랙트/프라임)을 보존하는지 검증.

수정 전 버그: XYZ 변화 없는 `G1 E.. F..`는 warp_len≈0 → scale=0 으로
ΔE가 0이 되어 리트랙션이 전부 사라졌다 (E가 5에서 안 내려감).

    python3 tests/test_backtransform_retraction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.backtransform import backtransform
from conical.gcode import parse


def test_retraction_preserved():
    g = """G1 X10 Y0 Z1 E0
G1 X20 Y0 Z1 E5
G1 E3 F2400
G1 X30 Y0 Z1 F6000
G1 E5 F2400
G1 X40 Y0 Z1 E10
""".splitlines()
    out, _ = backtransform(parse(g), 30, "outward")
    es = [p.e for k, p in out if k == "move" and p.e is not None]
    # E=5 도달 이후 3까지 내려갔다가(리트랙트) 10으로 끝나야 함.
    # (지시서 원안의 min(es)==3.0 은 첫 줄 E0(=0.0) 때문에 성립 불가 — 취지 유지해 수정)
    peak = next(i for i, e in enumerate(es) if abs(e - 5.0) < 1e-9)
    assert abs(min(es[peak:]) - 3.0) < 1e-9 and abs(es[-1] - 10.0) < 1e-9, es


if __name__ == "__main__":
    test_retraction_preserved()
    print("PASS: 리트랙션 보존 (E 3까지 하강 후 10으로 종료)")
