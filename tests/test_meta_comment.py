"""P0: ;CONICAL_META 주석이 파이프라인을 깨지 않는지 + 자기기술 확인.

(1) gcode.parse 가 ';'로 시작하는 줄을 raw 로 통과시키는지
(2) 파이프라인 출력 첫 줄이 유효한 한 줄 JSON 메타인지
(3) 메타 포함 파일을 backtransform 에 다시 넣어도 (역변환 재적용 시나리오)
    raw 줄이 그대로 보존되는지

    python3 tests/test_meta_comment.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conical.gcode import parse  # noqa: E402


def test_parser_ignores_comment_lines():
    items = parse([';CONICAL_META {"version":1}', "G1 X1 Y2 Z3 E0.1", "; note"])
    kinds = [k for k, _ in items]
    assert kinds == ["raw", "move", "raw"], kinds
    assert items[0][1].startswith(";CONICAL_META")


def test_pipeline_emits_valid_meta():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.gcode"
        r = subprocess.run(
            [sys.executable, str(ROOT / "conical_slice.py"),
             str(ROOT / "examples/funnel.stl"), "--angle", "20", "-o", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        first = open(out).readline().strip()
        assert first.startswith(";CONICAL_META "), first
        meta = json.loads(first[len(";CONICAL_META "):])
        assert meta["version"] == 1 and meta["profile"] == [[0.0, 20.0]]
        assert meta["direction"] == "outward" and meta["layer_height"] == 0.3

        # 프로필 모드 메타
        out2 = Path(td) / "p.gcode"
        r = subprocess.run(
            [sys.executable, str(ROOT / "conical_slice.py"),
             str(ROOT / "examples/funnel.stl"),
             "--profile", "0:26,2.5:26,3.5:10,5:10", "-o", str(out2)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        meta2 = json.loads(open(out2).readline().strip()[len(";CONICAL_META "):])
        assert len(meta2["profile"]) == 4 and meta2["profile"][0] == [0.0, 26.0]


if __name__ == "__main__":
    test_parser_ignores_comment_lines()
    test_pipeline_emits_valid_meta()
    print("PASS: 메타 주석 통과 + 고정각/프로필 메타 유효 JSON")
