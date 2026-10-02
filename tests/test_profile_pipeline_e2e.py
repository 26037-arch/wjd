"""T4: 가변각(2밴드) 전체 파이프라인 엔드투엔드.

examples/funnel.stl 에 2밴드 프로필로 conical_slice.py 실행 —
예외 없이 종료 + 출력 G-code 의 E 단조증가(내장 슬라이서는 리트랙션 없음).

    python3 tests/test_profile_pipeline_e2e.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conical.gcode import parse  # noqa: E402


def test_profile_pipeline_e2e():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "funnel_profile.gcode"
        r = subprocess.run(
            [sys.executable, str(ROOT / "conical_slice.py"),
             str(ROOT / "examples/funnel.stl"),
             "--profile", "0:26,2.5:26,3.5:10,5:10", "-o", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        with open(out) as fh:
            items = parse(fh.readlines())
        es = [p.e for k, p in items if k == "move" and p.e is not None]
        assert es, "E 없음"
        assert all(b >= a - 1e-9 for a, b in zip(es, es[1:], strict=False)), "E 비단조"
        # 메타에 프로필 기록 확인 (1줄=CONICAL_META JSON, 2줄=legacy)
        with open(out) as fh:
            head = fh.readline() + fh.readline()
        assert ";CONICAL_META" in head and "profile" in head, head


if __name__ == "__main__":
    test_profile_pipeline_e2e()
    print("PASS: 2밴드 프로필 파이프라인 완주, E 단조증가, 메타 기록")
