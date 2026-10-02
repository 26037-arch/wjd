"""T27: 검사기가 G-code 에서 **층고를 제대로 읽는가**.

왜 (2026-09-26): `toolpath_check.detect_layer_height` 가 `layer_h=` 패턴만 찾았는데
**우리 G-code 에 그런 문자열이 없다.** 자기기술 헤더는
`;CONICAL_META {... "layer_height":0.4 ...}` 다. 그래서 **항상 폴백 0.3 을 쓰고
있었고**, 0.3mm 로 뽑은 파일에서만 우연히 맞았다.

층고는 지지 창(`층고 × vwin_factor`)과 베드 판정에 직접 들어간다. 0.4mm 파일을
0.3 으로 재면 **전부 다른 것을 재는 것이다** — 실제로 램프 밴드2 가 0.89%p 로
보고되다가 고친 뒤 **0.77%p**(= compare_waist 와 일치)가 됐다.

`;CONICAL_META` 는 바로 이런 용도로 넣은 것인데(자기기술 G-code) 정작 검사기가
안 읽고 있었다. **자기기술 헤더를 넣는 것과 그것을 읽는 것은 다른 일이다.**

지키는 것:
  ① `;CONICAL_META` 의 layer_height 를 읽는다 (우선)
  ② 없으면 예전 `layer_h=` 패턴도 읽는다 (외부 파일 호환)
  ③ 둘 다 없으면 **못 읽었다고 알린다** (조용히 폴백하지 않는다)
  ④ 저장소의 예시 G-code 들이 실제로 읽힌다

    python3 -m pytest tests/test_layer_height_detect.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toolpath_check import detect_layer_height


def test_reads_conical_meta():
    lines = ['%s\n' % (
        ';CONICAL_META {"version":1,"direction":"outward","profile":[[0.0,24.0]],'
        '"layer_height":0.4,"extrusion_width":0.45}'), "G1 X1 Y1\n"]
    lh, known = detect_layer_height(lines)
    assert known and lh == 0.4, (lh, known)


def test_reads_legacy_pattern():
    lh, known = detect_layer_height(["; slicer layer_h=0.25\n", "G1 X1\n"])
    assert known and lh == 0.25


def test_reports_unknown_instead_of_silent_fallback():
    """③ 못 읽으면 known=False — 호출자가 경고할 수 있어야 한다."""
    lh, known = detect_layer_height(["G1 X1 Y1 E1\n", "G1 X2 Y2 E2\n"])
    assert known is False, "못 읽었는데 읽은 것처럼 보고한다"
    assert lh == 0.3, "폴백 값이 바뀌었다 — 의도한 것인지 확인할 것"


def test_broken_meta_falls_through():
    """깨진 메타에 걸려 죽지 않고 다음 후보로 넘어간다."""
    lines = [";CONICAL_META {not json at all\n", "; layer_h=0.35\n"]
    lh, known = detect_layer_height(lines)
    assert known and lh == 0.35


def test_repo_examples_are_readable():
    """④ 저장소 예시 G-code 가 실제로 읽힌다 (자기기술이 작동하는지)."""
    seen = 0
    for p in sorted((ROOT / "examples").glob("*.gcode")):
        lh, known = detect_layer_height(p.read_text(encoding="utf-8",
                                                    errors="replace").splitlines(True))
        assert known, f"{p.name}: 층고를 못 읽는다 — 자기기술 헤더가 빠졌나"
        assert 0.05 < lh < 2.0, f"{p.name}: 층고 {lh} 가 말이 안 된다"
        seen += 1
    assert seen > 0, "examples/ 에 .gcode 가 없다"


if __name__ == "__main__":
    test_reads_conical_meta()
    test_reads_legacy_pattern()
    test_reports_unknown_instead_of_silent_fallback()
    test_broken_meta_falls_through()
    test_repo_examples_are_readable()
    print("T27 OK")
