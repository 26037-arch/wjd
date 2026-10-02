"""T26: **가변각 + 기계 프로파일** 조합이 실제로 파일을 낸다.

왜 고정하나 (2026-09-24): 이 조합이 **통째로 죽어 있었다.**
`conical_slice.py` 가 기계 프로파일 치환 컨텍스트를 만들 때 `f"{angle:.1f}"` 를
썼는데, `--profile`/`--auto-bands` 면 `angle` 이 None 이라 TypeError 로 끝났다.

즉 **이 연구의 핵심 전략(부위별 각도)을 실물용으로 뽑는 경로가 막혀 있었다.**
고정각은 멀쩡했기 때문에 몇 달 안 드러났다 — 실물 출력을 한 번도 안 해봐서다.
첫 실물 실험을 준비하며 램프를 뽑다가 처음 터졌다.

지키는 것:
  ① 가변각 + 기계 프로파일 → 파일이 나온다 (예외 없음)
  ② 시작/종료 G-code 가 실제로 들어간다 (예열·호밍이 빠지면 실물을 못 뽑는다)
  ③ `{angle}` 계열 키가 채워진다 — 가변각이면 |θ| 최대가 대표값
  ④ 고정각도 여전히 된다 (회귀)

    python3 -m pytest tests/test_machine_profile_varangle.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROFILE_INI = """\
[machine]
name = TEST
bed_temp = 60
start_gcode =
    ; TEST-START mode={mode} h={layer_height} angle={angle}
    ; angle_max={angle_max} angle_min={angle_min}
    ; profile={profile}
    M140 S{bed_temp}
    G28
end_gcode =
    ; TEST-END
    M84
"""


def _run(extra_args):
    with tempfile.TemporaryDirectory() as td:
        ini = Path(td) / "m.ini"
        ini.write_text(PROFILE_INI, encoding="utf-8")
        out = Path(td) / "o.gcode"
        r = subprocess.run(
            [sys.executable, "conical_slice.py", "examples/lamp.stl",
             "--layer-height", "0.6", "--machine-profile", str(ini),
             "-o", str(out), *extra_args],
            cwd=ROOT, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, f"실패:\n{r.stdout}\n{r.stderr}"
        assert out.exists(), f"파일이 안 나왔다:\n{r.stdout}\n{r.stderr}"
        return out.read_text(encoding="utf-8")


def test_variable_angle_with_machine_profile_produces_file():
    """①② 가변각 + 기계 프로파일 → 파일이 나오고 시작/종료가 들어간다."""
    g = _run(["--auto-bands", "2"])
    assert "; TEST-START" in g, "시작 G-code 가 없다 — 예열·호밍 없이 실물을 못 뽑는다"
    assert "; TEST-END" in g, "종료 G-code 가 없다"
    assert "G28" in g


def test_angle_keys_filled_for_variable_angle():
    """③ 가변각이어도 {angle} 계열이 채워진다 (대표값 = |θ| 최대)."""
    g = _run(["--auto-bands", "2"])
    start = [ln for ln in g.splitlines() if "TEST-START" in ln][0]
    assert "angle={angle}" not in start, "치환이 안 됐다"
    assert "angle=None" not in start, "None 이 그대로 들어갔다"
    amax = [ln for ln in g.splitlines() if "angle_max=" in ln][0]
    assert "angle_max={" not in amax and "None" not in amax
    prof = [ln for ln in g.splitlines() if ln.startswith("; profile=")][0]
    assert ":" in prof, f"프로필 문자열이 비었다: {prof!r}"


def test_constant_angle_still_works():
    """④ 고정각 경로 회귀."""
    g = _run(["--angle", "20"])
    assert "; TEST-START" in g and "; TEST-END" in g
    start = [ln for ln in g.splitlines() if "TEST-START" in ln][0]
    assert "angle=20.0" in start, start


if __name__ == "__main__":
    test_variable_angle_with_machine_profile_produces_file()
    test_angle_keys_filled_for_variable_angle()
    test_constant_angle_still_works()
    print("T26 OK")
