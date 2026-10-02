"""T14: 기계 프로파일 주입과 G92 E 원점 처리.

왜 필요한가: 이 저장소의 파이프라인은 연구용이라 경로만 내보낸다 — 예열·호밍·
프라임·냉각이 없어 **그 파일로는 실물을 못 뽑는다.** 다음 달 프린터가 나오므로
기계 설정을 파일로 주입하는 경로를 만들었고, 그 과정에서 검사기의 G92 결함이
드러났다: 프라임 선(E12) 뒤 `G92 E0` 을 무시해 직후 한 구간을 '압출 아님'으로
흘렸다. 외부 슬라이서는 층마다 G92 를 내기도 하므로 조용히 두면 안 된다.

지키는 것:
  ① 프로파일이 `{key}` 를 채워 넣는다 (프로파일 값 + 슬라이싱 값)
  ② 채울 값이 없으면 **바로 알아볼 수 있는 오류**를 낸다 (조용히 비우지 않는다)
  ③ 시작 코드는 첫 이동보다 앞에, 종료 코드는 마지막 이동보다 뒤에 온다
  ④ `G92 E<n>` 뒤의 압출이 빠지지 않는다

    python3 tests/test_machine_profile.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from conical.gcode import Move
from conical.machine import MachineProfile
from conical.toolpath import sample_extrusions

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path, body):
    p = tmp_path / "m.ini"
    p.write_text(body, encoding="utf-8")
    return p


def test_placeholders_filled(tmp_path):
    p = _write(tmp_path, """
[machine]
name = 시험기
bed_temp = 55
start_gcode =
    M140 S{bed_temp}
    G1 Z{layer_height}
end_gcode =
    M84
""")
    mp = MachineProfile.from_file(p)
    head = [t for _, t in mp.start_items({"layer_height": 0.3})]
    assert "M140 S55" in head
    assert "G1 Z0.3" in head
    assert mp.name == "시험기"
    assert [t for _, t in mp.end_items()][-1] == "M84"


def test_missing_placeholder_is_loud(tmp_path):
    """채울 값이 없으면 조용히 비우지 말고 멈춰야 한다 — 기계로 나가는 파일이다."""
    p = _write(tmp_path, """
[machine]
start_gcode =
    M104 S{nozzle_temp}
end_gcode =
    M84
""")
    mp = MachineProfile.from_file(p)
    with pytest.raises(ValueError) as e:
        mp.start_items({})
    assert "nozzle_temp" in str(e.value)


def test_rejects_empty_and_sectionless(tmp_path):
    with pytest.raises(ValueError):
        MachineProfile.from_file(_write(tmp_path, "[other]\nx = 1\n"))
    with pytest.raises(ValueError):
        MachineProfile.from_file(_write(tmp_path, "[machine]\nname = 빈 것\n"))


def test_example_profile_loads():
    """레포에 든 템플릿이 실제로 읽히고 치환까지 끝나는지."""
    mp = MachineProfile.from_file(ROOT / "profiles" / "machine.example.ini")
    ctx = {"layer_height": 0.3, "angle": "20.0", "direction": "outward",
           "mode": "xyz", "source_stl": "x.stl"}
    head = [t for _, t in mp.start_items(ctx)]
    tail = [t for _, t in mp.end_items(ctx)]
    assert any(ln.startswith("M109") for ln in head), "노즐 온도 대기가 없다"
    assert any(ln.startswith("G28") for ln in head), "호밍이 없다"
    assert any(ln.startswith("M84") for ln in tail), "모터 해제가 없다"
    # 템플릿이 실기 확인 전이라는 표시가 이름에 남아 있어야 한다.
    assert "TEMPLATE" in mp.name


def test_injected_order_in_real_output(tmp_path):
    """실제 파이프라인을 돌려 시작/종료 코드가 이동 바깥에 오는지 본다.

    항목 목록을 이어붙이는 코드를 그대로 다시 쓰면 아무것도 검사하지 못하므로,
    conical_slice.py 를 실제로 실행해 나온 파일을 읽는다.
    """
    import subprocess
    out = tmp_path / "t.gcode"
    r = subprocess.run(
        [sys.executable, "conical_slice.py", "examples/funnel.stl",
         "--angle", "15", "--machine-profile",
         "profiles/machine.example.ini", "-o", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = out.read_text(encoding="utf-8").splitlines()

    def first(pred):
        return next(i for i, ln in enumerate(lines) if pred(ln))

    def last(pred):
        return max(i for i, ln in enumerate(lines) if pred(ln))

    def is_move(ln):
        return ln.startswith("G1 ") and " E" in ln

    start_marker = first(lambda ln: ln.startswith("; --- machine start:"))
    end_marker = first(lambda ln: ln.startswith("; --- machine end:"))
    assert start_marker < first(lambda ln: ln.startswith("; conical built-in"))
    assert end_marker > last(is_move), "종료 코드가 마지막 압출보다 앞에 있다"
    assert first(lambda ln: ln.startswith("G28")) < end_marker
    assert last(lambda ln: ln.startswith("M84")) > end_marker


def test_g92_reset_does_not_drop_extrusion():
    """프라임(E12) → G92 E0 → 본 압출. 리셋을 무시하면 첫 구간이 빠진다."""
    def items(with_reset):
        out = [("move", Move(g=1, x=0.0, y=0.0, z=0.3, e=0.0)),
               ("move", Move(g=1, x=10.0, y=0.0, z=0.3, e=12.0))]  # 프라임 10mm
        if with_reset:
            out.append(("raw", "G92 E0"))
        out += [("move", Move(g=0, x=10.0, y=5.0, z=0.3)),      # 트래블 (E 없음)
                ("move", Move(g=1, x=20.0, y=5.0, z=0.3, e=0.5))]  # 본 압출 10mm
        return out

    w_on = sample_extrusions(items(True))[2].sum()
    w_off = sample_extrusions(items(False))[2].sum()
    # 리셋을 처리하면 본 압출 10mm 가 살아난다.
    assert w_on - w_off == pytest.approx(10.0, abs=1e-6), \
        f"G92 처리 유무 차이가 {w_on - w_off:.3f}mm (10mm 여야 한다)"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_placeholders_filled(tmp)
        test_missing_placeholder_is_loud(tmp)
        test_rejects_empty_and_sectionless(tmp)
        test_injected_order_in_real_output(tmp)
    test_example_profile_loads()
    test_g92_reset_does_not_drop_extrusion()
    print("PASS: 치환·오류·템플릿 로드·G92 리셋")
