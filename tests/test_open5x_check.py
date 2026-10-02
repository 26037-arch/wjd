"""T15: Open5x 기계좌표 출력 사전 점검 (검사기 C).

왜 필요한가: `toolpath_check.py` 의 검사기 A/B 는 **3축 가정**(노즐 수직, 베드
고정)이라 5축 출력에는 안 맞는다. 다음 달 프린터가 나오면 이 G-code 가 실제
기계로 들어간다. 5축은 3축보다 사고가 쉽고, 사고가 나면 기계가 상한다.

모듈 docstring 의 '실기 체크리스트'(배선 감김, 축 근처 급회전)를 사람이 기억하는
대신 검사가 기억하게 만든 것이다. 되감기를 끈 생성물에서 셋이 걸린다 —
배선 감김 **44.5회전**, V 179° 급회전, 기계 Z 음수.

⚠ 감김은 **시작 기준 최대 이탈** `max|V − V₀|` 이지 Σ|ΔV| 가 아니다. 같은 파일의
Σ|ΔV| 는 175회전인데 실제 감김은 44.5다 — 왕복은 배선을 감지 않는다.
처음에 Σ|ΔV| 를 감김이라 부르는 실수를 했고, 그래서 이 구별을 테스트로 박아둔다.

    python3 tests/test_open5x_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.gcode import Move
from conical.open5x import PRUSA_UV, check_open5x

ROOT = Path(__file__).resolve().parent.parent


def _moves(vs, z=1.0, x=10.0, tilt=20.0):
    """틸트 한 번 + 주어진 V 값들로 이동을 만든다."""
    out = [("move", Move(g=1, x=0.0, y=0.0, z=z, f=600, extra=f"U{tilt:.3f}"))]
    out += [("move", Move(g=1, x=x, y=0.0, z=z, e=float(i) * 0.1, f=1200,
                          extra=f"V{v:.3f}"))
            for i, v in enumerate(vs)]
    return out


def _sev(findings):
    return {s for s, _ in findings}


def test_clean_output_has_no_fatal():
    """완만하게 도는 정상 출력은 아무것도 안 걸려야 한다."""
    findings, st = check_open5x(_moves([0, 10, 20, 30, 40]), PRUSA_UV)
    assert "치명" not in _sev(findings), findings
    assert st["tilt"] == 20.0
    assert st["v_max_step_deg"] == 10.0


def test_tilt_over_limit_is_fatal():
    findings, _ = check_open5x(_moves([0, 5], tilt=120.0), PRUSA_UV)
    assert any("틸트" in m for s, m in findings if s == "치명"), findings


def test_missing_tilt_is_fatal():
    items = [("move", Move(g=1, x=1.0, y=0.0, z=1.0, extra="V0.0"))]
    findings, _ = check_open5x(items, PRUSA_UV)
    assert any("틸트" in m for s, m in findings if s == "치명"), findings


def test_cable_wrap_is_fatal():
    """V 가 한 방향으로 계속 누적되면 배선이 감긴다 — 슬립링이 없으면 치명."""
    vs = [i * 30.0 for i in range(200)]          # 약 16.6회전
    findings, st = check_open5x(_moves(vs), PRUSA_UV, max_dv_deg=45.0)
    assert st["v_wind_turns"] > 15, st
    # 총 회전량과 감김을 구별해야 한다 — 한 방향으로만 돌았으니 둘이 같다.
    assert abs(st["v_total_turns"] - st["v_wind_turns"]) < 0.1, st
    assert any("배선" in m for s, m in findings if s == "치명"), findings


def test_wind_is_not_total_rotation():
    """+360 돌고 −360 돌면 배선은 제자리다. Σ|ΔV| 를 감김이라 부르면 안 된다."""
    vs = [0.0, 180.0, 360.0, 180.0, 0.0] * 20      # 왕복만
    _, st = check_open5x(_moves(vs), PRUSA_UV, max_dv_deg=200.0)
    assert st["v_wind_turns"] <= 1.01, st
    assert st["v_total_turns"] > 10, st


def test_azimuth_flip_is_flagged():
    """축 근처에서 V 가 한 이동에 180° 뛰는 것 — docstring 체크리스트 (b)."""
    findings, st = check_open5x(_moves([0.0, 179.0]), PRUSA_UV)
    assert st["v_max_step_deg"] == 179.0
    assert any("뛴다" in m for s, m in findings if s == "치명"), findings


def test_negative_machine_z_is_fatal():
    findings, _ = check_open5x(_moves([0, 5], z=-1.0), PRUSA_UV)
    assert any("소프트리밋" in m for s, m in findings if s == "치명"), findings


def test_off_bed_is_fatal():
    findings, _ = check_open5x(_moves([0, 5], x=200.0), PRUSA_UV, bed_radius=90)
    assert any("베드" in m for s, m in findings if s == "치명"), findings


def test_real_pipeline_output_trips_known_risks():
    """실제 파이프라인 출력에서 알려진 위험 셋이 잡히는지 (통합).

    이 셋은 '고쳐야 할 결함'이 아니라 **실기 전에 사람이 알아야 할 사실**이다.
    되감기나 Z 오프셋 없이 그대로 걸면 기계가 상한다.
    """
    import subprocess
    import tempfile

    from conical import gcode as gc

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "o5.gcode"
        r = subprocess.run(
            [sys.executable, "conical_slice.py", "examples/funnel.stl",
             "--angle", "20", "--mode", "open5x", "--v-rewind", "0",
             "-o", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        items = gc.parse(out.read_text().splitlines(keepends=True))

    findings, st = check_open5x(items, PRUSA_UV)
    assert st["v_wind_turns"] > 20, f"배선 감김이 안 잡혔다: {st}"
    assert st["v_max_step_deg"] > 90, f"급회전이 안 잡혔다: {st}"
    assert st["z_min"] < 0, f"기계 Z 음수가 안 잡혔다: {st}"
    assert len([1 for s, _ in findings if s == "치명"]) >= 3, findings


def test_default_pipeline_has_rewind_on():
    """기본 실행(되감기 켜짐)에서는 배선 감김이 치명에서 빠져야 한다.

    위 테스트는 `--v-rewind 0` 으로 날것을 본 것이고, 여기서는 **기본값으로
    돌렸을 때 실제로 안전해지는가**를 본다. 기본값이 조용히 바뀌면 여기서 걸린다.
    """
    import subprocess
    import tempfile

    from conical import gcode as gc

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "o5.gcode"
        r = subprocess.run(
            [sys.executable, "conical_slice.py", "examples/funnel.stl",
             "--angle", "20", "--mode", "open5x", "-o", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        items = gc.parse(out.read_text().splitlines(keepends=True))

    findings, st = check_open5x(items, PRUSA_UV)
    assert st["v_wind_turns"] <= 2.0 + 1e-6, f"기본값에서 감김이 크다: {st}"
    assert st["rewinds"] > 0, "기본값인데 되감기가 없다"
    assert not [m for s, m in findings if s == "치명" and "감김" in m], findings


if __name__ == "__main__":
    test_clean_output_has_no_fatal()
    test_tilt_over_limit_is_fatal()
    test_missing_tilt_is_fatal()
    test_cable_wrap_is_fatal()
    test_azimuth_flip_is_flagged()
    test_negative_machine_z_is_fatal()
    test_off_bed_is_fatal()
    test_wind_is_not_total_rotation()
    test_real_pipeline_output_trips_known_risks()
    test_default_pipeline_has_rewind_on()
    print("PASS: 정상 통과 / 틸트·배선·급회전·Z음수·베드이탈 검출 / 실제 출력 통합")
