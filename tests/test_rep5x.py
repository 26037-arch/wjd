"""T20: REP5X 출력 모드 (헤드 틸트+요, 펌웨어가 역기구학).

왜 필요한가: Open5x 와 **출력 계약이 다르다.** 여기서는 슬라이서가 기계좌표를
계산하지 않고 **노즐 팁 위치(부품좌표) + 공구 방향(B/C)** 만 낸다. 그래서 틀릴 수
있는 곳이 좌표가 아니라 **각도**인데, 각도는 **부호 하나가 틀려도 파일이 멀쩡해
보인다** — 노즐이 반대로 기운 채 출력이 시작될 때까지 아무도 모른다. 그래서
공구 방향을 해석식으로 되돌려 맞춘다.

그리고 **C 는 ±360° 다** (`Configuration.h` I_MIN/MAX_POS, 소프트 엔드스톱 켜짐).
"Continuous yaw rotation" 은 슬립링이 푸는 **기계적** 제약이지 소프트웨어 한계가
아니다. 원뿔 G-code 는 C 가 계속 누적되므로(funnel 44.5회전) 되감기 없이는 기계가
멈춘다. 이걸 놓치면 '되감기 필요 없다' 는 잘못된 결론으로 하드웨어를 고른다.

고정하는 성질:
  ① (B, C) 를 공구 방향으로 되돌리면 **원뿔 레이어 법선의 반대**와 정확히 같다
     (outward/inward, 여러 각도·방위각에서)
  ② X/Y/Z/E/F 를 **건드리지 않는다** (Open5x 와 달리 좌표 변환이 없다)
  ③ C 가 연속 누적된다 (한 이동에 360° 점프가 없다)
  ④ 되감기 뒤 C 가 **창 안에** 있다 — 실제 파이프라인에서
  ⑤ 되감기가 **툴패스를 보존한다** (압출 이동의 X/Y/Z/E 동일, C 만 360° 배수 차이)
  ⑥ 창을 넘으면 검사가 **치명**으로 잡는다, RTCP 가 없어도 잡는다
  ⑦ 한 압출 구간이 창보다 넓게 감기면 **못 고친다고 정직하게 알린다**

    python3 -m pytest tests/test_rep5x.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import trimesh

from conical.backtransform import backtransform
from conical.gcode import Move
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.rep5x import (
    REP5X,
    Rep5xProfile,
    _c_of,
    add_c_rewinds,
    check_rep5x,
    cone_tool_direction,
    to_rep5x,
    tool_direction,
)
from conical.transform import transform_cone

ROOT = Path(__file__).resolve().parent.parent


def _pipeline(name, angle, direction="outward", layer_height=0.3):
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / name, force="mesh"))
    warped = transform_cone(mesh.vertices, angle, direction)
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False),
                       layer_height=layer_height)
    real, _ = backtransform(items, angle, direction)
    return real


def _moves(items):
    return [p for k, p in items if k == "move"]


def _b_of(extra):
    for tok in (extra or "").split():
        if tok.upper().startswith("B"):
            return float(tok[1:])
    return None


def test_tool_direction_matches_cone_normal():
    """**가장 중요한 테스트.** (B, C) → 공구 방향이 원뿔 법선의 반대와 같은가.

    부호가 하나 틀리면 노즐이 반대로 기운 채 출력이 시작된다. G-code 만 봐서는
    멀쩡해 보이므로 여기서 잡아야 한다.
    """
    for direction in ("outward", "inward"):
        for angle in (0.0, 12.0, 20.0, 45.0, 70.0):
            items = [("move", Move(g=1, x=x, y=y, z=1.0, e=i * 0.1, f=1800.0))
                     for i, (x, y) in enumerate(
                         [(10, 0), (0, 10), (-10, 0), (0, -10),
                          (7, 7), (-3, 4), (2.5, -6)])]
            out, _st = to_rep5x(items, angle, direction)
            for mv in _moves(out):
                b, c = _b_of(mv.extra), _c_of(mv.extra, "C")
                got = tool_direction(b, c)
                want = cone_tool_direction(mv.x, mv.y, angle, direction)
                # G-code 는 각도를 소수 3자리로 적는다 (0.001°). LB=54.67mm 에서
                # 팁 변위 0.95µm 이므로 무해하지만, 허용오차는 그만큼 둬야 한다.
                for g, w in zip(got, want, strict=True):
                    assert abs(g - w) < 1e-5, (
                        f"{direction} {angle}° at ({mv.x},{mv.y}): "
                        f"공구 방향 {got} vs 원뿔 법선의 반대 {want}")


def test_tilt_is_constant_and_signed_by_direction():
    """B 는 상수이고 inward 에서 부호가 뒤집힌다 (B = c·θ)."""
    items = [("move", Move(g=1, x=5.0, y=0.0, z=1.0, e=0.1, f=1800.0))]
    out_o, so = to_rep5x(items, 30.0, "outward")
    out_i, si = to_rep5x(items, 30.0, "inward")
    assert so["b"] == pytest.approx(30.0)
    assert si["b"] == pytest.approx(-30.0)
    assert _b_of(_moves(out_o)[0].extra) == pytest.approx(30.0)


def test_coordinates_pass_through_unchanged():
    """좌표 변환이 **없다** — Open5x 와의 핵심 차이.

    펌웨어가 역기구학(RTCP)을 하므로 X/Y/Z 는 노즐 팁 위치 그대로다. 여기에
    변환이 끼어들면 펌웨어가 한 번 더 변환해 **두 번 변환된다.**
    """
    real = _pipeline("funnel.stl", 20.0)
    out, _st = to_rep5x(real, 20.0, "outward")
    src, dst = _moves(real), _moves(out)
    assert len(src) == len(dst)
    for a, b in zip(src, dst, strict=True):
        assert (a.x, a.y, a.z, a.e, a.f, a.g) == (b.x, b.y, b.z, b.e, b.f, b.g)


def test_c_is_continuously_unwrapped():
    """C 가 ±180° 경계에서 점프하지 않는다 (연속 누적)."""
    real = _pipeline("funnel.stl", 20.0)
    out, st = to_rep5x(real, 20.0, "outward")
    cs = [_c_of(m.extra, "C") for m in _moves(out)]
    steps = [abs(b - a) for a, b in zip(cs, cs[1:], strict=False)]
    assert max(steps) < 360.0, f"한 이동에 {max(steps):.0f}° — unwrap 이 깨졌다"
    assert st["c_turns"] > 10.0, "실제 출력이면 C 가 크게 누적돼야 한다"


def test_rewind_keeps_c_inside_firmware_window():
    """되감기 뒤 C 가 **소프트 엔드스톱 창 안**에 있어야 한다.

    창을 넘으면 기계가 그냥 멈춘다. 되감기 전 44.5회전(funnel) / 220회전(lamp)
    이므로 이 기능 없이는 REP5X 로 못 건다.
    """
    for name, ang in (("funnel.stl", 20.0), ("lamp.stl", 24.0)):
        real = _pipeline(name, ang)
        out, st0 = to_rep5x(real, ang, "outward")
        assert st0["c_turns"] > 10.0
        out2, rs = add_c_rewinds(out, REP5X)
        assert rs["fixable"], f"{name}: 못 고친 구간 {rs['unfixable_segments']}개"
        cs = [_c_of(m.extra, "C") for m in _moves(out2)]
        assert min(cs) >= REP5X.c_min - 1e-6, f"{name}: C 최소 {min(cs):.1f}"
        assert max(cs) <= REP5X.c_max + 1e-6, f"{name}: C 최대 {max(cs):.1f}"
        fatal = [m for sev, m in check_rep5x(out2)[0] if sev == "치명"]
        assert not fatal, f"{name}: {fatal}"


def test_rewind_preserves_toolpath():
    """되감기는 **툴패스를 건드리면 안 된다.**

    되는 이유: `d̂ = Rz(C)·Ry(B)·(0,0,−1)` 은 C 에 대해 360° 주기라, C 를 360°
    배수만큼 바꿔도 공구 방향이 **정확히 같다.** RTCP 가 켜져 있으면 팁도 제자리다.
    이 전제가 깨지면 출력물이 망가지므로 강제한다.
    """
    real = _pipeline("funnel.stl", 20.0)
    out, _ = to_rep5x(real, 20.0, "outward")
    out2, rs = add_c_rewinds(out, REP5X)
    assert rs["rewinds"] > 0

    def extruding_moves(items):
        got, e_prev = [], 0.0
        for mv in _moves(items):
            if mv.e is not None and mv.e > e_prev + 1e-9:
                got.append(mv)
            if mv.e is not None:
                e_prev = mv.e
        return got

    a, b = extruding_moves(out), extruding_moves(out2)
    assert len(a) == len(b), f"압출 이동 수가 달라졌다: {len(a)} → {len(b)}"
    for m1, m2 in zip(a, b, strict=True):
        assert (m1.x, m1.y, m1.z, m1.e) == (m2.x, m2.y, m2.z, m2.e)
        d = abs(_c_of(m1.extra, "C") - _c_of(m2.extra, "C"))
        assert abs(d - round(d / 360.0) * 360.0) < 1e-6, \
            f"C 차이가 360° 배수가 아니다: {d}"
        assert _b_of(m1.extra) == _b_of(m2.extra), "B 가 바뀌었다"


def test_check_catches_window_violation_and_missing_rtcp():
    """되감기를 안 하면 치명, RTCP 가 없어도 치명."""
    real = _pipeline("funnel.stl", 20.0)
    out, _ = to_rep5x(real, 20.0, "outward")
    findings, st = check_rep5x(out)
    msgs = [m for sev, m in findings if sev == "치명"]
    assert any("소프트 엔드스톱" in m for m in msgs), msgs
    assert st["rtcp"] is True

    no_rtcp = [it for it in out
               if not (it[0] == "raw" and REP5X.rtcp_on in str(it[1]))]
    msgs2 = [m for sev, m in check_rep5x(no_rtcp)[0] if sev == "치명"]
    assert any("RTCP" in m for m in msgs2), msgs2


def test_tilt_beyond_machine_limit_raises():
    """B 축 한계(±135°)를 넘는 각도는 조용히 내보내지 않는다."""
    items = [("move", Move(g=1, x=5.0, y=0.0, z=1.0, e=0.1, f=1800.0))]
    with pytest.raises(ValueError):
        to_rep5x(items, 140.0, "outward")


def test_unfixable_run_is_reported_not_hidden():
    """한 압출 구간이 창보다 넓게 감기면 **못 고친다고 알린다.**

    되감기는 트래블에서만 할 수 있다. 압출을 안 끊고 창(기본 2회전)보다 더 감기는
    경로가 있으면 되감기로는 못 고치고, 슬라이싱 쪽에서 끊어야 한다. 조용히
    넘어가면 기계 앞에서 알게 된다.
    """
    # 압출을 끊지 않고 3회전 감기는 경로 하나
    items = [("move", Move(g=0, x=10.0, y=0.0, z=1.0, f=3000.0))]
    e = 0.0
    for i in range(1, 217):                       # 216 스텝 × 5° = 1080°
        a = math.radians(i * 5.0)
        e += 0.05
        items.append(("move", Move(g=1, x=10.0 * math.cos(a),
                                   y=10.0 * math.sin(a), z=1.0, e=e, f=1800.0)))
    out, st = to_rep5x(items, 20.0, "outward")
    assert st["c_turns"] > 2.5
    _out2, rs = add_c_rewinds(out, REP5X)
    assert not rs["fixable"], "창보다 넓은 구간인데 고칠 수 있다고 했다"
    assert rs["unfixable_segments"] > 0


def test_profile_defaults_match_firmware_config():
    """기본값이 저장소 `Configuration.h` 와 같은지 고정한다.

    이 숫자들은 조사로 얻은 것이라 조용히 바뀌면 근거를 잃는다.
    """
    p = Rep5xProfile()
    assert (p.tilt_axis, p.rot_axis) == ("B", "C")   # AXIS5_NAME, AXIS4_NAME
    assert (p.c_min, p.c_max) == (-360.0, 360.0)     # I_MIN_POS / I_MAX_POS
    assert (p.b_min, p.b_max) == (-135.0, 135.0)     # J_MIN_POS / J_MAX_POS
    assert p.lb == pytest.approx(54.67)   # DEFAULT_ROTATIONAL_JOINT_OFFSET_Z
    assert p.lc == pytest.approx(1.6)     # DEFAULT_ROTATIONAL_JOINT_OFFSET_Y
    assert p.printable_radius == pytest.approx(100.0)   # PRINTABLE_RADIUS
    assert p.rtcp_on == "G43.4"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
