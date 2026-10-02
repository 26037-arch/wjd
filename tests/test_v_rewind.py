"""T16: V 되감기 — 배선 감김 대책이 툴패스를 바꾸지 않는가.

왜 필요한가: 원뿔 모드에서 회전축 V 는 압출점의 방위각을 따라가므로 한 방향으로
계속 감긴다. 실측(funnel 20°) **44.5회전** — 슬립링이 없으면 배선이 끊어진다.

대책은 '트래블 중에 V 를 360°의 배수만큼 되감기'다. 이게 성립하는 이유는
**Rz 가 360° 주기라 부품의 기계좌표가 변하지 않기** 때문이다. 그 전제가 깨지면
(예: 되감기 이동이 X/Y/Z 를 건드리거나, 압출 중에 되감으면) **출력물이 망가진다.**
그래서 여기서 고정한다.

지키는 것:
  ① 압출 이동의 기계 X/Y/Z/E 가 **정확히 같다**, V 는 360° 배수만큼만 다르다
  ② 감김이 `max_turns + 1` 회전 이내로 유지된다
  ③ 되감기 이동은 압출하지 않는다
  ④ 되감기 전에 **여태 퇴적한 것보다 위로** 들어올린다 (중간 각도에서 충돌 방지)
  ⑤ 되감기 구간이 표시돼 있어 검사기가 '사고 급회전'과 구별한다
  ⑥ 임계를 크게 줄수록 되감기 횟수가 줄고, 툴패스는 어느 값에서도 보존된다

    python3 tests/test_v_rewind.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.open5x import PRUSA_UV, add_v_rewinds, check_open5x, to_open5x
from conical.planar_slicer import slice_mesh
from conical.transform import transform_cone

ROOT = Path(__file__).resolve().parent.parent
ANGLE = 20.0


def _base_items():
    mesh = center_on_axis(trimesh.load(ROOT / "examples" / "funnel.stl",
                                       force="mesh"))
    v = transform_cone(mesh.vertices, ANGLE, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=v, faces=mesh.faces,
                                       process=False), layer_height=0.3)
    real, _ = backtransform(items, ANGLE, "outward")
    return to_open5x(real, ANGLE, "outward", PRUSA_UV)[0]


def _v_of(move):
    for t in (move.extra or "").split():
        if t.upper().startswith("V"):
            return float(t[1:])
    return None


def _extrusions(items):
    """압출 이동만 (x, y, z, e, V). 되감기는 압출이 아니라 여기 안 들어온다."""
    out, e_prev = [], 0.0
    for kind, p in items:
        if kind != "move":
            continue
        if p.e is not None and p.e > e_prev + 1e-9:
            out.append((p.x, p.y, p.z, p.e, _v_of(p)))
        if p.e is not None:
            e_prev = p.e
    return out


def _v_series(items):
    return np.array([v for kind, p in items if kind == "move"
                     for v in [_v_of(p)] if v is not None])


def _angdiff(d):
    """360° 주기 각도 차이. `d % 360` 만 쓰면 −360 근처에서 틀린다."""
    d = abs(d) % 360.0
    return min(d, 360.0 - d)


def test_toolpath_is_preserved_exactly():
    """이 기법의 전제. 깨지면 출력물이 망가진다."""
    base = _base_items()
    a = _extrusions(base)
    for turns in (0.5, 1.0, 3.0):
        b = _extrusions(add_v_rewinds(base, PRUSA_UV, max_turns=turns)[0])
        assert len(a) == len(b), f"turns={turns}: 압출 수 {len(a)} != {len(b)}"
        for i, (p, q) in enumerate(zip(a, b, strict=True)):
            assert abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 \
                and abs(p[2] - q[2]) < 1e-9, f"turns={turns} [{i}]: XYZ 가 바뀌었다"
            assert abs(p[3] - q[3]) < 1e-9, f"turns={turns} [{i}]: E 가 바뀌었다"
            assert _angdiff(p[4] - q[4]) < 1e-6, \
                f"turns={turns} [{i}]: V 차이가 360° 배수가 아니다 ({p[4]} vs {q[4]})"


def test_wind_is_bounded():
    """감김이 max_turns + 1 회전 이내 — 되감기는 트래블에서만 하므로 +1 이다."""
    base = _base_items()
    v0 = _v_series(base)
    assert np.abs(v0 - v0[0]).max() / 360.0 > 20, "기준 출력이 원래 많이 감겨야 한다"
    for turns in (0.5, 1.0, 2.0):
        v = _v_series(add_v_rewinds(base, PRUSA_UV, max_turns=turns)[0])
        wind = np.abs(v - v[0]).max() / 360.0
        assert wind <= turns + 1.0 + 1e-6, f"turns={turns}: 감김 {wind:.2f}회전"


def test_rewind_moves_do_not_extrude():
    base = _base_items()
    out, _ = add_v_rewinds(base, PRUSA_UV, max_turns=1.0)
    inside, checked = False, 0
    for kind, p in out:
        if kind != "move":
            if isinstance(p, str):
                up = p.upper()
                if "V_REWIND BEGIN" in up:
                    inside = True
                elif "V_REWIND END" in up:
                    inside = False
            continue
        if inside:
            checked += 1
            assert p.e is None, f"되감기 이동이 압출한다: E={p.e}"
    assert checked > 0, "되감기 이동이 하나도 없다"


def test_rewind_lifts_above_deposited():
    """중간 각도에서 부품이 돌아가므로, 들어올린 높이가 퇴적물 위여야 한다."""
    base = _base_items()
    out, _ = add_v_rewinds(base, PRUSA_UV, max_turns=1.0, clearance=2.0)

    # 각 되감기 블록의 '회전' 이동(들어올림 → **회전** → 내려오기 중 가운데)이
    # 그 시점까지 퇴적한 최고 높이 + clearance 위에 있어야 한다.
    z_dep, e_prev, block = None, 0.0, None
    worst = None
    for kind, p in out:
        if kind != "move":
            if isinstance(p, str) and "V_REWIND BEGIN" in p.upper():
                block = []
            elif isinstance(p, str) and "V_REWIND END" in p.upper() and block:
                spin_z = block[1].z if len(block) > 1 else None
                if spin_z is not None and z_dep is not None:
                    margin = spin_z - z_dep
                    worst = margin if worst is None else min(worst, margin)
                block = []
            continue
        if block is not None:
            block.append(p)
        if p.e is not None and p.e > e_prev + 1e-9 and p.z is not None:
            z_dep = p.z if z_dep is None else max(z_dep, p.z)
        if p.e is not None:
            e_prev = p.e
    assert worst is not None and worst >= 2.0 - 1e-6, \
        f"되감기 회전 높이가 퇴적물 위 2mm 를 확보 못 했다 (최소 여유 {worst})"


def test_threshold_trades_rewinds_for_wind():
    """임계를 올리면 되감기가 줄고 감김이 는다 — 시간 vs 배선 여유의 트레이드오프.

    실측(funnel 20°): 0.5 → 148회/1.0회전, 1.0 → 32회/2.0회전, 2.0 → 16회/3.0회전.
    0.5 는 1.0 보다 시간이 3배 드는데 감김은 1회전밖에 못 줄인다 (기본값이 1.0 인 이유).
    """
    base = _base_items()
    prev_n, prev_wind = None, None
    for turns in (0.5, 1.0, 2.0, 3.0):
        out, stats = add_v_rewinds(base, PRUSA_UV, max_turns=turns)
        wind = np.abs(_v_series(out) - _v_series(out)[0]).max() / 360.0
        if prev_n is not None:
            assert stats["rewinds"] <= prev_n, \
                f"임계를 올렸는데 되감기가 늘었다: {turns}"
            assert wind >= prev_wind - 1e-6, \
                f"임계를 올렸는데 감김이 줄었다: {turns}"
        prev_n, prev_wind = stats["rewinds"], wind


def test_checker_separates_rewind_from_slam():
    """되감기의 큰 회전을 '사고 급회전'으로 세면 안 된다."""
    base = _base_items()
    out, _ = add_v_rewinds(base, PRUSA_UV, max_turns=1.0)
    _, st = check_open5x(out, PRUSA_UV)
    assert st["rewinds"] > 0
    assert st["rewind_max_deg"] >= 360.0, "되감기 회전이 기록되지 않았다"
    # 되감기를 빼고 본 급회전 최대는 되감기 크기보다 작아야 한다.
    assert st["v_max_step_deg"] < st["rewind_max_deg"], \
        f"되감기가 급회전으로 섞였다: {st}"


def test_cable_wrap_finding_clears():
    """원래 치명이던 배선 감김이 되감기 뒤에는 치명에서 빠진다."""
    base = _base_items()
    before = [m for s, m in check_open5x(base, PRUSA_UV)[0]
              if s == "치명" and "감김" in m]
    after = [m for s, m in check_open5x(add_v_rewinds(base, PRUSA_UV,
                                                      max_turns=1.0)[0],
                                        PRUSA_UV)[0]
             if s == "치명" and "감김" in m]
    assert before, "기준 출력에서 감김이 치명으로 안 잡혔다"
    assert not after, f"되감기 뒤에도 감김이 치명이다: {after}"


if __name__ == "__main__":
    test_toolpath_is_preserved_exactly()
    test_wind_is_bounded()
    test_rewind_moves_do_not_extrude()
    test_rewind_lifts_above_deposited()
    test_threshold_trades_rewinds_for_wind()
    test_checker_separates_rewind_from_slam()
    test_cable_wrap_finding_clears()
    print("PASS: 툴패스 보존, 감김 상한, 비압출, 들어올림, 검사기 분리, 치명 해소")
