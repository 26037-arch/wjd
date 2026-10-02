"""T8: 압출 종류 태깅 (;TYPE:) 과 종류별 집계.

왜 필요했나: 희소 인필은 레이어마다 방향이 0/90° 로 바뀌어 '아래에 아무것도
없는' 구간이 원래 많다(브리징으로 정상 출력). 이걸 오버행 미지지와 같이 세면
우리가 재려던 표면(페리미터) 지지가 인필에 묻힌다 — 램프 모델에서 실제로
전체 수치의 순위가 뒤집혔다(compare_waist.py).

    (a) 내장 슬라이서가 Slic3r/PrusaSlicer 관례의 ;TYPE: 주석을 낸다
    (b) 역변환이 그 주석 줄을 보존한다 (raw 통과)
    (c) sample_extrusions(return_types=True) 가 페리미터/인필을 갈라낸다
    (d) 기본 호출(3-튜플)은 그대로 — 기존 호출자 회귀 방지

    python3 tests/test_extrusion_types.py
"""

import sys
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.backtransform import backtransform  # noqa: E402
from conical.meshio import center_on_axis  # noqa: E402
from conical.planar_slicer import slice_mesh  # noqa: E402
from conical.toolpath import sample_extrusions  # noqa: E402


def _items():
    m = center_on_axis(trimesh.creation.icosphere(subdivisions=2, radius=8.0))
    return slice_mesh(m, layer_height=0.4)


def test_slicer_emits_type_comments():
    raws = [p for k, p in _items() if k == "raw"]
    assert any(r == ";TYPE:PERIMETER" for r in raws), "페리미터 태그 없음"
    assert any(r == ";TYPE:FILL" for r in raws), "인필 태그 없음"


def test_backtransform_preserves_type_comments():
    items = _items()
    real, _ = backtransform(items, 20.0, "outward")
    raws = [p for k, p in real if k == "raw"]
    assert any(r == ";TYPE:PERIMETER" for r in raws), "역변환이 태그를 잃음"
    assert any(r == ";TYPE:FILL" for r in raws), "역변환이 태그를 잃음"


def test_types_split_and_backcompat():
    items = _items()
    # (d) 기본 호출은 3-튜플 그대로
    out3 = sample_extrusions(items)
    assert len(out3) == 3, f"기본 반환이 3-튜플이 아님: {len(out3)}"

    pts, mid, w, kinds = sample_extrusions(items, return_types=True)
    assert len(kinds) == len(pts) == len(w)
    peri = (kinds == 0).sum()
    fill = (kinds == 1).sum()
    assert peri > 0 and fill > 0, f"분류 실패 (페리미터 {peri}, 인필 {fill})"
    assert peri + fill == len(kinds), "분류 안 된 압출점이 있음"
    # 좌표·가중치는 기본 호출과 동일해야 한다
    assert (pts == out3[0]).all() and (w == out3[2]).all()


if __name__ == "__main__":
    test_slicer_emits_type_comments()
    test_backtransform_preserves_type_comments()
    test_types_split_and_backcompat()
    print("PASS: ;TYPE 태깅 / 역변환 보존 / 종류 분리 / 기존 반환 호환")
