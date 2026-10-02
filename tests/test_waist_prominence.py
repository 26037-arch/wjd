"""T11: 허리 두드러짐 지표가 **형상**을 제대로 재는가.

⚠ 범위 정정 (2026-09-21): 이 테스트는 원래 "지표가 '밴드가 이기는 조건'을
  가려내는가"를 지킨다고 썼다. 그 전제는 **반증됐다** — 허리 깊이를 그대로 둔 채
  XY 로만 λ 배 하면 두드러짐은 0.571 로 고정인데 승패가 갈린다(승 3·패 3,
  docs/verification.md 2026-09-21). 판정 기준은 `analyze_blend_ratio.py` 의
  ρ = 블렌드비용/잠재이득 이다.

  아래 성질 ①②③ 은 **지표 자신의 성질**이라 그대로 유효하고, 그래서 테스트도
  그대로 둔다. 바뀐 것은 '이 지표를 무엇으로 쓰느냐'다 — 형상 기술자이지 판정
  기준이 아니다.

지키는 성질:
  ① 단조 형상(구·원뿔)은 0 — 꼭대기 테이퍼를 허리로 세지 않는다
  ② 목이 가늘수록 값이 커진다 (waisted_model 의 r_neck 축에서 단조)
  ③ 값이 r_neck/7 에서 나오는 기하적 예측과 맞는다

    python3 tests/test_waist_prominence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from compare_waist import waisted_model
from conical.meshio import center_on_axis, waist_prominence


def test_monotone_shapes_have_no_waist():
    """구와 원뿔은 위로 갈수록 가늘어지기만 한다 — 허리가 아니다.

    min(r)/max(r) 로 재면 둘 다 1 에 가까운 값이 나와 '아주 깊은 허리'로 오독된다.
    위쪽 최대 반경을 기준에 넣는 정의라야 0 이 된다.
    """
    sphere = center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))
    cone = center_on_axis(trimesh.creation.cone(radius=10.0, height=20.0,
                                                sections=64))
    for name, mesh in (("구", sphere), ("원뿔", cone)):
        p, _, _ = waist_prominence(mesh)
        assert p < 0.02, f"{name}: 허리가 없어야 하는데 {p:.3f}"


def test_deeper_neck_gives_larger_prominence():
    """r_neck 을 줄이면 값이 단조증가해야 한다 (허리 깊이 축이 성립)."""
    necks = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    vals = [waist_prominence(waisted_model(r))[0] for r in necks]
    assert vals[0] < 0.02, f"r_neck=7 은 목이 사라진 형상인데 {vals[0]:.3f}"
    for a, b, ra, rb in zip(vals, vals[1:], necks, necks[1:], strict=False):
        assert b > a + 1e-3, f"r_neck {ra}→{rb}: {a:.3f} → {b:.3f} (증가해야 함)"


def test_prominence_matches_geometry():
    """waisted_model 은 목 위 상단이 r=7 로 고정이라 예측값이 1 − r_neck/7 이다.

    이 예측이 맞는다는 것은 지표가 '아래 최대'가 아니라 **작은 쪽(위 최대 7)**을
    기준으로 쓰고 있다는 뜻이다 — 몸통 최대 반경 8 을 기준으로 삼았다면
    1 − r_neck/8 이 나온다.
    """
    for r_neck in (5.0, 3.0, 2.0, 1.0):
        p, _, r_w = waist_prominence(waisted_model(r_neck))
        expected = 1.0 - r_neck / 7.0
        assert abs(p - expected) < 0.02, \
            f"r_neck={r_neck}: {p:.3f}, 예측 {expected:.3f}"
        assert abs(r_w - r_neck) < 0.1, \
            f"r_neck={r_neck}: 찾은 목 반경 {r_w:.2f}"


def test_degenerate_inputs():
    """납작한 메시·표본 부족에서 죽지 않고 '허리 없음'을 돌려준다."""
    flat = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                           faces=np.array([[0, 1, 2]]), process=False)
    assert waist_prominence(flat)[0] == 0.0
    assert waist_prominence(waisted_model(2.0), n_samples=2)[0] == 0.0


if __name__ == "__main__":
    test_monotone_shapes_have_no_waist()
    test_deeper_neck_gives_larger_prominence()
    test_prominence_matches_geometry()
    test_degenerate_inputs()
    print("PASS: 단조 형상 0, 목 깊이에 단조증가, 예측 1−r_neck/7 과 일치, 퇴화 입력 안전")
