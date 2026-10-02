"""T3: 블렌드 최소 폭(가역성) 검증.

w_min = c·r_max·(tanθ₂−tanθ₁) 보다 좁은 블렌드 → 명확한 에러(w_min 수치 포함).
경계값(w = w_min×1.01) → 통과.

    python3 tests/test_blend_min_width.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.profile import AngleProfile


def test_blend_min_width():
    r_max = 20.0
    th1, th2 = 10.0, 40.0
    w_min = r_max * (math.tan(math.radians(th2)) - math.tan(math.radians(th1)))

    # (a) 위반: w = w_min × 0.5 → 에러 + 메시지에 w_min 수치
    w_bad = w_min * 0.5
    try:
        AngleProfile([(0, th1), (10, th1), (10 + w_bad, th2), (40, th2)]) \
            .validate(r_max, "outward")
        raise AssertionError("가역성 위반인데 에러가 안 남")
    except ValueError as e:
        assert f"{w_min:.2f}" in str(e), f"에러 메시지에 w_min 수치 없음: {e}"

    # (b) 경계값: w = w_min × 1.01 → 통과
    w_ok = w_min * 1.01
    AngleProfile([(0, th1), (10, th1), (10 + w_ok, th2), (40, th2)]) \
        .validate(r_max, "outward")

    # (c) 안정 방향(각도 감소, c=+1)은 좁아도 제약 없음
    AngleProfile([(0, th2), (10, th2), (10.1, th1), (40, th1)]) \
        .validate(r_max, "outward")


if __name__ == "__main__":
    test_blend_min_width()
    print("PASS: w_min 위반 검출(수치 포함) / 경계값 통과 / 안정 방향 무제약")
