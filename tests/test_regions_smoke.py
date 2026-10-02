"""T28: 영역별 선택기(`conical.regions`)가 **돌기는 하는가**.

왜 (2026-09-28): 각도 후보를 `angle_candidates` 로 바꿀 때(실수 step 허용)
`regions.py` 에서 쓰기만 하고 import 를 빠뜨렸다. 이 경로를 부르는 테스트가 없어서
126 개가 전부 통과했고, main 에 ruff 가 들어온 뒤 F821(미정의 이름)로 처음 드러났다.

    python3 -m pytest tests/test_regions_smoke.py
"""

import sys
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conical.regions import select_regions  # noqa: E402


def test_select_regions_runs():
    m = trimesh.creation.icosphere(subdivisions=2, radius=10)
    m.apply_translation([0, 0, 10])
    out = select_regions(m, 0.2, 2, verbose=False)
    assert out is not None


if __name__ == "__main__":
    test_select_regions_runs()
    print("T28 OK")
