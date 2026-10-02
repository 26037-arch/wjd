"""strength.layer_normals 와 analytic.overhang_score 의 교차검증.

두 모듈은 같은 물리량을 독립적으로 계산한다:
    (면 법선)·(레이어 법선) = n_z·cosθ + d·n_r·sinθ = g(θ)
2026-07 리뷰에서 layer_normals 의 반경 성분 부호(−c)가 틀려 outward 가
inward 레이어를 계산하고 있었다. 수정(+c) 후 두 값이 정확히 일치해야 한다.

    python3 tests/test_strength_layer_normals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from conical import analytic
from conical.meshio import center_on_axis
from conical.strength import layer_normals


def test_layer_normal_matches_analytic():
    m = center_on_axis(trimesh.creation.icosphere(3, 10.0))
    for d in ("outward", "inward"):
        u = layer_normals(m, 30, d)
        g1 = (m.face_normals * u).sum(axis=1)
        g2 = analytic.overhang_score(m, 30, d)
        c = np.linalg.norm(m.vertices[m.faces].mean(axis=1)[:, :2], axis=1) > 1e-6
        assert np.allclose(g1[c], g2[c], atol=1e-9), \
            f"{d}: max diff {np.abs(g1[c]-g2[c]).max()}"


if __name__ == "__main__":
    test_layer_normal_matches_analytic()
    print("PASS: layer_normals · face_normals == analytic.overhang_score (양방향)")
