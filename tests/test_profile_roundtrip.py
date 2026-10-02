"""T1: 가변각 프로필 정변환↔역변환 왕복 정확성.

무작위 점 1,000개 × {상수, 2밴드, 3밴드} 프로필에 대해
solve_forward(정확 해) → 역변환 공식 → 원좌표 복원 오차 < 1e-9.

    python3 tests/test_profile_roundtrip.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conical.profile import AngleProfile


def roundtrip(profile, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-15, 15, n)
    y = rng.uniform(-15, 15, n)
    z = rng.uniform(0, 30, n)
    r = np.hypot(x, y)
    # 정변환: Z′ 정확 해 → 워프 좌표
    zw = profile.solve_forward(z, r, "outward")
    th = np.radians(profile.theta_at(zw))
    X = x / np.cos(th)
    Y = y / np.cos(th)
    # 역변환 공식: θ=θ(Zw), x=X·cosθ, z=Zw − √(X²+Y²)·sinθ  (c=+1, 부호 있는 θ)
    th2 = np.radians(profile.theta_at(zw))
    xb = X * np.cos(th2)
    yb = Y * np.cos(th2)
    zb = zw - np.hypot(X, Y) * np.sin(th2)
    err = max(np.abs(xb - x).max(), np.abs(yb - y).max(), np.abs(zb - z).max())
    return err


def test_roundtrip_profiles():
    r_max = np.hypot(15, 15) + 1
    profiles = {
        "상수 30°": AngleProfile.constant(30),
        "2밴드": AngleProfile.from_bands([(0, 12, 15), (12, 30, 35)], r_max),
        "3밴드(inward 포함)": AngleProfile.from_bands(
            [(0, 10, 30), (10, 20, 0), (20, 30, -20)], r_max),
    }
    for name, prof in profiles.items():
        err = roundtrip(prof)
        assert err < 1e-9, f"{name}: 왕복 오차 {err:.2e}"


if __name__ == "__main__":
    test_roundtrip_profiles()
    print("PASS: 상수/2밴드/3밴드 프로필 왕복 오차 < 1e-9 (1,000점)")
