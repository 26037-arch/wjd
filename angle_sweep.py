"""
angle_sweep.py — Stage 2 실행 스크립트 (CLI).

각도별 남은 서포트 곡선을 그리고 표로 출력한다. 실제 계산 로직은
conical/sweep.py 에 있고, 이 파일은 데모 실행 + 그래프 그리기만 담당한다.
    python3 angle_sweep.py    # 데모용 구로 실행, angle_sweep.png 저장

(기존처럼 `from angle_sweep import support_fraction` 도 계속 동작한다.)
"""

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 판정 기준 통일(2026-07 리뷰): sweep(변환공간 근사) → analytic(해석식).
from conical.analytic import sweep_table
from conical.plotstyle import L

if __name__ == "__main__":
    # 테스트 모델: 구 (아랫면이 곡면 오버행이라 원뿔 슬라이싱 효과를 보기 좋음)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=10.0)

    angles = np.arange(0, 46, 2)   # 0° ~ 44°
    for cone_type in ["inward", "outward"]:
        frac = sweep_table(mesh, angles, cone_type)
        plt.plot(angles, frac, marker="o", label=L(f"{cone_type} cone", f"{cone_type} 원뿔"))
        # 표로도 출력
        print(f"\n[{cone_type}]  angle -> remaining support (%)")
        for a, f in zip(angles, frac, strict=True):
            print(f"  {a:2d}°  {f:5.1f}%")

    plt.axvspan(0, 25, alpha=0.12, color="green")   # 3축으로 가능한 영역(작은 각도)
    plt.text(1, plt.ylim()[1]*0.05, "3-axis range (small angle)", fontsize=8)
    plt.xlabel("cone angle (deg)")
    plt.ylabel("remaining support area (%)")
    plt.title("Angle vs remaining support  (sphere)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("angle_sweep.png", dpi=130)
    print("\nsaved: angle_sweep.png")
