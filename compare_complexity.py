"""
compare_complexity.py — '복잡도 vs 성능' 가성비 비교 (발표 핵심 데이터).

같은 모델에 대해 각도 결정의 '복잡도'를 올려가며 성능을 잰다:
    균일(전역 1개) → 구간별(2개) → 구간별(3개) → 세밀(면마다)

성능 지표 (conical/metrics.py):
    · 서포트(%)   낮을수록 좋음  (재료·프린트 시간 = 속도와 연결)
    · 강도 proxy  낮을수록 좋음  (레이어-표면 정렬; 계단 적을수록 강함, 경향)
    · 평균각(°)   낮을수록 좋음  (왜곡·하드웨어 부담 = 복잡도 비용 proxy)
    · 계산시간    이 결정을 내리는 데 걸린 시간

핵심 질문: 복잡도를 올릴 때 성능이 '얼마나' 좋아지나? 어디서 가성비가 꺾이나?

    python3 compare_complexity.py            # 데모 모델들
    python3 compare_complexity.py model.stl  # 내 STL
"""

import time

import trimesh

from conical.config import THRESHOLD_DEG
from conical.meshio import center_on_axis
from conical.varangle import select_banded, select_fine, select_uniform


def profile_str(res):
    """프로파일을 짧은 문자열로 (아래→위 각도)."""
    if res.get("profile") is None:
        return "(면마다 다름)"
    parts = []
    for p in res["profile"]:
        parts.append("-" if p is None else f"{p[0]}{p[1][:3]}")
    return "[" + " ".join(parts) + "]"


def run_model(name, mesh, k=0.2):
    mesh = mesh.copy()
    mesh.merge_vertices()
    center_on_axis(mesh)  # 회전축(Z)에 XY 센터링 + 바닥 z=0 (off-axis 왜곡 방지)

    print(f"\n{'='*74}\n[{name}]  면수 {len(mesh.faces):,},  k={k},  임계각 {THRESHOLD_DEG:.0f}°\n{'='*74}")
    print(f"{'전략':>10} | {'서포트':>7} | {'강도proxy':>9} | {'평균각':>6} | {'계산시간':>8} | 프로파일(아래→위)")
    print("-" * 74)

    strategies = [
        ("균일(1)", lambda: select_uniform(mesh, k)),
        ("구간별(2)", lambda: select_banded(mesh, k, 2)),
        ("구간별(3)", lambda: select_banded(mesh, k, 3)),
        ("세밀(면)", lambda: select_fine(mesh, k)),
    ]
    rows = []
    for label, fn in strategies:
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        rows.append((label, res, dt))
        print(f"{label:>10} | {res['support_pct']:6.1f}% | {res['staircase']:9.3f} | "
              f"{res['avg_angle']:5.1f}° | {dt*1000:6.0f}ms | {profile_str(res)}")

    # 균일 대비 개선폭 요약
    base = rows[0][1]
    print("-" * 74)
    for label, res, _ in rows[1:]:
        print(f"  {label} vs 균일: 서포트 {base['support_pct']-res['support_pct']:+.1f}%p, "
              f"강도 {base['staircase']-res['staircase']:+.3f}, "
              f"평균각 {base['avg_angle']-res['avg_angle']:+.1f}°")
    return rows


def demo_models():
    models = {}
    models["sphere"] = trimesh.creation.icosphere(subdivisions=4, radius=10)
    # 아래=오버행 구 + 위=긴 기둥(오버행 없음): 균일각이 타협할 수밖에 없는 대표 사례
    s = trimesh.creation.icosphere(subdivisions=3, radius=8)
    cyl = trimesh.creation.cylinder(radius=3, height=20)
    cyl.apply_translation([0, 0, 18])
    models["sphere+stalk"] = trimesh.util.concatenate([s, cyl])
    return models


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mesh = trimesh.load(sys.argv[1], force="mesh")
        run_model(sys.argv[1], mesh)
    else:
        print("[STL 없음 → 데모 모델들로 실행]")
        for name, mesh in demo_models().items():
            run_model(name, mesh)
    print("\n해석 팁: '균일→구간별'에서 서포트가 크게 줄면 부위별 각도의 이득이 크다는 뜻.")
    print("         '구간별→세밀'에서 서포트가 더 안 줄면 거기서 가성비가 꺾이는 것.")
    print("  ※ 세밀(면)은 '이론적 바닥'일 뿐 물리적으로 못 찍는다(면마다 다른 각도 =")
    print("     유효한 θ(z) 아님). 실제로 찍을 수 있는 최적은 '구간별'이다.")
