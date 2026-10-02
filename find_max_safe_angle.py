"""
find_max_safe_angle.py — 이 모델·이 기계의 3축 최대 안전 원뿔각 산출.

각도 0~44°를 스윕하며 파이프라인(내장 슬라이서) + 노즐 간섭 검사(검사기 B)를
돌려, 간섭이 처음 나타나는 각도 '직전'을 MAX_ANGLE 로 보고한다.
config.MAX_ANGLE_DEG(전역 상수)를 모델별 계산값으로 대체할 수 있게 하는 도구.

    python3 find_max_safe_angle.py [model.stl] [--step 4] [--layer-height 0.4]

⚠ HotendProfile 기본값은 Ender 3 V2 계열 '추정치' — 실측 전 추정값이며
  반드시 캘리퍼스로 잴 것. 3축(노즐 수직) 가정.
"""

import argparse

import trimesh

from conical.backtransform import backtransform
from conical.meshio import center_on_axis
from conical.planar_slicer import slice_mesh
from conical.toolpath import HotendProfile, check_nozzle, sample_extrusions
from conical.transform import transform_cone


def run_pipeline(mesh, angle, direction="outward", layer_height=0.4):
    if angle > 0:
        v = transform_cone(mesh.vertices, angle, direction)
        warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    else:
        warped = mesh
    items = slice_mesh(warped, layer_height=layer_height)
    real, _ = backtransform(items, float(angle), direction)
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", nargs="?", default=None)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--layer-height", type=float, default=0.4)
    ap.add_argument("--direction", choices=["outward", "inward"], default="outward")
    args = ap.parse_args()

    if args.stl:
        mesh = center_on_axis(trimesh.load(args.stl, force="mesh"))
        name = args.stl
    else:
        mesh = center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10))
        name = "demo sphere (subdiv3, r10)"
    # 세분화: 원뿔면 근사 (파이프라인과 동일 취지, 러닝타임 위해 완만하게)
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=2.0)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    hot = HotendProfile()
    print("=" * 62)
    print(f"[find_max_safe_angle] {name}  ({args.direction}, layer {args.layer_height})")
    print(f"  HotendProfile(추정): tip_r={hot.tip_radius} cone={hot.cone_half_deg}°"
          f"×{hot.cone_height}mm block_r={hot.block_radius}@z{hot.block_z0}")
    print("-" * 62)
    print(f"  {'각도':>5} | {'간섭 샘플%':>9} | 첫 간섭 z")
    max_safe = None
    first_hit = None
    for angle in range(0, 45, args.step):
        items = run_pipeline(mesh, angle, args.direction, args.layer_height)
        pts, mid, w = sample_extrusions(items)
        col, st = check_nozzle(pts, mid, hot, clearance=args.layer_height)
        z = st["first_collision_z"]
        print(f"  {angle:4d}° | {st['collision_pct']:8.2f}% | "
              f"{'-' if z is None else f'{z:.1f}'}")
        if st["collision_pct"] == 0.0 and first_hit is None:
            max_safe = angle
        elif first_hit is None:
            first_hit = angle
    print("-" * 62)
    if first_hit is None:
        print(f"  결과: {max_safe}°까지 간섭 없음 (스윕 상한) → MAX_ANGLE ≥ {max_safe}°")
    else:
        print(f"  결과: 이 모델·이 기계의 3축 MAX_ANGLE = {max_safe}° "
              f"(첫 간섭 {first_hit}°)")
    print("  → config.MAX_ANGLE_DEG(전역 상수)를 이 모델별 계산값으로 대체 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
