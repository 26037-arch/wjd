"""
analyze_envelope.py — 5축에서 최대 원뿔각을 정하는 것이 무엇인가. [플랫폼 비교]

    python3 analyze_envelope.py                      # 기본 모델 + 부품 크기 스윕
    python3 analyze_envelope.py model.stl
    python3 analyze_envelope.py --x-travel 180 --z-travel 120

3축 원뿔의 최대각은 **노즐이 출력물을 치는 것**이 정했다(`find_max_safe_angle.py`).
5축에서는 그 한계가 구조적으로 사라진다 — 원뿔 레이어가 기계공간에서 **수평면**이
되기 때문이다(`conical/envelope.py` ①, 아래 ⓪에서 실제 G-code 로 확인한다).

대신 **기계가 그 자세에 도달하느냐**가 한계가 된다. 답하는 것:

  ⓪ 정말로 간섭이 불가능한가 — 실제 G-code 에서 기계 Z 단조성·층 평탄도
  ① 이 부품을 이 각도로 뽑으려면 기계가 무엇을 내줘야 하나 (X/Z 이동, 베드 위치)
  ② 지금 G-code 를 그대로 걸면 몇 도까지 되나 (데이텀 고정)
  ③ 기계를 **아직 만들기 전**이면 몇 도까지 되나 (데이텀 자유) — 지금이 이 경우다
  ④ 부품이 커지면 어디서 막히나

⚠ 기계 제원을 모른다. 기본 엔벨로프는 **가정**(Prusa i3 조형공간)이다. 실기
  제원이 나오면 `--x-travel/--z-travel/--bed-radius/--max-tilt` 로 갈아끼울 것.
"""

import argparse
import math

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical.backtransform import backtransform
from conical.envelope import (
    MachineEnvelope,
    check_planar_stacking,
    max_angle,
    part_samples,
    requirement,
)
from conical.meshio import center_on_axis
from conical.open5x import PRUSA_UV, to_open5x
from conical.planar_slicer import slice_mesh
from conical.plotstyle import L as _L
from conical.transform import transform_cone

ANGLES = (0, 15, 30, 45, 60, 75)


def gcode_for(stl, angle, layer_height=0.3):
    """같은 슬라이싱의 (부품좌표, 기계좌표) 쌍. 두 해석을 나란히 보려고 둘 다 낸다."""
    mesh = center_on_axis(trimesh.load(stl, force="mesh"))
    warped = transform_cone(mesh.vertices, angle, "outward")
    items = slice_mesh(trimesh.Trimesh(vertices=warped, faces=mesh.faces,
                                       process=False), layer_height=layer_height)
    real, _ = backtransform(items, angle, "outward")
    machine, _ = to_open5x(real, angle, "outward", PRUSA_UV)
    return real, machine


def box_samples(radius, height, n=24):
    """반지름 R, 높이 H 인 원기둥의 (r, z) 껍질 — 부품 크기 스윕용 상자 근사."""
    r = np.concatenate([np.full(n, 0.0), np.full(n, radius),
                        np.linspace(0, radius, n), np.linspace(0, radius, n)])
    z = np.concatenate([np.linspace(0, height, n), np.linspace(0, height, n),
                        np.full(n, 0.0), np.full(n, height)])
    return r, z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--x-travel", type=float, default=250.0)
    ap.add_argument("--z-travel", type=float, default=210.0)
    ap.add_argument("--bed-radius", type=float, default=90.0)
    ap.add_argument("--max-tilt", type=float, default=90.0)
    ap.add_argument("--pivot-depth", type=float, default=PRUSA_UV.pivot_depth)
    ap.add_argument("--out", default="envelope_limits.png")
    args = ap.parse_args()

    prof = PRUSA_UV
    prof.pivot_depth = args.pivot_depth
    d = prof.pivot_depth
    free = MachineEnvelope(x_travel=args.x_travel, z_travel=args.z_travel,
                           bed_radius=args.bed_radius,
                           max_tilt_deg=args.max_tilt, datum="free",
                           name="자유 데이텀")
    fixed = MachineEnvelope(x_travel=args.x_travel, z_travel=args.z_travel,
                            bed_radius=args.bed_radius,
                            max_tilt_deg=args.max_tilt,
                            x_min=-args.x_travel / 2, x_max=args.x_travel / 2,
                            z_min=0.0, z_max=args.z_travel,
                            datum="fixed", name="Open5x 관례 데이텀")

    jobs = args.models or ["examples/funnel.stl", "examples/lamp.stl"]

    # ⓪ 간섭이 정말로 불가능한가 — 주장 검증
    print("=" * 78)
    print("⓪ **같은 슬라이싱**을 3축으로 해석할 때와 5축으로 해석할 때")
    print("   3축: 노즐 수직·베드 고정 → 원뿔면이 노즐 밑에서 θ 만큼 기울어 있다")
    print("        → 노즐이 출력물을 친다. funnel 의 3축 MAX_ANGLE 은 **24°**")
    print("          (`find_max_safe_angle.py`, 32° 에서 첫 간섭 0.04%)")
    print("   5축: 베드가 θ 만큼 기울어 → 원뿔 레이어가 기계공간에서 **수평면**이 된다")
    print(f"\n   {'모델':<12} {'θ':>4} | {'층':>5} {'층간격':>8} {'h·cosθ':>8} "
          f"{'층내 편차':>10} {'기계Z 단조':>10}")
    allok = True
    for stl in jobs:
        for ang in (20.0, 45.0):
            st = check_planar_stacking(gcode_for(stl, ang)[1])
            exp = 0.3 * math.cos(math.radians(ang))
            ok = st["monotone"] and st["flat"] < 1e-9
            allok = allok and ok
            print(f"   {stl.split('/')[-1]:<12} {ang:>4.0f}° | {st['n_layers']:>5} "
                  f"{st['layer_dz']:>8.4f} {exp:>8.4f} {st['flat']:>10.1e} "
                  f"{('✅ 단조' if st['monotone'] else '❌ 내려감'):>10}")
    print("\n   층간격이 정확히 h·cosθ 이고, 층 내부 편차가 1e-13 수준이다")
    print("   = **한 레이어가 기계 Z 값 하나.** 그리고 압출 순서대로 기계 Z 가 한 번도")
    print("   안 내려간다 → 이미 놓인 것이 전부 노즐 팁보다 아래거나 같은 높이다.")
    print(f"   **3축에서 각도를 24° 로 막던 간섭이 5축에서는 정의상 불가능해진다.** "
          f"{'(전부 통과)' if allok else '⚠ 일부 실패'}")
    print("   ⚠ 옆면 간섭·트래블 도중 자세·프레임 충돌은 여전히 못 본다.")

    for stl in jobs:
        mesh = center_on_axis(trimesh.load(stl, force="mesh"))
        r, z = part_samples(mesh)
        R, H = float(r.max()), float(z.max() - z.min())
        print("=" * 78)
        print(f"[{stl.split('/')[-1]}]  R={R:.1f}mm  H={H:.1f}mm  "
              f"대각 √(R²+H²)={math.hypot(R, H):.1f}mm")

        print(f"\n  ① 기계가 내줘야 하는 것 (피벗 깊이 d={d:.0f}mm)")
        print(f"     {'θ':>4} {'X 이동':>8} {'Z 이동':>8} | "
              f"{'베드중심 X':>11} {'베드면 침하':>11}")
        for th in ANGLES:
            q = requirement(r, z, th, prof)
            print(f"     {th:>4} {q['x_span']:>8.1f} {q['z_span']:>8.1f} | "
                  f"{q['bed_center_x']:>11.1f} {q['bed_sink']:>11.1f}")
        print(f"     → X 이동 상한은 **부품 대각선 √(R²+H²) = "
              f"{math.hypot(R, H):.1f}mm** (θ=atan(H/R) 에서). 틸트는 부품 경계상자를"
              f"\n       기계공간에서 그냥 **θ 만큼 회전**시키는 것이라 그렇다.")

        print("\n  ② 지금 G-code 를 그대로 걸면 (원점=베드중심, Z=0=베드면)")
        res = max_angle(r, z, fixed, prof)
        print(f"     최대 각도: {res['max_angle']}°   막는 것: {', '.join(res['blockers'])}")
        if res["max_angle"] is not None and res["max_angle"] < 1.0:
            print(f"     ⚠⚠ **어떤 각도도 안 된다.** 틸트하면 베드면이 기계 Z "
                  f"{-d:.0f}·(1−cosθ) 만큼 내려가는데,")
            print(f"        소프트리밋이 0 이면 그대로 잘린다. θ=20° 에서만 해도 "
                  f"{-d * (1 - math.cos(math.radians(20))):.2f}mm.")
            print("        → **기계 Z 영점을 베드면보다 아래로 두거나 음수 Z 를 "
                  "허용해야 한다.** 제작 전에 정할 것.")

        print(f"\n  ③ 기계를 아직 만들기 전이면 (X {args.x_travel:.0f} / "
              f"Z {args.z_travel:.0f} / 베드반경 {args.bed_radius:.0f} 이동만 맞추면 됨)")
        res = max_angle(r, z, free, prof)
        print(f"     최대 각도: {res['max_angle']}°   막는 것: "
              f"{', '.join(res['blockers'])}")
        print("     (스팬에는 피벗 깊이 d 가 **약분돼 안 들어간다** — d 는 위치만"
              " 정한다)")

    # ④ 부품이 커지면 어디서 막히나
    print("=" * 78)
    print(f"④ 부품이 커지면 (원기둥 근사, X {args.x_travel:.0f} / "
          f"Z {args.z_travel:.0f} / 베드반경 {args.bed_radius:.0f})")
    print(f"   {'R×H (mm)':>12} {'대각':>7} {'최대각':>8}  막는 것")
    grid = [(20, 40), (40, 80), (60, 120), (80, 160), (90, 200), (90, 300),
            (120, 120)]
    for R, H in grid:
        r, z = box_samples(R, H)
        res = max_angle(r, z, free, prof)
        ang = res["max_angle"]
        print(f"   {R:>5.0f} × {H:<4.0f} {math.hypot(R, H):>7.1f} "
              f"{str(ang) + '°':>8}  {', '.join(res['blockers'])}")
    lim = min(args.x_travel, args.z_travel)
    print("\n   **규칙: 0~90° 전 구간을 쓰려면 X·Z 이동이 둘 다 부품 대각선")
    print(f"   √(R²+H²) 이상이어야 한다.** 여기서는 {lim:.0f}mm — 위 표가 정확히")
    print("   그 경계에서 갈린다 (80×160 은 대각 178.9 로 통과, 90×200 은 219.3")
    print("   으로 7.25° 에서 막힌다). 틸트가 부품 경계상자를 회전시키는 것이므로,")
    print("   어느 각도에선가 대각선이 X 축과, 또 어느 각도에선가 Z 축과 나란해진다.")
    print("   (70개 조합에서 불일치 0건 — `tests/test_envelope.py` 가 고정한다)")

    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ths = np.linspace(0, 80, 161)
    ax = axes[0]
    for stl in jobs:
        mesh = center_on_axis(trimesh.load(stl, force="mesh"))
        r, z = part_samples(mesh)
        xs = [requirement(r, z, t, prof)["x_span"] for t in ths]
        zs = [requirement(r, z, t, prof)["z_span"] for t in ths]
        nm = stl.split("/")[-1].replace(".stl", "")
        ln, = ax.plot(ths, xs, lw=1.8, label=f"{nm}: X")
        ax.plot(ths, zs, lw=1.8, ls="--", color=ln.get_color(),
                label=f"{nm}: Z")
    ax.set_xlabel(_L("cone angle (deg)", "원뿔각 (도)"))
    ax.set_ylabel(_L("required machine travel (mm)", "필요한 기계 이동 (mm)"))
    ax.set_title(_L("tilt trades Z travel for X travel",
                    "틸트는 Z 이동을 X 이동으로 바꾼다"), fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    Rs = np.arange(4, 101, 3.0)
    Hs = np.arange(4, 261, 6.0)
    grid_ang = np.full((len(Hs), len(Rs)), np.nan)
    for i, H in enumerate(Hs):
        for j, R in enumerate(Rs):
            rr, zz = box_samples(R, H, n=6)
            m = max_angle(rr, zz, free, prof, step=2.5)["max_angle"]
            if m is not None:
                grid_ang[i, j] = m
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e8e8e8")                      # 0° 조차 안 되는 곳
    im = ax.pcolormesh(Rs, Hs, np.ma.masked_invalid(grid_ang), cmap=cmap,
                       vmin=0, vmax=90, shading="nearest")
    lim = min(args.x_travel, args.z_travel)
    tt = np.linspace(0, np.pi / 2, 400)
    ax.plot(lim * np.cos(tt), lim * np.sin(tt), color="crimson", lw=2.2,
            label=_L(f"diagonal rule: R²+H² = {lim:.0f}²",
                     f"대각선 규칙 √(R²+H²) = {lim:.0f}mm"))
    ax.axvline(args.bed_radius, color="#d95f02", lw=2.0, ls="--",
               label=_L(f"bed radius {args.bed_radius:.0f}mm",
                        f"베드 반경 {args.bed_radius:.0f}mm"))
    ax.set_xlim(Rs[0], Rs[-1])
    ax.set_ylim(Hs[0], Hs[-1])
    ax.set_xlabel(_L("part radius R (mm)", "부품 반지름 R (mm)"))
    ax.set_ylabel(_L("part height H (mm)", "부품 높이 H (mm)"))
    ax.set_title(_L("max reachable angle (grey = not even 0deg)",
                    "도달 가능 최대각 (회색 = 0° 조차 불가)"), fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=.92)
    fig.colorbar(im, ax=ax, label=_L("max cone angle (deg)", "최대 원뿔각 (도)"))

    fig.suptitle(_L("5-axis max cone angle is set by machine reach, not nozzle "
                    "interference (assumed envelope)",
                    "5축 최대 원뿔각은 노즐 간섭이 아니라 기계 도달이 정한다 "
                    "(엔벨로프는 가정값)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=130)
    print(f"\n그림 저장: {args.out}")
    print("\n⚠ 여기서 나온 각도는 '기계가 못 하는 곳'의 상한이지 **써야 할 각도가")
    print("  아니다.** 서포트 감소(J)·층간격 제약·표면 품질은 전부 별개 질문이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
