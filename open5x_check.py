"""
open5x_check.py — Open5x 기계좌표 G-code 사전 점검 (검사기 C).

    python3 open5x_check.py out_open5x.gcode
    python3 open5x_check.py out.gcode --bed-radius 90 --max-turns 2

왜 따로 있나: `toolpath_check.py` 의 검사기 A/B 는 **3축 가정**(노즐 수직, 베드
고정)이다. 5축은 베드가 돌고 기울어서 그 가정이 깨진다. 여기서는 기계좌표 자체를
본다 — 틸트 한계, 회전 누적(배선 감김), 급회전, 베드 이탈, 노즐이 베드 아래로.

⚠ 이 검사를 통과해도 **첫 출력은 사람이 지켜봐야 한다.** 축 가속도 한계와 실제
  기구 충돌은 보지 못한다. 5축은 3축보다 사고가 쉽고, 사고가 나면 기계가 상한다.
"""

import argparse

from conical import gcode as gc
from conical.open5x import PRUSA_UV, VORON_BC, check_open5x

SEVERITY_ORDER = {"치명": 0, "경고": 1, "정보": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("gcode")
    ap.add_argument("--machine", choices=["prusa-uv", "voron-bc"],
                    default="prusa-uv")
    ap.add_argument("--bed-radius", type=float, default=None,
                    help="베드 반경 mm (주면 기계좌표가 베드를 벗어나는지 본다)")
    ap.add_argument("--max-turns", type=float, default=None,
                    help="허용 배선 감김 (기본 1.5회전). 슬립링이 있으면 크게 준다")
    ap.add_argument("--max-step", type=float, default=30.0,
                    help="한 이동에 허용할 V 변화(도). 기본 30")
    ap.add_argument("--min-z", type=float, default=0.0)
    ap.add_argument("--max-feed", type=float, default=None,
                    help="기계 최대 피드 (mm/min)")
    args = ap.parse_args()

    prof = PRUSA_UV if args.machine == "prusa-uv" else VORON_BC
    items = gc.parse(open(args.gcode).readlines())
    findings, st = check_open5x(items, prof, bed_radius=args.bed_radius,
                                max_dv_deg=args.max_step, min_z=args.min_z,
                                max_feed=args.max_feed,
                                max_turns=args.max_turns)

    print("=" * 66)
    print(f"[open5x_check] {args.gcode}  ({args.machine})")
    tilt = "없음" if st["tilt"] is None else f"{st['tilt']:.1f}°"
    print(f"  이동          : {st['moves']:,}   틸트 {prof.tilt_axis}={tilt}")
    print(f"  배선 감김     : {st['v_wind_turns']:.1f}회전  "
          f"(시작 기준 최대 이탈 — 이게 배선이 감기는 양)")
    print(f"  총 회전량     : {st['v_total_turns']:.1f}회전  "
          f"(Σ|Δ{prof.rot_axis}| — 마모·시간, 감김 아님)")
    print(f"  V 최대 한 걸음: {st['v_max_step_deg']:.1f}°")
    if st["z_min"] is not None:
        print(f"  Z 최소        : {st['z_min']:.2f}mm")
    if st["xy_radius_max"] is not None:
        print(f"  XY 최대 반경  : {st['xy_radius_max']:.1f}mm")

    if not findings:
        print("  ✓ 걸린 것 없음 (그래도 첫 출력은 지켜볼 것)")
    else:
        print(f"  걸린 것 {len(findings)}건:")
        for sev, msg in sorted(findings, key=lambda f: SEVERITY_ORDER[f[0]]):
            mark = {"치명": "✖", "경고": "⚠", "정보": "·"}[sev]
            print(f"    {mark} [{sev}] {msg}")

    print("  ⚠ 축 가속도·기구 충돌은 보지 못한다. 첫 출력은 사람이 지켜볼 것.")
    return 1 if any(s == "치명" for s, _ in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
