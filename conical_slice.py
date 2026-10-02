"""
conical_slice.py — 형상 적응형 원뿔 슬라이싱 파이프라인 (분석 → 실제 G-code).

RotBot 의 검증된 3단계 구조에 우리 '자동 각도 결정'을 앞단으로 붙인 것:

    [1] 분석/결정 : STL 오버행 분석 → 최적 (방향, 각도) 자동 선택  (우리 기여)
    [2] 정변환    : 메시를 원뿔 변환 (+ 사전 세분화)               (RotBot 차용)
    [3] 평면 슬라이싱 :
          · 기본: 내장 미니 슬라이서 (외부 의존 없음, 연구용)
          · 옵션: --slicer-cmd 로 외부 슬라이서 CLI 를 꽂음
                  (예: "prusa-slicer -g -o {gcode} {stl}")
    [4] 역변환    : 적응 현 분할 L=2√(2rε) 로 실공간 G-code 생성   (우리 개선)

사용:
    python3 conical_slice.py model.stl                      # 각도 자동
    python3 conical_slice.py model.stl --angle 30 --direction outward
    python3 conical_slice.py model.stl --layer-height 0.2 --chord-tol 0.05
    python3 conical_slice.py model.stl --slicer-cmd "prusa-slicer -g -o {gcode} {stl}"

출력: <입력이름>_conical.gcode  (+ 요약 리포트 stdout)
⚠ 슬라이서 자동 서포트는 끄고 쓸 것 — 변환공간의 45° 판정은 물리와 다르다
   (docs/warped_threshold_finding.md). 서포트 필요 여부는 [1]의 해석식이 판단.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from conical import analytic
from conical import gcode as gc
from conical.backtransform import backtransform
from conical.config import (
    BLEND_COST_K,
    BLEND_SHIFT_RATIO,
    DEFAULT_K,
    MAX_SPACING_FACTOR,
    THRESHOLD_DEG,
)
from conical.machine import MachineProfile
from conical.meshio import RadiusProfile, center_on_axis
from conical.planar_slicer import slice_mesh
from conical.profile import AngleProfile
from conical.selector import select_cone
from conical.transform import transform_cone, transform_cone_profile
from conical.varangle import select_banded, select_banded_j


def auto_select(mesh, k=DEFAULT_K):
    """평가함수 J = (서포트 감소) − k×각도 로 (방향, 각도) 자동 선택.

    이전에는 k 없이 순수 서포트 최소화였는데, 그건 프로젝트 핵심 통찰인
    '최소화의 함정'(항상 최대각 선택)과 모순이라 selector 의 J 로 교체.
    (selector 는 판정 통일로 해석식을 쓴다.)
    """
    best, _ = select_cone(mesh, k, verbose=False)
    return best["support"], best["angle"], best["direction"], best["J"]


def refine(mesh, max_edge):
    """정변환 전 세분화: 긴 변을 쪼개야 '변환 후 평면 슬라이스'가 원뿔면을 잘 근사."""
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces,
                                            max_edge=max_edge)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def run_external_slicer(cmd_template, warped_mesh, workdir):
    stl_path = Path(workdir) / "warped.stl"
    gcode_path = Path(workdir) / "warped.gcode"
    warped_mesh.export(stl_path)
    cmd = cmd_template.format(stl=stl_path, gcode=gcode_path)
    subprocess.run(cmd, shell=True, check=True)
    with open(gcode_path) as fh:
        return gc.parse(fh.readlines())


def main():
    ap = argparse.ArgumentParser(description="adaptive conical slicing pipeline")
    ap.add_argument("stl")
    ap.add_argument("--angle", type=float, default=None, help="원뿔 각도(도). 생략=자동")
    ap.add_argument("--direction", choices=["outward", "inward"], default=None)
    ap.add_argument("--profile", default=None,
                    help='가변각 θ(Z′) 수동 지정 "Z1:deg1,Z2:deg2,..." '
                         '(예 "0:15,10:15,14:35,30:35"; 음수=inward)')
    ap.add_argument("--auto-bands", type=int, default=None,
                    help="밴드 N개 자동 탐색(select_banded) → 가변각 프로필")
    ap.add_argument("--k-blend", type=float, default=BLEND_COST_K,
                    help=f"J의 블렌드 비용 가중치 (기본 {BLEND_COST_K}; "
                         f"analyze_blend_k.py 참조). 0=블렌드 비용 무시")
    ap.add_argument("--band-select", choices=["j", "independent"], default="j",
                    help="j=프로필 J로 공동 선택(블렌드 비용 포함) / "
                         "independent=밴드별 독립 선택(옛 방식, 비교용)")
    ap.add_argument("--spacing-limit", type=float, default=MAX_SPACING_FACTOR,
                    help=f"블렌드 층간격 배율 상한 (기본 {MAX_SPACING_FACTOR}; "
                         f"config.MAX_SPACING_FACTOR). 0=제약 끄기(옛 동작)")
    ap.add_argument("--layer-height", type=float, default=0.3)
    ap.add_argument("--perimeters", type=int, default=2)
    ap.add_argument("--infill-spacing", type=float, default=2.5,
                    help="인필 간격 mm (0=인필 없음)")
    ap.add_argument("--chord-tol", type=float, default=0.05,
                    help="역변환 허용 현 오차 ε (mm)")
    ap.add_argument("--max-edge", type=float, default=1.5,
                    help="정변환 전 최대 변 길이 (세분화)")
    ap.add_argument("--slicer-cmd", default=None,
                    help='외부 슬라이서 CLI 템플릿. 예 "prusa-slicer -g -o {gcode} {stl}"')
    ap.add_argument("--k", type=float, default=DEFAULT_K,
                    help=f"J의 각도 비용 가중치 (기본 {DEFAULT_K}; analyze_k 참조)")
    ap.add_argument("--mode", choices=["xyz", "open5x", "rep5x"], default="xyz",
                    help="xyz=3축(작은 각도) / open5x=베드 틸트+회전 기계좌표 / "
                         "rep5x=헤드 틸트+요, 부품좌표+B/C (펌웨어가 IK) [실험적]")
    ap.add_argument("--machine", choices=["prusa-uv", "voron-bc"], default="prusa-uv")
    ap.add_argument("--v-rewind", type=float, default=1.0,
                    help="배선 감김을 (이 값+1) 회전 이내로 유지 (트래블 중 되감기). "
                         "0 이면 끔. 진짜 한계는 기계의 배선 여유에서 온다")
    ap.add_argument("--v-rewind-feed", type=float, default=1200.0,
                    help="되감기 회전 속도 (deg/min). 올리면 추가 시간이 비례해 준다 "
                         "— 축 속도 한계 확인 후")
    ap.add_argument("--v-rewind-clearance", type=float, default=2.0,
                    help="되감기 전 들어올릴 여유 mm (퇴적물 최고점 위로)")
    ap.add_argument("--pivot-depth", type=float, default=50.0,
                    help="베드면→틸트축 거리 mm (Open5x 스탠드오프별, 실기 보정)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--machine-profile", default=None,
                    help="기계 프로파일 INI (시작/종료 G-code). 없으면 경로만 "
                         "나오고 실물로는 못 뽑는다 — profiles/machine.example.ini 참고")
    args = ap.parse_args()

    mesh = trimesh.load(args.stl, force="mesh")
    mesh = center_on_axis(mesh)
    base = analytic.support_fraction(mesh, 0.0, "outward", THRESHOLD_DEG)

    # [1] 각도 결정 (고정각 / 가변각 프로필)
    profile = None
    banded_info = None
    if args.profile is not None or args.auto_bands is not None:
        if args.angle is not None:
            raise SystemExit("--angle 과 --profile/--auto-bands 는 동시 사용 불가")
        if args.mode in ("open5x", "rep5x"):
            raise SystemExit("가변각 + 5축 모드는 향후 과제 (틸트가 상수라는 "
                             "가정이 깨짐) — xyz 모드만 지원")
        r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
        rprof = RadiusProfile(mesh)
        spacing_limit = args.spacing_limit if args.spacing_limit > 1.0 else None
        if args.profile is not None:
            profile = AngleProfile.parse(args.profile)
            if (args.direction or "outward") == "inward":
                profile = AngleProfile(list(zip(profile.zs, -profile.thetas_deg, strict=True)))
            why = "수동 프로필"
        else:
            # 이동 예산은 모델 높이 기준 — 밴드 수와 무관 (config 주석 참조)
            shift = BLEND_SHIFT_RATIO * (mesh.bounds[1][2] - mesh.bounds[0][2])
            if args.band_select == "j":
                banded_info = select_banded_j(
                    mesh, args.k, args.auto_bands, r_max, rprof,
                    spacing_limit, shift, args.k_blend)
                profile = banded_info["profile_obj"]
                why = (f"--auto-bands {args.auto_bands} "
                       f"(프로필 J, k={args.k}, k_blend={args.k_blend})")
            else:
                banded_info = select_banded(mesh, args.k, args.auto_bands)
                profile = AngleProfile.from_banded_result(
                    banded_info, r_max, radius_profile=rprof,
                    spacing_limit=spacing_limit, max_shift=shift)
                why = f"--auto-bands {args.auto_bands} (밴드 독립 J, k={args.k})"
        profile.validate(r_max, "outward")
        direction = "outward"          # 부호 있는 각도 규약 (음수=inward)
        angle = None
        after = analytic.support_fraction_profile(mesh, profile, THRESHOLD_DEG)
    elif args.angle is not None:
        direction = args.direction or "outward"
        angle = args.angle
        after = analytic.support_fraction(mesh, angle, direction, THRESHOLD_DEG)
        why = "사용자 지정"
    else:
        after, angle, direction, j_score = auto_select(mesh, args.k)
        why = f"J 기준, k={args.k} (J={j_score:.2f})"

    print("=" * 62)
    print(f"[conical_slice] {args.stl}")
    if profile is not None:
        print(f"  각도 결정   : 가변각 프로필 ({why}, 가역성 검증 통과)")
        print(profile.describe())
        for note in getattr(profile, "notes", []):
            print(f"    ↳ {note}")
        mf = profile.max_spacing_factor(r_max, "outward", rprof)
        print(f"  층간격 배율 : 최대 {mf:.2f}배 "
              f"(상한 {spacing_limit if spacing_limit else '없음'})")
        if spacing_limit is not None:
            for bad in profile.check_spacing(r_max, "outward", spacing_limit, rprof):
                print(f"  ⚠ 층간격 위반: [{bad['lo']:.2f},{bad['hi']:.2f}] "
                      f"θ {bad['theta1']:.0f}°→{bad['theta2']:.0f}° r={bad['r']:.1f} "
                      f"→ 배율 {bad['factor']:.1f}배 (레이어가 떠서 지지가 사라짐)")
        if banded_info is not None and "thetas" in banded_info:
            uni = banded_info["uniform_best"]
            print(f"  밴드 선택   : 독립선택 {banded_info['start_thetas']} → "
                  f"프로필 J {banded_info['thetas']}  "
                  f"(J={banded_info['J']:.2f}, 블렌드비용 {banded_info['blend_penalty']:.1f})")
            print(f"  균일 대조   : 최선 균일 {uni:.0f}° J={banded_info['uniform_J']:.2f}"
                  + ("  → 균일이 더 나아 균일로 수렴"
                     if banded_info["J"] <= banded_info["uniform_J"] + 1e-9
                     else f"  → 부위별이 {banded_info['J'] - banded_info['uniform_J']:.2f} 만큼 이득"))
        elif banded_info is not None:
            print(f"  밴드 서포트 추정: {banded_info['support_pct']:.1f}% "
                  f"(select_banded, 이상적 추정)")
    else:
        print(f"  각도 결정   : {direction} {angle:.0f}°  ({why})")
    print(f"  서포트 예측 : {base:.1f}% (평면) → {after:.1f}% "
          f"({'프로필' if profile is not None else '선택 각도'}"
          f"{', 축상 근사' if profile is not None else ''})")

    # [2] 세분화 + 정변환
    fine = refine(mesh, args.max_edge)
    if profile is not None:
        warped = trimesh.Trimesh(
            vertices=transform_cone_profile(fine.vertices, profile, "outward"),
            faces=fine.faces, process=False)
    elif angle > 0:
        warped = trimesh.Trimesh(
            vertices=transform_cone(fine.vertices, angle, direction),
            faces=fine.faces, process=False)
    else:
        warped = fine
    print(f"  메시        : {len(mesh.faces):,} → 세분화 {len(fine.faces):,} 면")

    # [3] 평면 슬라이싱 (변환공간)
    if args.slicer_cmd:
        with tempfile.TemporaryDirectory() as td:
            items = run_external_slicer(args.slicer_cmd, warped, td)
        print(f"  슬라이서    : 외부 CLI ({args.slicer_cmd.split()[0]})")
    else:
        items = slice_mesh(warped, layer_height=args.layer_height,
                           perimeters=args.perimeters,
                           infill_spacing=args.infill_spacing)
        print(f"  슬라이서    : 내장 (layer {args.layer_height}mm, "
              f"perim {args.perimeters}, infill {args.infill_spacing}mm)")

    # [4] 역변환 (적응 현 분할; 프로필이면 점별 θ(Zw) + 블렌드 분할 2배)
    real_items, stats = backtransform(items, profile if profile is not None else angle,
                                      direction, chord_tol=args.chord_tol)
    print(f"  역변환      : 이동 {stats['moves_in']:,} → {stats['moves_out']:,} "
          f"(확장 {stats['expansion']:.2f}배, ε={args.chord_tol}mm 적응 분할)")

    # [5] 출력 모드
    if args.mode == "open5x":
        from conical.open5x import PRUSA_UV, VORON_BC, add_v_rewinds, to_open5x
        prof = PRUSA_UV if args.machine == "prusa-uv" else VORON_BC
        prof.pivot_depth = args.pivot_depth
        real_items, o5 = to_open5x(real_items, angle, direction, prof)
        print(f"  Open5x      : 틸트 {prof.tilt_axis}={angle:.0f}° 고정 "
              f"(pivot {args.pivot_depth}mm) [실험적 — 부호·피벗 실기보정 필요]")
        if args.v_rewind > 0:
            real_items, rw = add_v_rewinds(real_items, prof,
                                           max_turns=args.v_rewind,
                                           clearance=args.v_rewind_clearance,
                                           rot_feed=args.v_rewind_feed)
            print(f"  되감기      : 트래블 중 {rw['rewinds']}회, "
                  f"감김 ≤{args.v_rewind + 1:.1f}회전 "
                  f"(들어올림 +{args.v_rewind_clearance}mm, "
                  f"추가 시간 약 {rw['rewind_minutes']:.0f}분)")
        else:
            print("  ⚠ 되감기 꺼짐 — 배선 감김을 직접 확인할 것")
    elif args.mode == "rep5x":
        from conical.rep5x import REP5X, add_c_rewinds, to_rep5x
        real_items, r5 = to_rep5x(real_items, angle, direction, REP5X)
        print(f"  REP5X       : 헤드 틸트 B={r5['b']:.0f}° 고정, C 가 방위각 추종 "
              f"(펌웨어가 IK — X/Y/Z 는 노즐 팁) [실험적 — 축 부호 실기보정 필요]")
        print(f"                C 누적 {r5['c_turns']:.1f}회전 "
              f"(펌웨어 소프트 엔드스톱 창 "
              f"{REP5X.c_min:.0f}~{REP5X.c_max:.0f}°)")
        if args.v_rewind > 0:
            real_items, rw = add_c_rewinds(real_items, REP5X,
                                           clearance=args.v_rewind_clearance,
                                           rot_feed=args.v_rewind_feed)
            note = "" if rw["fixable"] else \
                f"  ⚠⚠ 못 고친 구간 {rw['unfixable_segments']}개 (창보다 넓게 감긴다)"
            print(f"  C 되감기    : 트래블 중 {rw['rewinds']}회 "
                  f"(시작 오프셋 {rw['start_offset']:+.0f}°, "
                  f"들어올림 +{args.v_rewind_clearance}mm){note}")
        else:
            print("  ⚠⚠ 되감기 꺼짐 — C 가 소프트 엔드스톱을 넘어 기계가 멈춘다")
    out_path = args.output or (Path(args.stl).stem +
                               {"open5x": "_open5x.gcode",
                                "rep5x": "_rep5x.gcode"}.get(args.mode,
                                                             "_conical.gcode"))
    # 뷰어/후처리 도구가 읽는 메타데이터 (tools/slicing_simulator.html 등)
    # ;CONICAL_META — 검증 탭이 파일 하나로 자기 계산을 할 수 있게 하는
    # 한 줄 JSON (자기기술 G-code, 사이드카 파일 없음).
    import json as _json
    if profile is not None:
        bps = [[float(z), float(t)] for z, t in zip(profile.zs, profile.thetas_deg, strict=True)]
        meta_dir = "outward"          # 부호 각도 규약 (음수=inward)
        prof_txt = ",".join(f"{z:g}:{t:g}"
                            for z, t in zip(profile.zs, profile.thetas_deg, strict=True))
        legacy = f"; conical: profile={prof_txt} direction=outward " \
                 f"mode={args.mode} chord_tol={args.chord_tol}"
    else:
        bps = [[0.0, float(angle)]]
        meta_dir = direction
        legacy = f"; conical: angle={angle:.1f} direction={direction} " \
                 f"mode={args.mode} chord_tol={args.chord_tol}"
    meta_json = _json.dumps({
        "version": 1, "direction": meta_dir, "profile": bps,
        "layer_height": args.layer_height, "extrusion_width": 0.45,
        "chord_tol": args.chord_tol, "source_stl": str(args.stl),
        "mode": args.mode}, separators=(",", ":"))
    meta = [("raw", f";CONICAL_META {meta_json}"), ("raw", legacy)]

    # 기계 프로파일(예열·호밍·프라임·냉각)은 경로 생성과 분리해 파일로 받는다.
    # 없으면 경로만 나오므로 실물로는 못 뽑는다 — 그 사실을 조용히 넘기지 않는다.
    head, tail = [], []
    if args.machine_profile:
        mp = MachineProfile.from_file(args.machine_profile)
        # ⚠ 여기는 오래 **상수각만 가정**하고 있었다 — 가변각(`--profile`/`--auto-bands`)
        #   이면 `angle` 이 None 이라 f"{angle:.1f}" 에서 죽었다. 즉 **이 연구의 핵심
        #   전략(부위별 각도)을 실물용으로 뽑는 경로가 통째로 막혀 있었다.**
        #   (tests/test_machine_profile_varangle.py 가 고정)
        if profile is not None:
            th = [float(t) for t in profile.thetas_deg]
            a_max = max(th, key=abs)
            angle_ctx = {"angle": f"{a_max:.1f}",      # 대표값 = |θ| 최대 (안전 기준)
                         "angle_max": f"{a_max:.1f}",
                         "angle_min": f"{min(th, key=abs):.1f}",
                         "profile": prof_txt}
        else:
            angle_ctx = {"angle": f"{angle:.1f}", "angle_max": f"{angle:.1f}",
                         "angle_min": f"{angle:.1f}",
                         "profile": f"0:{angle:g}"}
        ctx = {"layer_height": args.layer_height,
               "direction": direction, "mode": args.mode,
               "source_stl": Path(args.stl).name, **angle_ctx}
        head, tail = mp.start_items(ctx), mp.end_items(ctx)
        print(f"  기계        : {mp.name}  ({args.machine_profile})")
    gc.write(meta + head + real_items + tail, out_path)
    print(f"  출력        : {out_path}")
    if not args.machine_profile:
        print("  ⚠ 기계 프로파일이 없다 — 예열·호밍·프라임·냉각이 빠져 있어 "
              "이 파일로는 실물을 못 뽑는다.")
        print("    --machine-profile profiles/machine.example.ini "
              "(템플릿: 값은 실기 확인 전)")
    if args.mode == "open5x":
        # 검사기 A/B 는 3축 가정이라 5축 출력에는 안 맞는다. 기계좌표 검사로 보낸다.
        print(f"  검증        : python3 open5x_check.py {out_path} --bed-radius <mm>")
        print("                (3축용 toolpath_check 는 5축 출력에 맞지 않는다)")
    elif args.mode == "rep5x":
        print(f"  검증        : python3 rep5x_check.py {out_path}")
        print("                (3축용 toolpath_check 는 5축 출력에 맞지 않는다)")
    else:
        print(f"  검증        : python3 toolpath_check.py {out_path}")
    if args.mode == "xyz":
        print("  ⚠ 3축 프린터는 작은 각도만 안전 (노즐-출력물 간섭). "
              "큰 각도는 틸트 하드웨어 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
