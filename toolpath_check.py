"""
toolpath_check.py — 출력 G-code 의 툴패스 가상 검증 (하드웨어 없이).

검사기 A(기본): 압출 세그먼트 지지 검사 — 미지지 압출 길이 비율(%), 층별 통계,
미지지 지점 산점 PNG(위+옆 2뷰).
검사기 B(--nozzle): 3축 노즐 간섭 검사 (HotendProfile 추정 기하).

    python3 toolpath_check.py out.gcode
    python3 toolpath_check.py out.gcode --nozzle
    python3 toolpath_check.py out.gcode --layer-height 0.4 --width 0.45

⚠ 이 검사는 브리징·수축·유변학을 무시한 기하 판정이며, 메시 기반 예측과 측정
  대상이 다르다(툴패스에는 인필·트래블·시임이 있음) — 절대값이 아니라 각도 간
  순위·경향 비교가 목적이다.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conical import gcode as gc
from conical.toolpath import (
    HotendProfile,
    check_nozzle,
    check_support,
    sample_extrusions,
    support_breakdown,
)


def detect_layer_height(lines, fallback=0.3):
    """G-code 에서 층고를 읽는다. 못 읽으면 `(fallback, False)`.

    ⚠⚠ 2026-09-26 정정: 예전에는 `layer_h=` 패턴만 찾았는데 **우리 G-code 에 그런
      문자열이 없다.** 자기기술 헤더는 `;CONICAL_META {... "layer_height":0.4 ...}`
      이고 레거시 줄은 `; conical: angle=… chord_tol=…` 이다. 그래서 **항상 폴백
      0.3 을 쓰고 있었고**, 0.3mm 로 뽑은 파일에서만 우연히 맞았다.

      층고는 지지 창(`층고 × vwin_factor`)과 베드 판정에 직접 들어가므로, 0.4mm
      파일을 0.3 으로 재면 **전부 다른 것을 재는 것이다.** 실제로 층고를 0.2→0.5 로
      훑으면 진짜 오버행이 0.02 → 1.63 %p 로 움직인다(analyze_layer_height.py).

      `;CONICAL_META` 는 바로 이런 용도로 넣은 것인데(P0, 자기기술 G-code) 정작
      검사기가 안 읽고 있었다. 이제 그것을 **먼저** 본다.
    """
    head = lines[:40]
    for ln in head:                                    # ① 자기기술 메타 (우선)
        if ";CONICAL_META" in ln:
            try:
                meta = json.loads(ln.split(";CONICAL_META", 1)[1].strip())
                v = float(meta["layer_height"])
                if v > 0:
                    return v, True
            except (ValueError, KeyError, TypeError):
                pass                                   # 깨진 메타는 조용히 넘기고 아래로
    for ln in head:                                    # ② 예전 패턴 (외부 파일 호환)
        m = re.search(r"layer_h(?:eight)?\s*[=:]\s*([\d.]+)", ln)
        if m:
            return float(m.group(1)), True
    return fallback, False


def main():
    ap = argparse.ArgumentParser(description="toolpath virtual checker")
    ap.add_argument("gcode")
    ap.add_argument("--width", type=float, default=0.45)
    ap.add_argument("--layer-height", type=float, default=None)
    ap.add_argument("--nozzle", action="store_true", help="노즐 간섭 검사도 수행")
    ap.add_argument("--png", default=None)
    ap.add_argument("--export-json", default=None,
                    help="샘플점·판정·파라미터·집계를 JSON으로 저장 (검증 탭 대조용 정답지)")
    args = ap.parse_args()

    lines = open(args.gcode).readlines()
    if args.layer_height:
        lh, known = args.layer_height, True
    else:
        lh, known = detect_layer_height(lines)
    if not known:
        # 조용히 틀린 층고로 재는 것을 막는다 — 지지 창과 베드 판정이 층고에 비례한다.
        print(f"⚠⚠ 층고를 G-code 에서 못 읽었다 — 폴백 {lh} 을 쓴다. 이 값이 실제와")
        print("   다르면 지지 창·베드 판정이 전부 어긋난다. --layer-height 로 줄 것.")
    items = gc.parse(lines)
    pts, mid, w, kinds, lay = sample_extrusions(items, width=args.width,
                                                return_types=True,
                                                return_layers=True)
    if len(pts) == 0:
        raise SystemExit("압출 세그먼트 없음")

    # 지지 판정은 '퇴적된 모든 재료'로 한다(인필도 실제로 받쳐준다).
    # 종류 구분은 '무엇을 보고할지'에만 쓴다.
    sup, st = check_support(pts, mid, w, layer_height=lh, width=args.width)
    print("=" * 62)
    print(f"[toolpath_check] {args.gcode}  (layer_h={lh}, width={args.width})")
    print(f"  샘플점        : {len(pts):,}  (압출 경로 {w.sum()/1000:.2f} m)")
    print(f"  미지지 압출   : {st['unsupported_pct']:.2f} %  (길이 가중, 전체)")

    def pct(mask):
        tot = w[mask].sum()
        return (w[mask & ~sup].sum() / tot * 100.0) if tot > 0 else float("nan")

    peri, fill = kinds == 0, kinds == 1
    if peri.any() or fill.any():
        print(f"    ├ 페리미터  : {pct(peri):.2f} %  "
              f"(전체 길이의 {w[peri].sum()/w.sum()*100:.0f}%)")
        # 페리미터 미지지에는 오버행이 아닌 몫이 섞인다 — 위로 좁아지는 형상은
        # 페리미터가 아랫층 희소 인필 위에 놓인다(평범한 브리징). 갈라서 보고한다.
        if (lay >= 0).any():
            b = support_breakdown(pts, mid, kinds, lay, sup, w, width=args.width)
            print(f"    │  ├ 진짜 오버행   : {b['overhang_pct']:.2f} %p"
                  f"   ← **서포트 판단은 이 값으로**")
            print(f"    │  └ 아랫층 단면 안: {b['inside_pct']:.2f} %p"
                  f"   (오버행 아님 — 브리징 또는 층간격 팽창)")
        else:
            print("    │  (층 주석이 없어 오버행/브리징을 가르지 못했다 — "
                  "`; layer N` 또는 `;LAYER:N` 필요)")
        print(f"    └ 인필      : {pct(fill):.2f} %  "
              f"(희소 인필은 층마다 방향이 바뀌어 원래 브리징 — 참고용)")
    worst = sorted(st["layers"].items(), key=lambda kv: -kv[1])[:5]
    if worst and worst[0][1] > 0:
        print("  최악 층(z bin → 미지지%): " +
              ", ".join(f"z≈{k * lh:.1f}:{v:.0f}%" for k, v in worst if v > 0))

    if args.nozzle:
        col, ns = check_nozzle(pts, mid, HotendProfile(), clearance=lh)
        print(f"  노즐 간섭     : {ns['collision_pct']:.2f} % "
              f"(첫 간섭 z={ns['first_collision_z']})"
              "  [HotendProfile은 실측 전 추정값]")

    if args.export_json:
        import json
        payload = {
            "params": {"layer_height": lh, "width": args.width},
            "unsupported_pct": st["unsupported_pct"],
            "layers": st["layers"],
            "points": [[round(float(x), 4), round(float(y), 4),
                        round(float(z), 4), int(s), round(float(ww), 5)]
                       for (x, y, z), s, ww in zip(pts, sup, w, strict=True)],
        }
        with open(args.export_json, "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"  JSON 저장     : {args.export_json} (점 {len(pts):,}개)")

    png = args.png or (Path(args.gcode).stem + "_check.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ok = sup
    for ax, (a, b), name in zip(axes, [(0, 1), (0, 2)], ["top (XY)", "side (XZ)"], strict=True):
        ax.scatter(pts[ok][::5, a], pts[ok][::5, b], s=1, c="#9aa7c4", label="supported")
        if (~ok).any():
            ax.scatter(pts[~ok][:, a], pts[~ok][:, b], s=2, c="#d9534f", label="unsupported")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.suptitle(f"unsupported {st['unsupported_pct']:.2f}%")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"  산점 저장     : {png}")
    print("  ⚠ 기하 판정(브리징·수축 무시) — 절대값 아닌 각도 간 경향 비교용")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
