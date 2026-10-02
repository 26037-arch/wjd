"""
gcode_stats.py — G-code 진단 도구 (의존성 없음, 표준 라이브러리만).

왜 만들었나:
    원뿔 변환 파이프라인(변환→Slic3r→역변환)은 G-code가 수십~수백만 줄로
    폭발하기 쉽다. 역변환이 '직선을 잘게 쪼갠 곡선'으로 바꾸기 때문이다.
    근데 줄 수 자체가 문제가 아니라, **너무 짧은 세그먼트**가 문제다:
    펌웨어가 초당 처리할 수 있는 세그먼트 수(예: 8bit 보드 ~200/s,
    32bit ~1000-2000/s)보다 요구율이 높으면 프린터가 버벅이며(stutter)
    실제 출력이 느려진다. 이 도구는 그걸 수치로 진단한다.

사용:
    python tools/gcode_stats.py print.gcode
    python tools/gcode_stats.py print.gcode --chord-tol 0.05 --axis-x 0 --axis-y 0

출력:
    · 줄 수 분해(이동/압출/주석), 세그먼트 길이 분포
    · 명령 피드레이트 기준 '요구 세그먼트 처리율'(초당) 분포
    · 현(chord) 허용오차 ε 기준, 축 반경별 적응 세그먼트로 다시 나누면
      몇 줄로 줄일 수 있는지 추정  (L_max ≈ 2·√(2·r·ε): 반경 클수록 길게 가능)
"""

import argparse
import math
import re
import sys

WORD = re.compile(r"([A-Za-z])([-+]?[0-9]*\.?[0-9]+)")


def parse_args():
    p = argparse.ArgumentParser(description="conical G-code size/stutter diagnosis")
    p.add_argument("gcode", help="G-code file path")
    p.add_argument("--chord-tol", type=float, default=0.05,
                   help="허용 현 오차 ε (mm). 역변환 재분할 추정에 사용. 기본 0.05")
    p.add_argument("--axis-x", type=float, default=None,
                   help="원뿔 축 X (기본: 모든 XY 점의 평균)")
    p.add_argument("--axis-y", type=float, default=None,
                   help="원뿔 축 Y (기본: 모든 XY 점의 평균)")
    p.add_argument("--seg-rate", type=float, default=500.0,
                   help="펌웨어가 처리 가능한 초당 세그먼트 수 가정. 기본 500 (32bit 보수적)")
    return p.parse_args()


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main():
    args = parse_args()
    total = comments = moves = extrude_moves = arcs = other = 0
    x = y = z = None
    feed = None                      # mm/min
    seg_len = []                     # 압출 이동의 XY+Z 길이
    seg_rate_req = []                # 그 이동을 명령 속도로 찍을 때 필요한 세그/초
    pts = []                         # 반경 추정용 (x,y,len)

    with open(args.gcode, "r", errors="replace") as fh:
        for line in fh:
            total += 1
            s = line.split(";", 1)[0].strip()
            if not s:
                comments += 1
                continue
            words = dict((m[0].upper(), float(m[1])) for m in WORD.findall(s))
            g = words.get("G")
            if g in (2.0, 3.0):
                arcs += 1
                continue
            if g not in (0.0, 1.0):
                other += 1
                continue
            moves += 1
            nx, ny = words.get("X", x), words.get("Y", y)
            nz = words.get("Z", z)
            if "F" in words:
                feed = words["F"]
            has_e = "E" in words and words["E"] > 0
            if x is not None and nx is not None and ny is not None:
                dx = nx - x
                dy = ny - y
                dz = (nz - z) if (z is not None and nz is not None) else 0.0
                L = math.sqrt(dx * dx + dy * dy + dz * dz)
                if has_e and L > 1e-9:
                    extrude_moves += 1
                    seg_len.append(L)
                    pts.append((0.5 * (x + nx), 0.5 * (y + ny), L))
                    if feed:
                        seg_rate_req.append((feed / 60.0) / L)
            x, y = (nx if nx is not None else x), (ny if ny is not None else y)
            z = nz if nz is not None else z

    if not seg_len:
        print("압출 이동을 찾지 못했습니다. 파일이 G-code가 맞는지 확인하세요.")
        return 1

    # 축 위치: 지정 없으면 압출 경로의 평균점 (모델이 축 근처에 있다는 가정)
    ax = args.axis_x if args.axis_x is not None else sum(p[0] * p[2] for p in pts) / sum(p[2] for p in pts)
    ay = args.axis_y if args.axis_y is not None else sum(p[1] * p[2] for p in pts) / sum(p[2] for p in pts)

    seg_sorted = sorted(seg_len)
    total_path = sum(seg_len)
    eps = args.chord_tol
    # 적응 재분할 추정: 반경 r에서 허용 현 길이 L=2√(2rε) → 필요한 세그 수 = len/L
    est = 0.0
    for px, py, L in pts:
        r = math.hypot(px - ax, py - ay)
        allowed = 2.0 * math.sqrt(max(2.0 * r * eps, 1e-12))
        est += L / max(allowed, 1e-9)
    est = max(int(est), 1)

    short01 = sum(1 for v in seg_len if v < 0.1)
    short02 = sum(1 for v in seg_len if v < 0.2)
    short05 = sum(1 for v in seg_len if v < 0.5)
    rate_sorted = sorted(seg_rate_req)

    print("=" * 64)
    print(f"[G-code 진단] {args.gcode}")
    print("=" * 64)
    print(f"총 줄수           : {total:,}  (이동 {moves:,} / 압출이동 {extrude_moves:,} / 호 G2·G3 {arcs:,})")
    print(f"압출 경로 총길이  : {total_path/1000:.1f} m")
    print(f"세그먼트 길이     : p50 {percentile(seg_sorted,50):.3f}  p10 {percentile(seg_sorted,10):.3f}  "
          f"p90 {percentile(seg_sorted,90):.3f} mm")
    print(f"짧은 세그먼트     : <0.1mm {short01/len(seg_len)*100:.0f}%  <0.2mm {short02/len(seg_len)*100:.0f}%  "
          f"<0.5mm {short05/len(seg_len)*100:.0f}%")
    if rate_sorted:
        p50r = percentile(rate_sorted, 50)
        p95r = percentile(rate_sorted, 95)
        print(f"요구 세그 처리율  : p50 {p50r:.0f}/s  p95 {p95r:.0f}/s   (펌웨어 가정 {args.seg_rate:.0f}/s)")
        over = sum(1 for v in rate_sorted if v > args.seg_rate)
        print(f"  → 버벅임 위험 이동 비율: {over/len(rate_sorted)*100:.0f}%  "
              f"(요구율이 펌웨어 처리율을 초과; 실제로는 느려짐)")
    print(f"축 가정           : ({ax:.1f}, {ay:.1f})  현 허용오차 ε={eps} mm")
    print(f"적응 재분할 추정  : 압출이동 {extrude_moves:,} → 약 {est:,} 세그먼트 "
          f"({extrude_moves/max(est,1):.1f}배 감소 가능)")
    print("  근거: 반경 r에서 최대 현 길이 L=2√(2rε). 축에서 멀수록 긴 직선 허용.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
