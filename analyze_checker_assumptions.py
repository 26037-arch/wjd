"""analyze_checker_assumptions.py — 검사기가 슬라이서와 **공유하는 가정**을 끊어 본다.

## 왜

외부 리뷰(2026-09-23)가 "검증기가 슬라이서와 일부 가정을 공유한다"고 지적했다.
저장소의 검증 계층 ③(브라우저 독립 재구현)·⑤(팀메 독립 구현)는 이 지적에 **답이
되지 않는다** — 둘 다 **같은 판정 기준**을 다시 구현한 것이기 때문이다.

    구현 독립 ≠ 가정 독립.   우리가 가진 것은 앞쪽뿐이었다.

공유 가정이 구체적으로 둘 있다.

**(a) 미지지 재료가 지지로 센다.** `check_support` 는 이전에 퇴적된 **모든** 점을
    지지 후보로 쓰고 `supported` 로 거르지 않는다. 처짐이 연쇄되는 자리를 못 본다.

**(b) `1.5` 가 양쪽에 같은 수로 들어간다.**
    · 계획기: `config.MAX_SPACING_FACTOR = 1.5` (블렌드 층간격 상한)
    · 검사기: `toolpath.check_support` 의 `vwin = 층고 × 1.5` (지지 창)
    계획기가 폭을 제약의 **최소값**으로 잡아 `m = 1.5000` 을 정확히 물리므로
    (16/16, analyze_blend_cost.py), **계획기가 검사기의 합격선에 붙여서 계획하고
    검사기가 그걸 통과시킨다.** 블렌드 구간 안 미지지 밀도가 밖의 0.48 배로
    나온 것이 그 증상일 수 있다.

## 무엇을 재나

같은 툴패스에 **판정만 바꿔서** 다시 잰다 (슬라이싱은 모델·밴드수마다 한 번뿐 —
비싼 쪽은 슬라이싱이고, 바꾸는 것은 `check_support` 뿐이다).

    ① 연쇄 끊기      require_supported_below=True
    ② 창 스윕        vwin_factor ∈ {1.2, 1.3, 1.5, 1.8}
    ③ 둘 다

판정 기준(결과 보기 전에 적는다):

    · ①에서 수치가 거의 안 움직이면 → 이 근사는 무해하다
    · 크게 움직이면 → **지금까지의 오버행 수치가 전부 낙관적이었다**
    · ②에서 승/패가 1.5 근처에서만 유지되면 → 결론이 **공유 상수의 산물**이다
    · 넓은 구간에서 유지되면 → 공유는 사실이지만 결과를 만들지는 않았다

⚠ ① 이 '옳은' 판정이라고 주장하지 않는다. 처진 비드도 부분적으로는 받친다.
  두 값의 **차이**가 이 근사의 크기이고, 그걸 모르고 있었다는 것이 문제였다.

실행:
    python3 analyze_checker_assumptions.py              # 전체 표본
    python3 analyze_checker_assumptions.py --quick      # 대표 5 모델
"""

import argparse
import json
import sys
import time

import numpy as np
import trimesh

from analyze_blend_ratio import build_specs
from compare_waist import LAYER_H
from conical.backtransform import backtransform
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.planar_slicer import slice_mesh
from conical.toolpath import check_support, sample_extrusions, support_breakdown
from conical.transform import transform_cone_profile
from conical.varangle import select_banded_j

QUICK = ("구", "허리 r=5", "허리 r=3", "허리3 ×1.2", "허리3 로브0.2")

# (이름, vwin_factor, 연쇄)
SETTINGS = [
    ("기준 1.5", 1.5, False),        # 지금 쓰는 값
    ("연쇄 1.5", 1.5, True),         # ①
    ("창 1.2", 1.2, False),          # ②
    ("창 1.3", 1.3, False),
    ("창 1.8", 1.8, False),
    ("연쇄+1.2", 1.2, True),         # ③
]


def toolpath_once(mesh, prof, direction="outward"):
    """슬라이싱 → 역변환 → 압출 샘플링. **판정은 하지 않는다** (재사용용)."""
    v = transform_cone_profile(mesh.vertices, prof, direction)
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    real, _ = backtransform(slice_mesh(warped, layer_height=LAYER_H),
                            prof, direction)
    return sample_extrusions(real, return_types=True, return_layers=True)


def judge(sample, vwin_factor, chained):
    """같은 툴패스에 판정만 다시 — '진짜 오버행 %p' 를 돌려준다."""
    pts, mid, w, kinds, lay = sample
    sup, _st = check_support(pts, mid, w, layer_height=LAYER_H,
                             vwin_factor=vwin_factor,
                             require_supported_below=chained)
    b = support_breakdown(pts, mid, kinds, lay, sup, w)
    return b["overhang_pct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="대표 5 모델만")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="checker_assumptions_results.json")
    args = ap.parse_args()

    specs = [(n, b) for n, b in build_specs(asym=True)
             if (not args.quick or n in QUICK)]

    names = [s[0] for s in SETTINGS]
    print("같은 툴패스에 판정만 바꿔 다시 잰다 (진짜 오버행 %p)")
    print(f"{'모델':<13}{'n':>2}" + "".join(f"{nm:>11}" for nm in names))
    print("-" * (15 + 11 * len(names)))

    rows = []
    for name, build in specs:
        mesh = build()
        rp = RadiusProfile(mesh)
        r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
        for n in (1, 2):
            t0 = time.time()
            r = select_banded_j(mesh, args.k, n, r_max, rp, MAX_SPACING_FACTOR)
            sample = toolpath_once(mesh, r["profile_obj"])
            vals = {nm: judge(sample, vf, ch) for nm, vf, ch in SETTINGS}
            rows.append(dict(model=name, n=n, seconds=round(time.time() - t0, 1),
                             thetas=[round(t, 1) for t in r["thetas"]], **vals))
            print(f"{name:<13}{n:>2}" + "".join(f"{vals[nm]:>11.3f}" for nm in names),
                  flush=True)

    # ── 판정 ────────────────────────────────────────────────
    def verdict(row_u, row_b, key):
        """그 설정에서 밴드2 가 균일을 오버행 축에서 이기나."""
        return row_b[key] < row_u[key] - 1e-9

    pairs = {}
    for r in rows:
        pairs.setdefault(r["model"], {})[r["n"]] = r
    pairs = {m: v for m, v in pairs.items() if 1 in v and 2 in v}

    print("\n[① 연쇄를 끊으면 수치가 얼마나 움직이나]")
    base = np.array([r["기준 1.5"] for r in rows])
    chain = np.array([r["연쇄 1.5"] for r in rows])
    d = chain - base
    print(f"  {len(rows)} 개 측정: 차이 중앙 {np.median(d):+.3f}%p, "
          f"최대 {d.max():+.3f}%p, 평균 {d.mean():+.3f}%p")
    worse = int((d > 1e-9).sum())
    print(f"  연쇄를 끊었더니 더 나빠진 측정 {worse}/{len(rows)}")
    if abs(np.median(d)) < 0.01 and d.max() < 0.05:
        print("  → 이 근사는 **무해하다**. 미지지 재료가 지지로 세어지는 몫이 거의 없다.")
    else:
        print("  → ⚠ 지금까지의 오버행 수치가 **낙관적**이었다. 크기를 문서에 박을 것.")

    print("\n[② 지지 창을 바꾸면 승/패가 뒤집히나]")
    print(f"  {'모델':<13}" + "".join(f"{nm:>11}" for nm in names))
    flips = 0
    for m, v in pairs.items():
        marks = []
        for nm, _vf, _ch in SETTINGS:
            marks.append("승" if verdict(v[1], v[2], nm) else "패")
        if len(set(marks)) > 1:
            flips += 1
        print(f"  {m:<13}" + "".join(f"{x:>11}" for x in marks))
    print(f"\n  설정에 따라 판정이 바뀐 모델: **{flips}/{len(pairs)}**")

    # 어느 손잡이가 뒤집었는지 **갈라서** 귀속한다 — 뭉뚱그리면 엉뚱한 상수를 범인으로 만든다.
    win_names = [nm for nm, _vf, ch in SETTINGS if not ch]      # 창만 바꾼 설정들
    chain_names = [nm for nm, _vf, ch in SETTINGS if ch]        # 연쇄를 켠 설정들
    base = SETTINGS[0][0]

    def flips_within(group):
        return sum(1 for v in pairs.values()
                   if len({verdict(v[1], v[2], nm) for nm in group}) > 1)

    fw, fc = flips_within(win_names), flips_within([base] + chain_names)
    print(f"  · 지지 창만 바꿨을 때({', '.join(win_names)}) 뒤집힌 모델: **{fw}/{len(pairs)}**")
    print(f"  · 연쇄를 켰을 때 뒤집힌 모델: **{fc}/{len(pairs)}**")
    if fw == 0:
        print("  → 창 값(= MAX_SPACING_FACTOR 와 공유하는 상수)은 **결과를 만들지 않는다.**")
    else:
        print("  → ⚠ 결론 일부가 검사기 **창 값**에 딸려 있다.")
    if fc:
        print("  → ⚠⚠ 결론이 **지지 연쇄 가정**에 딸려 있다. 이건 결함이 아니라 모델링")
        print("       선택이라 어느 쪽도 '옳다' 고 못 한다 — 현재값은 낙관적 하한,")
        print("       연쇄는 비관적 상한이다. **실물 출력으로만 좁혀진다.**")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1,
                  default=str)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
