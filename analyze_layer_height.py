"""analyze_layer_height.py — 층고가 '밴드 vs 균일' 판정을 바꾸는가.

## 왜

2026-09-24 에 램프를 실물용으로 뽑다가 **층고가 결론을 바꾼다**는 것이 나왔다
(진짜 오버행 %p):

| 층고 | 균일 24° | 밴드2 | 낙관 판정 | 비관 판정 |
|---|---|---|---|---|
| 0.3mm | 0.267 / 4.113 | 0.000 / 0.000 | 밴드2 승 | 밴드2 승 |
| 0.4mm | 1.210 / 7.644 | 0.767 / 8.528 | 밴드2 승 | **균일 승** |

**두 점뿐이었다.** 그런데 이 두 점으로 첫 실물 실험의 층고를 0.4mm 로 정했다
(거기서 두 지지 모델이 갈리므로 한 번의 출력으로 판정된다는 논리).
**실물을 뽑기 전에 이 축을 넓혀야 한다** — 두 점이 우연이면 실험 설계가 틀린다.

## 설계

층고를 바꿀 때 **프로필은 다시 고르지 않는다.** 층간격 제약이 비율
(`m = 1 − c·r·s`, `MAX_SPACING_FACTOR`)이라 층고에 무관하고, 실제로 0.3/0.4 에서
같은 `[30°, 0°]` / 블렌드 2.31mm 가 나왔다. 그래서 **모델당 한 번 고르고 층고별로
슬라이싱만 다시 한다** — 원인을 층고 하나로 분리하고 비용도 아낀다.

각 층고에서 검사기의 `layer_height` 도 같이 맞춘다(지지 창과 베드 판정이 층고에
비례하므로 안 맞추면 다른 걸 재는 것이 된다).

## 판정 기준 (**결과 보기 전에** 적는다)

* **낙관 판정이 층고 전체에서 밴드2 승** → 밴드 우세는 층고에 둔감하다(좋은 결과)
* **비관 판정이 특정 층고 아래에서만 밴드2 승** → 0.3/0.4 의 뒤집힘이 **경향**이고,
  실험 A 는 **갈리는 층고**로 뽑아야 한다(지금 계획대로)
* **뒤집힘이 0.3↔0.4 에만 있고 다른 데선 없다** → **두 점의 우연**이다.
  실험 A 의 층고 근거를 다시 세워야 한다
* **판정이 층고마다 들쭉날쭉** → 층고가 교란변수다. 어떤 결론도 층고를 고정해
  말해야 하고, 지금까지의 표 전부에 층고를 명시해야 한다

실행:
    python3 analyze_layer_height.py                 # 램프 6점 + 대조 2모델
    python3 analyze_layer_height.py --lamp-only
"""

import argparse
import json
import sys
import time

import numpy as np
import trimesh

from analyze_blend_ratio import _sphere
from compare_waist import waisted_model
from conical.backtransform import backtransform
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.planar_slicer import slice_mesh
from conical.toolpath import check_support, sample_extrusions, support_breakdown
from conical.transform import transform_cone_profile
from conical.varangle import select_banded_j

HEIGHTS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
CONTROL_HEIGHTS = (0.25, 0.3, 0.4, 0.5)


def measure(mesh, prof, layer_height, direction="outward"):
    """이 층고로 자르고, 같은 툴패스를 두 지지 모델로 판정한다."""
    v = transform_cone_profile(mesh.vertices, prof, direction)
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    real, _ = backtransform(slice_mesh(warped, layer_height=layer_height),
                            prof, direction)
    pts, mid, w, kinds, lay = sample_extrusions(real, return_types=True,
                                               return_layers=True)
    out = {}
    for label, chained in (("낙관", False), ("비관", True)):
        sup, _st = check_support(pts, mid, w, layer_height=layer_height,
                                 require_supported_below=chained)
        out[label] = support_breakdown(pts, mid, kinds, lay, sup, w)["overhang_pct"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lamp-only", action="store_true")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--json", default="layer_height_results.json")
    args = ap.parse_args()

    models = [("램프 (=허리 r=2)", lambda: waisted_model(2.0), HEIGHTS)]
    if not args.lamp_only:
        models += [("허리 r=3", lambda: waisted_model(3.0), CONTROL_HEIGHTS),
                   ("구", _sphere, CONTROL_HEIGHTS)]

    rows = []
    for name, build, heights in models:
        mesh = build()
        rp = RadiusProfile(mesh)
        r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
        # 프로필은 층고와 무관하므로 **한 번만** 고른다.
        profs = {}
        for n in (1, 2):
            r = select_banded_j(mesh, args.k, n, r_max, rp, MAX_SPACING_FACTOR)
            profs[n] = (r["profile_obj"], [round(t, 1) for t in r["thetas"]])
        print(f"\n[{name}]  균일={profs[1][1]}  밴드2={profs[2][1]}"
              f"  블렌드={[(round(a,2), round(b,2)) for a, b in profs[2][0].blend_intervals()]}")
        print(f"{'층고':>6}{'균일 낙관':>10}{'밴드2 낙관':>11}{'낙관 판정':>11}"
              f"{'균일 비관':>11}{'밴드2 비관':>11}{'비관 판정':>11}")
        print("-" * 72)
        for h in heights:
            t0 = time.time()
            u = measure(mesh, profs[1][0], h)
            b = measure(mesh, profs[2][0], h)
            v_opt = "밴드2" if b["낙관"] < u["낙관"] - 1e-9 else (
                "균일" if u["낙관"] < b["낙관"] - 1e-9 else "무승부")
            v_pes = "밴드2" if b["비관"] < u["비관"] - 1e-9 else (
                "균일" if u["비관"] < b["비관"] - 1e-9 else "무승부")
            rows.append(dict(model=name, layer_height=h,
                             uniform_opt=u["낙관"], band2_opt=b["낙관"],
                             uniform_pes=u["비관"], band2_pes=b["비관"],
                             verdict_opt=v_opt, verdict_pes=v_pes,
                             seconds=round(time.time() - t0, 1)))
            print(f"{h:>6.2f}{u['낙관']:>10.3f}{b['낙관']:>11.3f}{v_opt:>11}"
                  f"{u['비관']:>11.3f}{b['비관']:>11.3f}{v_pes:>11}", flush=True)

    # ── 판정 ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    for name, _b, _h in models:
        sub = [r for r in rows if r["model"] == name]
        vo = {r["verdict_opt"] for r in sub}
        vp = {r["verdict_pes"] for r in sub}
        hs = [r["layer_height"] for r in sub]
        print(f"[{name}]  층고 {min(hs)}~{max(hs)} ({len(hs)}점)")
        print(f"  낙관 판정 : {'일관 ' + vo.pop() if len(vo) == 1 else '**층고에 따라 바뀐다** ' + str(sorted(vo))}")
        print(f"  비관 판정 : {'일관 ' + vp.pop() if len(vp) == 1 else '**층고에 따라 바뀐다**'}")
        if len(vp) != 1:
            flip = [(r["layer_height"], r["verdict_pes"]) for r in sub]
            print(f"      {flip}")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    sys.exit(main())
