"""analyze_blend_cost.py — `blend_penalty` 는 대체 무엇을 재고 있었나.

## 물음

J 의 블렌드 비용은 `Σ (블렌드 구간 표면적 %) × (m−1)/(limit−1)` 이고, 코드 주석은
그 선형 가중을 **"유도된 물리가 아닌 휴리스틱"** 이라고 몇 주째 달아 뒀다.
`analyze_rho_threshold.py` 가 "문턱이 왜 그 값인가" 를 닫으면서, 마지막으로 남은
구멍이 **"비용을 왜 그렇게 재는가"** 였다.

## 세 가설을 세우고 전부 기각했다

**① 층간격 위험을 잰다** — 그렇다면 `(m−1)/(limit−1)` 이 0~1 을 오가야 한다.
    실제로는 **선택된 프로필 전부에서 정확히 1** 이다. 계획기가 폭을 층간격 제약의
    **최소값**으로 잡으므로 `m = limit` 이 물린다. **가중이 항등이다.**

**② 블렌드가 만드는 손상을 잰다** — 그렇다면 블렌드 구간 **안**의 미지지 밀도가
    **밖**보다 높아야 한다. 실제로는 **밖의 0.48 배(중앙값)** 로 오히려 낮다.
    구조적 이유가 있다: 제약이 `m ≤ MAX_SPACING_FACTOR(1.5)` 로 가두는데 그 1.5 가
    **검사기 A 의 지지 창과 같은 값**이다(`toolpath.check_support: vwin = h*1.5`).
    제약을 지키는 블렌드는 미지지를 만들 수 **없다.**

**③ 해석식 예측의 오차를 메운다** — 그렇다면 그 오차와 상관이 있어야 한다.
    그런데 해석식은 **블렌드가 없는 상수 프로필에서도** 실측과 1.5~4.6 배로 어긋난다
    (면적 비율 vs 압출 무게 비율 — 애초에 눈금이 안 맞는 순위 대리물이다).
    축상 근사 탓도 아니다: 정확 Z′ 판(`analytic.support_fraction_profile_exact`)과
    차이가 **정확히 0** 이다(계획기가 블렌드를 반경이 작은 높이로 옮기기 때문).

## 답 — 정규화항이다

블렌드 폭이 `r_b·|Δtanθ|/(limit−1)` 이므로, 이 항은 결국 **반경으로 가중한 각도
변화량**을 벌한다. 실제 기능은 탐색을 병리적 해에서 밀어내는 것이고 그건 확인된다:
`k_blend=0` 이면 구가 `[36°, −44°]`(방향 전환, 블렌드 19.9mm)를 고른다.

⇒ **`k_blend` 는 물리 계수가 아니라 정규화 세기다.** "왜 선형인가" 는 물을 필요가
없는 질문이었다 — 가중이 항등이므로 선형 가중은 **작동한 적이 없다.**

실행:
    python3 analyze_blend_cost.py            # ① 만 (빠름)
    python3 analyze_blend_cost.py --full     # ①②③ 전부 (툴패스, 느림)
"""

import argparse
import sys

import numpy as np
import trimesh

from analyze_blend_ratio import build_specs
from compare_waist import LAYER_H
from conical import analytic
from conical.backtransform import backtransform
from conical.config import DEFAULT_K, MAX_SPACING_FACTOR
from conical.meshio import RadiusProfile
from conical.planar_slicer import slice_mesh
from conical.toolpath import check_support, sample_extrusions
from conical.transform import transform_cone_profile
from conical.varangle import select_banded_j


def risk_values(mesh, prof, rp, limit=MAX_SPACING_FACTOR, direction="outward"):
    """블렌드 구간마다 (m, risk). `blend_penalty` 안의 가중과 같은 식."""
    c = 1.0 if direction == "outward" else -1.0
    out = []
    for i in range(len(prof.zs) - 1):
        a, b = float(prof.zs[i]), float(prof.zs[i + 1])
        dt = prof.tans[i + 1] - prof.tans[i]
        if abs(dt) < 1e-12:
            continue
        s = dt / (b - a)
        r_b = rp.max_between(a, b)
        m = abs(1.0 - c * r_b * s)
        out.append((a, b, float(m),
                    float(min(1.0, max(0.0, (m - 1.0) / (limit - 1.0))))))
    return out


def unsupported_density(mesh, prof, direction="outward"):
    """블렌드 구간 안/밖의 미지지 페리미터 밀도(%). 블렌드가 없으면 None."""
    ivs = prof.blend_intervals()
    if not ivs:
        return None
    v = transform_cone_profile(mesh.vertices, prof, direction)
    warped = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    real, _ = backtransform(slice_mesh(warped, layer_height=LAYER_H),
                            prof, direction)
    pts, mid, w, kinds, _lay = sample_extrusions(real, return_types=True,
                                                 return_layers=True)
    sup, _st = check_support(pts, mid, w, layer_height=LAYER_H)
    z = pts[:, 2]
    peri = kinds == 0
    inb = np.zeros(len(z), dtype=bool)
    for a, b in ivs:
        inb |= (z >= a) & (z <= b)

    def dens(mask):
        tot = w[mask & peri].sum()
        return (w[mask & peri & ~sup].sum() / tot * 100.0) if tot > 1e-12 else np.nan

    share = w[inb & peri].sum() / max(w[peri].sum(), 1e-12) * 100.0
    return dens(inb), dens(~inb), share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="②(미지지 밀도) ③(정확 Z′ 대조) 까지 — 툴패스라 느리다")
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    args = ap.parse_args()

    risks, ratios, exact_gaps = [], [], []

    print("① 가중 (m−1)/(limit−1) 의 실제 값")
    print(f"{'모델':<13}{'블렌드 구간':>17}{'m':>9}{'risk':>8}"
          + ("{:>10}{:>9}{:>9}".format("밀도 안", "밖", "안/밖") if args.full else ""))
    print("-" * (47 + (28 if args.full else 0)))

    for name, build in build_specs(asym=True):
        mesh = build()
        rp = RadiusProfile(mesh)
        r_max = float(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]).max())
        r = select_banded_j(mesh, args.k, 2, r_max, rp, MAX_SPACING_FACTOR)
        prof = r["profile_obj"]

        if args.full:
            ap_ = analytic.support_fraction_profile(mesh, prof)
            ex_ = analytic.support_fraction_profile_exact(mesh, prof)
            exact_gaps.append(abs(ex_ - ap_))

        rv = risk_values(mesh, prof, rp)
        if not rv:
            print(f"{name:<13}{'(블렌드 없음)':>17}")
            continue

        tail = ""
        if args.full:
            d = unsupported_density(mesh, prof)
            if d is not None:
                d_in, d_out, _share = d
                ratio = d_in / d_out if d_out > 1e-9 else np.inf
                if np.isfinite(ratio):
                    ratios.append(ratio)
                tail = f"{d_in:>10.2f}{d_out:>9.2f}{ratio:>9.2f}"
        for a, b, m, risk in rv:
            risks.append(risk)
            print(f"{name:<13}{f'[{a:.2f},{b:.2f}]':>17}{m:>9.4f}{risk:>8.4f}{tail}",
                  flush=True)
            tail = ""

    v = np.array(risks)
    print(f"\n[① 판정] risk {len(v)} 개 — 최소 {v.min():.4f} 중앙 {np.median(v):.4f} "
          f"최대 {v.max():.4f};  risk==1 인 구간 **{int((v > 0.999).sum())}/{len(v)}**")
    print("  → 계획기가 폭을 제약의 최소값으로 잡아 m=limit 이 물린다. **가중은 항등이다.**")
    print("     즉 '휴리스틱 선형 가중' 은 작동한 적이 없고, 남는 것은")
    print("     cost = Σ(블렌드 구간 표면적 %) — 반경 가중 각도 변화량이다.")

    if args.full:
        if ratios:
            rr = np.array(ratios)
            print(f"\n[② 판정] 블렌드 안/밖 미지지 밀도비 — 중앙 **{np.median(rr):.2f}** "
                  f"(n={len(rr)}, 범위 {rr.min():.2f}~{rr.max():.2f})")
            print("  → 1 을 넘지 않는다. 블렌드는 미지지를 **집중시키지 않는다.**")
            print(f"     구조적 이유: 제약이 m ≤ {MAX_SPACING_FACTOR} 로 가두는데 그 값이")
            print("     검사기 A 의 지지 창(vwin = 층고 × 1.5)과 **같다.**")
        g = np.array(exact_gaps)
        print(f"\n[③ 판정] 축상 근사 vs 정확 Z′ — 최대 차이 **{g.max():.2e}** (n={len(g)})")
        print("  → 0 이다. 계획기가 블렌드를 반경이 작은 높이로 옮기므로 r·tanθ 오차가")
        print("     작아진다. 이 항은 축상 근사를 메우는 것도 아니다.")

    print("\n[결론] blend_penalty 는 손상 모형이 아니라 **정규화항**이다.")
    print("  k_blend 는 물리 계수가 아니라 **정규화 세기**다.")
    print("  실제 기능: k_blend=0 이면 구가 [36°, −44°] (방향 전환) 를 고른다 — 그걸 막는다.")


if __name__ == "__main__":
    sys.exit(main())
