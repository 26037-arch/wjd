"""
compare_kappa_rules.py — 임계 κ 규칙 둘 중 어느 쪽이 실제 툴패스를 잘 예측하나.

팀 안에 서로 다른 κ 규칙이 있다 (conical/kappa.py 참조):
  ① 고정   κ = −sin(45°) ≈ −0.7071      — 실제 슬라이서 관례(Cura)에 앵커
  ② 형상의존 κ_c = −κ_max·q/(q+q₀)      — 팀메 규칙, 성분 둘레/면적에 반응

의견으로 정할 일이 아니라 **실측으로 판정할 일**이다. 두 규칙의 예측을
독립 지표(툴패스 검사기의 페리미터 미지지 %)와 대조한다.

    python3 compare_kappa_rules.py

⚠ 두 규칙 모두 '메시 표면'을 재고 툴패스는 '압출 경로'를 잰다 — 절대값이 아니라
  각도에 따른 순위·경향이 맞는지가 판정 기준이다.
"""

import sys

import numpy as np
import trimesh
from scipy.stats import spearmanr

from compare_waist import run_pipeline, waisted_model
from conical.kappa import component_kappa, fixed_kappa, support_fraction_kappa
from conical.meshio import center_on_axis

ANGLES = [0, 8, 16, 24, 32, 40]


def main():
    models = [
        ("구 (허리 없음)",
         center_on_axis(trimesh.creation.icosphere(subdivisions=3, radius=10.0))),
        ("램프 (허리 있음)", waisted_model()),
    ]
    if len(sys.argv) > 1:
        models = [(sys.argv[1],
                   center_on_axis(trimesh.load(sys.argv[1], force="mesh")))]

    m_rows = {}
    print("=" * 82)
    print("임계 κ 규칙 비교 — 예측(메시) vs 실측(툴패스 페리미터 미지지)")
    for name, m in models:
        kap_fixed = np.full(len(m.faces), fixed_kappa())
        kap_comp, info = component_kappa(m)
        ths = sorted({round(i["threshold_deg"], 1) for i in info})
        print(f"\n[{name}]  오버행 성분 {len(info)}개 → "
              f"팀메 규칙 임계각 {ths}°  (우리 고정 45°)")
        print(f"  {'각도':>4} | {'실측 페리미터':>12} | {'①고정 예측':>10} | {'②형상 예측':>10}")
        rows = []
        for a in ANGLES:
            peri, _, _ = run_pipeline(m, float(a))
            p1 = support_fraction_kappa(m, a, "outward", kap_fixed)
            p2 = support_fraction_kappa(m, a, "outward", kap_comp)
            rows.append((a, peri, p1, p2))
            print(f"  {a:4d}° | {peri:11.2f}% | {p1:9.2f}% | {p2:9.2f}%")
        meas = [r[1] for r in rows]
        m_rows[name] = meas
        r1 = spearmanr(meas, [r[2] for r in rows]).statistic
        r2 = spearmanr(meas, [r[3] for r in rows]).statistic
        print(f"  순위상관(실측 대비)  ①고정 ρ={r1:+.3f}   ②형상 ρ={r2:+.3f}")
        # 각 규칙이 '서포트 최소'로 고르는 각도와, 그 각도의 실측 성능
        b1 = min(rows, key=lambda r: r[2])
        b2 = min(rows, key=lambda r: r[3])
        best = min(rows, key=lambda r: r[1])
        print(f"  ①이 고르는 각도 {b1[0]}° → 실측 {b1[1]:.2f}%   "
              f"②가 고르는 각도 {b2[0]}° → 실측 {b2[1]:.2f}%   "
              f"(실측 최적 {best[0]}° → {best[1]:.2f}%)")
    # ── κ_max 눈금 스윕: 형상 의존 규칙이 '어느 눈금에서' 고정 규칙과 만나나 ──
    print("\n" + "=" * 82)
    print("κ_max 눈금 스윕 — 형상 의존 규칙은 눈금만 맞추면 고정 규칙과 만난다")
    print(f"  {'κ_max':>6} | " + " | ".join(f"{n:>10}" for n, _ in models)
          + "     (칸: 유효 임계각 / 실측 대비 ρ)")
    measured = {name: m_rows[name] for name in m_rows}
    for kmax in (0.3, 0.5, 0.707, 0.85, 1.0):
        cells = []
        for name, m in models:
            kap, info = component_kappa(m, kappa_max=kmax)
            th = info[0]["threshold_deg"] if info else float("nan")
            pred = [support_fraction_kappa(m, a, "outward", kap) for a in ANGLES]
            rho = spearmanr(measured[name], pred).statistic
            cells.append(f"{th:5.1f}° ρ={rho:+.3f}")
        print(f"  {kmax:6.3f} | " + " | ".join(cells))
    print("  (참고) 고정 45° 규칙의 ρ 는 위 모델별 표의 ①고정 값과 같다.")

    print("\n" + "=" * 82)
    print("⚠ 표본 2모델 · 각도 6개의 경향이다. 실물 출력이 불가능한 조건에서")
    print("  '어느 κ 가 옳은가'의 최종 근거는 이 툴패스 실측과 선행연구 관례뿐이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
