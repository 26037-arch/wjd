# 선행연구 분석 — 형상 적응형 원뿔 슬라이싱 R&E

> 목적: **우리 방법의 각 부분이 어느 선행연구에서 왔는지, 무엇이 겹치고 무엇이
> 진짜 우리 것인지** 정직하게 매핑한다. 원칙: "이미 있는 걸 이상하게 재창작하면
> 마이너스"(네모난 바퀴 재발명 금지). 선행연구가 문제를 *어떻게 정의/해결*했는지
> 정확히 알고 그 위에 좁은 기여를 얹는다.
>
> ⚠️ **정직성 플래그**: 아래 **서지(저자·연도·DOI·저장소)는 1차 출처로 확인**했다.
> 그러나 조사 환경의 네트워크 차단으로 **일부 목적함수 '수식'은 원문 인용이 아니라
> 구조 재구성**이다. 수식을 논문/발표에 옮기기 전 반드시 오픈액세스 원문을 직접
> 대조할 것(§6 TODO). 확실히 교차확인된 수치는 그렇게 표시했다.

---

## 1. 지형: 다축/비평면 슬라이싱의 4개 층 + 우리 자리

| 층 | 대표 | 각도/방향 결정 | 무게 | 투명성 |
|---|---|---|---|---|
| 무거운 최적화 | **S³-Slicer**(2022), CurviSlicer(2019) | 자동(쿼터니언 필드 최적화) | 무거움 | 수학 무거움 |
| 딥러닝 | **Neural Slicer**(2024) | 자동(신경 변형장) | 매우 무거움(GPU 학습) | 블랙박스 |
| 분해 + 탐색 | **Wu 2020**, **Han 2025**, Gao 2019, Wei 2018 | 자동(부위별 **이산 방향**) | 중~무거움(beam search/Pareto) | 투명하나 무거움 |
| 접근형 경량(수동/고정) | RotBot(고정 45°; `var_angle` 스크립트도 **상수각**, §7 확인), Fractal Cortex·Open5x(수동) | 고정 또는 사람이 지정 | 경량 | 투명 |
| **← 우리** | 이 프로젝트 | **자동 + 부위별 연속 원뿔각** | **경량 휴리스틱** | **투명·재현** |

우리 자리 = "자동 *이면서* 싸고 *투명하게*, 그리고 그걸 가성비로 정량화". 무거운 것들은
계산으로 성능을 사고, 경량인 것들은 사람이 각도를 고른다. 우리는 그 사이.

---

## 2. 방법 대응표 — 우리 코드 ↔ 선행연구

| 우리 코드/개념 | 어느 선행연구가 이미 했나 | 그들의 정의/해결 | 우리 차이 |
|---|---|---|---|
| `transform.py` 원뿔 변환 | **RotBot**(Wüthrich 2021), **slicer4rtn**(Mueller) | 변환→평면슬라이스→역변환. f=(x/cosθ,y/cosθ,z+c·r·tanθ) | **우리 것 아님.** 그대로 차용, 반드시 인용 |
| `overhang.py` 면법선 판정 | **S³**의 support-free 항 | 면법선 n과 빌드방향 d의 각이 자기지지각 α(≈45°) 안인가 (per-face) | 동일 개념. 우리는 여기에 Cura/Prusa 관례까지 맞춤 |
| `overhang_layers.py` 레이어 2D | Cura/PrusaSlicer 서포트 생성 | 층 윤곽 − 아래층 윤곽(오프셋) | 실제 슬라이서 재현(검증용) |
| `regions.py`·`varangle.py` 부위별 각도 | **Wu 2020, Han 2025, Gao 2019, Wei 2018** | 모델을 부위로 분해, **각 부위에 이산 빌드 *방향*** | **★ 우리는 연속 *원뿔각 θ*** — 설계변수가 다름 |
| `compare_complexity.py`·`analyze_k.py` 가성비 곡선 | **Wu 2020(빔폭 B)**, **Han 2025(Pareto)** | 복잡도↑→서포트↓ 곡선/전선 | **개념은 그들 것.** 우리는 그 *원뿔각 버전* |
| `selector.py` 평가함수 J | **S³**(3목적 가중합), **Singularity**(2021) | E=w_SF·SF+w_SR·SR+w_SQ·SQ+w_reg | 우리 J를 이 구조에 맞춰 단순화 |
| `strength.py` 강도 | **Reinforced FDM**(Fang 2020) | 필라멘트를 주응력에 정렬(응력크기 가중) | 우리는 Hankinson으로 각도→강도 점수화 |

---

## 3. 무엇이 **새롭지 않은가** (발표에서 먼저 인정할 것)

1. **원뿔/RTN 변환 자체** — RotBot(2021), slicer4rtn. 우리는 사용자·확장자.
2. **자동 곡면레이어 방향 결정** — CurviSlicer(2019), S³(2022)이 이미.
3. **부위별 방향 분해** — Wu 2020, Gao 2019, Wei 2018, Han 2025.
4. **복잡도–성능 곡선 그 자체** — **Wu 2020이 빔폭 B로 이미** 냈다. 확인된 수치:
   **B 10→50 ⇒ 서포트 필요영역 17.34%→2.64%** (Wu 2020 및 후속 arXiv:2004.03450에서 교차확인).
   후속 논문 하나가 통째로 "그 곡선 위를 더 싸게 이동"을 다룸. → 곡선을 *발명*했다고
   말하면 안 됨.
5. **서포트 감소 목표** — 위 전부의 공통 목표.

## 4. 우리의 **좁고 방어 가능한 빈칸**

네 분해 논문 전부 **부위마다 "이산 빌드 방향"**을 고른다. **연속 파라미터(원뿔 반각 θ)를
부위별로 바꾸는 연구는 없다.** 그리고 원뿔 계열 쪽(RotBot)도 θ 를 **상수**로 둔다
(§7 원문 대조로 확정 — 'var_angle' 은 높이의 함수가 아니라 실행마다 고르는 상수다).
여기가 우리 자리:

1. **부위별 연속 원뿔각 θ 선택** — 이산 방향/절단평면이 아니라 conical 변환의 자유도(θ)를
   부위별로. (Wu·Han의 이산 방향에는 없는 축)
2. **그 θ/구간수 손잡이에 대한 가성비 곡선** — 단, **"Wu의 빔폭 곡선·Han의 Pareto 전선의
   원뿔 슬라이싱 대응물"**로 명시하고 둘 다 인용할 때만 정당.
3. **경량·투명·재현(고등학생이 노트북에서)** — S³/Neural(무거운 solver/GPU)과의 대비.

> 한 줄 주장: *"우리는 복잡도–성능 곡선을 발명한 게 아니라, 확립된 그 곡선(Wu의 빔폭,
> Han의 Pareto)을 **연속 원뿔각**이라는 새 축 위에서, **경량·투명하게** 다시 그린다."*

---

## 5. J를 선행연구에 **충실하게** 정의하기 (우리 J 업그레이드 설계)

⚠️ 아래 수식 구조는 재구성이다(§6에서 원문 대조 필요). 확립된 **3목적 조 = 서포트프리 +
강도 + 표면품질**이며 전부 각도 기반 무차원 항의 가중합 + 정규화항이다(S³).

우리 J(보상, 최대화):
```
J = w_s·Support + w_r·Strength − w_q·Staircase − w_g·Singularity
```
- **Support**(S³ 충실): 자기지지 면적 비율 = (1/A)·Σ A_f·[ (−d)와 n_f 각 ≤ α_self(≈45°) ].
  (지금 우리 서포트 지표의 물리적 정식화)
- **Strength**(Reinforced FDM 충실 + Hankinson): 고응력 요소에서 필라멘트 방향과 주응력 σ₁의
  각 φ를 Hankinson으로 점수화, 응력크기 가중:
  `σ_φ = σ∥σ⊥/(σ∥sinⁿφ + σ⊥cosⁿφ)`, Strength = Σ‖σ₁‖σ_φ / Σ‖σ₁‖σ∥.
  (우리 `strength.py`가 이 항의 경량 버전 — 지금은 하중을 FEA 대신 가정값으로)
- **Staircase/표면**(S³ SQ): 품질면에서 |cos(d, n_f)| 가중합(레이어가 표면과 나란=계단=나쁨).
- **Singularity**(옵션, Zhang 2021): 기계 특이점(짐벌락) 원뿔에 다가갈수록 폭증하는 장벽항.
  우리 하드웨어에 짐벌락 극이 있을 때만.

핵심: 모든 항이 무차원(면적÷총면적, 강도÷σ∥, 각의 sin/cos)이라 가중치 w는 순수 교환값.
**"단순 J(각도 페널티) vs 물리기반 J(위 3항)"** 비교가 그대로 인수인계 목표.

---

## 6. 검증 TODO & 확인된 서지

**⚠️ 원문 대조 필요(수식)**: 이 세션은 arXiv/MDPI/IEEE/ACM 전문이 차단되어 §5 수식은
구조 재구성이다. 아래 오픈액세스로 **정확한 식·임계값·가중치**를 직접 옮길 것:
- S³-Slicer 프리프린트: mewangcl.github.io/pubs/SIGAsia22S3Slicer.pdf
- Reinforced FDM 프리프린트: mewangcl.github.io/pubs/SIGAsia2020ReinforcedFDM.pdf
- Neural Slicer: arxiv.org/abs/2404.15061
- Singularity-aware: arxiv.org/abs/2103.00273
- Han 2025(완전공개): mdpi.com/2072-666X/16/12/1316
- Wu 2020 프리프린트: arXiv:1812.00606 · 후속: arXiv:2004.03450

**확인된 서지(1차 출처)**:
- Wüthrich, Gubser, Elspass, Jaeger. "A Novel Slicing Strategy…" *Applied Sciences* 11(18):8760, 2021. DOI 10.3390/app11188760. 코드: github.com/RotBotSlicer/Transform (GPL-3.0).
- Wu, Dai, Fang, Liu, Wang. "General Support-Effective Decomposition for Multi-Directional 3-D Printing." *IEEE T-ASE*, 2020. DOI 10.1109/TASE.2019.2938219. 프리프린트 arXiv:1812.00606. **B 10→50 ⇒ 17.34%→2.64% (교차확인)**.
- Wu, Liu, Wang. "Learning to Accelerate Decomposition for Multi-Directional 3D Printing." arXiv:2004.03450 (IEEE T-ASE 2021).
- Han, Qin, Chen, Liu, Cui. "Support-Free 3D Printing Based on Model Decomposition." *Micromachines* 16(12):1316, 2025. DOI 10.3390/mi16121316 (오픈액세스). Pareto(서포트면적 vs 부품수)+beam search.
- Gao, Wu, Yan, Nan. "Near support-free multi-directional 3D printing via global-optimal decomposition." *Graphical Models* 104:101034, 2019. DOI 10.1016/j.gmod.2019.101034.
- Wei et al. "Toward Support-Free 3D Printing: A Skeletal Approach…" *IEEE TVCG* 24(10):2799–2812, 2018. DOI 10.1109/TVCG.2017.2767047.
- Zhang, Fang, Huang, Dutta, Lefebvre, Kilic, Wang. "S³-Slicer." *ACM TOG* 41(6):277, 2022. DOI 10.1145/3550454.3555516. 코드: github.com/zhangty019/S3_DeformFDM.
- Fang, Zhang, Zhong, Chen, Zhong, Wang. "Reinforced FDM." *ACM TOG* 39(6):204, 2020. DOI 10.1145/3414685.3417834. 코드: github.com/GuoxinFang/ReinforcedFDM.
- Liu, Zhang, Chen, Huang, Wang. "Neural Slicer for Multi-Axis 3D Printing." *ACM TOG* 43(4):85, 2024. DOI 10.1145/3658212. arXiv:2404.15061. 코드: github.com/RyanTaoLiu/NeuralSlicer.
- Zhang, Chen, Fang, Tian, Wang. "Singularity-aware motion planning…" *IEEE RA-L* 6(4):6172–6179, 2021. DOI 10.1109/LRA.2021.3091109. arXiv:2103.00273.
- Ahn et al. "Anisotropic material properties of FDM ABS." *Rapid Prototyping J.* 8(4):248–257, 2002. DOI 10.1108/13552540210441166.
- Hankinson (1921); Kollmann (1934) — 강도-각도 경험식(지수 n).

**오픈소스(검증)**: S4 Slicer `github.com/jyjblrd/S4_Slicer`(GPL-3.0, Jupyter; 사면체 거리장→변형→Cura;
동반 프린터 `Core-R-Theta-4-Axis-Printer`; 별도 `Radial_Non_Planar_Slicer`). Rep5x
`github.com/dennisklappe/Rep5x`(GPL-3.0, JS, 2026, Ender 5 Pro/Ender 3 V3 SE 개조).
Fractal Cortex `github.com/fractalrobotics/Fractal-Cortex`(GPL-3.0, Python, **수동** 슬라이스평면).
slicer4rtn `github.com/Spiritdude/Slicer4RTN`(**LGPL-3.0**, Perl, 3/4/5축). Open5x
`github.com/FreddieHong19/Open5x`(**MIT**, Rhino/Grasshopper 의존; CHI EA 2022, DOI
10.1145/3491101.3519782, arXiv:2202.11426).

---

## 7. RotBot 원문 대조 (2026-09-21) — **'var_angle' 은 θ(z) 가 아니다**

§6 의 "원문 대조 필요" TODO 중 **RotBot 항목을 닫는다.** 저장소
`github.com/RotBotSlicer/transform` (커밋 `ed8128e`) 를 직접 읽었다.

### 무엇을 확인했나

**'Scripts for Variable Angle' 의 'variable' 은 '실행마다 사용자가 고르는 상수각'
이라는 뜻이고, 높이의 함수가 아니다.** 저장소 README 원문:

> ### Scripts for variable angle
> With this scripts, the cone angle **can be changed**. So it does not only work for
> 45° angle as used for RotBot, but can also be used with much smaller angles
> (e.g. 15°) to do a conical slicing for any printer.

코드가 확증한다:

| 위치 | 증거 |
|---|---|
| `Scripts for Variable Angle/Transformation_STL_var_angle.py:13` | `CONE_ANGLE = 16` — 스칼라 상수 |
| 같은 파일 `:18`, `:35` | `transformation_kegel(points, cone_angle_rad, cone_type)` → `np.tan(cone_angle_rad)`. **z 의존성 없음** |
| `Scripts for Variable Angle/Backtransformation_GCode_var_angle.py:215–228` | `z_layer + c*sqrt(x²+y²)*tan(cone_angle_rad)` — 역시 스칼라 |
| 기본 `Backtransformation_GCode.py:40` | *"has to be divided by **sqrt(2)**"* = cos(45°) **하드코딩** |

즉 기본 스크립트는 RotBot 실기의 45° 틸트 노즐에 고정돼 있고, `var_angle` 스크립트는
그것을 **임의의 상수각으로 일반화**한 것이다. 그 이상은 없다.

### 따름 — 이 저장소의 기여 범위가 넓어진다

1. **θ(Z′) 가변각 프로필은 RotBot 에 없다.** 차용한 것은 **변환식과 3단계 구조**이고
   (`conical/transform.py`), θ(Z′) 로의 일반화는 이 저장소의 확장이다.
2. **블렌드(각도 전환 구간)와 층간격 배율 `m` 은 RotBot 의 문제공간에 존재하지
   않는다.** 상수각이면 `s = dT/dZ′ = 0` 이라 `m = 1 − c·r·s ≡ 1` 이다. 이 제약은
   θ 를 높이의 함수로 허용해야 비로소 **생겨난다** — 따라서 `w ≥ r_b·|Δtanθ|/(limit−1)`
   유도(`conical/profile.py`)는 "선행연구가 안 쓴 것"이 아니라 **선행연구에 없는 문제**다.
3. 이 확장은 §"새로움을 만드는 세 패턴" 중 **파라미터화**의 형태다: RotBot 의 상수각이
   우리 상수 프로필의 **특수경우로 포함**되고, 그 포함이
   `tests/test_profile_constant_equals_fixed.py` 로 **회귀 테스트에 고정**돼 있다.
   기존 결과가 검증 기준 역할을 하므로 가장 안전한 형태의 일반화다.

### 이 대조가 고친 우리 쪽 오류

`conical/varangle.py` 와 `README.md` 가 θ(z) 를 "RotBot 의 var_angle 방식"이라고
서술했다. **틀렸고, 자기 기여를 선행연구로 돌리는 방향의 오류였다.** 두 곳 모두
정정했다. 반면 §1 표가 RotBot 을 "고정 45°" 로 분류한 것은 **맞았다** — 두 서술이
충돌했는데 선행연구 문서 쪽이 옳았다.

### ⚠ 아직 안 닫힌 것

**논문 본문**(Wüthrich et al., *Appl. Sci.* 11(18):8760)은 이 환경의 망 차단으로
확인하지 못했다. 특히 **Conclusion / future work 절**에 "각도를 높이에 따라 바꾸는
것"이 향후 과제로 적혀 있는지 확인할 것. 적혀 있다면 우리 확장이 **저자들이 지목한
빈칸**에 들어가는 형태가 되므로 방어가 더 쉬워진다. 위 코드 기반 결론은 이 확인과
무관하게 유지된다(코드가 1차 출처다).
