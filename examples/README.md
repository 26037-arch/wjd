# examples — 예시 입력/출력

| 파일 | 내용 |
|---|---|
| `funnel.stl` | 데모 모델: 팁이 아래인 깔때기 (벽 오버행 70.3° — 평면 슬라이싱 시 서포트 51.5%) |
| `funnel_conical.gcode` | `python3 conical_slice.py examples/funnel.stl` 출력 (자동 outward 26°, 서포트 0%) |
| `funnel_open5x.gcode` | `--mode open5x` 출력 (U=26° 고정, V 방위각 추적) [실험적] |
| `lamp.stl` | 대조 모델: 구 + 가는 목(r=2) + 위로 벌어짐 = **'허리'가 있는 형상**. 밴드(부위별 각도)가 균일 원뿔을 이기는 조건을 보여주는 모델 (`compare_waist.py`, 페리미터 미지지 0.84% vs 균일 3.01%). 목에서 각도를 싸게 바꿀 수 있어서다. |

`tools/slicing_simulator.html` 의 "G-code 실행" 탭에서 STL과 G-code를 함께
로드하면 분석(빨간 오버행) → 왜곡 → 출력 재생을 한 화면에서 볼 수 있다.
| `funnel_profile2band.gcode` | 가변각 2밴드(`--profile "0:26,2.5:26,3.5:10,5:10"`) 출력 — θ(Z′) 파이프라인 예시 |
| `lamp_banded.gcode` | `python3 conical_slice.py examples/lamp.stl --auto-bands 2 --layer-height 0.4` 출력. 프로필 J(블렌드 비용 포함)가 [24°, 0°] 를 고르고, 경계가 자동으로 목(z 15.00→17.25)으로 옮겨가 블렌드가 1.8mm 로 끝난다 — 층간격 배율 1.50배(상한 준수) |

모든 G-code 는 `;CONICAL_META` 자기기술 헤더와 `;TYPE:` 압출 종류 주석
(Slic3r/PrusaSlicer 관례)을 담고 있어, 검사기와 시뮬레이터 검증 탭이 그대로 읽는다.
