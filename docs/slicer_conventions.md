# 실제 오픈소스 슬라이서의 오버행/서포트 판정 방식 (조사 정리)

우리 코드가 실제 슬라이서와 어떻게 맞물리는지 정리한 문서. 발표/논문에서
"우리 방식이 근거 없는 게 아니라 실제 슬라이서 관례와 대응된다"를 보일 때 쓴다.
모든 사실은 각 슬라이서의 **소스코드/공식 설정 정의**에서 확인했다(2차 블로그 아님).

## 1. 오버행 각도 관례 — 슬라이서마다 다르다

우리 정의: `overhang_angle = arcsin(-n_z)` → **0° = 수직 벽, 90° = 수평 천장,
임계각을 초과하면 서포트.** ("수직 기준 각도, 초과 시 서포트" 관례)

| 슬라이서 | 설정(내부 키) | 각도 기준 | 서포트 조건 | 기본값 | 우리와의 관계 |
|---|---|---|---|---|---|
| **Cura** | `support_angle` | 수직 기준 (0°=수직벽, 90°=수평) | 임계각 **이상** | **50°** (과거 45°) | **완전 동일** |
| **PrusaSlicer** | `support_material_threshold` | 수평 기준 경사각 (90°=수직) | 임계각 **미만** | 0=자동 | 여집합: `90 − 우리` |
| **SuperSlicer** | `support_material_threshold` | Prusa와 동일 | Prusa와 동일 | 0=자동 | 여집합 |
| **OrcaSlicer** | `support_threshold_angle` | 수평 기준 경사각 | 임계각 **미만** | **30°** | 여집합 |

방향이 반대인 건 물리적으로 실제다: Cura는 값이 **클수록 서포트가 줄고**,
Prusa/Orca는 값이 **클수록 서포트가 는다**. (한 면의 '수직 기준 각도' + '수평 기준
경사각' = 90° 이므로 서로 여집합.)

변환 헬퍼는 `conical/config.py`의 `ours_to_prusa()`, `prusa_to_ours()`.

## 2. 판정 방식 — 실제로는 '레이어별 2D', 면 법선 아님

네 슬라이서 **모두** 각 레이어의 2D 윤곽을 '아래 레이어'와 비교해 서포트를 만든다.
면 법선(face normal)을 서포트 생성에 쓰지 않는다.

- **Cura** (`CuraEngine/src/support.cpp`): `max_dist = tan(support_angle) × layer_height`,
  `overhang = outline.difference(outline_below.offset(max_dist))`.
- **PrusaSlicer** (`Support/SupportMaterial.cpp`, `detect_overhangs`):
  `lower_layer_offset = layer_height / tan(threshold)`, 그 뒤 윤곽 차집합.
- **Orca/Super**: 같은 Slic3r 계열 방식.

차이(우리 글에 쓸 포인트):
- **레이어 2D 방식은 '아래에 받쳐주는 게 있는지'를 안다.** 밑에 재료가 있는
  아래보기 면(구멍 윗면, 계단 윗단, 바닥에 닿는 면)은 서포트로 안 잡는다.
- **면 법선 방식은 그걸 모른다** → 아래보기 면을 전부 오버행으로 과대평가.
  (우리 `overhang.py`가 정육면체 밑면을 16.7%로 오판하는 게 바로 이 한계.)
- 대신 면 법선은 빠르고 레이어높이와 무관 → '표면 경사 미리보기/분석'엔 충분.

→ 그래서 우리는 두 가지를 **둘 다** 둔다:
  `overhang.py`(빠른 면법선, 영역 분할·미리보기용) +
  `overhang_layers.py`(슬라이서 충실, 서포트 넓이 추정용).

## 3. 브릿지

브릿지(두 지지대 사이 평평한 다리)는 별도 단계로 서포트에서 제외한다.
- PrusaSlicer/Super: `dont_support_bridges`(기본 켜짐)가 브릿지 영역을 빼줌.
- Orca: 동일 취지 옵션.
- Cura: 전용 토글 없이 각도 임계 + 브릿지 설정으로 처리.

우리 `overhang_layers.py`는 이 브릿지 제외 단계는 아직 안 넣었다(정직한 한계).

## 4. 비평면/원뿔 슬라이싱

Cura/Prusa/Super/Orca 모두 **평면 평행 레이어만** 슬라이싱한다. 원뿔 슬라이싱은
이들을 '감싸서' 구현한다: STL을 원뿔 변환 → 일반 슬라이서로 슬라이싱 → G-code
역변환. RotBot(ZHAW)이 대표적이며 내부적으로 PrusaSlicer 2.7.0을 사용했다.

## 출처
- PrusaSlicer 설정: https://github.com/prusa3d/PrusaSlicer/blob/master/src/libslic3r/PrintConfig.cpp
- PrusaSlicer 판정/브릿지: https://github.com/prusa3d/PrusaSlicer/blob/master/src/libslic3r/Support/SupportMaterial.cpp
- OrcaSlicer 설정: https://github.com/SoftFever/OrcaSlicer/blob/main/src/libslic3r/PrintConfig.cpp
- Cura 설정: https://github.com/Ultimaker/Cura/blob/main/resources/definitions/fdmprinter.def.json
- CuraEngine 판정: https://github.com/Ultimaker/CuraEngine/blob/main/src/support.cpp
- Prusa 서포트 문서: https://help.prusa3d.com/article/support-material_1698
- RotBot: https://github.com/RotBotSlicer/Transform · https://www.cnckitchen.com/blog/the-rotbot-4-axis-non-planar-3d-printing
