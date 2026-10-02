# 슬라이서 CLI 분석 — 파이프라인 백엔드 선정 (소스코드 검증)

`conical_slice.py`의 [3]단계(평면 슬라이싱)에 꽂을 외부 슬라이서 6종을 분석했다.
전부 각 프로젝트의 `PrintConfig.cpp`(CLI 정의)·공식 위키에서 확인(2026-07).

## 계보 — 백엔드는 딱 2종류만 필요

| 계열 | 슬라이서 | CLI 스타일 |
|---|---|---|
| Slic3r계 | Slic3r(원조), **PrusaSlicer**, SuperSlicer | `-g --load cfg.ini -o out.gcode` + 평평한 INI |
| Bambu계 | Bambu Studio, **OrcaSlicer** | `--slice N --load-settings "m.json;p.json" --outputdir` + JSON 프로필 |
| 독립 | S4 Slicer | CLI 없음 (Jupyter 노트북) — 백엔드 불가, 비교 대상만 |

## 추천

1. **PrusaSlicer (1순위)** — 수년간 안정된 공식 CLI, 단일 INI(`--load`),
   모든 옵션 CLI 재정의 가능(`--help-fff`), `--dont-arrange`로 좌표 보존,
   `resolution`/`gcode_resolution` 손잡이, 리눅스 headless 확실.
   RotBot 원본도 PrusaSlicer를 썼다.
   ```
   prusa-slicer -g --load profiles/conical_pipeline.ini --dont-arrange -o out.gcode warped.stl
   ```
2. **SuperSlicer (2순위)** — 플래그 100% 동일(바이너리 경로만 교체). 단 개발 정체.
3. **OrcaSlicer** — 활발히 유지되는 2번째 생태계가 필요하면. 단 CLI가 다름:
   `--slice 1 --load-settings "machine.json;process.json" --arrange 0 --outputdir out/`
   → `out/plate_1.gcode`. headless는 AppImage 추출(`squashfs-root/AppRun`) 권장.
4. **Bambu Studio** — Orca와 같은 계열이지만 CLI 문서 최소, GUI와 결과 다른
   이슈(#1704) 보고 → 비추천. **Slic3r(원조)** — 2018년 이후 미유지, 레거시만.

## 파이프라인 필수 설정 (백엔드 무관 — 이유가 중요)

| 설정 | Prusa계 | Orca/Bambu계 | 왜 필수인가 |
|---|---|---|---|
| 자동배치 끄기 | `--dont-arrange` | `--arrange 0` | 변환 메시의 XY 좌표가 옮겨지면 역변환 좌표계가 깨짐 |
| 아크 피팅 끄기 | `arc_fitting = disabled` (기본) | `"enable_arc_fitting": "0"` (**켜진 프로필 많음 주의**) | G2/G3 호는 점 단위 역변환 불가 |
| 서포트 끄기 | `support_material = 0` | `"enable_support": "0"` | 변환공간 45° 자동서포트는 물리와 다름 → 서포트 판단은 우리 해석식 |

## 주요 키 대응표

| 목적 | Prusa INI | Orca/Bambu JSON |
|---|---|---|
| 레이어 높이 | `layer_height` | `layer_height` |
| 페리미터 수 | `perimeters` | `wall_loops` |
| 인필 밀도 | `fill_density` (`15%`) | `sparse_infill_density` (`"15%"`) |
| G-code 단순화 | `gcode_resolution` (기본 0.0125) | `resolution` (기본 0.01) |
| 시작/끝 G-code | `start_gcode`/`end_gcode` | `machine_start_gcode`/`machine_end_gcode` (machine 프로필) |

메모: Orca/Bambu JSON은 값이 전부 문자열(`"0"`), `--load-settings`는 세미콜론
구분 한 인자(machine 먼저). Bambu 출력은 `.gcode.3mf`(zip 내부 `Metadata/plate_1.gcode`).

## 이 저장소에서 바로 쓰기

```
# PrusaSlicer가 설치돼 있다면:
python3 conical_slice.py model.stl \
  --slicer-cmd "prusa-slicer -g --load profiles/conical_pipeline.ini --dont-arrange -o {gcode} {stl}"
# 없으면 내장 미니 슬라이서로 자동 동작 (연구 검증용)
```
