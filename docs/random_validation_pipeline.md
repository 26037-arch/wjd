# 무작위 모델 기반 원뿔각 검증 파이프라인

이 파이프라인은 `select_cone`을 수정하지 않고, 예측한 각도를 G-code까지 실행해 검사한다. 입력 메시, G-code, REP5X 기구학, 충돌, 퇴적 형상, 물리 출력 가능성은 서로 다른 판정이다. `status=OK`는 선택된 경로가 **실행된 내부 검사**를 통과했다는 뜻이며, 실제 프린터에서 안전하거나 출력 가능하다는 뜻이 아니다.

## 재현

```powershell
python -m pip install -r requirements.txt
python -m pip install pymeshlab pytest
python tools/run_random_validation.py --count 100 --seed 20261002 --output-dir results/random_validation
```

외부 도구는 현재 저장소에 복사하지 않는다. PyMeshLab은 설치된 Python wheel을 사용한다. ADMesh는 공식 CLI를 설치한 후 `--admesh-exe`로 위치를 지정한다. Toolkit은 [공식 저장소](https://github.com/AdametherzLab/gcode-toolkit)를 별도 디렉터리에 clone하고 `npm install` 및 `node node_modules/typescript/bin/tsc --outDir dist`로 컴파일한다. `--toolkit-dir` 또는 환경 변수 `GCODE_TOOLKIT_DIR`은 이 디렉터리를 가리킨다. VOLCO도 [공식 저장소](https://github.com/FullControlXYZ/volco)를 별도 clone하고 그 의존성을 설치한 후 `--volco-dir` 또는 `VOLCO_DIR`로 지정한다. MAGE와 MultiAxis의 조사 소스 경로는 각각 `MAGE_SIMULATOR_DIR`, `MULTI_AXIS_DIR`로 설정할 수 있지만, REP5X 판정 상태는 여전히 지원 불가이다. 이 작업 환경에서는 Python 3.14 및 Trimesh 5.1.0으로 시범 실행했지만 VOLCO 저장소의 `requirements.txt`는 Trimesh 3.21.7을 고정하므로, 다른 환경에서는 호환성을 다시 확인해야 한다.

```powershell
python tools/run_random_validation.py --count 100 --seed 20261002 `
  --mesh-validator pymeshlab --mesh-cross-validator admesh `
  --admesh-exe C:\path\to\admesh.exe `
  --gcode-validator gcode-toolkit --toolkit-dir C:\path\to\gcode-toolkit `
  --collision-validator mage --collision-cross-validator multi-axis `
  --geometry-validator volco --volco-dir C:\path\to\volco `
  --output-dir results/external_validation
```

각도 스윕에는 `--angle-sweep --angle-step 4`를 추가한다. 스윕은 선택된 방향에서 기본적으로 0°부터 `MAX_ANGLE_DEG`까지 탐색하고 선택각을 반드시 포함한다. 표본 실험에서는 `--angle-max`로 상한을 줄일 수 있다. `--layer-height`, `--selector-k`, `--max-generation-retries`, `--volco-voxel-size` 등을 조정할 수 있다. 100개와 VOLCO 전체 스윕은 상당한 시간과 디스크를 쓸 수 있으므로, 먼저 `--count 1`로 환경을 확인하는 것이 좋다.

## 실제 검사 범위와 독립성

| Validator | 검증 대상 | 입력 → 출력 | 공유 가정 | 독립성 | 지원되지 않는 항목 | 실행 방법 | 이번 조사 버전/commit |
|---|---|---|---|---|---|---|---|
| [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) | STL 위상 | STL → 경계, 홀, 연결 성분, 비다양체, 자기 교차 면 | STL 삼각형 | 독립 구현 | 퇴적·기구학, 법선 및 퇴화 면은 이 어댑터에서 미검사 | `--mesh-validator pymeshlab` | wheel 2025.7.post1; 조사 소스 `9c30334547750bcbe7126278b16739349643d04b` |
| [ADMesh](https://github.com/admesh/admesh) | STL 구조·법선 | STL → 공식 CLI Original 열 및 처리 통계 | STL 삼각형 | 독립 구현 | 자기 교차·REP5X; 이번 환경에는 실행 파일 없음 | `--mesh-cross-validator admesh --admesh-exe ...` | 조사 소스 `e1b296b575ba574cc91480587a9876a36bbe2c0d` |
| [Sutura](https://github.com/Krateian/Sutura) | STL 결함/수리 | STL → 자체 triage | 내부적으로 PyMeshLab 사용 | PyMeshLab과 독립 아님 | 이번 파이프라인 미연결; Linux/macOS 중심 | 선택하지 않음 | 조사 소스 `63a05859c5998f05c244455172a994668aaa9eab` |
| [gcode-toolkit](https://github.com/AdametherzLab/gcode-toolkit) | G-code 기본 구조 | G-code → 파싱 오류, 통계, 이슈 | 일반 XYZ/E/F 문법 | 부분 독립 | B/C와 RTCP 및 REP5X 기계 판정 미지원; centered XY에 맞는 베드 원점을 알 수 없어 bed-boundary 검사는 끔 | 빌드 후 `--gcode-validator gcode-toolkit --toolkit-dir ...` | package 0.4.0; `052e1558c84248af7658eb68508c47a3b5eda5c7` |
| [MAGE Simulator](https://github.com/gear2nd-droid/MageSimulator) | 다축 기계·충돌 후보 | machine.xml·G-code → GUI 시뮬레이션 | 현재 공개 CoreXY-BC는 베드 회전식 | REP5X에 적용 불가 | 헤드 회전식 REP5X 모델 및 기계 판독 가능한 배치 결과 미확인 | 요청 시 `UNSUPPORTED_MACHINE_MODEL` | `10985b9971d36b00d836a92827f4d1346d13cb89` |
| [MultiAxis_3DP_MotionPlanning](https://github.com/zhangty019/MultiAxis_3DP_MotionPlanning) | 연구용 다축 충돌 후보 | 자체 waypoint/layer → Qt GUI 검사 | 다른 기계와 경로 정의 | REP5X에 적용 불가 | REP5X G-code 입력·배치 API 미확인 | 요청 시 `UNSUPPORTED_MACHINE_MODEL` | `006fca9fca915aa4a32e7aef7b851c08467d29d1` |
| [VOLCO](https://github.com/FullControlXYZ/volco) | 퇴적 형상 | 노즐 팁 XYZ/E/F 경로 → voxel/STL | REP5X의 XYZ가 RTCP 노즐 팁이라는 가정 | 퇴적 엔진은 독립 | B/C 자세, 헤드 충돌, 유동·열·실물 출력; 열린 STL의 체적은 신뢰 불가 | `--geometry-validator volco --volco-dir ...` | `0ea1398a824a5b67df91a4af3bf8775de10fcfbd` |
| `conical.toolpath` | 기하학적 지지 | 기존 실공간 경로 → optimistic/chained 미지지율 | 슬라이서와 층고·비드 폭 공유 | 내부 | 브리지 처짐, 접착, 열·재료 거동 | 기본 실행 | slicer 기준 `02f6430e3a7fed947c0517d079a216aa7c0a8424` |
| `conical.rep5x` | REP5X 범위·간섭 | REP5X 경로 → 축 한계, 되감기, 근사 간섭 | G-code 생성기와 기계 상수 공유 | 내부 | 실측되지 않은 B-arm 형상, 펌웨어 IK/축 부호 실기 검증 | 기본 실행 | 동일 |

gcode-toolkit의 `GCodeValidator`는 호출하지만 REP5X의 실측 feedrate, 베드 원점, 온도 한계가 없으므로 기계 의존 규칙은 끈다. 파싱 결과와 통계는 B/C를 무시한 구조 교차 검사로만 해석한다. toolkit의 layer/압출 통계도 비평면 REP5X에서 의미가 다를 수 있다.

MAGE README의 CoreXY-BC는 B/C가 베드를 움직이는 구조이고, REP5X는 헤드 틸트/요 구조이다. 이름이 같은 축을 쓰더라도 운동학은 다르다. `MultiAxis_3DP_MotionPlanning`도 자체 Qt/Visual Studio 워크플로를 전제한다. 두 엔진을 REP5X 검증에 적용했다고 주장할 수 없으므로 결과는 지원 불가로 저장한다. 물리적 미지지 압출 비율을 독립적으로 계산하는 범용 외부 도구도 확인되지 않았다.

## 판정 규칙과 출력

모델은 `SeedSequence([seed, model_id, attempt])`로 생성한다. 빈 메시, 비유한 좌표, 0 이하 체적, 열린 메시가 있으면 Trimesh 기본 판정에서 탈락한다. 선택한 외부 검사기가 실제로 `valid=false`를 내면 재생성한다. 외부 검사기가 없거나 오류이면 그 사실을 남기고 내부 판정으로 진행한다. 다만 `mesh_validated_externally=false`이다. `max_generation_retries`로 반복 횟수를 제한한다.

`cases.jsonl`은 실패 모델도 포함하고 각 case의 STL 및 G-code 경로, 예측(`prediction`), 내부·외부 결과, `validator_disagreement`를 기록한다. `summary.json`은 상태별 개수와 외부 검사 상태를 집계한다. 사용할 수 없는 숫자는 JSON `null`로 저장한다. 모델 생성 이후 실패는 다른 모델로 대체하지 않는다.

VOLCO는 REP5X의 `X/Y/Z`가 노즐 팁이라는 전제 아래 B/C를 제거한 복사본을 받는다. `orientation_not_modeled=true`는 반드시 유지한다. VOLCO의 voxel 배열용 XY 이동량은 재구성 STL에서 제거한 뒤 원본과 비교한다. Chamfer와 Hausdorff는 각각 고정 seed로 표면에서 1,500점씩 샘플링해 근사하므로 정확한 연속 표면 거리와 다르다. VOLCO STL이 닫혀 있지 않으면 `relative_volume_error=null`이고, 원시 메시 체적 차이는 참고값으로만 남긴다. 또한 VOLCO의 짧은 선분에서 `round(length / step_size)=0`이 되는 것을 피하려고 입력 경로의 최소 양의 압출 선분보다 작은 `step_size`를 설정한다. 이 설정이 지나치게 작아질 수 있어 대규모 스윕에는 실행 시간 제한이 필요하다.

스윕의 `internally_eligible_angles`는 **내부** G-code/운동학/충돌 통과 집합이다. 그 안에서 형상 Chamfer와 내부 chained 미지지율의 최솟값, 두 지표의 Pareto 집합을 별도로 제시한다. 물리적으로 유일한 최적각을 가정하거나 임의 가중합을 만들지 않는다. MAGE와 MultiAxis의 REP5X 검증이 불가능한 현재 상태에서는 `external_validation_complete=false`이고, 최종 신뢰도에는 물리 출력 실험이 필요하다.
