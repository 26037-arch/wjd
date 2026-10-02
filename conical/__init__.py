"""
conical — 형상 적응형 원뿔(conical) 슬라이싱 알고리즘 패키지.

이 패키지 하나로 주요 기능을 바로 가져다 쓸 수 있다:

    from conical import analyze_overhangs, support_fraction, select_cone

각 단계별 모듈:
    config    : 설정값(임계각, 최대 각도, 탐색 간격) 한 곳에 모음
    transform : 원뿔 좌표 변환식 (RotBot 방식)
    overhang  : Stage 1 — 오버행 분석
    sweep     : Stage 2 — 각도별 남은 서포트 계산
    selector  : Stage 3 — 평가함수 J로 각도/방향 자동 결정
    meshio    : STL 로드 / 데모 구 생성 도우미
"""

from . import analytic, config

# 판정 기준 통일(2026-07 리뷰): 패키지 기본 export 는 해석식(analytic).
# 레거시(변환공간)는 conical.sweep.support_fraction 처럼 모듈 경로로만 접근.
from .analytic import face_support_and_staircase, support_fraction, sweep_table
from .meshio import center_on_axis, load_mesh_or_demo
from .overhang import analyze_overhangs, summarize, support_area_fraction
from .selector import evaluate_J, k_sensitivity, select_cone
from .strength import face_strength, part_strength
from .transform import transform_cone
from .varangle import (
                       blend_penalty,
                       profile_objective,
                       select_banded,
                       select_banded_j,
                       select_fine,
                       select_uniform,
)

__all__ = [
    "config",
    "transform_cone",
    "analyze_overhangs", "support_area_fraction", "summarize",
    "support_fraction", "sweep_table",
    "evaluate_J", "select_cone", "k_sensitivity",
    "load_mesh_or_demo", "center_on_axis",
    "face_support_and_staircase",
    "select_uniform", "select_banded", "select_fine",
    "select_banded_j", "profile_objective", "blend_penalty",
    "face_strength", "part_strength",
    "analytic",
]
