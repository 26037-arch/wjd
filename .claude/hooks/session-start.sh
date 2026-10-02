#!/bin/bash
# Claude Code on the web 세션 시작 훅.
# 목적: pytest·ruff 와 conical 패키지가 바로 돌아가는 상태로 세션을 시작한다.
set -euo pipefail

# 웹(원격) 세션에서만 돈다. 로컬에서는 각자 venv 를 쓰므로 건드리지 않는다.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# 실행 의존성 (trimesh, numpy, matplotlib, shapely, networkx, scipy, rtree, koreanize-matplotlib)
python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt

# 개발 의존성. requirements.txt 는 실행용이라 pytest·ruff 가 없다.
# ruff 설정은 ruff.toml 에 고정돼 있다 (`ruff check .`).
python3 -m pip install --quiet --disable-pip-version-check pytest ruff

# tests/ 가 리포지토리 루트의 conical 패키지를 import 할 수 있게 한다.
echo 'export PYTHONPATH="${CLAUDE_PROJECT_DIR}:${PYTHONPATH:-}"' >> "$CLAUDE_ENV_FILE"

# matplotlib 은 헤드리스. 그림 생성 스크립트가 디스플레이를 찾다 죽지 않게 한다.
echo 'export MPLBACKEND=Agg' >> "$CLAUDE_ENV_FILE"
