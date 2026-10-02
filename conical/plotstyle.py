"""
plotstyle.py — 그림 라벨을 한글로 낼지 영문으로 낼지 자동 결정.

왜 필요한가: 개발 환경(리눅스 컨테이너)에는 한글 폰트가 없어서 matplotlib 이
한글을 두부(□)로 그린다. 그렇다고 라벨을 영문으로 고정하면 발표 자료를 만들 때
매번 손으로 고쳐야 한다. 그래서 **폰트가 있으면 한글, 없으면 영문**으로
자동 분기한다 — 같은 스크립트가 두 환경에서 각각 옳게 동작한다.

사용법:

    from conical.plotstyle import L
    ax.set_ylabel(L("unsupported perimeter (%)", "페리미터 미지지 (%)"))

확인:

    python3 -m conical.plotstyle      # 이 환경에서 한글이 되는지 + 쓸 폰트 이름

폰트를 직접 지정하려면 환경변수 CONICAL_PLOT_FONT 에 폰트 이름을 넣는다.
"""

import os

import matplotlib
from matplotlib import font_manager

# 흔한 한글 폰트 (윈도우 → macOS → 리눅스 순)
CANDIDATES = [
    "Malgun Gothic",          # 윈도우 기본
    "AppleGothic", "Apple SD Gothic Neo",
    "NanumGothic", "NanumBarunGothic", "NanumSquare",
    "Noto Sans KR", "Noto Sans CJK KR", "Source Han Sans KR",
    "Pretendard", "D2Coding",
]


def _register_bundled():
    """koreanize_matplotlib 가 설치돼 있으면 import 해서 번들 폰트를 등록시킨다.

    이 패키지는 나눔고딕 TTF 를 site-packages 안에 넣고 import 시점에
    matplotlib 폰트 목록에 등록한다. site-packages 는 시스템 폰트 경로가 아니라
    import 를 해야만 잡힌다 — 그래서 감지 전에 한 번 불러준다.
    시스템에 한글 폰트가 이미 있으면 굳이 필요 없다(없어도 조용히 넘어간다).
    """
    try:
        import koreanize_matplotlib  # noqa: F401
    except Exception:
        pass


def find_korean_font():
    """설치된 한글 폰트 이름 (없으면 None). 환경변수가 있으면 그것을 우선."""
    forced = os.environ.get("CONICAL_PLOT_FONT")
    _register_bundled()
    available = {f.name for f in font_manager.fontManager.ttflist}
    if forced:
        return forced if forced in available else None
    for name in CANDIDATES:
        if name in available:
            return name
    return None


def use_korean():
    """한글 폰트가 있으면 rcParams 를 설정하고 True, 없으면 아무것도 안 하고 False."""
    name = find_korean_font()
    if not name:
        return False
    # 폰트 '폴백 체인': 한글 폰트에 없는 글리프(⟨ ⟩ 같은 기호)는 뒤 폰트가 그린다.
    # 나눔고딕엔 각괄호가 없어 두부(□)로 나오던 것을 이렇게 막는다. (matplotlib ≥3.6)
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False   # 한글 폰트는 −(U+2212)가 없다
    return True


KOREAN = use_korean()       # import 시 1회 결정


def L(en, ko):
    """한글 폰트가 있으면 ko, 없으면 en."""
    return ko if KOREAN else en


if __name__ == "__main__":
    name = find_korean_font()
    if name:
        print(f"한글 폰트 있음: {name!r} → 그림 라벨이 한글로 나옵니다.")
    else:
        print("한글 폰트 없음 → 그림 라벨이 영문으로 나옵니다.")
        print("  설치된 폰트 중 후보를 못 찾았습니다. 다음 중 하나를 설치하거나,")
        print(f"  CONICAL_PLOT_FONT 환경변수로 직접 지정하세요: {', '.join(CANDIDATES[:6])}")
        names = sorted({f.name for f in font_manager.fontManager.ttflist})
        print(f"  (이 환경에 있는 폰트 {len(names)}종: {', '.join(names[:8])} ...)")
