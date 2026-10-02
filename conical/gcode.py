"""
gcode.py — 파이프라인이 공유하는 아주 작은 G-code 데이터 모델.

슬라이서 전체를 흉내내는 게 아니라, 변환 파이프라인에 필요한 최소만 다룬다:
  · 이동(G0/G1)의 X Y Z E F 를 읽고 쓴다
  · 그 외 줄(온도, 팬, 주석 등)은 '그대로 통과'시킨다 (역변환이 건드리면 안 되니까)
"""

import re

_WORD = re.compile(r"([A-Za-z])([-+]?[0-9]*\.?[0-9]+)")


class Move:
    """G0/G1 한 줄. 지정 안 된 좌표는 None (모달: 이전 값 유지)."""
    __slots__ = ("g", "x", "y", "z", "e", "f", "extra")

    def __init__(self, g=1, x=None, y=None, z=None, e=None, f=None, extra=""):
        self.g, self.x, self.y, self.z, self.e, self.f = g, x, y, z, e, f
        self.extra = extra          # A/B/C 회전축 등 추가 워드 문자열

    def to_line(self):
        parts = [f"G{self.g}"]
        for k, v, fmt in (("X", self.x, ".3f"), ("Y", self.y, ".3f"),
                          ("Z", self.z, ".3f"), ("E", self.e, ".5f"),
                          ("F", self.f, ".0f")):
            if v is not None:
                parts.append(f"{k}{format(v, fmt)}")
        if self.extra:
            parts.append(self.extra)
        return " ".join(parts)


def parse(lines):
    """G-code 줄들을 [(kind, payload)] 로 파싱.
    kind = "move" → payload Move / kind = "raw" → payload 원본 문자열."""
    out = []
    for line in lines:
        code = line.split(";", 1)[0].strip()
        if not code:
            out.append(("raw", line.rstrip("\n")))
            continue
        words = _WORD.findall(code)
        if not words or words[0][0].upper() != "G" or float(words[0][1]) not in (0.0, 1.0):
            out.append(("raw", line.rstrip("\n")))
            continue
        mv = Move(g=int(float(words[0][1])))
        extras = []
        for k, v in words[1:]:
            k = k.upper()
            if k == "X":
                mv.x = float(v)
            elif k == "Y":
                mv.y = float(v)
            elif k == "Z":
                mv.z = float(v)
            elif k == "E":
                mv.e = float(v)
            elif k == "F":
                mv.f = float(v)
            else:
                extras.append(f"{k}{v}")
        mv.extra = " ".join(extras)
        out.append(("move", mv))
    return out


def write(items, path):
    with open(path, "w") as fh:
        for kind, payload in items:
            fh.write((payload.to_line() if kind == "move" else payload) + "\n")
