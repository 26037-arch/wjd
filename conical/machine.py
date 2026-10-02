"""
machine.py — 기계 프로파일(시작/종료 G-code, 온도)을 G-code 에 주입한다.

왜 필요한가: 이 저장소의 파이프라인은 연구용이라 `G21/G90/M82` 만 내보낸다.
**그 파일로는 실물을 못 뽑는다** — 예열·호밍·프라임·냉각·모터 해제가 없다.
슬라이싱 결과(경로)와 기계 설정(예열·호밍)은 성질이 다르므로, 경로 생성 코드에
기계 상수를 섞지 않고 **파일 하나로 분리**한다.

형식은 INI 다 (`configparser`). 여러 줄 값은 들여쓰기로 잇는다.

    [machine]
    name = Open5x (Prusa 개조)
    nozzle_temp = 215
    bed_temp = 60
    start_gcode =
        M140 S{bed_temp}
        M104 S{nozzle_temp}
        G28
        M190 S{bed_temp}
        M109 S{nozzle_temp}
    end_gcode =
        M104 S0
        M140 S0
        M84

치환은 `{key}` 하나뿐이다 — 프로파일 자신의 키와, 슬라이싱이 정한 값
(`layer_height`, `angle`, `direction`, `mode`, `source_stl`)을 쓸 수 있다.
조건문·수식은 일부러 넣지 않았다: 학생이 읽고 손으로 고칠 수 있어야 한다.

⚠ **이 모듈은 G-code 를 검사하지 않는다.** 프로파일에 적은 대로 그대로 내보낸다.
  틀린 시작 코드는 기계를 상하게 할 수 있다. 첫 사용 전에 반드시 사람이 읽고,
  가능하면 노즐을 띄운 채로 한 번 돌려 볼 것.
"""

import configparser
from dataclasses import dataclass, field

REQUIRED_HINT = ("[machine] 절에 start_gcode / end_gcode 를 적는다. "
                 "예시는 profiles/machine.example.ini 참고.")


@dataclass
class MachineProfile:
    """기계 하나의 시작/종료 G-code 와 치환에 쓸 값들."""

    name: str = "unnamed"
    start_gcode: str = ""
    end_gcode: str = ""
    values: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path):
        # ⚠ `;` 를 주석 접두사에서 **뺀다.** 기본값은 ('#', ';') 인데 G-code 의
        #   주석이 바로 `;` 라, 그대로 두면 start_gcode/end_gcode 안의 주석 줄이
        #   **조용히 사라진다.** 실제로 템플릿의 `; TODO(5축): U/V 축 호밍 …` 이
        #   출력 파일에서 없어지고 있었다(2026-09-24 발견). 기계로 나가는 파일에서
        #   사람이 적은 줄이 말없이 빠지는 것은 이 저장소가 막기로 한 종류의 일이다.
        #   → INI 자체의 주석은 `#` 로 쓴다 (machine.example.ini 가 그렇게 돼 있다).
        cp = configparser.ConfigParser(comment_prefixes=("#",))
        # 키 이름의 대소문자를 보존한다 (치환 키와 1:1 로 맞추려고).
        cp.optionxform = str
        with open(path, encoding="utf-8") as fh:
            cp.read_file(fh)
        if not cp.has_section("machine"):
            raise ValueError(f"{path}: [machine] 절이 없다. {REQUIRED_HINT}")
        sec = dict(cp["machine"])
        start = sec.pop("start_gcode", "")
        end = sec.pop("end_gcode", "")
        name = sec.pop("name", "unnamed")
        if not start.strip() and not end.strip():
            raise ValueError(f"{path}: start_gcode·end_gcode 가 둘 다 비었다. "
                             + REQUIRED_HINT)
        return cls(name=name, start_gcode=start, end_gcode=end, values=sec)

    def _render(self, text, context):
        """`{key}` 치환. 프로파일 값이 먼저, 슬라이싱 값이 그 뒤에 덮어쓴다."""
        table = dict(self.values)
        table.update(context)
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                lines.append(line.format(**table))
            except KeyError as e:
                raise ValueError(
                    f"기계 프로파일 '{self.name}' 의 줄 `{line}` 에 있는 "
                    f"{e} 를 채울 값이 없다. 프로파일에 그 키를 적거나 줄을 고쳐라."
                ) from None
        return lines

    def start_items(self, context=None):
        """시작 G-code 를 `gcode.write` 가 받는 ('raw', str) 목록으로."""
        ctx = context or {}
        out = [("raw", f"; --- machine start: {self.name} ---")]
        out += [("raw", ln) for ln in self._render(self.start_gcode, ctx)]
        out.append(("raw", "; --- machine start end ---"))
        return out

    def end_items(self, context=None):
        ctx = context or {}
        out = [("raw", f"; --- machine end: {self.name} ---")]
        out += [("raw", ln) for ln in self._render(self.end_gcode, ctx)]
        return out
