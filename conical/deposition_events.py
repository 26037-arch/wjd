"""Modal G-code motion events for deposition and REP5X playback.

XYZ are RTCP nozzle-tip coordinates in REP5X mode.  B/C are commanded angles;
the recorded C trajectory is never implicitly wrapped or shortened.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from .rep5x import tool_direction

_WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


@dataclass(frozen=True)
class MotionEvent:
    line: int
    command: str
    start_position: tuple[float, float, float]
    end_position: tuple[float, float, float]
    start_b: float
    start_c: float
    end_b: float
    end_c: float
    delta_e: float
    feedrate: float
    segment_length: float
    estimated_duration: float
    start_time: float
    end_time: float
    extruding: bool
    retracting: bool

    def pose(self, fraction: float) -> dict:
        s = min(1.0, max(0.0, fraction))
        xyz = tuple(a + (b - a) * s for a, b in
                    zip(self.start_position, self.end_position, strict=True))
        b = self.start_b + (self.end_b - self.start_b) * s
        c = self.start_c + (self.end_c - self.start_c) * s
        return {"position": xyz, "b": b, "c": c,
                "direction": tool_direction(b, c),
                "time_s": self.start_time + self.estimated_duration * s}

    def json(self) -> dict:
        return asdict(self)


def parse_motion_events(lines) -> list[MotionEvent]:
    """Read G0/G1, modal distance/extrusion, G92 and G20/G21.

    Timing is linear distance/F.  For a pure rotary move F is interpreted as
    degrees/minute; this is a visualization estimate, not Marlin's planner.
    """
    xyz = [0.0, 0.0, 0.0]
    b = c = e = 0.0
    absolute_xyz = absolute_e = True
    units = 1.0
    feed = None
    time_s = 0.0
    events = []
    for line_no, raw in enumerate(lines, 1):
        code = raw.split(";", 1)[0].strip().upper()
        words = _WORD.findall(code)
        if not words:
            continue
        letter, number = words[0][0], float(words[0][1])
        args = {key: float(value) for key, value in words[1:]}
        if letter == "G" and number in (2, 3):
            raise ValueError(f"Arc G{number:g} on line {line_no} is unsupported")
        if letter == "G" and number == 20:
            units = 25.4
        elif letter == "G" and number == 21:
            units = 1.0
        elif letter == "G" and number == 90:
            absolute_xyz = True
        elif letter == "G" and number == 91:
            absolute_xyz = False
        elif letter == "M" and number == 82:
            absolute_e = True
        elif letter == "M" and number == 83:
            absolute_e = False
        elif letter == "G" and number == 92:
            for i, axis in enumerate("XYZ"):
                if axis in args:
                    xyz[i] = args[axis] * units
            if "B" in args:
                b = args["B"]
            if "C" in args:
                c = args["C"]
            if "E" in args:
                e = args["E"] * units
        elif letter == "G" and number in (0, 1):
            start_xyz, start_b, start_c, start_e = tuple(xyz), b, c, e
            for i, axis in enumerate("XYZ"):
                if axis in args:
                    value = args[axis] * units
                    xyz[i] = value if absolute_xyz else xyz[i] + value
            for axis in "BC":
                if axis in args:
                    value = args[axis] if absolute_xyz else (
                        (b if axis == "B" else c) + args[axis])
                    if axis == "B":
                        b = value
                    else:
                        c = value
            if "E" in args:
                value = args["E"] * units
                e = value if absolute_e else e + value
            if "F" in args:
                feed = args["F"] * units
            length = math.dist(start_xyz, xyz)
            rotary = max(abs(b - start_b), abs(c - start_c))
            if (length or rotary or e != start_e) and (feed is None or feed <= 0):
                raise ValueError(f"Missing positive modal feedrate at line {line_no}")
            duration = 60 * (length if length else rotary) / feed if feed else 0.0
            event = MotionEvent(
                line_no, f"G{int(number)}", start_xyz, tuple(xyz),
                start_b, start_c, b, c, e - start_e, feed or 0.0, length,
                duration, time_s, time_s + duration, e > start_e, e < start_e)
            events.append(event)
            time_s += duration
    return events
