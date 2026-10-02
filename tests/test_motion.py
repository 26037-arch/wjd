"""T18: 가속도를 넣은 출력 시간 모델 (플랫폼 비교의 '속도' 축).

왜 필요한가: `analyze_sync.py` 가 주던 시간은 **가속도를 무시한 하한**이었다.
그 하한으로 기계를 비교하면 '회전축이 병목' 이라는 결론의 크기를 못 잰다. 그래서
사다리꼴 운동 계획을 넣었는데, 운동 계획은 **조용히 틀리기 아주 쉬운 코드**다
(패스 방향, 부호, 삼각/사다리꼴 분기). 그래서 손으로 푼 답과 맞춰 고정한다.

실제로 이 테스트가 필요했던 이유: 처음 돌렸을 때 lamp 가 89분이 나와 하한(6.4분)의
14배였다. 모델 버그로 의심했으나 **되감기 75분**이 섞여 있던 것이었다. 출력 시간과
되감기 시간을 섞으면 기계 비교가 통째로 망가진다 — ⑤가 그걸 고정한다.

고정하는 성질:
  ① 사다리꼴 구간 시간이 해석해와 일치한다
  ② 짧은 구간은 정속에 못 가고 **삼각** 프로파일이 된다 (t = 2√(d/a))
  ③ 축 속도 한계가 명령 피드를 실제로 깎는다 (Z, 회전축)
  ④ 한계를 올리면 시간이 **단조감소**하고, 하한 Σ L/F 아래로는 절대 안 내려간다
  ⑤ `; V_REWIND BEGIN/END` 구간이 출력 시간에서 **분리**된다
  ⑥ 되감기 시간 ≈ 되돌린 총 각도 / 되감기 속도 (횟수가 아니라 속도가 지배)

    python3 -m pytest tests/test_motion.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from conical.gcode import Move
from conical.motion import AxisLimits, _displacements, limits_from_spec, plan_motion

# 저크·다른 축을 크게 열어 '한 축 사다리꼴' 만 남긴 한계
FREE = AxisLimits(max_vel={"X": 1e9, "Y": 1e9, "Z": 1e9, "V": 1e9, "E": 1e9},
                  max_accel={"X": 1e9, "Y": 1e9, "Z": 1e9, "V": 1e9, "E": 1e9},
                  jerk={"X": 1e9, "Y": 1e9, "Z": 1e9, "V": 1e9, "E": 1e9},
                  name="free")


def _x_move(dist, feed_mm_s, z=0.2):
    """원점에서 X 로 `dist` 가는 이동 하나 (정지 → 정지)."""
    return [("move", Move(g=0, x=0.0, y=0.0, z=z, f=feed_mm_s * 60.0)),
            ("move", Move(g=1, x=dist, y=0.0, z=z, f=feed_mm_s * 60.0))]


def _limits(**kw):
    base = dict(max_vel={}, max_accel={}, jerk={})
    base.update(kw)
    return AxisLimits(**base)


def test_trapezoid_time_matches_analytic():
    """정지→정지 한 구간: t = 2·v/a + (d − v²/a)/v.

    v = 100mm/s, a = 1000mm/s², d = 50mm 이면 가속거리 5mm ×2 = 10mm 이므로
    사다리꼴이 성립하고 t = 0.2 + 0.4 = 0.6s.
    """
    v, a, d = 100.0, 1000.0, 50.0
    lim = AxisLimits(max_vel={"X": 1e9}, max_accel={"X": a}, jerk={"X": 1e9},
                     feed_axes=("X", "Y", "Z", "V"))
    got = plan_motion(_x_move(d, v), lim)["seconds"]
    want = 2.0 * v / a + (d - v * v / a) / v
    assert got == pytest.approx(want, rel=1e-9), f"{got} vs 해석해 {want}"
    assert want == pytest.approx(0.6, abs=1e-12)


def test_short_segment_becomes_triangle():
    """가속거리보다 짧으면 정속에 못 간다 — t = 2√(d/a), 명령 피드와 무관.

    출력 G-code 는 구간 중앙값이 6mm 남짓이라 **대부분이 이 경우**다. 그래서
    '명령 피드대로 나간다' 는 가정이 실제 시간과 2배 어긋난다.
    """
    v, a, d = 100.0, 1000.0, 1.0      # v²/a = 10mm ≫ 1mm
    lim = AxisLimits(max_vel={"X": 1e9}, max_accel={"X": a}, jerk={"X": 1e9})
    got = plan_motion(_x_move(d, v), lim)["seconds"]
    want = 2.0 * math.sqrt(d / a)
    assert got == pytest.approx(want, rel=1e-9), f"{got} vs 삼각 {want}"
    # 피드를 10배로 올려도 안 빨라진다 (가속도가 지배)
    got2 = plan_motion(_x_move(d, v * 10), lim)["seconds"]
    assert got2 == pytest.approx(want, rel=1e-9), "가속 지배 구간인데 피드가 먹었다"


def test_axis_velocity_cap_binds():
    """축 최대 속도가 명령 피드를 깎고, 그 축이 병목으로 기록된다."""
    items = [("move", Move(g=0, x=0.0, y=0.0, z=0.0, f=6000.0)),
             ("move", Move(g=1, x=0.0, y=0.0, z=100.0, f=6000.0))]  # 100mm/s 명령
    lim = AxisLimits(max_vel={"Z": 10.0}, max_accel={"Z": 1e9}, jerk={"Z": 1e9})
    res = plan_motion(items, lim)
    assert res["seconds"] == pytest.approx(10.0, rel=1e-6), \
        f"Z 10mm/s 로 100mm 면 10s 인데 {res['seconds']}"
    assert res["binding"] == ["Z"], f"병목 축이 Z 가 아니다: {res['binding']}"


def test_rotary_cap_binds_on_v_only_move():
    """회전만 하는 이동은 회전축 한계가 정한다 (RRF 는 V 를 노름에 넣는다)."""
    items = [("move", Move(g=0, x=0.0, y=0.0, z=1.0, f=60000.0, extra="V0")),
             ("move", Move(g=0, x=0.0, y=0.0, z=1.0, f=60000.0, extra="V360"))]
    lim = AxisLimits(max_vel={"V": 90.0}, max_accel={"V": 1e9}, jerk={"V": 1e9})
    res = plan_motion(items, lim)
    assert res["seconds"] == pytest.approx(4.0, rel=1e-6), \
        f"90deg/s 로 360° 면 4s 인데 {res['seconds']}"
    assert res["binding"] == ["V"]


def test_raising_limits_never_slows_down():
    """한계를 올렸는데 느려지면 패스(역/정방향)가 틀린 것이다."""
    items = []
    z = 0.2
    for i in range(40):                      # 지그재그 — 접합부에서 방향이 꺾인다
        items.append(("move", Move(g=1, x=(i % 2) * 5.0, y=i * 0.4, z=z,
                                   e=i * 0.1, f=3600.0)))
    prev = math.inf
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0):
        t = plan_motion(items, limits_from_spec(
            "s", xy_vel=60.0 * mult, xy_acc=1000.0 * mult, z_vel=12.0,
            z_acc=300.0, rot_vel=360.0, rot_acc=1800.0))["seconds"]
        assert t <= prev + 1e-9, f"한계 {mult}× 에서 시간이 늘었다: {t} > {prev}"
        prev = t


def test_never_faster_than_feed_lower_bound():
    """Σ L/F 는 가속도를 무시한 하한이다 — 모델이 그 아래로 가면 버그다."""
    items = []
    for i in range(60):
        items.append(("move", Move(g=1, x=(i % 3) * 4.0, y=i * 0.5, z=0.2,
                                   e=i * 0.1, f=1800.0)))
    lb = sum(math.sqrt(sum(d * d for a, d in disp.items()
                           if a in ("X", "Y", "Z", "V"))) / feed
             for disp, feed, _p, _r in _displacements(items))
    t = plan_motion(items, limits_from_spec(
        "s", xy_vel=150.0, xy_acc=3000.0, z_vel=12.0, z_acc=300.0,
        rot_vel=360.0, rot_acc=1800.0))["seconds"]
    assert t >= lb - 1e-9, f"하한 {lb:.3f}s 보다 빠르다: {t:.3f}s"


def test_rewind_region_is_split_out():
    """되감기는 **출력이 아니다** — 합쳐 보고하면 기계 비교가 망가진다.

    lamp 에서 실제로 출력 13.6분 + 되감기 75.4분이 나왔고, 둘을 합친 89분을
    출력 시간이라 믿으면 '회전축이 느려서 느리다' 는 엉뚱한 결론이 나온다.
    """
    # V 는 첫 이동부터 적어 둔다 — 모달 좌표라 처음 나타나는 축은 '이전 위치를
    # 모르므로' 변위가 잡히지 않는다 (`to_open5x` 는 매 이동에 V 를 적는다).
    items = [("move", Move(g=1, x=0.0, y=0.0, z=1.0, e=0.0, f=1800.0,
                           extra="V0")),
             ("move", Move(g=1, x=30.0, y=0.0, z=1.0, e=1.0, f=1800.0,
                           extra="V0")),
             ("comment", "; V_REWIND BEGIN -1 turn(s): 배선 되감기"),
             ("move", Move(g=0, x=30.0, y=0.0, z=3.0, f=3000.0)),
             ("move", Move(g=0, x=30.0, y=0.0, z=3.0, f=1200.0, extra="V360")),
             ("move", Move(g=0, x=30.0, y=0.0, z=1.0, f=3000.0)),
             ("comment", "; V_REWIND END"),
             ("move", Move(g=1, x=60.0, y=0.0, z=1.0, e=2.0, f=1800.0))]
    lim = limits_from_spec("s", xy_vel=150.0, xy_acc=3000.0, z_vel=12.0,
                           z_acc=300.0, rot_vel=360.0, rot_acc=1800.0)
    res = plan_motion(items, lim)
    t_rw = res["region_time"]["V_REWIND"]
    assert t_rw > 0.0
    assert res["print_seconds"] + t_rw == pytest.approx(res["seconds"], rel=1e-9)
    # 되감기 360° 를 1200mm/min(=20deg/s) 로 → 18s 가 그 안에 들어 있어야 한다
    assert t_rw > 17.0, f"되감기 시간이 너무 작다: {t_rw:.2f}s"
    # 출력 구간 60mm 를 30mm/s 로 → 2s 남짓
    assert 2.0 <= res["print_seconds"] < 3.0, \
        f"출력 시간이 이상하다: {res['print_seconds']:.2f}s"


def test_rewind_time_is_governed_by_speed_not_count():
    """되돌릴 각도는 감긴 만큼 정해져 있다 → 시간 ≒ 각도/속도.

    그래서 `--v-rewind` 문턱(횟수)을 만져도 시간이 거의 안 변하고, 줄이려면
    되감기 **속도**를 올려야 한다. REP5X 처럼 연속 회전이면 이 항 자체가 0 이다.
    """
    def job(feed_mm_min):
        items = [("move", Move(g=1, x=0.0, y=0.0, z=1.0, e=0.0, f=1800.0,
                               extra="V0"))]
        for k in range(4):                      # 4번 나눠 되감아도 총 1440°
            items += [("comment", "; V_REWIND BEGIN -1 turn(s)"),
                      ("move", Move(g=0, x=0.0, y=0.0, z=1.0,
                                    f=feed_mm_min,
                                    extra=f"V{(k + 1) * 360.0}")),
                      ("comment", "; V_REWIND END")]
        lim = limits_from_spec("s", xy_vel=150.0, xy_acc=3000.0, z_vel=12.0,
                               z_acc=300.0, rot_vel=1e9, rot_acc=1e9)
        return plan_motion(items, lim)["region_time"]["V_REWIND"]

    slow, fast = job(1200.0), job(12000.0)
    assert slow == pytest.approx(1440.0 / 20.0, rel=0.01), \
        f"1440° ÷ 20deg/s = 72s 인데 {slow:.1f}s"
    assert fast == pytest.approx(slow / 10.0, rel=0.02), \
        f"속도 10배인데 시간이 1/10 이 아니다: {slow:.1f} → {fast:.1f}"


def test_empty_input_is_safe():
    res = plan_motion([], FREE)
    assert res["seconds"] == 0.0 and res["n_segments"] == 0
    assert res["print_seconds"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
