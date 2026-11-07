import pytest

from sim.junction import Phase
from sim.signals import FixedTimePolicy, SignalController, Stage

PHASES = [Phase(n, {n: frozenset({"left", "straight", "right"})}) for n in "NESW"]


def run(ctrl, policy, seconds, dt=0.1):
    for _ in range(int(round(seconds / dt))):
        policy.update(ctrl)
        ctrl.step(dt)


def test_fixed_time_cycle_order_and_timing():
    ctrl = SignalController(PHASES, amber_s=3, all_red_s=2, min_green_s=7)
    run(ctrl, FixedTimePolicy([25] * 4), 130)
    greens = [(round(e.time, 1), e.phase) for e in ctrl.events if e.stage == "green"]
    # cycle = 4 * (25 + 3 + 2) = 120 s
    assert greens == [(0.0, "N"), (30.0, "E"), (60.0, "S"), (90.0, "W"), (120.0, "N")]


def test_min_green_blocks_early_switch():
    ctrl = SignalController(PHASES, amber_s=3, all_red_s=2, min_green_s=7)
    run(ctrl, FixedTimePolicy([25] * 4), 5)
    assert ctrl.request_switch() is False
    run(ctrl, FixedTimePolicy([25] * 4), 2.1)
    assert ctrl.request_switch() is True
    assert ctrl.stage is Stage.AMBER


def test_never_two_approaches_green_and_all_red_is_all_red():
    ctrl = SignalController(PHASES, amber_s=3, all_red_s=2, min_green_s=7)
    policy = FixedTimePolicy([10] * 4)
    for _ in range(1500):
        policy.update(ctrl)
        ctrl.step(0.1)
        states = ctrl.arm_states("NESW")
        assert sum(s != "R" for s in states.values()) <= 1
        if ctrl.stage is Stage.ALL_RED:
            assert set(states.values()) == {"R"}


def test_switch_ignored_during_amber():
    ctrl = SignalController(PHASES, amber_s=3, all_red_s=2, min_green_s=0)
    assert ctrl.request_switch()
    assert ctrl.request_switch() is False


# ------------------------------------------------------------------ exclusive pedestrian phase
def ped_controller(min_greens=1):
    return SignalController(PHASES, amber_s=3, all_red_s=2, min_green_s=7,
                            walk_s=7, ped_clear_s=13, min_greens_between_peds=min_greens)


def test_pedestrian_phase_is_inserted_only_when_called_and_holds_all_traffic():
    c, pol = ped_controller(), FixedTimePolicy([20.0] * 4)
    run(c, pol, 60)
    assert all(e.stage not in ("walk", "ped_clear") for e in c.events)  # nobody pressed the button
    c.call_pedestrian()
    seen = []
    for _ in range(600):
        pol.update(c)
        c.step(0.1)
        if c.stage in (Stage.WALK, Stage.PED_CLEAR):
            assert all(c.movement_state(a, m) == "R" for a in "NESW" for m in ("left", "straight", "right"))
            assert c.phase_name == "PED"
        if not seen or seen[-1] != c.stage:
            seen.append(c.stage)
    i = seen.index(Stage.WALK)
    assert seen[i - 1] is Stage.ALL_RED and seen[i - 2] is Stage.AMBER   # vehicles cleared first
    assert seen[i + 1:i + 4] == [Stage.PED_CLEAR, Stage.ALL_RED, Stage.GREEN]
    walk = next(e for e in c.events if e.stage == "walk")
    clear = next(e for e in c.events if e.stage == "ped_clear")
    assert clear.time - walk.time == pytest.approx(7)
    assert c.ped_phases_served == 1 and not c.ped_call


def test_pedestrian_phase_at_most_once_per_cycle():
    c, pol = ped_controller(min_greens=4), FixedTimePolicy([10.0] * 4)
    greens_between = 0
    counts = []
    for _ in range(6000):
        c.call_pedestrian()  # someone is always waiting
        prev = c.stage
        pol.update(c)
        c.step(0.1)
        if c.stage is Stage.GREEN and prev is not Stage.GREEN:
            greens_between += 1
        if c.stage is Stage.WALK and prev is not Stage.WALK:
            counts.append(greens_between)
            greens_between = 0
    assert len(counts) >= 3 and all(n >= 4 for n in counts[1:])
