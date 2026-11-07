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
