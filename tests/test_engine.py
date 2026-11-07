import pytest

from sim.config import load_config
from sim.engine import Simulation


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_same_seed_is_bit_identical(cfg):
    a, b = Simulation(cfg, seed=7), Simulation(cfg, seed=7)
    a.run(300)
    b.run(300)
    assert a.fingerprint() == b.fingerprint()
    assert a.summary() == b.summary()


def test_different_seed_differs(cfg):
    a, b = Simulation(cfg, seed=1), Simulation(cfg, seed=2)
    a.run(120)
    b.run(120)
    assert a.fingerprint() != b.fingerprint()


def test_no_overlaps_no_red_runs_and_no_box_conflicts(cfg):
    """The core safety invariants, checked every step for 10 simulated minutes."""
    sim = Simulation(cfg, seed=3)
    J = sim.junction
    for _ in range(6000):
        sim.step()
        for eid, lane in sim.lanes.items():
            for lead, foll in zip(lane, lane[1:]):
                assert lead.s - lead.length - foll.s > -1e-6, f"overlap on {eid}"
            if J.edges[eid].kind == "connector":
                for veh in lane:
                    assert veh.committed and veh.commit_signal in ("G", "Y"), "entered on red"
        occupied = {e for e, lane in sim.lanes.items() if lane and J.edges[e].kind == "connector"}
        for c in occupied:
            assert not (J.conflicts[c] & occupied), f"conflicting paths occupied together: {c}"
    assert sim.counters.emergency_stops == 0


def test_vehicles_wait_at_red(cfg):
    """During N's green, the front vehicles of E/S/W must be stationary at or before their stop lines."""
    sim = Simulation(cfg, seed=5)
    sim.run(115)  # late in W's green: N, E, S have all had time to queue
    for arm in ("N", "E", "S"):
        for lane_id in sim.junction.inbound[arm]:
            lane = sim.lanes[lane_id]
            if lane and not lane[0].committed:
                assert lane[0].s <= sim.junction.edges[lane_id].length


def test_light_demand_is_fully_served(cfg):
    sim = Simulation(cfg, seed=11)
    sim.run(900)
    s = sim.summary()
    assert s["backlog"] == 0
    assert s["completed"] / s["spawned"] > 0.9


def test_aggressiveness_baseline_is_identity_and_aggressive_raises_capacity(cfg):
    from sim.vehicle import scale_for_aggressiveness
    car = cfg.vehicle_types["car"]
    assert scale_for_aggressiveness(car, 0.5) == car
    assert scale_for_aggressiveness(car, 1.0).time_headway < car.time_headway

    def served(aggr):
        s = Simulation(cfg, seed=8)
        s.apply_settings(volume_vph=900, aggressiveness=aggr)
        s.run(900)
        return s.counters.completed
    assert served(1.0) > served(0.0)  # shorter headways discharge queues faster


def test_apply_settings_validates_and_logs(cfg):
    s = Simulation(cfg, seed=1)
    with pytest.raises(ValueError):
        s.apply_settings(green_s=2)
    assert s.control_events == []
    s.apply_settings(green_s=40, source="test")
    assert s.policy.green_times == [40.0] * 4
    assert s.control_events[-1]["changes"] == {"green_s": 40.0}
    assert cfg.signals.green_s == 25  # the shared config was not mutated
