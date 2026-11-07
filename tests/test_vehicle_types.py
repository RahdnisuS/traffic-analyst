"""Mixed traffic: vehicle types, the bus kerb-lane preference, and bike filtering."""

from collections import Counter

import pytest

from sim.config import load_config
from sim.engine import Simulation


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c.pedestrians.rate_ph = 0  # isolate vehicle behaviour
    return c


def run_collect(cfg, seed, seconds, setup=None):
    sim = Simulation(cfg, seed=seed)
    if setup:
        setup(sim)
    trips = []

    class Obs:
        def after_step(self, s):
            trips.extend(s.recent_trips)

    sim.observers.append(Obs())
    sim.run(seconds)
    return sim, trips


def test_mix_proportions_follow_config(cfg):
    sim, trips = run_collect(cfg, 5, 1800)
    counts = Counter(v.vtype for v in sim.vehicles.values()) + Counter(t.vtype for t in trips)
    total = sum(counts.values())
    mix = cfg.demand.vehicle_mix
    for t, w in mix.items():
        assert counts[t] / total == pytest.approx(w / sum(mix.values()), abs=0.05)


def test_types_differ_in_behaviour(cfg):
    vt = cfg.vehicle_types
    assert vt["bike"].max_accel > vt["car"].max_accel > vt["bus"].max_accel
    assert vt["bus"].length > vt["car"].length > vt["auto"].length > vt["bike"].length
    assert vt["bike"].filters and not vt["car"].filters
    assert vt["bus"].kerb_lane


def test_buses_keep_to_the_kerb_lane(cfg):
    lanes = Counter()

    def setup(sim):
        orig = sim._try_insert

        def spy(arm, pv):
            before = set(sim.vehicles)
            p = sim._params_cache[pv.vtype]
            kerb_room = sim._entry_space(f"in_{arm}_0") >= p.length + p.min_gap + 0.5
            ok = orig(arm, pv)
            # Right turns are only allowed from the median lane, so only left/straight buses choose.
            if ok and pv.vtype == "bus" and pv.movement != "right":
                veh = sim.vehicles[(set(sim.vehicles) - before).pop()]
                in_kerb = veh.edge_id.endswith("_0")
                assert in_kerb or not kerb_room, "bus skipped a kerb lane that had room"
                lanes[in_kerb] += 1
            return ok
        sim._try_insert = spy

    run_collect(cfg, 4, 1200, setup)
    assert lanes[True] > 2 * lanes[False]  # mostly kerbside; the rest found the kerb lane full


def test_bikes_filter_to_the_front_of_a_red_queue(cfg):
    sim = Simulation(cfg, seed=7)
    real = sim.signals.movement_state
    sim.signals.movement_state = lambda arm, m: "R" if arm == "N" else real(arm, m)
    sim.run(150)
    flane = sim.lanes["flt_N_0"]
    assert flane, "no bike filtered past the stationary queue"
    assert all(v.vtype == "bike" and v.movement == "straight" for v in flane)
    stop = sim.junction.edges["flt_N_0"].length
    cars_queue_rear = min(v.s - v.length for lid in sim.junction.inbound["N"] for v in sim.lanes[lid])
    assert flane[0].s > stop - 10           # the leading bike waits right at the stop line
    assert flane[0].s > cars_queue_rear     # ...ahead of vehicles that were queued in the lanes


def test_filtering_saves_bikes_time_without_costing_cars(cfg):
    def mean(xs):
        return sum(xs) / len(xs)

    with_f, without = [], []
    for seed in (1, 2):
        _, t1 = run_collect(cfg, seed, 900)
        _, t2 = run_collect(cfg, seed, 900, lambda s: setattr(s.junction, "filters_beside", {}))
        with_f += t1
        without += t2
    bikes_f = [t.delay for t in with_f if t.vtype == "bike" and t.filtered]
    bikes_lane = [t.delay for t in without if t.vtype == "bike"]
    assert bikes_f and mean(bikes_f) < mean(bikes_lane)
    cars_f = mean([t.delay for t in with_f if t.vtype == "car"])
    cars_lane = mean([t.delay for t in without if t.vtype == "car"])
    assert cars_f < cars_lane * 1.1  # cars are not made noticeably worse off


def test_vehicle_mix_setting_validates_and_applies(cfg):
    sim = Simulation(cfg, seed=1)
    with pytest.raises(ValueError):
        sim.apply_settings(vehicle_mix={"tractor": 1})
    with pytest.raises(ValueError):
        sim.apply_settings(vehicle_mix={"car": 0, "bike": 0})
    mix = sim.apply_settings(vehicle_mix={"bike": 3, "car": 1})["vehicle_mix"]
    assert mix == {"auto": 0.0, "bike": 0.75, "bus": 0.0, "car": 0.25}
    sim.run(600)
    assert {v.vtype for v in sim.vehicles.values()} <= {"bike", "car"}
