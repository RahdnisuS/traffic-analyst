"""Social force model and the pedestrian crossing rules."""

import sys

import numpy as np
import pytest

from sim.config import PedestrianSettings, load_config
from sim.engine import Simulation
from sim.pedestrian import PedState, social_force

P = PedestrianSettings()
NO_WALLS = (np.zeros((1, 0, 2)), np.zeros((1, 0)), np.zeros((1, 0)))


def walls_for(n):
    return np.zeros((n, 0, 2)), np.zeros((n, 0)), np.zeros((n, 0))


def walk(pos, goals, v0, steps=200, dt=0.1):
    """Integrate free walkers (no walls) towards their goals; return the position history."""
    pos, vel = np.array(pos, float), np.zeros((len(pos), 2))
    hist = [pos.copy()]
    for _ in range(steps):
        to = np.array(goals, float) - pos
        dist = np.hypot(to[:, 0], to[:, 1])
        desired = to / np.maximum(dist, 1e-9)[:, None] * (np.asarray(v0) * np.minimum(1, dist))[:, None]
        vel = vel + social_force(pos, vel, desired, P, *walls_for(len(pos))) * dt
        pos = pos + vel * dt
        hist.append(pos.copy())
    return np.array(hist)


def test_lone_walker_reaches_desired_speed_within_a_few_tau():
    hist = walk([[0, 0]], [[100, 0]], [1.34], steps=30)
    speed = np.hypot(*(hist[-1, 0] - hist[-2, 0])) / 0.1
    assert speed == pytest.approx(1.34, rel=0.02)  # after 3 s = 6 tau the gap has decayed to ~0


def test_two_walkers_head_on_sidestep_instead_of_colliding(monkeypatch):
    """Why we use the elliptical (anticipating) specification: with the plain circular force
    two people walking straight at each other only brake, and their bodies overlap."""
    def closest(anticipation):
        monkeypatch.setattr(sys.modules[__name__], "P", PedestrianSettings(anticipation_s=anticipation))
        # A tiny vertical offset breaks the perfect symmetry, as in any real encounter.
        hist = walk([[0, 0], [10, 0.05]], [[10, 0], [0, 0]], [1.3, 1.3], steps=150)
        assert hist[-1, 0, 0] > 9 and hist[-1, 1, 0] < 1   # both still get where they were going
        assert np.abs(hist[:, :, 1]).max() > 0.2           # by stepping aside
        return np.hypot(*(hist[:, 0] - hist[:, 1]).T).min()

    circular, elliptical = closest(0.0), closest(PedestrianSettings().anticipation_s)
    assert circular < 2 * P.radius < elliptical  # only the anticipating model keeps bodies apart


def test_someone_ahead_matters_more_than_someone_behind():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    desired = np.array([[1.3, 0.0], [0.0, 0.0], [0.0, 0.0]])
    a_both = social_force(pos, np.zeros((3, 2)), desired, P, *walls_for(3))
    # Walker 0 faces +x: the person ahead (x=1) pushes back harder than the one behind pushes on.
    only_ahead = social_force(pos[:2], np.zeros((2, 2)), desired[:2], P, *walls_for(2))[0]
    only_behind = social_force(pos[[0, 2]], np.zeros((2, 2)), desired[[0, 2]], P, *walls_for(2))[0]
    driving = desired[0] / P.relaxation_time
    assert abs((only_ahead - driving)[0]) > abs((only_behind - driving)[0])
    assert np.allclose(a_both[0], only_ahead + only_behind - driving)


def test_wall_pushes_walker_back():
    wn, wo, wm = np.array([[[0.0, 1.0]]]), np.array([[0.0]]), np.array([[1.0]])  # walkable: y >= 0
    acc = social_force(np.array([[0.0, 0.2]]), np.zeros((1, 2)), np.zeros((1, 2)), P, wn, wo, wm)
    assert acc[0, 1] > 0 and acc[0, 0] == pytest.approx(0)


# ------------------------------------------------------------------ in the full simulation
@pytest.fixture(scope="module")
def busy():
    cfg = load_config()
    cfg.pedestrians.rate_ph = 600
    return cfg


def test_crossing_rules_hold_every_step(busy):
    """Walkers start only on WALK and only when their crosswalk is free of vehicles; drivers never
    drive onto a crosswalk while someone is on it; waiting walkers stay on the sidewalk."""
    sim = Simulation(busy, seed=12)
    J = sim.junction
    on_prev: set[int] = set()
    crossing_prev: frozenset = frozenset()
    for _ in range(6000):
        active_before = sim.active_connectors()
        sim.step()
        if sim.peds.recent_starts:
            assert sim.signals.ped_signal() == "walk"
        for p in sim.peds.peds:
            cw = J.crosswalks[p.arm]
            across = float(cw.n @ p.pos)
            if p.state in (PedState.APPROACH, PedState.WAIT):
                assert p.side * (across - (cw.hi if p.side > 0 else cw.lo)) >= -1e-6, "waiting walker on the road"
            if p.state is PedState.CROSS and p.cross_start == pytest.approx(sim.time - sim.dt):
                assert not (J.crosswalk_connectors[p.arm] & active_before), "stepped out in front of a vehicle"
        on_now = {}
        for a in J.crosswalks:
            on_now[a] = {v.id for c in J.crosswalk_connectors[a] for v in sim.lanes[c]}
        for a in crossing_prev:
            assert not (on_now[a] - on_prev), f"a vehicle drove onto crosswalk {a} while someone was on it"
        on_prev = set().union(*on_now.values())
        crossing_prev = sim.peds.crossing_arms
    assert sim.counters.emergency_stops == 0


def test_everyone_gets_across_and_waits_at_most_about_a_cycle(busy):
    sim = Simulation(busy, seed=3)
    waits = []

    class Obs:
        def after_step(self, s):
            waits.extend(w for _, w in s.peds.recent_starts)

    sim.observers.append(Obs())
    sim.run(900)
    ps = busy.pedestrians
    cycle = 4 * (25 + 3 + 2) + ps.walk_s + sim.ped_clear_s + 2
    assert sim.peds.completed > 0.8 * sim.peds.spawned
    assert max(waits) < cycle + 5
    assert sim.signals.ped_phases_served >= 900 / cycle - 1  # roughly one pedestrian phase per cycle


def test_no_pedestrians_means_no_pedestrian_phase():
    cfg = load_config()
    cfg.pedestrians.rate_ph = 0
    sim = Simulation(cfg, seed=1)
    sim.run(600)
    assert sim.signals.ped_phases_served == 0 and sim.peds.spawned == 0
    assert all(e.stage != "walk" for e in sim.signals.events)


def test_pedestrian_rate_setting(busy):
    sim = Simulation(busy, seed=2)
    with pytest.raises(ValueError):
        sim.apply_settings(ped_rate_ph=-1)
    assert sim.apply_settings(ped_rate_ph=0)["ped_rate_ph"] == 0
    sim.run(300)
    assert sim.peds.spawned == 0
