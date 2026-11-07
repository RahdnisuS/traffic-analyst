"""Vehicle BODIES never overlap: the 2D invariant behind "no collisions in the junction".

The older safety test (test_engine) checks spacing along each lane. That missed collisions
between vehicles on different paths through the box: a 12 m bus's body sweeping across the
neighbouring path, a car following a right-turning bus into its tail, a turning bus's tail
swinging into a bike in the filter gap. These tests check every pair of vehicle rectangles,
built from exactly the poses the server streams to the renderers.
"""

import math

import numpy as np
import pytest

from sim.config import load_config
from sim.engine import Simulation
from sim.geometry import segment_distances

TOLERANCE = 0.15  # m of overlap ignored (sub-step discretisation and box-vs-capsule rounding)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def bodies(sim):
    """One rectangle (4 corners) per vehicle, from the streamed pose: front bumper + heading."""
    out = []
    for eid in sim.junction.edge_order:
        lane = sim.lanes[eid]
        if not lane:
            continue
        xs, ys, hs = sim.lane_poses(eid)
        for veh, x, y, h in zip(lane, xs, ys, hs):
            u = np.array([math.cos(h), math.sin(h)])
            n = np.array([-u[1], u[0]]) * veh.params.width / 2
            f = np.array([x, y])
            r = f - u * veh.length
            out.append((veh, np.array([f + n, f - n, r - n, r + n])))
    return out


def overlap_depth(a, b):
    """Separating-axis test for two rectangles: overlap depth along the best axis (<= 0: apart)."""
    best = math.inf
    for poly in (a, b):
        for i in range(2):
            e = poly[i + 1] - poly[i]
            ax = np.array([-e[1], e[0]]) / np.hypot(*e)
            pa, pb = a @ ax, b @ ax
            best = min(best, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
    return best


def collisions(sim):
    bs = bodies(sim)
    hits = []
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            (va, pa), (vb, pb) = bs[i], bs[j]
            if np.hypot(*(pa[0] - pb[0])) < 30 and overlap_depth(pa, pb) > TOLERANCE:
                hits.append((va.id, va.vtype, va.edge_id, vb.id, vb.vtype, vb.edge_id))
    return hits


@pytest.mark.parametrize("seed,vph,bus", [(7, 500, 20), (9, 450, 30), (11, 700, 10)])
def test_no_vehicle_bodies_overlap(cfg, seed, vph, bus):
    """Busy junction, many buses (the long vehicles that exposed the bugs), 8 simulated minutes."""
    sim = Simulation(cfg, seed=seed)
    sim.apply_settings(volume_vph=vph, vehicle_mix={"car": 35, "bike": 50 - bus, "auto": 15, "bus": bus})
    for step in range(4800):
        sim.step()
        if step % 10 == 0:  # once per simulated second
            hits = collisions(sim)
            assert not hits, f"t={sim.time:.1f}: overlapping bodies {hits[:3]}"
    assert sim.counters.emergency_stops == 0


def test_long_vehicle_keeps_the_box_until_its_rear_leaves(cfg):
    """A connector stays 'in use' while any part of a vehicle is on it, not just its nose."""
    sim = Simulation(cfg, seed=7)
    sim.apply_settings(volume_vph=500, vehicle_mix={"bus": 1})
    seen = False
    for _ in range(3000):
        sim.step()
        for veh in sim.vehicles.values():
            prev = veh.route[veh.route_idx - 1] if veh.route_idx else None
            if prev and sim.junction.edges[prev].kind == "connector" and veh.s < veh.length:
                assert prev in sim.active_connectors()
                seen = True
    assert seen


def test_bodies_are_drawn_along_the_chord(cfg):
    """On a turn, front AND rear of the drawn body lie on the path (rear = length behind)."""
    sim = Simulation(cfg, seed=7)
    sim.apply_settings(volume_vph=500, vehicle_mix={"bus": 1})
    checked = 0
    for _ in range(3000):
        sim.step()
        for eid in sim.junction.edge_order:
            if sim.junction.edges[eid].kind != "connector" or not sim.lanes[eid]:
                continue
            xs, ys, hs = sim.lane_poses(eid)
            for veh, x, y, h in zip(sim.lanes[eid], xs, ys, hs):
                rx, ry = sim.rear_point(veh)
                drawn_rear = (x - math.cos(h) * math.hypot(x - rx, y - ry), y - math.sin(h) * math.hypot(x - rx, y - ry))
                assert drawn_rear == pytest.approx((rx, ry), abs=1e-6)
                checked += 1
    assert checked > 50


def test_segment_distances_match_brute_force():
    rng = np.random.default_rng(0)
    p0, p1, q0, q1 = (rng.uniform(-5, 5, (6, 2)) for _ in range(4))
    q0[0], q1[0] = p0[0] + [0.1, 0], p1[0] + [0.1, 0]  # include a parallel pair
    got = segment_distances(p0, p1, q0, q1)
    t = np.linspace(0, 1, 801)
    for i in range(6):
        a = p0[i] + (p1[i] - p0[i]) * t[:, None]
        for j in range(6):
            b = q0[j] + (q1[j] - q0[j]) * t[:, None]
            brute = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min()
            assert got[i, j] == pytest.approx(brute, abs=0.02)


def test_buses_get_extra_conflicts_only_where_their_body_needs_them(cfg):
    sim = Simulation(cfg, seed=1)
    J = sim.junction
    bc = J.body_conflicts
    assert set(bc) >= {("car", "car"), ("bus", "bike"), ("bike", "bus")}
    extra = lambda a, b: sum(len(v - J.conflicts[c]) for c, v in bc[(a, b)].items())  # noqa: E731
    assert extra("bike", "bus") > 0          # a bike's filter path vs a bus's swept body
    assert extra("car", "car") == 0          # cars: the centreline rule already suffices
    for (a, b), table in bc.items():         # symmetric
        for cx, others in table.items():
            for o in others:
                assert cx in bc[(b, a)][o]
