import numpy as np
import pytest

from sim.config import PROJECT_ROOT
from sim.junction import Junction, JunctionSpec

J = Junction.from_file(PROJECT_ROOT / "junctions/four_way_2x2.json")


def test_lane_counts():
    assert sum(1 for _ in J.inbound_edges()) == 8
    assert all(len(v) == 2 for v in J.outbound.values())


def test_left_hand_traffic_lane_sides():
    # Southbound traffic (from N) drives on the east half (x > 0) in left-hand traffic.
    assert all(J.edges[e].path.points[:, 0].min() > 0 for e in J.inbound["N"])
    # Eastbound traffic (from W) drives on the north half.
    assert all(J.edges[e].path.points[:, 1].min() > 0 for e in J.inbound["W"])


@pytest.mark.parametrize("arm,movement,exit_arm", [
    ("N", "left", "E"), ("N", "right", "W"), ("N", "straight", "S"),
    ("E", "left", "S"), ("W", "left", "N"), ("S", "right", "E"),
])
def test_turn_destinations(arm, movement, exit_arm):
    lane = J.lanes_for_movement[(arm, movement)][0]
    out = J.routes[(lane, movement)][2]
    assert J.edges[out].arm == exit_arm


def test_turns_are_slower_than_straight_and_left_is_tightest():
    lim = {m: J.edges[f"cx_N_{0 if m == 'left' else 1}_{m}"].speed_limit
           for m in ("left", "straight", "right")}
    assert lim["left"] < lim["right"] < lim["straight"]


def test_conflicts_are_symmetric_and_cross_traffic_conflicts():
    for a, cs in J.conflicts.items():
        for b in cs:
            assert a in J.conflicts[b]
    # N straight crosses W straight
    assert "cx_W_0_straight" in J.conflicts["cx_N_0_straight"]
    # same-approach movements never conflict, so a split phase can run them together
    n_cx = [c for c in J.conflicts if c.startswith("cx_N_")]
    for c in n_cx:
        assert not any(o.startswith("cx_N_") for o in J.conflicts[c])


def test_connectors_join_lane_ends_exactly():
    for cx in J.connectors():
        assert np.allclose(cx.path.points[0], J.edges[cx.source_id].path.points[-1])
        assert np.allclose(cx.path.points[-1], J.edges[cx.target_id].path.points[0])


def test_invalid_layout_rejected():
    spec = JunctionSpec.model_validate({
        "name": "bad", "arms": {"N": {"in_lanes": [["left"]], "out_lanes": 1}},
        "phases": [{"name": "N", "green": {"N": ["left"]}}]})
    with pytest.raises(ValueError, match="no exit lanes"):
        Junction(spec)
