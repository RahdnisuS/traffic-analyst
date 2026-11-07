"""Junction layout: built from a JSON spec into a graph of lane "edges".

Coordinate system: junction centre at (0, 0), x points east, y points north, metres.
LEFT-HAND TRAFFIC (India): a vehicle drives on the left half of the road
relative to its own heading, so a left turn is the short kerb-side turn and a
right turn crosses the path of oncoming traffic.

The road network is a small directed graph with three kinds of edge:
  inbound   - approach lane, from far upstream to the stop line
  connector - path through the junction box for one (lane, movement) pair
  outbound  - exit lane, from the box to a sink where vehicles despawn
A vehicle's route is always [inbound, connector, outbound], fixed at spawn.
Lane index 0 is the kerb-side (leftmost) lane; higher indices are nearer the median.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path as FsPath

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import Movement
from .geometry import Path, line_intersection, min_distance, quadratic_bezier

ARM_VECTORS: dict[str, np.ndarray] = {
    "N": np.array([0.0, 1.0]),
    "E": np.array([1.0, 0.0]),
    "S": np.array([0.0, -1.0]),
    "W": np.array([-1.0, 0.0]),
}
MOVEMENTS: tuple[str, ...] = ("left", "straight", "right")


# ---------------------------------------------------------------- JSON schema
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmSpec(_Strict):
    in_lanes: list[list[Movement]] = Field(default_factory=list)  # movements allowed per lane
    out_lanes: int = Field(0, ge=0)


class PhaseSpec(_Strict):
    name: str
    green: dict[str, list[Movement]]


class JunctionSpec(_Strict):
    name: str
    description: str = ""
    lane_width: float = Field(3.5, gt=0)
    approach_length: float = Field(150.0, gt=20)
    exit_length: float = Field(80.0, gt=10)
    crosswalk_width: float = Field(3.0, ge=0)
    corner_margin: float = Field(1.5, ge=0)
    speed_limit: float = Field(13.9, gt=0)
    arms: dict[str, ArmSpec]
    phases: list[PhaseSpec] = Field(min_length=1)


# ---------------------------------------------------------------- runtime types
@dataclass(eq=False)
class Edge:
    id: str
    kind: str                     # "inbound" | "connector" | "outbound"
    path: Path
    speed_limit: float
    arm: str                      # inbound/outbound: its arm; connector: the source arm
    lane_index: int = -1
    movement: str | None = None   # connectors only
    source_id: str | None = None  # connectors only: inbound edge id
    target_id: str | None = None  # connectors only: outbound edge id

    @property
    def length(self) -> float:
        return self.path.length


@dataclass
class Phase:
    name: str
    green: dict[str, frozenset[str]] = field(default_factory=dict)

    def allows(self, arm: str, movement: str) -> bool:
        return movement in self.green.get(arm, ())


def _left_of(h: np.ndarray) -> np.ndarray:
    """Unit vector pointing to the driver's left (90 degrees counter-clockwise)."""
    return np.array([-h[1], h[0]])


def _turn(h: np.ndarray, movement: str) -> np.ndarray:
    if movement == "left":
        return _left_of(h)
    if movement == "right":
        return np.array([h[1], -h[0]])
    return h.copy()


def _arm_for_direction(v: np.ndarray) -> str:
    for name, u in ARM_VECTORS.items():
        if np.allclose(u, v):
            return name
    raise ValueError(f"no arm points along {v}")


class Junction:
    """Immutable road geometry plus derived lookup tables used by the engine."""

    def __init__(self, spec: JunctionSpec, turn_lateral_accel: float = 2.5,
                 conflict_distance: float = 2.5) -> None:
        unknown = set(spec.arms) - set(ARM_VECTORS)
        if unknown:
            raise ValueError(f"unknown arm names {sorted(unknown)}; use N/E/S/W")
        self.spec = spec
        self.name = spec.name
        self.lane_width = spec.lane_width
        w = spec.lane_width
        max_lanes = max(max(len(a.in_lanes), a.out_lanes) for a in spec.arms.values())
        # The box must contain the widest crossing road, plus a margin so turns fit.
        self.box_half = max_lanes * w + spec.corner_margin
        # The crosswalk sits between the box and the stop line (pedestrians use it in Phase 4).
        self.stop_offset = self.box_half + spec.crosswalk_width

        self.edges: dict[str, Edge] = {}
        self.inbound: dict[str, list[str]] = {}
        self.outbound: dict[str, list[str]] = {}
        self.routes: dict[tuple[str, str], list[str]] = {}
        self.lanes_for_movement: dict[tuple[str, str], list[str]] = {}

        self._build_lanes()
        self._build_connectors(turn_lateral_accel)
        self.conflicts = self._compute_conflicts(conflict_distance)
        # feeders[outbound_id] = connectors that empty into that exit lane
        self.feeders: dict[str, list[str]] = {eid: [] for eid in self.edges
                                              if self.edges[eid].kind == "outbound"}
        for e in self.connectors():
            self.feeders[e.target_id].append(e.id)
        self.phases = self._build_phases()
        # Fixed iteration order makes the engine deterministic regardless of dict quirks.
        self.edge_order: list[str] = list(self.edges)

    # ------------------------------------------------------------------ builders
    def _build_lanes(self) -> None:
        s, w = self.spec, self.spec.lane_width
        stop = self.stop_offset
        for arm_name, arm in s.arms.items():
            u = ARM_VECTORS[arm_name]
            h_in = -u  # inbound traffic drives towards the centre
            n_in, n_out = len(arm.in_lanes), arm.out_lanes
            self.inbound[arm_name] = []
            self.outbound[arm_name] = []
            for i in range(n_in):
                lat = _left_of(h_in) * (n_in - i - 0.5) * w  # lane 0 farthest from centreline
                start, end = u * (stop + s.approach_length) + lat, u * stop + lat
                eid = f"in_{arm_name}_{i}"
                self.edges[eid] = Edge(eid, "inbound", Path([start, end]), s.speed_limit, arm_name, i)
                self.inbound[arm_name].append(eid)
            for j in range(n_out):
                lat = _left_of(u) * (n_out - j - 0.5) * w
                start, end = u * stop + lat, u * (stop + s.exit_length) + lat
                eid = f"out_{arm_name}_{j}"
                self.edges[eid] = Edge(eid, "outbound", Path([start, end]), s.speed_limit, arm_name, j)
                self.outbound[arm_name].append(eid)

    def _build_connectors(self, a_lat: float) -> None:
        s = self.spec
        for arm_name, arm in s.arms.items():
            h_in = -ARM_VECTORS[arm_name]
            for m in MOVEMENTS:
                lanes_m = [i for i, mv in enumerate(arm.in_lanes) if m in mv]
                if not lanes_m:
                    continue
                exit_arm = _arm_for_direction(_turn(h_in, m))
                n_out = s.arms[exit_arm].out_lanes if exit_arm in s.arms else 0
                if n_out == 0:
                    raise ValueError(f"arm {arm_name} allows '{m}' but arm {exit_arm} has no exit lanes")
                self.lanes_for_movement[(arm_name, m)] = []
                for k, i in enumerate(lanes_m):
                    # Lane mapping: left turns hug the kerb, right turns hug the median,
                    # straight keeps its lane index where possible.
                    if m == "left":
                        j = min(k, n_out - 1)
                    elif m == "right":
                        j = max(n_out - len(lanes_m) + k, 0)
                    else:
                        j = min(i, n_out - 1)
                    src = self.edges[f"in_{arm_name}_{i}"]
                    dst = self.edges[f"out_{exit_arm}_{j}"]
                    p0, p2 = src.path.points[-1], dst.path.points[0]
                    if m == "straight":
                        path = Path([p0, p2])
                        limit = s.speed_limit
                    else:
                        ctrl = line_intersection(p0, h_in, p2, _turn(h_in, m))
                        path = Path(quadratic_bezier(p0, ctrl, p2))
                        # Average turn radius = arc length / angle turned. A comfortable
                        # lateral acceleration a_lat then gives v = sqrt(a_lat * R).
                        radius = path.length / max(path.total_turn(), 1e-6)
                        limit = min(s.speed_limit, float(np.sqrt(a_lat * radius)))
                    eid = f"cx_{arm_name}_{i}_{m}"
                    self.edges[eid] = Edge(eid, "connector", path, limit, arm_name, i,
                                           movement=m, source_id=src.id, target_id=dst.id)
                    self.routes[(src.id, m)] = [src.id, eid, dst.id]
                    self.lanes_for_movement[(arm_name, m)].append(src.id)

    def _compute_conflicts(self, threshold: float) -> dict[str, frozenset[str]]:
        """Two connectors conflict if they merge into the same exit lane or their
        paths come within ``threshold`` metres of each other. Computed once, from
        geometry, so any junction layout works without hand-written conflict tables."""
        cxs = list(self.connectors())
        samples = {c.id: c.path.sample(0.5) for c in cxs}
        out: dict[str, set[str]] = {c.id: set() for c in cxs}
        for a, b in combinations(cxs, 2):
            if a.source_id == b.source_id:
                continue  # same lane: car-following already orders these vehicles
            if a.target_id == b.target_id or min_distance(samples[a.id], samples[b.id]) < threshold:
                out[a.id].add(b.id)
                out[b.id].add(a.id)
        return {k: frozenset(v) for k, v in out.items()}

    def _build_phases(self) -> list[Phase]:
        phases = []
        for p in self.spec.phases:
            for arm, moves in p.green.items():
                if arm not in self.spec.arms:
                    raise ValueError(f"phase {p.name} references missing arm {arm}")
                for m in moves:
                    if (arm, m) not in self.lanes_for_movement:
                        raise ValueError(f"phase {p.name}: arm {arm} has no lane allowing '{m}'")
            phases.append(Phase(p.name, {a: frozenset(ms) for a, ms in p.green.items()}))
        return phases

    # ------------------------------------------------------------------ queries
    def connectors(self):
        return (e for e in self.edges.values() if e.kind == "connector")

    def inbound_edges(self):
        return (e for e in self.edges.values() if e.kind == "inbound")

    @classmethod
    def from_file(cls, path: str | FsPath, turn_lateral_accel: float = 2.5,
                  conflict_distance: float = 2.5) -> "Junction":
        with open(path, encoding="utf-8") as fh:
            spec = JunctionSpec.model_validate(json.load(fh))
        return cls(spec, turn_lateral_accel, conflict_distance)

    def describe(self) -> dict:
        """Static layout for renderers (sent once to the frontend in Phase 3)."""
        return {
            "name": self.name,
            "lane_width": self.lane_width,
            "box_half": self.box_half,
            "stop_offset": self.stop_offset,
            "edges": {eid: {"kind": e.kind, "arm": e.arm, "movement": e.movement,
                            "points": e.path.points.round(3).tolist()}
                      for eid, e in self.edges.items()},
            "phases": [{"name": p.name, "green": {a: sorted(m) for a, m in p.green.items()}}
                       for p in self.phases],
        }
