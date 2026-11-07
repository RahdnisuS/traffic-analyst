"""Junction layout: built from a JSON spec into a graph of lane "edges".

Coordinate system: junction centre at (0, 0), x points east, y points north, metres.
LEFT-HAND TRAFFIC (India): a vehicle drives on the left half of the road
relative to its own heading, so a left turn is the short kerb-side turn and a
right turn crosses the path of oncoming traffic.

The road network is a small directed graph with four kinds of edge:
  inbound   - approach lane, from far upstream to the stop line
  filter    - a narrow virtual lane along the divider between two inbound lanes. Only
              bikes use it: it models riders squeezing between queued vehicles (Phase 4).
              Exits get one too (kind "outbound", id oflt_*), so a bike that filtered can ride
              straight on beside the cars instead of cutting in front of them in the box
  connector - path through the junction box for one (lane, movement) pair
  outbound  - exit lane, from the box to a sink where vehicles despawn
A vehicle's route is [approach, connector, outbound]: fixed at spawn, except that a bike may
swap its inbound lane for the adjacent filter lane (same arc length, since both are parallel).
Lane index 0 is the kerb-side (leftmost) lane; higher indices are nearer the median.

Crosswalks sit between the box edge and the stop line on every arm. Each connector crosses
two of them (its origin arm's and its exit arm's), which is how vehicles know whom to yield to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path as FsPath

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import Movement
from .geometry import Path, line_intersection, min_distance, quadratic_bezier, segment_distances

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
    sidewalk_width: float = Field(3.0, gt=0)
    filter_lanes: bool = True     # build bike filter lanes between adjacent inbound lanes
    corner_margin: float = Field(1.5, ge=0)
    speed_limit: float = Field(13.9, gt=0)
    arms: dict[str, ArmSpec]
    phases: list[PhaseSpec] = Field(min_length=1)


# ---------------------------------------------------------------- runtime types
@dataclass(eq=False)
class Edge:
    id: str
    kind: str                     # "inbound" | "filter" | "connector" | "outbound"
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

    @property
    def is_approach(self) -> bool:
        """Ends at a stop line (vehicles on it obey the signal)."""
        return self.kind in ("inbound", "filter")


@dataclass(frozen=True, eq=False)
class Crosswalk:
    """Zebra crossing over one arm, described in that arm's own axes.

    u: unit vector pointing out along the arm; n: unit vector across it.
    The walkable strip is u-coordinate in [u0, u1]; the carriageway spans n in [lo, hi].
    Pedestrians wait on the sidewalk just beyond lo or hi and walk across to the other side.
    """
    arm: str
    u: np.ndarray
    n: np.ndarray
    u0: float
    u1: float
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo

    @property
    def centre_u(self) -> float:
        return 0.5 * (self.u0 + self.u1)

    def point(self, along: float, across: float) -> np.ndarray:
        return self.u * along + self.n * across


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


_BODY_CONFLICT_CACHE: dict = {}


class Junction:
    """Immutable road geometry plus derived lookup tables used by the engine."""

    def __init__(self, spec: JunctionSpec, turn_lateral_accel: float = 2.5,
                 conflict_distance: float = 2.5, filter_conflict_distance: float = 1.5) -> None:
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
        self.filters: dict[str, list[str]] = {}          # arm -> filter lane ids
        self.out_filters: dict[str, list[str]] = {}      # arm -> exit-side filter lane ids
        self.filters_beside: dict[str, list[str]] = {}   # inbound lane id -> adjacent filter lanes

        self._build_lanes()
        self._build_connectors(turn_lateral_accel)
        self.conflicts = self._compute_conflicts(conflict_distance, filter_conflict_distance)
        # Filled by compute_body_conflicts() once vehicle sizes are known (the engine does it).
        self.body_conflicts: dict[tuple[str, str], dict[str, frozenset[str]]] = {}
        # feeders[outbound_id] = connectors that empty into that exit lane
        self.feeders: dict[str, list[str]] = {eid: [] for eid in self.edges
                                              if self.edges[eid].kind == "outbound"}
        for e in self.connectors():
            self.feeders[e.target_id].append(e.id)
        # filter_movements[filter lane] = movements a bike may filter for. Only those whose path
        # out of the gap crosses neither neighbouring lane's traffic: a bike that must cut across a
        # stream of cars waits for a gap at the line and loses more time than filtering saved.
        # On the default layout this works out as "straight only"; derived from geometry, so it
        # adapts to other layouts.
        self.filter_movements: dict[str, frozenset[str]] = {}
        for arm, flts in self.filters.items():
            for k, fid in enumerate(flts):
                beside = {c.id for c in self.connectors()
                          if c.source_id in (f"in_{arm}_{k}", f"in_{arm}_{k + 1}")}
                self.filter_movements[fid] = frozenset(
                    c.movement for c in self.connectors()
                    if c.source_id == fid and not (self.conflicts[c.id] & beside))
        self.phases = self._build_phases()
        self.crosswalks = self._build_crosswalks()
        # crosswalk_connectors[arm] = connectors that drive over that arm's crosswalk
        self.crosswalk_connectors: dict[str, frozenset[str]] = {
            a: frozenset(c.id for c in self.connectors()
                         if c.arm == a or self.edges[c.target_id].arm == a)
            for a in self.crosswalks}
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
            self.filters[arm_name] = []
            if s.filter_lanes:
                for k in range(n_in - 1):
                    # On the divider between lane k and lane k+1.
                    lat = _left_of(h_in) * (n_in - k - 1) * w
                    start, end = u * (stop + s.approach_length) + lat, u * stop + lat
                    eid = f"flt_{arm_name}_{k}"
                    # Listed before the arm's car lanes on purpose: the engine evaluates edges
                    # in order, so bikes at the front of the filter lane launch first on green.
                    self.edges[eid] = Edge(eid, "filter", Path([start, end]), s.speed_limit, arm_name, k)
                    self.filters[arm_name].append(eid)
            for i in range(n_in):
                lat = _left_of(h_in) * (n_in - i - 0.5) * w  # lane 0 farthest from centreline
                start, end = u * (stop + s.approach_length) + lat, u * stop + lat
                eid = f"in_{arm_name}_{i}"
                self.edges[eid] = Edge(eid, "inbound", Path([start, end]), s.speed_limit, arm_name, i)
                self.inbound[arm_name].append(eid)
                self.filters_beside[eid] = [f"flt_{arm_name}_{k}" for k in (i - 1, i)
                                            if f"flt_{arm_name}_{k}" in self.edges]
            for j in range(n_out):
                lat = _left_of(u) * (n_out - j - 0.5) * w
                start, end = u * stop + lat, u * (stop + s.exit_length) + lat
                eid = f"out_{arm_name}_{j}"
                self.edges[eid] = Edge(eid, "outbound", Path([start, end]), s.speed_limit, arm_name, j)
                self.outbound[arm_name].append(eid)
            self.out_filters[arm_name] = []
            if s.filter_lanes:
                for k in range(n_out - 1):
                    lat = _left_of(u) * (n_out - k - 1) * w
                    start, end = u * stop + lat, u * (stop + s.exit_length) + lat
                    eid = f"oflt_{arm_name}_{k}"
                    self.edges[eid] = Edge(eid, "outbound", Path([start, end]), s.speed_limit, arm_name, k)
                    self.out_filters[arm_name].append(eid)

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
                    self._add_connector(f"cx_{arm_name}_{i}_{m}", src, dst, m, h_in, a_lat)
                    self.lanes_for_movement[(arm_name, m)].append(src.id)
            # A filter lane allows every movement of the two lanes it sits between.
            for k, fid in enumerate(self.filters[arm_name]):
                moves = set(arm.in_lanes[k]) | set(arm.in_lanes[k + 1])
                for m in MOVEMENTS:
                    if m not in moves:
                        continue
                    exit_arm = _arm_for_direction(_turn(h_in, m))
                    n_out = s.arms[exit_arm].out_lanes
                    oflt = self.out_filters.get(exit_arm, [])
                    if m == "straight" and oflt:
                        # Straight on into the matching exit gap: parallel to the cars, no conflict.
                        dst = self.edges[oflt[min(k, len(oflt) - 1)]]
                    else:
                        j = 0 if m == "left" else n_out - 1 if m == "right" else min(k, n_out - 1)
                        dst = self.edges[f"out_{exit_arm}_{j}"]
                    self._add_connector(f"cx_{fid}_{m}", self.edges[fid], dst, m, h_in, a_lat)

    def _add_connector(self, eid: str, src: Edge, dst: Edge, m: str, h_in: np.ndarray,
                       a_lat: float) -> None:
        s = self.spec
        p0, p2 = src.path.points[-1], dst.path.points[0]
        ctrl = None if m == "straight" else line_intersection(p0, h_in, p2, _turn(h_in, m))
        if ctrl is None:
            path = Path([p0, p2])
            limit = s.speed_limit
        else:
            path = Path(quadratic_bezier(p0, ctrl, p2))
            # Average turn radius = arc length / angle turned. A comfortable
            # lateral acceleration a_lat then gives v = sqrt(a_lat * R).
            radius = path.length / max(path.total_turn(), 1e-6)
            limit = min(s.speed_limit, float(np.sqrt(a_lat * radius)))
        self.edges[eid] = Edge(eid, "connector", path, limit, src.arm, src.lane_index,
                               movement=m, source_id=src.id, target_id=dst.id)
        self.routes[(src.id, m)] = [src.id, eid, dst.id]

    def _compute_conflicts(self, threshold: float, filter_threshold: float) -> dict[str, frozenset[str]]:
        """Two connectors conflict if they merge into the same exit lane or their
        paths come within ``threshold`` metres of each other. Computed once, from
        geometry, so any junction layout works without hand-written conflict tables.

        Paths out of a filter lane use the smaller ``filter_threshold``: they only carry bikes
        (0.8 m wide), so a bike 1.75 m from a car's path still clears it by about half a metre.
        That is what lets a bike ride through the box beside a car, the way it does in real life."""
        cxs = list(self.connectors())
        samples = {c.id: c.path.sample(0.5) for c in cxs}
        out: dict[str, set[str]] = {c.id: set() for c in cxs}
        for a, b in combinations(cxs, 2):
            if a.source_id == b.source_id:
                continue  # same lane: car-following already orders these vehicles
            limit = filter_threshold if "filter" in (self.edges[a.source_id].kind,
                                                     self.edges[b.source_id].kind) else threshold
            if a.target_id == b.target_id or min_distance(samples[a.id], samples[b.id]) < limit:
                out[a.id].add(b.id)
                out[b.id].add(a.id)
        return {k: frozenset(v) for k, v in out.items()}

    # ------------------------------------------------------------------ body-swept conflicts
    def _swept_body(self, cx: Edge, length: float, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Where a vehicle's BODY is while it uses connector ``cx``: rear and front points of the
        segment rear->front, sampled from "front bumper reaches the stop line" until "rear leaves
        the box". The rear follows the same path as the front (no trailer maths), so on a turn the
        body is the CHORD of the curve and cuts inside it, like a real long vehicle."""
        src, dst = self.edges[cx.source_id], self.edges[cx.target_id]
        chain = [(src, 0.0), (cx, src.length), (dst, src.length + cx.length)]

        def points(s_route: np.ndarray) -> np.ndarray:
            out = np.empty((len(s_route), 2))
            for edge, start in chain:  # later edges overwrite shared boundary points: harmless
                m = (s_route >= start - 1e-9) & (s_route <= start + edge.length + 1e-9)
                if m.any():
                    x, y, _ = edge.path.poses_at(s_route[m] - start)
                    out[m, 0], out[m, 1] = x, y
            return out

        front = np.arange(src.length, src.length + cx.length + length + 1e-9, step)
        return points(np.maximum(front - length, 0.0)), points(front)

    def compute_body_conflicts(self, bodies: dict[str, tuple[float, float]], clearance: float,
                               filtering_types: set[str] | frozenset[str] = frozenset()) -> None:
        """Conflicts that depend on WHO is driving: ``body_conflicts[(ta, tb)][cx_a]`` = connectors
        that a vehicle of type ``tb`` must not be using while one of type ``ta`` uses ``cx_a``.

        WHY: the centreline test above (``conflicts``) treats every vehicle as a point on its path.
        A 12 m bus turning right sweeps its body across the neighbouring lane's path, which a
        4.5 m car does not; so whether two paths can be used together depends on the vehicles.
        Two paths conflict for (ta, tb) when, at any moment of either crossing, the bodies would
        come closer than ``clearance`` (half-widths included).

        Pairs from the SAME source lane are left out: those vehicles are kept apart by
        car-following (the engine's follower keeps tracking a leader that has turned off).
        Filter-lane paths only ever carry ``filtering_types`` (bikes), so only those are computed.
        The result is cached per layout + vehicle sizes: every Simulation builds a Junction, and
        the benchmark builds hundreds.
        """
        key = (self.spec.model_dump_json(), tuple(sorted(bodies.items())), clearance, tuple(sorted(filtering_types)))
        cached = _BODY_CONFLICT_CACHE.get(key)
        if cached is None:
            cached = self._body_conflicts(bodies, clearance, frozenset(filtering_types))
            _BODY_CONFLICT_CACHE[key] = cached
        self.body_conflicts = cached

    def _body_conflicts(self, bodies, clearance, filtering_types) -> dict:
        cxs = list(self.connectors())
        types = sorted(bodies)
        allowed = {c.id: (filtering_types if self.edges[c.source_id].kind == "filter" else set(types)) for c in cxs}
        swept = {(c.id, t): self._swept_body(c, bodies[t][0]) for c in cxs for t in types if t in allowed[c.id]}
        boxes = {k: (np.minimum(r.min(0), f.min(0)), np.maximum(r.max(0), f.max(0))) for k, (r, f) in swept.items()}
        out = {(ta, tb): {c.id: set() for c in cxs} for ta in types for tb in types}
        for i, a in enumerate(cxs):
            for b in cxs[i + 1:]:
                if a.source_id == b.source_id:
                    continue
                for ta in sorted(allowed[a.id]):
                    for tb in sorted(allowed[b.id]):
                        need = (bodies[ta][1] + bodies[tb][1]) / 2 + clearance
                        (lo_a, hi_a), (lo_b, hi_b) = boxes[(a.id, ta)], boxes[(b.id, tb)]
                        if np.any(lo_a - need > hi_b) or np.any(lo_b - need > hi_a):
                            continue  # bounding boxes too far apart: cannot touch
                        (ra, fa), (rb, fb) = swept[(a.id, ta)], swept[(b.id, tb)]
                        if segment_distances(ra, fa, rb, fb).min() < need:
                            out[(ta, tb)][a.id].add(b.id)
                            out[(tb, ta)][b.id].add(a.id)
        return {k: {c: frozenset(v) for c, v in m.items()} for k, m in out.items()}

    def conflicts_for(self, cx: str, vtype: str, other_cx: str, other_type: str) -> bool:
        """May a ``vtype`` vehicle NOT use ``cx`` while an ``other_type`` vehicle uses ``other_cx``?
        The centreline conflicts always apply; body conflicts add to them for big vehicles."""
        if other_cx in self.conflicts[cx]:
            return True
        bc = self.body_conflicts.get((vtype, other_type))
        return bc is not None and other_cx in bc.get(cx, ())

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

    def _build_crosswalks(self) -> dict[str, Crosswalk]:
        """One crosswalk per arm, spanning the arm's full carriageway (both directions)."""
        out = {}
        w = self.lane_width
        for arm in self.spec.arms:
            u = ARM_VECTORS[arm]
            n = _left_of(u)
            lats = [float(n @ p) for e in self.edges.values()
                    if e.kind in ("inbound", "outbound") and e.arm == arm for p in e.path.points]
            if not lats:
                continue
            out[arm] = Crosswalk(arm, u, n, self.box_half, self.stop_offset,
                                 min(lats) - w / 2, max(lats) + w / 2)
        return out

    # ------------------------------------------------------------------ queries
    def connectors(self):
        return (e for e in self.edges.values() if e.kind == "connector")

    def inbound_edges(self):
        return (e for e in self.edges.values() if e.kind == "inbound")

    def approach_edges(self):
        """Inbound and filter lanes: everything that ends at a stop line."""
        return (e for e in self.edges.values() if e.is_approach)

    def approach_lanes(self, arm: str) -> list[str]:
        return self.inbound[arm] + self.filters.get(arm, [])

    @classmethod
    def from_file(cls, path: str | FsPath, turn_lateral_accel: float = 2.5,
                  conflict_distance: float = 2.5, filter_conflict_distance: float = 1.5) -> "Junction":
        with open(path, encoding="utf-8") as fh:
            spec = JunctionSpec.model_validate(json.load(fh))
        return cls(spec, turn_lateral_accel, conflict_distance, filter_conflict_distance)

    def describe(self) -> dict:
        """Static layout for renderers (sent once to the frontend in Phase 3)."""
        return {
            "name": self.name,
            "lane_width": self.lane_width,
            "box_half": self.box_half,
            "stop_offset": self.stop_offset,
            "sidewalk_width": self.spec.sidewalk_width,
            "crosswalks": {a: {"u": cw.u.tolist(), "n": cw.n.tolist(), "u0": cw.u0, "u1": cw.u1,
                               "lo": round(cw.lo, 3), "hi": round(cw.hi, 3)}
                           for a, cw in self.crosswalks.items()},
            "edges": {eid: {"kind": e.kind, "arm": e.arm, "movement": e.movement,
                            "points": e.path.points.round(3).tolist()}
                      for eid, e in self.edges.items()},
            "phases": [{"name": p.name, "green": {a: sorted(m) for a, m in p.green.items()}}
                       for p in self.phases],
        }
