"""Headless simulation engine. No rendering, no I/O, no wall-clock time.

One call to ``step()`` advances the world by ``dt`` seconds:

  1. signal policy decides; signal controller advances (green/amber/all-red/pedestrian phase)
  2. spawner samples arrivals; pedestrians spawn, walk (social force) and cross
  3. queued vehicle arrivals are inserted where there is room
  4. bikes stuck behind a queue move into the filter gap between lanes
  5. every driver finds its leader (a vehicle, or a red stop line)
  6. IDM gives every acceleration in one vectorised NumPy call
  7. positions/speeds integrate; vehicles move to the next edge or despawn

Determinism: all randomness comes from one seeded ``np.random.Generator``, and
every loop iterates in a fixed order (sorted arms, fixed edge order, vehicles
sorted by position). Same seed + same config gives a bit-identical run.

Junction entry uses a RESERVATION rule (the "box-junction" rule, as used in many
real intersections and in SUMO's junction model): a driver at the stop line
commits to entering only if
  (a) the light is green (or amber and they can no longer stop comfortably),
  (b) no vehicle is on, or has reserved, a conflicting path through the box,
  (c) the exit lane has room for them, so they never block the box, and
  (d) nobody is still walking on a crosswalk their path drives over (yield to pedestrians).
Once committed, the decision is final, like a real driver past the point of no return.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Config, VehicleTypeParams
from .idm import ballistic_step, idm_acceleration
from .junction import Junction
from .pedestrian import PedestrianSystem
from .signals import FixedTimePolicy, SignalController, SignalPolicy
from .spawner import PendingVehicle, Spawner
from .vehicle import TripRecord, Vehicle, scale_for_aggressiveness

WAIT_SPEED = 0.1      # m/s; below this a vehicle counts as "waiting" (same threshold as SUMO)
STOP_BUFFER = 1.0     # virtual red-light obstacle sits up to this far past the stop line, so the
                      # front bumper rests about (s0 - 1) m before the line...
STOP_MIN_MARGIN = 0.5 # ...but never closer than this: a bike's small s0 (1 m) would otherwise
                      # park its front wheel on the line itself
COMMIT_MARGIN = 1.0   # extra metres of look-ahead when deciding to commit on green
INSERT_CLEARANCE = 0.5
RESTART_SPEED = 1.0   # a stopped vehicle must exceed this before a new halt counts as another stop
                      # (hysteresis: creeping forward in a queue is not a new stop)
FILTER_MIN_ROOM = 3.0  # m before the stop line: closer than this there is no room left to squeeze past
FILTER_LAG_HEADWAY = 0.5  # s: a bike won't cut in front of a filter-lane follower closer than this
FILTER_TAIL_MARGIN = 10.0  # m: start slowing to filter speed this far before the queue's tail


class EntryDecision(Enum):
    COMMIT = "commit"    # enter the box; decision is final
    PROCEED = "proceed"  # light is fine but still far away; re-evaluate next step
    STOP = "stop"        # treat the stop line as a wall


@dataclass
class RunCounters:
    """Minimal Phase 1 statistics. Phase 2 replaces these with a full metrics module."""
    spawned: int = 0
    completed: int = 0
    total_travel_time: float = 0.0
    total_wait_time: float = 0.0
    emergency_stops: int = 0  # safety net fired: should stay ~0 in normal operation


class Simulation:
    def __init__(self, config: Config, seed: int | None = None,
                 junction: Junction | None = None, policy: SignalPolicy | None = None):
        # Deep copy: live setting changes (sliders) must not leak into other sims sharing a config.
        config = config.model_copy(deep=True)
        self.cfg = config
        self.dt = config.sim.dt
        self.seed = config.sim.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        ix = config.intersection
        self.junction = junction or Junction.from_file(
            config.junction_path(), ix.turn_lateral_accel, ix.conflict_distance_m,
            ix.filter_conflict_distance_m)
        self.peds = PedestrianSystem(self.junction, config.pedestrians, self.rng)
        ps = config.pedestrians
        # Flashing DON'T WALK lasts long enough for someone who starts on the last second of
        # WALK to cross the longest crosswalk at the design speed (rounded up to 0.1 s).
        self.ped_clear_s = math.ceil(self.peds.max_crossing_length() / ps.clearance_speed * 10) / 10
        sg = config.signals
        self.signals = SignalController(self.junction.phases, sg.amber_s, sg.all_red_s, sg.min_green_s,
                                        walk_s=ps.walk_s, ped_clear_s=self.ped_clear_s,
                                        min_greens_between_peds=ps.min_greens_between)
        self.policy: SignalPolicy = policy or FixedTimePolicy(sg.green_times(len(self.junction.phases)))
        self.spawner = Spawner(self.junction, config.demand, config.vehicle_types, self.rng)
        self.tick = 0
        self._next_id = 0
        self.vehicles: dict[int, Vehicle] = {}
        # lanes[edge_id] = vehicles on that edge, FRONT FIRST (descending s)
        self.lanes: dict[str, list[Vehicle]] = {eid: [] for eid in self.junction.edge_order}
        self.counters = RunCounters()
        # Trips completed during the latest step (observers read this, then it is reset).
        self.recent_trips: list[TripRecord] = []
        # Observer pattern: anything with ``after_step(sim)`` (metrics, logger, websocket
        # streamer) can watch the sim without the sim knowing what it is.
        self.observers: list = []
        # Settings changes made while running (sliders, analyst scenarios). Logged as events so
        # the analyst can say "wait rose after you raised volume at 3:40".
        self.control_events: list[dict] = []
        self._base_vph = dict(config.demand.per_approach_vph)
        self._params_cache: dict[str, VehicleTypeParams] = {}
        self._refresh_params()
        J = self.junction
        if not J.body_conflicts:
            J.compute_body_conflicts({t: (q.length, q.width) for t, q in config.vehicle_types.items()},
                                     ix.body_clearance_m,
                                     {t for t, q in config.vehicle_types.items() if q.filters})
        self._cx_exit_arm = {c.id: J.edges[c.target_id].arm for c in J.connectors()}
        self.filter_moves = 0  # how many times a bike moved into a filter lane
        # filter lane -> the inbound lanes on either side of it
        self._filter_sides: dict[str, list[str]] = {}
        for lane_id, flts in J.filters_beside.items():
            for f in flts:
                self._filter_sides.setdefault(f, []).append(lane_id)
        self._queue_tail: dict[str, float] = {}
        # Sideways distance between a filter lane's centreline and each lane beside it (measured
        # at the stop line, from geometry, so other layouts work too).
        self._body_clearance = ix.body_clearance_m
        self._lateral_spacing: dict[tuple[str, str], float] = {}
        for f, sides in self._filter_sides.items():
            fx, fy, _ = J.edges[f].path.pose_at(J.edges[f].length)
            for side in sides:
                sx, sy, _ = J.edges[side].path.pose_at(J.edges[side].length)
                self._lateral_spacing[(f, side)] = math.hypot(fx - sx, fy - sy)

    def _refresh_params(self) -> None:
        agg = self.cfg.driver.aggressiveness
        self._params_cache = {name: scale_for_aggressiveness(p, agg)
                              for name, p in self.cfg.vehicle_types.items()}

    # ================================================================== live settings
    def apply_settings(self, *, volume_vph: float | None = None, aggressiveness: float | None = None,
                       green_s: float | None = None, vehicle_mix: dict[str, float] | None = None,
                       ped_rate_ph: float | None = None, source: str = "user") -> dict:
        """Change settings mid-run. Returns the changes that were applied.

        volume_vph:   mean demand per approach; per-approach ratios from the config are kept
                      (so a busier N-S corridor stays busier).
        green_s:      green time for every phase of the fixed-time plan (from the next switch).
        vehicle_mix:  relative weights per vehicle type (normalised; types left out get 0).
        ped_rate_ph:  pedestrian arrivals per hour, all crosswalks together.
        Raises ValueError on invalid values, leaving the sim unchanged.
        """
        changes: dict = {}
        if vehicle_mix is not None:
            unknown = set(vehicle_mix) - set(self.cfg.vehicle_types)
            if unknown:
                raise ValueError(f"unknown vehicle types: {sorted(unknown)}")
            if any(w < 0 for w in vehicle_mix.values()) or sum(vehicle_mix.values()) <= 0:
                raise ValueError("vehicle_mix weights must be >= 0 and not all zero")
        if ped_rate_ph is not None and ped_rate_ph < 0:
            raise ValueError("ped_rate_ph must be >= 0")
        if green_s is not None:
            if not hasattr(self.policy, "green_times"):
                raise ValueError("the active signal policy has no adjustable green time")
            if green_s < self.cfg.signals.min_green_s:
                raise ValueError(f"green_s must be >= min_green_s ({self.cfg.signals.min_green_s})")
        if aggressiveness is not None and not 0.0 <= aggressiveness <= 1.0:
            raise ValueError("aggressiveness must be in [0, 1]")
        if volume_vph is not None and volume_vph < 0:
            raise ValueError("volume_vph must be >= 0")

        if volume_vph is not None:
            base_mean = sum(self._base_vph.values()) / max(len(self._base_vph), 1)
            new = {a: round(v * volume_vph / base_mean, 1) if base_mean > 0 else float(volume_vph)
                   for a, v in self._base_vph.items()}
            self.cfg.demand.per_approach_vph = new
            self.spawner.set_demand(self.cfg.demand)
            changes["per_approach_vph"] = new
        if aggressiveness is not None:
            self.cfg.driver.aggressiveness = float(aggressiveness)
            self._refresh_params()
            changes["aggressiveness"] = float(aggressiveness)
        if green_s is not None:
            self.policy.green_times = [float(green_s)] * len(self.junction.phases)
            self.cfg.signals.green_s = float(green_s)
            changes["green_s"] = float(green_s)
        if vehicle_mix is not None:
            total = sum(vehicle_mix.values())
            mix = {t: round(vehicle_mix.get(t, 0.0) / total, 4) for t in sorted(self.cfg.vehicle_types)}
            self.cfg.demand.vehicle_mix = mix
            self.spawner.set_demand(self.cfg.demand)
            changes["vehicle_mix"] = mix
        if ped_rate_ph is not None:
            self.cfg.pedestrians.rate_ph = float(ped_rate_ph)
            self.peds.set_rate(ped_rate_ph)
            changes["ped_rate_ph"] = float(ped_rate_ph)
        if changes:
            self.control_events.append({"t": round(self.time, 1), "source": source, "changes": changes})
        return changes

    def mean_volume_vph(self) -> float:
        v = self.cfg.demand.per_approach_vph
        return sum(v.values()) / max(len(v), 1)

    @property
    def time(self) -> float:
        # tick * dt instead of summing dt: no floating-point drift over long runs
        return self.tick * self.dt

    # ================================================================== main loop
    def step(self) -> None:
        dt = self.dt
        self.recent_trips = []
        self.policy.update(self.signals, self)
        self.signals.step(dt)
        self.spawner.generate(self.time, dt)
        self.peds.update(self, self.time, dt)
        self._insert_from_backlog()
        self._filter_bikes()
        self._move_vehicles(dt)
        self.tick += 1
        for obs in self.observers:
            obs.after_step(self)

    def run(self, duration_s: float) -> None:
        for _ in range(int(round(duration_s / self.dt))):
            self.step()

    # ================================================================== spawning
    def _entry_space(self, lane_id: str) -> float:
        lane = self.lanes[lane_id]
        return math.inf if not lane else lane[-1].s - lane[-1].length

    def _insert_from_backlog(self) -> None:
        for arm in self.spawner.arms:
            queue = self.spawner.backlog[arm]
            # FIFO: if the head can't enter, nobody behind it can either (like a real queue).
            while queue and self._try_insert(arm, queue[0]):
                queue.popleft()

    def _try_insert(self, arm: str, pv: PendingVehicle) -> bool:
        params = self._params_cache[pv.vtype]
        candidates = self.junction.lanes_for_movement[(arm, pv.movement)]
        need = params.length + params.min_gap + INSERT_CLEARANCE
        if params.kerb_lane and self._entry_space(candidates[0]) >= need:
            lane_id = candidates[0]  # buses keep to the kerb-side lane when it has room
        else:
            # Drivers pick the allowed lane with the most free road (ties: lower index).
            lane_id = max(candidates, key=lambda lid: (self._entry_space(lid), -candidates.index(lid)))
        space = self._entry_space(lane_id)
        if space < need:
            return False
        desired = params.desired_speed * pv.speed_factor
        edge = self.junction.edges[lane_id]
        v = min(desired, edge.speed_limit)
        lane = self.lanes[lane_id]
        if lane:
            # Enter no faster than a speed IDM would consider safe for this gap:
            # s* = s0 + v*T <= gap  ->  v <= (gap - s0) / T, and not faster than the leader.
            gap_ok_v = max(0.0, (space - params.min_gap) / params.time_headway)
            v = min(v, gap_ok_v, lane[-1].v + 2.0)
        route = list(self.junction.routes[(lane_id, pv.movement)])
        free_flow = sum(self.junction.edges[e].length / min(desired, self.junction.edges[e].speed_limit)
                        for e in route)
        veh = Vehicle(id=self._next_id, vtype=pv.vtype, params=params, route=route,
                      origin=arm, movement=pv.movement, desired_speed=desired,
                      arrival_time=pv.arrival_time, insert_time=self.time, v=v,
                      free_flow_time=free_flow)
        self._next_id += 1
        self.vehicles[veh.id] = veh
        lane.append(veh)  # s=0 is behind everyone already on the lane
        self.counters.spawned += 1
        return True

    # ================================================================== bike filtering
    def _filter_bikes(self) -> None:
        """Bikes stuck behind a queue move into the filter gap between lanes.

        A bike filters when (1) its leader is queued (slower than queue_speed) and close,
        (2) it has slowed to filter speed itself, (3) there is still room before the stop line,
        (4) its movement can leave the gap without cutting across the cars beside it
        (``junction.filter_movements``), and (5) gap acceptance passes in the filter lane: room
        ahead for the bike, and the filter-lane follower (if any) is not too close behind. That is the same gap-acceptance
        idea as any lane change (as in the MOBIL model), with fixed thresholds.
        The switch is logically instant; ``lat_shift`` makes it look like a smooth swerve.
        """
        J, ix = self.junction, self.cfg.intersection
        queue_speed = self.cfg.metrics.queue_speed
        turned_off = self._turned_off_leaders()
        for eid in J.edge_order:
            flts = J.filters_beside.get(eid)
            lane = self.lanes[eid]
            if not flts or len(lane) < 2:
                continue
            edge = J.edges[eid]
            keep = [lane[0]]
            for veh in lane[1:]:
                lead = keep[-1]
                target = None
                if (veh.params.filters and not veh.committed
                        and lead.v < queue_speed
                        and lead.s - lead.length - veh.s <= ix.filter_trigger_gap
                        and edge.length - veh.s > FILTER_MIN_ROOM):
                    target = next((f for f in flts if veh.v <= ix.filter_speed
                                   and veh.movement in J.filter_movements[f]
                                   and self._filter_gap_ok(veh, self.lanes[f])
                                   and self._wide_side_gap(veh, f, turned_off)[0] >= veh.params.min_gap), None)
                if target is None:
                    keep.append(veh)
                    continue
                flane = self.lanes[target]
                # Lane lists are front-first (descending s): insert at the matching position.
                idx = bisect.bisect_left([-x.s for x in flane], -veh.s)
                flane.insert(idx, veh)
                x0, y0, _ = edge.path.pose_at(veh.s)
                x1, y1, h1 = J.edges[target].path.pose_at(veh.s)
                veh.lat_shift = (x0 - x1) * -math.sin(h1) + (y0 - y1) * math.cos(h1)
                veh.route = list(J.routes[(target, veh.movement)])
                veh.filtered = True
                self.filter_moves += 1
            self.lanes[eid] = keep

    def _too_wide_beside(self, veh: Vehicle, flt: str, side: str, other: Vehicle) -> bool:
        """Is ``other`` (in lane ``side``) too wide for bike ``veh`` to pass in filter lane ``flt``?"""
        return (other.params.width + veh.params.width) / 2 + self._body_clearance > self._lateral_spacing[(flt, side)]

    def _wide_side_gap(self, veh: Vehicle, flt: str, turned_off: dict) -> tuple[float, float]:
        """Gap from a bike in filter lane ``flt`` to the rear of the nearest vehicle AHEAD in a lane
        beside it that is too wide to squeeze past (a bus leaves ~5 cm to the divider). Such a
        vehicle closes the filter gap, so the bike won't overtake it and treats its rear like a
        leader. Filter and lane share the same ``s``.

        Only vehicles whose REAR is already ahead count. A bus that drives past a bike is safe
        while both go straight (the only danger is a turning tail swing, which the bus avoids by
        waiting, see ``_bike_beside_body``); counting it would make the bike brake hard and the
        bikes behind it rear-end it (an earlier version did exactly that)."""
        gap, dv = math.inf, 0.0
        for side in self._filter_sides.get(flt, ()):
            for other in self.lanes[side]:
                if other.s - other.length >= veh.s and self._too_wide_beside(veh, flt, side, other):
                    g = other.s - other.length - veh.s
                    if g < gap:
                        gap, dv = g, veh.v - other.v
            # ...and the same for one that has turned off into the box: its rear is measured along
            # its path from the stop line. Only if that rear is AHEAD of the bike; a turning wide
            # vehicle never starts while a bike is ahead of its rear (_bike_beside_body).
            to_line = self.junction.edges[flt].length - veh.s
            for other, rear_past_line in turned_off.get(side, ()):
                g = to_line + rear_past_line
                if 0.0 <= g < gap and self._too_wide_beside(veh, flt, side, other):
                    gap, dv = g, veh.v - other.v
        return gap, dv

    def _bike_beside_body(self, veh: Vehicle, lane_id: str) -> bool:
        """Is a filtering vehicle in the gap beside ``veh`` and not entirely behind it, in a gap
        ``veh`` is too wide to share? "Not entirely behind" includes a bike waiting AHEAD at the
        stop line: the turning body sweeps past it and the tail swings into it."""
        for flt in self.junction.filters_beside.get(lane_id, ()):
            for bike in self.lanes[flt]:
                if bike.s > veh.s - veh.length and self._too_wide_beside(bike, flt, lane_id, veh):
                    return True
        return False

    @staticmethod
    def _filter_gap_ok(veh: Vehicle, flane: list[Vehicle]) -> bool:
        ahead = [x for x in flane if x.s >= veh.s - veh.length]
        behind = [x for x in flane if x.s < veh.s - veh.length]
        if ahead:
            lead = ahead[-1]
            # Room to stop behind it with comfortable braking, not just the standstill gap: a bike
            # swerving in at 4 m/s 2 m behind a STOPPED bike cannot avoid it.
            closing = max(0.0, veh.v * veh.v - lead.v * lead.v) / (2.0 * veh.params.comfort_decel)
            if lead.s - lead.length - veh.s < veh.params.min_gap + closing:
                return False
        if behind:
            lag = behind[0]
            if veh.s - veh.length - lag.s < lag.params.min_gap + lag.v * FILTER_LAG_HEADWAY:
                return False
        return True

    # ================================================================== driving
    def connector_users(self) -> dict[str, list[str]]:
        """Connectors in use -> vehicle types using them.

        "Using" means any part of the BODY is on it, not just the front bumper: a 12 m bus whose
        nose has reached the exit lane still blocks the box with the other 10 m. Also counted:
        drivers who have committed to a connector but not reached it yet (a reservation).
        """
        J = self.junction
        users: dict[str, list[str]] = {}
        for eid in J.edge_order:
            edge = J.edges[eid]
            for veh in self.lanes[eid]:
                if edge.kind == "connector":
                    users.setdefault(eid, []).append(veh.vtype)
                elif veh.route_idx > 0 and veh.s < veh.length:
                    prev = veh.route[veh.route_idx - 1]
                    if J.edges[prev].kind == "connector":  # rear still in the box
                        users.setdefault(prev, []).append(veh.vtype)
                if edge.is_approach and veh is self.lanes[eid][0] and veh.committed:
                    users.setdefault(veh.route[1], []).append(veh.vtype)
        return users

    def active_connectors(self) -> set[str]:
        """Connectors in use: occupied by any part of a vehicle, or reserved by a committed driver."""
        return set(self.connector_users())

    def _turned_off_leaders(self) -> dict[str, list[tuple[Vehicle, float]]]:
        """Per approach lane: vehicles that left it onto a connector and whose body is still in
        the box, with the distance their REAR has travelled past the stop line.

        WHY: a driver normally follows the vehicle ahead on its own route. If the car ahead turns
        right while I go straight, it vanishes from my route the moment its nose passes the stop
        line, yet its tail is still right in front of me. Without this, the follower drove into the
        tail of turning buses (seen as overlapping bodies in the 3D view).
        """
        J = self.junction
        out: dict[str, list[tuple[Vehicle, float]]] = {}
        for eid in J.edge_order:
            edge = J.edges[eid]
            for veh in self.lanes[eid]:
                if edge.kind == "connector":
                    cx, along = edge, veh.s
                elif veh.route_idx > 0 and veh.s < veh.length and J.edges[veh.route[veh.route_idx - 1]].kind == "connector":
                    cx = J.edges[veh.route[veh.route_idx - 1]]
                    along = cx.length + veh.s
                else:
                    continue
                out.setdefault(cx.source_id, []).append((veh, along - veh.length))
        return out

    def _lookahead(self, veh: Vehicle, edge_len: float) -> tuple[float, float]:
        """Gap and approach rate to the nearest vehicle ahead on the next edges of the route."""
        dist = edge_len - veh.s
        limit = self.cfg.intersection.lookahead_m
        for eid in veh.route[veh.route_idx + 1:]:
            lane = self.lanes[eid]
            if lane:
                last = lane[-1]
                return dist + last.s - last.length, veh.v - last.v
            dist += self.junction.edges[eid].length
            if dist > limit:
                break
        return math.inf, 0.0

    def _exit_has_space(self, veh: Vehicle, out_id: str) -> bool:
        lane = self.lanes[out_id]
        free = self.junction.edges[out_id].length if not lane else lane[-1].s - lane[-1].length
        # Vehicles already in the box heading for this exit will need room too.
        pending = sum(v.length + v.params.min_gap
                      for cx in self.junction.feeders[out_id] for v in self.lanes[cx])
        return free - pending >= veh.length + veh.params.min_gap

    def _entry_decision(self, veh: Vehicle, edge, users: dict[str, list[str]]) -> tuple[EntryDecision, str]:
        J = self.junction
        cx = J.edges[veh.route[1]]
        light = self.signals.movement_state(edge.arm, cx.movement)
        if light == "R":
            return EntryDecision.STOP, light
        # Blocked if anyone is using a path that conflicts with ours FOR THIS PAIR OF VEHICLES
        # (a bus sweeps more of the box than a bike; see Junction.compute_body_conflicts).
        for other, types in users.items():
            if other != cx.id and any(J.conflicts_for(cx.id, veh.vtype, other, t) for t in types):
                return EntryDecision.STOP, light
        if not self._exit_has_space(veh, cx.target_id):
            return EntryDecision.STOP, light
        if cx.movement != "straight" and self._bike_beside_body(veh, edge.id):
            # A turning vehicle's tail swings outwards. A bus leaves ~5 cm to a bike waiting in
            # the filter gap beside it, so it lets the bike go first (a mirror check).
            return EntryDecision.STOP, light
        walking = self.peds.crossing_arms
        if walking and (edge.arm in walking or self._cx_exit_arm[cx.id] in walking):
            return EntryDecision.STOP, light  # someone is still on a crosswalk this path crosses
        dist = edge.length - veh.s
        stop_dist = veh.v * veh.v / (2.0 * veh.params.comfort_decel)
        if light == "G":
            near = dist <= stop_dist + veh.params.min_gap + COMMIT_MARGIN
            return (EntryDecision.COMMIT if near else EntryDecision.PROCEED), light
        # Amber, the "dilemma zone" rule: go only if stopping would need harder than
        # comfortable braking; otherwise stop.
        return (EntryDecision.COMMIT if dist < stop_dist else EntryDecision.STOP), light

    def _queue_tails(self) -> dict[str, float]:
        """Per filter lane: arc length of the rear of the slowest-moving queue beside it.
        Squeezing between stationary vehicles is slow; once they move off, the gap opens up."""
        qs = self.cfg.metrics.queue_speed
        tails = {}
        for f, sides in self._filter_sides.items():
            rear = math.inf
            for lane_id in sides:
                for veh in self.lanes[lane_id]:
                    if veh.v < qs:
                        rear = min(rear, veh.s - veh.length)
            tails[f] = rear
        return tails

    def _desired_speed(self, veh: Vehicle, edge) -> float:
        v0 = min(veh.desired_speed, edge.speed_limit)
        if edge.kind == "filter" and veh.s >= self._queue_tail.get(edge.id, math.inf) - FILTER_TAIL_MARGIN:
            v0 = min(v0, self.cfg.intersection.filter_speed)  # alongside a queue: ride slowly
        nxt = veh.route_idx + 1
        if nxt < len(veh.route):
            nlim = self.junction.edges[veh.route[nxt]].speed_limit
            if nlim < v0:
                # Slow down smoothly before a turn: the fastest speed from which you can
                # still reach the turn speed using comfortable braking, v^2 = v_t^2 + 2*b*d.
                d = max(edge.length - veh.s, 0.0)
                v0 = min(v0, math.sqrt(nlim * nlim + 2.0 * veh.params.comfort_decel * d))
        return max(v0, 0.1)

    def _move_vehicles(self, dt: float) -> None:
        edges = self.junction.edges
        users = self.connector_users()
        turned_off = self._turned_off_leaders()
        self._queue_tail = self._queue_tails()

        order: list[Vehicle] = []
        gaps, dvs, v0s = [], [], []
        for eid in self.junction.edge_order:
            lane, edge = self.lanes[eid], edges[eid]
            for k, veh in enumerate(lane):
                if k > 0:
                    lead = lane[k - 1]
                    gap, dv = lead.s - lead.length - veh.s, veh.v - lead.v
                else:
                    gap, dv = self._lookahead(veh, edge.length)
                    if edge.is_approach:
                        # Keep following a vehicle that left this lane on another path (see
                        # _turned_off_leaders): the gap is to its rear, measured along the road.
                        to_line = edge.length - veh.s
                        for lead, rear_past_line in turned_off.get(eid, ()):
                            if lead.route[1] != veh.route[1] and to_line + rear_past_line < gap:
                                gap, dv = to_line + rear_past_line, veh.v - lead.v
                    if edge.is_approach and not veh.committed:
                        decision, light = self._entry_decision(veh, edge, users)
                        if decision is EntryDecision.COMMIT:
                            veh.committed, veh.commit_signal = True, light
                            users.setdefault(veh.route[1], []).append(veh.vtype)
                        elif decision is EntryDecision.STOP:
                            buffer = min(STOP_BUFFER, veh.params.min_gap - STOP_MIN_MARGIN)
                            stop_gap = edge.length + buffer - veh.s
                            if stop_gap < gap:
                                gap, dv = stop_gap, veh.v  # stop line is a stationary "leader"
                if edge.kind == "filter":
                    # Every bike in a filter gap (not only the first: the ones behind would
                    # otherwise just follow it past the obstacle) must not overtake a vehicle too
                    # wide to share the gap, nor ride into the swinging tail of one that is
                    # turning off from a lane beside it. Narrow vehicles (cars) leave room.
                    side_gap, side_dv = self._wide_side_gap(veh, eid, turned_off)
                    if side_gap < gap:
                        gap, dv = side_gap, side_dv
                order.append(veh)
                gaps.append(gap)
                dvs.append(dv)
                v0s.append(self._desired_speed(veh, edge))
        if not order:
            return

        p = [veh.params for veh in order]
        v = np.fromiter((veh.v for veh in order), float, len(order))
        acc = idm_acceleration(
            v, np.array(v0s), np.array(gaps), np.array(dvs),
            np.array([q.max_accel for q in p]), np.array([q.comfort_decel for q in p]),
            np.array([q.time_headway for q in p]), np.array([q.min_gap for q in p]),
            np.array([q.accel_exponent for q in p]))
        acc = np.maximum(acc, -np.array([q.max_decel for q in p]))  # brakes have physical limits
        ds, v_new = ballistic_step(v, acc, dt)

        new_lanes: dict[str, list[Vehicle]] = {eid: [] for eid in self.junction.edge_order}
        now = self.time + dt
        lat_step = self.cfg.intersection.lateral_speed * dt
        for veh, a_i, ds_i, v_i in zip(order, acc, ds, v_new):
            veh.a, veh.v = float(a_i), float(v_i)
            if veh.lat_shift:
                veh.lat_shift = math.copysign(max(0.0, abs(veh.lat_shift) - lat_step), veh.lat_shift)
            edge = edges[veh.edge_id]
            new_s = veh.s + float(ds_i)
            if edge.is_approach and not veh.committed and new_s > edge.length - 0.05:
                # Safety net: an uncommitted driver never enters the box (only possible
                # if a conflict appeared after it was too close to brake). Counted.
                new_s, veh.v, veh.a = edge.length - 0.05, 0.0, 0.0
                self.counters.emergency_stops += 1
            veh.s = new_s
            if veh.v < WAIT_SPEED:
                veh.wait_time += dt
                if not veh.is_stopped:
                    veh.is_stopped = True
                    veh.stops += 1
            elif veh.v > RESTART_SPEED:
                veh.is_stopped = False
            finished = False
            while veh.s > edges[veh.edge_id].length:
                veh.s -= edges[veh.edge_id].length
                veh.route_idx += 1
                if veh.route_idx >= len(veh.route):
                    finished = True
                    break
            if finished:
                self._complete_trip(veh, now)
            else:
                new_lanes[veh.edge_id].append(veh)
        for lane in new_lanes.values():
            lane.sort(key=lambda x: -x.s)
        self.lanes = new_lanes

    def _complete_trip(self, veh: Vehicle, now: float) -> None:
        del self.vehicles[veh.id]
        travel = now - veh.arrival_time  # includes any time spent in the backlog
        c = self.counters
        c.completed += 1
        c.total_travel_time += travel
        c.total_wait_time += veh.wait_time
        self.recent_trips.append(TripRecord(
            id=veh.id, vtype=veh.vtype, origin=veh.origin, movement=veh.movement,
            exit_arm=self.junction.edges[veh.route[-1]].arm,
            arrival_time=round(veh.arrival_time, 3), insert_time=round(veh.insert_time, 3),
            exit_time=round(now, 3), travel_time=round(travel, 3),
            entry_delay=round(veh.insert_time - veh.arrival_time, 3),
            wait_time=round(veh.wait_time, 3),
            delay=round(max(0.0, travel - veh.free_flow_time), 3), stops=veh.stops,
            filtered=veh.filtered))

    # ================================================================== outputs
    def rear_point(self, veh: Vehicle) -> tuple[float, float]:
        """Where the vehicle's rear is: ``length`` metres back along its route (may be the
        previous edge, e.g. a bus whose nose is in the box while its tail is still on the lane)."""
        idx, s = veh.route_idx, veh.s - veh.length
        while s < 0 and idx > 0:
            idx -= 1
            s += self.junction.edges[veh.route[idx]].length
        x, y, _ = self.junction.edges[veh.route[idx]].path.pose_at(max(s, 0.0))
        return x, y

    def lane_poses(self, eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, heading) of every vehicle on one edge, front first, including the drawing-only
        sideways offset of bikes moving into a filter gap.

        x, y is the front bumper. On a curve the heading is that of the CHORD from the rear point
        to the front point, not the path's direction at the bumper: a body drawn along the
        tangent at its nose swings its tail far off the road on a tight turn (a 12 m bus by
        several metres). Along the chord, front and rear both sit on the path, the way a real
        vehicle's axles do. On straight lanes the two are identical."""
        lane = self.lanes[eid]
        xs, ys, hs = self.junction.edges[eid].path.poses_at([veh.s for veh in lane])
        edge = self.junction.edges[eid]
        if edge.kind == "connector" or any(veh.s < veh.length and veh.route_idx > 0 for veh in lane):
            hs = hs.copy()
            for i, veh in enumerate(lane):
                if edge.kind == "connector" or (veh.s < veh.length and veh.route_idx > 0):
                    rx, ry = self.rear_point(veh)
                    if math.hypot(xs[i] - rx, ys[i] - ry) > 1e-6:
                        hs[i] = math.atan2(ys[i] - ry, xs[i] - rx)
        lat = np.array([veh.lat_shift for veh in lane])
        if lat.any():
            xs, ys = xs - np.sin(hs) * lat, ys + np.cos(hs) * lat
        return xs, ys, hs

    def state(self) -> dict:
        """World snapshot for renderers and debugging (positions are derived, not stored)."""
        out = []
        for eid in self.junction.edge_order:
            lane = self.lanes[eid]
            if not lane:
                continue
            xs, ys, hs = self.lane_poses(eid)
            for veh, x, y, h in zip(lane, xs, ys, hs):
                out.append({"id": veh.id, "type": veh.vtype, "x": round(float(x), 3),
                            "y": round(float(y), 3), "heading": round(float(h), 4),
                            "v": round(veh.v, 3), "edge": eid})
        return {
            "t": round(self.time, 3),
            "signals": self.signals.arm_states(self.junction.inbound),
            "ped_signal": self.signals.ped_signal(),
            "phase": self.signals.phase_name,
            "stage": self.signals.stage.value,
            "vehicles": out,
            "pedestrians": [{"id": p.id, "x": round(float(p.pos[0]), 3), "y": round(float(p.pos[1]), 3),
                             "state": p.state.name.lower(), "crosswalk": p.arm} for p in self.peds.peds],
        }

    def summary(self) -> dict:
        c = self.counters
        hours = max(self.time, 1e-9) / 3600.0
        return {
            "time_s": round(self.time, 1),
            "in_network": len(self.vehicles),
            "backlog": self.spawner.backlog_size(),
            "spawned": c.spawned,
            "completed": c.completed,
            "throughput_vph": round(c.completed / hours, 1),
            "mean_travel_time_s": round(c.total_travel_time / c.completed, 2) if c.completed else None,
            "mean_wait_time_s": round(c.total_wait_time / c.completed, 2) if c.completed else None,
            "emergency_stops": c.emergency_stops,
            "peds_spawned": self.peds.spawned,
            "peds_completed": self.peds.completed,
            "ped_phases": self.signals.ped_phases_served,
        }

    def fingerprint(self) -> str:
        """Hash of the full dynamic state, used to prove determinism in tests."""
        import hashlib
        h = hashlib.sha256()
        for eid in self.junction.edge_order:
            for veh in self.lanes[eid]:
                h.update(f"{veh.id}:{eid}:{veh.s:.9f}:{veh.v:.9f};".encode())
        for p in self.peds.peds:
            h.update(f"p{p.id}:{p.state.value}:{p.pos[0]:.9f}:{p.pos[1]:.9f};".encode())
        h.update(f"{self.signals.phase_idx}:{self.signals.stage.value}:{self.tick}".encode())
        return h.hexdigest()
