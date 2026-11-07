"""Headless simulation engine. No rendering, no I/O, no wall-clock time.

One call to ``step()`` advances the world by ``dt`` seconds:

  1. signal policy decides; signal controller advances (green/amber/all-red)
  2. spawner samples arrivals; queued arrivals are inserted where there is room
  3. every driver finds its leader (a vehicle, or a red stop line)
  4. IDM gives every acceleration in one vectorised NumPy call
  5. positions/speeds integrate; vehicles move to the next edge or despawn

Determinism: all randomness comes from one seeded ``np.random.Generator``, and
every loop iterates in a fixed order (sorted arms, fixed edge order, vehicles
sorted by position). Same seed + same config gives a bit-identical run.

Junction entry uses a RESERVATION rule (the "box-junction" rule, as used in many
real intersections and in SUMO's junction model): a driver at the stop line
commits to entering only if
  (a) the light is green (or amber and they can no longer stop comfortably),
  (b) no vehicle is on, or has reserved, a conflicting path through the box, and
  (c) the exit lane has room for them, so they never block the box.
Once committed, the decision is final, like a real driver past the point of no return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Config, VehicleTypeParams
from .idm import ballistic_step, idm_acceleration
from .junction import Junction
from .signals import FixedTimePolicy, SignalController, SignalPolicy
from .spawner import PendingVehicle, Spawner
from .vehicle import TripRecord, Vehicle, scale_for_aggressiveness

WAIT_SPEED = 0.1      # m/s; below this a vehicle counts as "waiting" (same threshold as SUMO)
STOP_BUFFER = 1.0     # virtual red-light obstacle sits this far past the stop line, so the
                      # front bumper rests about (s0 - 1) m before the line
COMMIT_MARGIN = 1.0   # extra metres of look-ahead when deciding to commit on green
INSERT_CLEARANCE = 0.5
RESTART_SPEED = 1.0   # a stopped vehicle must exceed this before a new halt counts as another stop
                      # (hysteresis: creeping forward in a queue is not a new stop)


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
            config.junction_path(), ix.turn_lateral_accel, ix.conflict_distance_m)
        sg = config.signals
        self.signals = SignalController(self.junction.phases, sg.amber_s, sg.all_red_s, sg.min_green_s)
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

    def _refresh_params(self) -> None:
        agg = self.cfg.driver.aggressiveness
        self._params_cache = {name: scale_for_aggressiveness(p, agg)
                              for name, p in self.cfg.vehicle_types.items()}

    # ================================================================== live settings
    def apply_settings(self, *, volume_vph: float | None = None, aggressiveness: float | None = None,
                       green_s: float | None = None, source: str = "user") -> dict:
        """Change settings mid-run. Returns the changes that were applied.

        volume_vph: mean demand per approach; per-approach ratios from the config are kept
                    (so a busier N-S corridor stays busier).
        green_s:    green time for every phase of the fixed-time plan (from the next switch).
        Raises ValueError on invalid values, leaving the sim unchanged.
        """
        changes: dict = {}
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
        self._insert_from_backlog()
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
        # Drivers pick the allowed lane with the most free road (ties: lower index).
        lane_id = max(candidates, key=lambda lid: (self._entry_space(lid), -candidates.index(lid)))
        space = self._entry_space(lane_id)
        if space < params.length + params.min_gap + INSERT_CLEARANCE:
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

    # ================================================================== driving
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

    def _entry_decision(self, veh: Vehicle, edge, active_cx: set[str]) -> tuple[EntryDecision, str]:
        cx = self.junction.edges[veh.route[1]]
        light = self.signals.movement_state(edge.arm, cx.movement)
        if light == "R":
            return EntryDecision.STOP, light
        if self.junction.conflicts[cx.id] & active_cx or not self._exit_has_space(veh, cx.target_id):
            return EntryDecision.STOP, light
        dist = edge.length - veh.s
        stop_dist = veh.v * veh.v / (2.0 * veh.params.comfort_decel)
        if light == "G":
            near = dist <= stop_dist + veh.params.min_gap + COMMIT_MARGIN
            return (EntryDecision.COMMIT if near else EntryDecision.PROCEED), light
        # Amber, the "dilemma zone" rule: go only if stopping would need harder than
        # comfortable braking; otherwise stop.
        return (EntryDecision.COMMIT if dist < stop_dist else EntryDecision.STOP), light

    def _desired_speed(self, veh: Vehicle, edge) -> float:
        v0 = min(veh.desired_speed, edge.speed_limit)
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
        # Connectors currently in use: occupied, or reserved by a committed driver.
        active_cx = {eid for eid in self.junction.edge_order
                     if edges[eid].kind == "connector" and self.lanes[eid]}
        for e in self.junction.inbound_edges():
            lane = self.lanes[e.id]
            if lane and lane[0].committed:
                active_cx.add(lane[0].route[1])

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
                    if edge.kind == "inbound" and not veh.committed:
                        decision, light = self._entry_decision(veh, edge, active_cx)
                        if decision is EntryDecision.COMMIT:
                            veh.committed, veh.commit_signal = True, light
                            active_cx.add(veh.route[1])
                        elif decision is EntryDecision.STOP:
                            stop_gap = edge.length + STOP_BUFFER - veh.s
                            if stop_gap < gap:
                                gap, dv = stop_gap, veh.v  # stop line is a stationary "leader"
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
        for veh, a_i, ds_i, v_i in zip(order, acc, ds, v_new):
            veh.a, veh.v = float(a_i), float(v_i)
            edge = edges[veh.edge_id]
            new_s = veh.s + float(ds_i)
            if edge.kind == "inbound" and not veh.committed and new_s > edge.length - 0.05:
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
            delay=round(max(0.0, travel - veh.free_flow_time), 3), stops=veh.stops))

    # ================================================================== outputs
    def state(self) -> dict:
        """World snapshot for renderers and debugging (positions are derived, not stored)."""
        out = []
        for eid in self.junction.edge_order:
            lane = self.lanes[eid]
            if not lane:
                continue
            xs, ys, hs = self.junction.edges[eid].path.poses_at([veh.s for veh in lane])
            for veh, x, y, h in zip(lane, xs, ys, hs):
                out.append({"id": veh.id, "type": veh.vtype, "x": round(float(x), 3),
                            "y": round(float(y), 3), "heading": round(float(h), 4),
                            "v": round(veh.v, 3), "edge": eid})
        return {
            "t": round(self.time, 3),
            "signals": self.signals.arm_states(self.junction.inbound),
            "phase": self.signals.phase.name,
            "stage": self.signals.stage.value,
            "vehicles": out,
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
        }

    def fingerprint(self) -> str:
        """Hash of the full dynamic state, used to prove determinism in tests."""
        import hashlib
        h = hashlib.sha256()
        for eid in self.junction.edge_order:
            for veh in self.lanes[eid]:
                h.update(f"{veh.id}:{eid}:{veh.s:.9f}:{veh.v:.9f};".encode())
        h.update(f"{self.signals.phase_idx}:{self.signals.stage.value}:{self.tick}".encode())
        return h.hexdigest()
