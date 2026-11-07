"""Traffic metrics and notable-event detection. Pure computation, no file I/O.

``MetricsCollector`` watches a Simulation (observer pattern) and produces:
  * one row of metrics per simulated second (queues, waits, throughput, LOS, ...)
  * events: signal phase changes, plus *episodes* (queue spikes, spillback,
    gridlock) that have a start and an end
  * a run summary

The same collector feeds the run logger (Phase 2), the live dashboard (Phase 3)
and the RL reward/observation (Phase 7), so every part of the system uses one
definition of "queue" or "delay".

Definitions:
  queued vehicle  - on an approach lane (inbound or bike filter lane), moving slower than
                    ``queue_speed`` (2 m/s)
  queue length    - metres from the stop line to the rear of the farthest-back queued vehicle
  delay           - travel time minus free-flow travel time ("control delay")
  Level of Service- HCM signalised-intersection bands on mean control delay:
                    A<=10s, B<=20, C<=35, D<=55, E<=80, F>80
  gridlock        - no vehicle has left the junction for ``gridlock_window_s`` while at least
                    ``gridlock_min_vehicles`` are present and >=90% of them are stopped
  pedestrian wait - seconds a walker stood at the kerb before starting to cross
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .config import MetricsSettings
from .engine import WAIT_SPEED
from .pedestrian import PedState, PedTripRecord
from .vehicle import TripRecord

LOS_BANDS = ((10.0, "A"), (20.0, "B"), (35.0, "C"), (55.0, "D"), (80.0, "E"))
GRIDLOCK_CLEAR_S = 10.0  # gridlock ends once a vehicle has exited within this many seconds


def level_of_service(delay: float | None) -> str | None:
    if delay is None:
        return None
    for limit, letter in LOS_BANDS:
        if delay <= limit:
            return letter
    return "F"


def clock(t: float) -> str:
    """Sim seconds as m:ss, the way a person would ask: 'what happened at 3:20?'"""
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


@dataclass(slots=True)
class LaneQueue:
    vehicles: int = 0
    queued: int = 0
    queue_m: float = 0.0
    wait_sum: float = 0.0  # accumulated wait of every vehicle on the lane (RL uses this)


def lane_queues(sim, queue_speed: float) -> dict[str, LaneQueue]:
    """Queue state of every approach lane. Cheap enough to call every RL decision."""
    out: dict[str, LaneQueue] = {}
    for e in sim.junction.approach_edges():
        q = LaneQueue()
        for veh in sim.lanes[e.id]:
            q.vehicles += 1
            q.wait_sum += veh.wait_time
            if veh.v < queue_speed:
                q.queued += 1
                q.queue_m = max(q.queue_m, min(e.length, e.length - (veh.s - veh.length)))
        out[e.id] = q
    return out


@dataclass
class _Episode:
    id: int
    start: float
    peak: float
    peak_t: float


class _Snapshots:
    """Per-second history of a cumulative counter, so we can ask 'how many in the last N s?'"""

    def __init__(self, initial, keep_s: float):
        self.keep_s = keep_s
        self.items: deque = deque([(0.0, initial)])

    def push(self, t: float, value) -> None:
        self.items.append((t, value))
        while len(self.items) > 1 and self.items[1][0] < t - self.keep_s:
            self.items.popleft()

    def at_or_after(self, t: float):
        """Value of the oldest snapshot taken at or after ``t``."""
        for ts, v in self.items:
            if ts >= t - 1e-9:
                return ts, v
        return self.items[-1]


class MetricsCollector:
    def __init__(self, sim, settings: MetricsSettings, history_s: int = 900):
        self.s = settings
        self.junction = sim.junction
        self.arms = sorted(sim.junction.inbound)
        self.lane_ids = [e.id for e in sim.junction.approach_edges()]
        self.lane_arm = {e.id: e.arm for e in sim.junction.approach_edges()}
        self.vtypes = sorted(sim.cfg.vehicle_types)
        self.phase_arms = {p.name: sorted(p.green) for p in sim.junction.phases}
        self.steps_per_sample = max(1, round(1.0 / sim.dt))

        keep = max(settings.window_s, settings.gridlock_window_s, 60.0) + 1
        self._completed = _Snapshots(0, keep)
        self._arrivals = _Snapshots(dict(sim.spawner.arrivals_total), keep)
        self._prev_arrivals = dict(sim.spawner.arrivals_total)
        self._trip_window: deque[tuple[float, float]] = deque()  # (exit_time, delay)
        self._ped_window: deque[tuple[float, float]] = deque()   # (cross_start, wait)
        self._walk_idx = len(sim.peds.walk_log)
        self._sig_idx = 0
        self._ctl_idx = len(sim.control_events)
        self._last_green: dict[str, float | None] = {a: None for a in self.arms}
        self._episodes: dict[tuple[str, str], _Episode] = {}
        self._next_event_id = 0
        self._los: str | None = None

        self.history: deque[dict] = deque(maxlen=history_s)  # recent rows for live charts
        self.latest: dict | None = None
        self.trips: list[TripRecord] = []
        self.ped_trips: list[PedTripRecord] = []
        self.ped_waits: list[float] = []  # every crossing started, for the run summary
        self.event_counts: Counter = Counter()
        self.max_queue: dict[str, tuple[int, float]] = {a: (0, 0.0) for a in self.arms}
        self.columns = self._columns()

    # ------------------------------------------------------------------ schema
    def _columns(self) -> list[str]:
        cols = ["t", "clock", "phase", "stage", "time_in_stage", "in_network", "backlog",
                "completed", "arrivals", "throughput_vph", "mean_speed", "stopped_frac",
                "mean_wait_active", "max_wait_active", "mean_delay_window", "los", "queue_total",
                "ped_signal", "peds_in_network", "peds_waiting", "peds_crossing",
                "ped_wait_window", "ped_longest_wait"]
        cols += [f"veh_{t}" for t in self.vtypes]
        for a in self.arms:
            cols += [f"arrivals_{a}", f"queue_{a}", f"wait_{a}", f"backlog_{a}"]
        for lid in self.lane_ids:
            cols += [f"q_{lid}", f"qm_{lid}"]
        return cols

    # ------------------------------------------------------------------ observer hook
    def after_step(self, sim) -> tuple[dict | None, list[dict], list[TripRecord], list[PedTripRecord]]:
        """Returns (per-second row or None, new events, vehicle trips and pedestrian trips
        finished this step)."""
        events: list[dict] = []
        trips = sim.recent_trips
        for tr in trips:
            self._trip_window.append((tr.exit_time, tr.delay))
            self.trips.append(tr)
        ped_trips = sim.peds.recent_trips
        self.ped_trips.extend(ped_trips)
        for t0, wait in sim.peds.recent_starts:
            self._ped_window.append((t0, wait))
            self.ped_waits.append(wait)
        for w in sim.peds.walk_log[self._walk_idx:]:
            # Why the cycle just got longer, and for whom: the analyst can cite this.
            events.append(self._event("ped_phase", w["t"], {"waiting": w["waiting"], "max_wait": w["max_wait"]}))
        self._walk_idx = len(sim.peds.walk_log)

        new = sim.signals.events[self._sig_idx:]
        self._sig_idx = len(sim.signals.events)
        for pe in new:
            if pe.stage == "green":
                for arm in self.phase_arms[pe.phase]:
                    self._last_green[arm] = pe.time
            events.append(self._event("phase_change", pe.time, {"phase": pe.phase, "stage": pe.stage}))
        new_ctl = sim.control_events[self._ctl_idx:]
        self._ctl_idx = len(sim.control_events)
        for ce in new_ctl:
            events.append(self._event("config_change", ce["t"],
                                      {"source": ce["source"], "changes": ce["changes"]}))

        row = None
        if sim.tick % self.steps_per_sample == 0:
            row = self._sample(sim, events)
            self.latest = row
            self.history.append(row)
        return row, events, trips, ped_trips

    # ------------------------------------------------------------------ sampling
    def _sample(self, sim, events: list[dict]) -> dict:
        t = sim.time
        s = self.s
        while self._trip_window and self._trip_window[0][0] < t - s.window_s:
            self._trip_window.popleft()
        while self._ped_window and self._ped_window[0][0] < t - s.window_s:
            self._ped_window.popleft()
        mean_delay = float(np.mean([d for _, d in self._trip_window])) if self._trip_window else None
        ped_wait = float(np.mean([w for _, w in self._ped_window])) if self._ped_window else None

        completed = sim.counters.completed
        self._completed.push(t, completed)
        arrivals_now = dict(sim.spawner.arrivals_total)
        self._arrivals.push(t, arrivals_now)
        t0, c0 = self._completed.at_or_after(t - s.window_s)
        throughput = (completed - c0) / max(t - t0, 1e-9) * 3600.0 if t > t0 else 0.0

        speeds = np.fromiter((v.v for v in sim.vehicles.values()), float, len(sim.vehicles))
        waits = np.fromiter((v.wait_time for v in sim.vehicles.values()), float, len(sim.vehicles))
        stopped_frac = float((speeds < WAIT_SPEED).mean()) if len(speeds) else 0.0
        lq = lane_queues(sim, s.queue_speed)
        los = level_of_service(mean_delay)

        row = {
            "t": round(t, 1), "clock": clock(t),
            "phase": sim.signals.phase_name, "stage": sim.signals.stage.value,
            "time_in_stage": round(sim.signals.time_in_stage, 1),
            "in_network": len(sim.vehicles), "backlog": sim.spawner.backlog_size(),
            "completed": completed,
            "arrivals": sum(arrivals_now.values()) - sum(self._prev_arrivals.values()),
            "throughput_vph": round(throughput, 1),
            "mean_speed": round(float(speeds.mean()), 3) if len(speeds) else 0.0,
            "stopped_frac": round(stopped_frac, 3),
            "mean_wait_active": round(float(waits.mean()), 2) if len(waits) else 0.0,
            "max_wait_active": round(float(waits.max()), 2) if len(waits) else 0.0,
            "mean_delay_window": round(mean_delay, 2) if mean_delay is not None else "",
            "los": los or "",
            "queue_total": sum(q.queued for q in lq.values()),
            "ped_signal": sim.signals.ped_signal(),
            "peds_in_network": len(sim.peds.peds),
            "peds_waiting": sim.peds.count(PedState.WAIT),
            "peds_crossing": sim.peds.count(PedState.CROSS),
            "ped_wait_window": round(ped_wait, 2) if ped_wait is not None else "",
            "ped_longest_wait": round(sim.peds.longest_wait(t), 1),
        }
        by_type = Counter(v.vtype for v in sim.vehicles.values())
        for vt in self.vtypes:
            row[f"veh_{vt}"] = by_type.get(vt, 0)
        for a in self.arms:
            lanes = [lq[lid] for lid in sim.junction.approach_lanes(a)]
            n_veh = sum(q.vehicles for q in lanes)
            queued = sum(q.queued for q in lanes)
            row[f"arrivals_{a}"] = arrivals_now[a] - self._prev_arrivals[a]
            row[f"queue_{a}"] = queued
            row[f"wait_{a}"] = round(sum(q.wait_sum for q in lanes) / n_veh, 2) if n_veh else 0.0
            row[f"backlog_{a}"] = len(sim.spawner.backlog[a])
            if queued > self.max_queue[a][0]:
                self.max_queue[a] = (queued, round(t, 1))
        for lid in self.lane_ids:
            row[f"q_{lid}"] = lq[lid].queued
            row[f"qm_{lid}"] = round(lq[lid].queue_m, 1)
        self._prev_arrivals = arrivals_now

        self._detect(sim, row, lq, events)
        if los != self._los and los is not None:
            events.append(self._event("los_change", t, {
                "from": self._los, "to": los, "mean_delay_window": row["mean_delay_window"]}))
            self._los = los
        return row

    # ------------------------------------------------------------------ event detection
    def _detect(self, sim, row: dict, lq: dict, events: list[dict]) -> None:
        t, s = sim.time, self.s
        arr_t, arr_base = self._arrivals.at_or_after(t - 60.0)
        for a in self.arms:
            def ctx(a=a):
                last = self._last_green[a]
                return {"phase": row["phase"], "stage": row["stage"],
                        "approach_light": sim.signals.movement_state(a, "straight"),
                        "seconds_since_approach_green": round(t - last, 1) if last is not None else None,
                        "arrivals_last_60s": sim.spawner.arrivals_total[a] - arr_base[a],
                        "backlog": row[f"backlog_{a}"], "wait_mean": row[f"wait_{a}"]}
            self._hysteresis("queue_spike", "approach", a, row[f"queue_{a}"],
                             s.queue_spike_on, s.queue_spike_off, t, events, ctx)
        for lid in self.lane_ids:
            frac = lq[lid].queue_m / self.junction.edges[lid].length
            self._hysteresis("spillback", "lane", lid, round(frac, 3),
                             s.spillback_on_frac, s.spillback_off_frac, t, events,
                             lambda lid=lid: {"approach": self.lane_arm[lid], "queue_m": row[f"qm_{lid}"],
                                              "phase": row["phase"]})

        # Gridlock is not a single threshold, so it gets its own start/end conditions.
        _, c_base = self._completed.at_or_after(t - s.gridlock_window_s)
        _, c_recent = self._completed.at_or_after(t - GRIDLOCK_CLEAR_S)
        completed = sim.counters.completed
        key = ("gridlock", "network")
        ep = self._episodes.get(key)
        if ep is None:
            if (t >= s.gridlock_window_s and completed - c_base == 0
                    and row["in_network"] >= s.gridlock_min_vehicles
                    and row["stopped_frac"] >= s.gridlock_stopped_frac):
                ep = _Episode(self._next_event_id, t, row["in_network"], t)
                self._episodes[key] = ep
                events.append(self._event("gridlock", t, {
                    "status": "start", "episode": ep.id, "in_network": row["in_network"],
                    "backlog": row["backlog"], "stopped_frac": row["stopped_frac"],
                    "phase": row["phase"], "stage": row["stage"]}))
        else:
            if row["in_network"] > ep.peak:
                ep.peak, ep.peak_t = row["in_network"], t
            if completed - c_recent > 0 or row["in_network"] == 0:
                events.append(self._end_event("gridlock", key, ep, t, {}))

    def _hysteresis(self, kind, key_name, key, value, on, off, t, events, context) -> None:
        """Start an episode when ``value >= on``, end it when ``value <= off``.
        The gap between on and off stops a value hovering at the threshold from
        producing dozens of start/end pairs."""
        k = (kind, key)
        ep = self._episodes.get(k)
        if ep is None:
            if value >= on:
                ep = _Episode(self._next_event_id, t, value, t)
                self._episodes[k] = ep
                events.append(self._event(kind, t, {"status": "start", "episode": ep.id,
                                                    key_name: key, "value": value, **context()}))
        else:
            if value > ep.peak:
                ep.peak, ep.peak_t = value, t
            if value <= off:
                events.append(self._end_event(kind, k, ep, t, {key_name: key}))

    def _end_event(self, kind, k, ep: _Episode, t: float, extra: dict, truncated=False) -> dict:
        del self._episodes[k]
        return self._event(kind, t, {"status": "end", "episode": ep.id, **extra,
                                     "started": round(ep.start, 1), "started_clock": clock(ep.start),
                                     "duration_s": round(t - ep.start, 1), "peak": ep.peak,
                                     "peak_t": round(ep.peak_t, 1), "peak_clock": clock(ep.peak_t),
                                     **({"truncated": True} if truncated else {})})

    def _event(self, kind: str, t: float, fields: dict) -> dict:
        ev = {"id": self._next_event_id, "type": kind, "t": round(t, 1), "clock": clock(t), **fields}
        self._next_event_id += 1
        if fields.get("status") != "end":
            self.event_counts[kind] += 1
        return ev

    def flush_open_episodes(self, sim) -> list[dict]:
        """Close any episode still running when the run stops (marked truncated)."""
        out = []
        for (kind, key), ep in list(self._episodes.items()):
            key_name = {"queue_spike": "approach", "spillback": "lane"}.get(kind)
            extra = {key_name: key} if key_name else {}
            out.append(self._end_event(kind, (kind, key), ep, sim.time, extra, truncated=True))
        return out

    # ------------------------------------------------------------------ summary
    def summary(self, sim) -> dict:
        trips = self.trips
        hours = max(sim.time, 1e-9) / 3600.0

        def stats(ts: list[TripRecord]) -> dict:
            if not ts:
                return {"trips": 0}
            d = np.array([x.delay for x in ts])
            return {"trips": len(ts),
                    "mean_delay_s": round(float(d.mean()), 2),
                    "p95_delay_s": round(float(np.percentile(d, 95)), 2),
                    "mean_wait_s": round(float(np.mean([x.wait_time for x in ts])), 2),
                    "los": level_of_service(float(d.mean()))}

        out = {
            "sim_time_s": round(sim.time, 1),
            "spawned": sim.counters.spawned,
            "completed": len(trips),
            "in_network_end": len(sim.vehicles),
            "backlog_end": sim.spawner.backlog_size(),
            "throughput_vph": round(len(trips) / hours, 1),
            "emergency_stops": sim.counters.emergency_stops,
            **{k: v for k, v in stats(trips).items() if k != "trips"},
        }
        if trips:
            out["mean_travel_time_s"] = round(float(np.mean([x.travel_time for x in trips])), 2)
            out["mean_entry_delay_s"] = round(float(np.mean([x.entry_delay for x in trips])), 2)
            out["mean_stops"] = round(float(np.mean([x.stops for x in trips])), 2)
        out["per_type"] = {vt: stats([x for x in trips if x.vtype == vt]) for vt in self.vtypes}
        filtered = [x for x in trips if x.filtered]
        if filtered:
            out["per_type"]["bike"]["filtered_trips"] = len(filtered)
            out["per_type"]["bike"]["filtered_mean_delay_s"] = round(float(np.mean([x.delay for x in filtered])), 2)
        pw = np.array(self.ped_waits) if self.ped_waits else None
        out["pedestrians"] = {
            "spawned": sim.peds.spawned,
            "completed": len(self.ped_trips),
            "crossings_started": len(self.ped_waits),
            "mean_wait_s": round(float(pw.mean()), 2) if pw is not None else None,
            "p95_wait_s": round(float(np.percentile(pw, 95)), 2) if pw is not None else None,
            "max_wait_s": round(float(pw.max()), 2) if pw is not None else None,
            "mean_crossing_s": (round(float(np.mean([x.crossing_time for x in self.ped_trips])), 2)
                                if self.ped_trips else None),
            "ped_phases": sim.signals.ped_phases_served,
        }
        out["per_approach"] = {
            a: {**stats([x for x in trips if x.origin == a]),
                "max_queue": self.max_queue[a][0], "max_queue_t": self.max_queue[a][1],
                "max_queue_clock": clock(self.max_queue[a][1])}
            for a in self.arms}
        out["event_counts"] = dict(sorted(self.event_counts.items()))
        return out
