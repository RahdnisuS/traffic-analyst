"""Structured run logging to data/runs/<run_id>/.

Layout of one run directory:
  manifest.json  - who/what/when: seed, full config, junction spec, git commit,
                   metric definitions, status, and the final summary
  metrics.csv    - one row per simulated second (see MetricsCollector.columns)
  events.jsonl   - one JSON object per line: phase changes, queue spikes,
                   spillback, gridlock, LOS changes (episodes have start + end)
  trips.csv      - one row per completed vehicle trip (with vehicle type)
  peds.csv       - one row per completed pedestrian trip (kerb wait, crossing time)

WHY these formats: CSV loads directly into pandas/LightGBM (Phase 9). JSONL
is append-only, so a crash loses at most the last line, and each line is
self-contained for chunking into the RAG knowledge base (Phase 11). The manifest
makes every run reproducible: config + seed + commit is enough to re-run it exactly.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT
from .metrics import MetricsCollector
from .pedestrian import PedTripRecord
from .vehicle import TripRecord

SCHEMA_VERSION = 2  # 2: Phase 4 added vehicle types, pedestrians, peds.csv and the PED phase

METRIC_DEFINITIONS = {
    "queued_vehicle": "vehicle on an inbound lane moving slower than metrics.queue_speed (m/s)",
    "queue_m": "metres from the stop line to the rear of the farthest-back queued vehicle",
    "wait_time": "seconds a vehicle spent on the road at speed below 0.1 m/s",
    "delay": "travel time minus free-flow travel time (control delay), seconds",
    "entry_delay": "seconds a vehicle waited off-network because its entry lane was full",
    "throughput_vph": "vehicles leaving the junction per hour over the rolling window",
    "los": "HCM Level of Service from mean control delay: A<=10s B<=20 C<=35 D<=55 E<=80 F>80",
    "queue_spike": "episode while an approach has >= queue_spike_on queued vehicles (ends at <= queue_spike_off)",
    "spillback": "episode while a lane's queue tail is beyond spillback_on_frac of the lane length",
    "config_change": "a setting changed mid-run (e.g. a dashboard slider); 'changes' lists the new values",
    "gridlock": "episode while no vehicle has exited for gridlock_window_s and >= gridlock_stopped_frac are stopped",
    "wait_<arm>": "mean accumulated wait of vehicles currently on that approach's lanes (incl. bike filter lane)",
    "filter_lane": "flt_<arm>_<k>: virtual bike lane on the divider between inbound lanes k and k+1",
    "filtered": "trips.csv: the bike rode a filter lane past queued vehicles to the stop line",
    "PED": "phase name during the exclusive pedestrian phase: every vehicle movement is red",
    "ped_signal": "walk | flash (flashing DON'T WALK: finish crossing, don't start) | dont_walk",
    "ped_phase": "event when a pedestrian phase starts: 'waiting' walkers, longest 'max_wait' (s)",
    "ped_wait": "seconds a pedestrian stood at the kerb before starting to cross",
    "ped_wait_window": "mean ped_wait of crossings started in the rolling window",
    "veh_<type>": "vehicles of that type currently in the network",
}


def _git_commit() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=3)
        if rev.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3)
        return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None  # logging must never fail just because git is missing


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written manifest."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class RunLogger:
    """Observer that records one simulation run. Use as a context manager:

        with RunLogger(sim, controller="fixed") as log:
            sim.run(600)
        print(log.dir)
    """

    def __init__(self, sim, controller: str = "fixed", runs_dir: str | Path | None = None,
                 run_id: str | None = None, tags: dict | None = None,
                 collector: MetricsCollector | None = None, attach: bool = True):
        """``attach=False``: the caller (e.g. the dashboard session, which also streams the
        same rows to browsers) drives the collector and passes results to ``record()``."""
        self.sim = sim
        cfg = sim.cfg
        self.collector = collector or MetricsCollector(sim, cfg.metrics)
        self._attached = attach
        base = Path(runs_dir) if runs_dir is not None else cfg.runs_path()
        base.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        self.run_id = run_id or self._unique_id(base, f"{now:%Y%m%d-%H%M%S}_{controller}_s{sim.seed}")
        self.dir = base / self.run_id
        self.dir.mkdir(exist_ok=False)
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "created_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "controller": controller,
            "seed": sim.seed,
            "dt": sim.dt,
            "git_commit": _git_commit(),
            "tags": tags or {},
            "junction": {"name": sim.junction.name, "file": cfg.sim.junction,
                         "spec": sim.junction.spec.model_dump(mode="json")},
            "config": cfg.model_dump(mode="json"),
            "metric_definitions": METRIC_DEFINITIONS,
            "files": {"metrics": "metrics.csv", "events": "events.jsonl", "trips": "trips.csv",
                      "pedestrians": "peds.csv"},
            "summary": None,
        }
        _write_json_atomic(self.dir / "manifest.json", self.manifest)

        self._metrics_fh = open(self.dir / "metrics.csv", "w", newline="", encoding="utf-8")
        self._metrics = csv.DictWriter(self._metrics_fh, fieldnames=self.collector.columns)
        self._metrics.writeheader()
        self._events_fh = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self._trips_fh = open(self.dir / "trips.csv", "w", newline="", encoding="utf-8")
        self._trips = csv.DictWriter(self._trips_fh,
                                     fieldnames=[f.name for f in dataclasses.fields(TripRecord)])
        self._trips.writeheader()
        self._peds_fh = open(self.dir / "peds.csv", "w", newline="", encoding="utf-8")
        self._peds = csv.DictWriter(self._peds_fh,
                                    fieldnames=[f.name for f in dataclasses.fields(PedTripRecord)])
        self._peds.writeheader()
        self.closed = False
        if attach:
            sim.observers.append(self)

    @staticmethod
    def _unique_id(base: Path, stem: str) -> str:
        rid, n = stem, 2
        while (base / rid).exists():
            rid, n = f"{stem}-{n}", n + 1
        return rid

    # ------------------------------------------------------------------ observer
    def after_step(self, sim) -> None:
        self.record(*self.collector.after_step(sim))

    def record(self, row: dict | None, events: list[dict], trips: list[TripRecord],
               ped_trips: list[PedTripRecord] = ()) -> None:
        self._write_events(events)
        for tr in trips:
            self._trips.writerow(dataclasses.asdict(tr))
        for pt in ped_trips:
            self._peds.writerow(dataclasses.asdict(pt))
        if row is not None:
            self._metrics.writerow(row)

    def _write_events(self, events: list[dict]) -> None:
        for ev in events:
            self._events_fh.write(json.dumps(ev, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------ lifecycle
    def close(self, status: str = "completed", error: str | None = None) -> dict:
        if self.closed:
            return self.manifest
        self._write_events(self.collector.flush_open_episodes(self.sim))
        for fh in (self._metrics_fh, self._events_fh, self._trips_fh, self._peds_fh):
            fh.close()
        if self in self.sim.observers:
            self.sim.observers.remove(self)
        self.manifest.update(
            status=status, error=error,
            ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sim_duration_s=round(self.sim.time, 1),
            summary=self.collector.summary(self.sim))
        _write_json_atomic(self.dir / "manifest.json", self.manifest)
        self.closed = True
        return self.manifest

    def discard(self) -> None:
        """Delete this run's directory (used for trivially short dashboard runs, e.g. a
        reset two seconds after starting), so data/runs/ isn't cluttered with noise."""
        for fh in (self._metrics_fh, self._events_fh, self._trips_fh, self._peds_fh):
            fh.close()
        if self in self.sim.observers:
            self.sim.observers.remove(self)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.closed = True

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Keep partial logs: a crashed run is still evidence worth inspecting.
            self.close(status="failed", error=f"{exc_type.__name__}: {exc}")


def list_runs(runs_dir: str | Path) -> list[dict]:
    """Lightweight index of logged runs, newest first (used by the server in Phase 3)."""
    out = []
    for mf in Path(runs_dir).glob("*/manifest.json"):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # skip a corrupt run instead of breaking the whole listing
        out.append({k: m.get(k) for k in ("run_id", "created_at", "status", "controller",
                                           "seed", "sim_duration_s", "summary")})
    return sorted(out, key=lambda r: r["created_at"] or "", reverse=True)
