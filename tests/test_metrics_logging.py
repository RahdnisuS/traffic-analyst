import csv
import json

import pytest

from sim.config import load_config
from sim.engine import Simulation
from sim.logger import RunLogger, list_runs
from sim.metrics import MetricsCollector, clock, lane_queues, level_of_service


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def heavy(cfg, vph=900):
    c = cfg.model_copy(deep=True)
    c.demand.per_approach_vph = {a: vph for a in "NESW"}
    return c


def read_events(run_dir):
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("delay,los", [(None, None), (0, "A"), (10, "A"), (10.1, "B"),
                                       (35, "C"), (55, "D"), (80, "E"), (80.1, "F")])
def test_level_of_service_bands(delay, los):
    assert level_of_service(delay) == los


def test_clock_format():
    assert clock(200) == "3:20" and clock(59.6) == "1:00" and clock(0) == "0:00"


def test_red_approach_builds_a_queue(cfg):
    sim = Simulation(cfg, seed=5)
    sim.run(115)  # N, E and S have been red for a while
    lq = lane_queues(sim, cfg.metrics.queue_speed)
    assert sum(lq[lid].queued for lid in sim.junction.inbound["E"]) > 0
    for q in lq.values():
        assert 0 <= q.queued <= q.vehicles
        assert 0 <= q.queue_m <= 150.0


def test_logger_writes_consistent_files(cfg, tmp_path):
    sim = Simulation(cfg, seed=9)
    with RunLogger(sim, runs_dir=tmp_path) as log:
        sim.run(300)
    d = log.dir
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["seed"] == 9
    assert manifest["config"]["signals"]["green_s"] == 25

    rows = list(csv.DictReader(open(d / "metrics.csv")))
    assert len(rows) == 300 and rows[0]["t"] == "1.0" and rows[-1]["t"] == "300.0"
    trips = list(csv.DictReader(open(d / "trips.csv")))
    assert len(trips) == manifest["summary"]["completed"] == sim.counters.completed
    assert all(float(t["delay"]) >= 0 for t in trips)

    events = read_events(d)
    greens = [e for e in events if e["type"] == "phase_change" and e["stage"] == "green"]
    assert greens[0]["t"] == 0.0 and [g["phase"] for g in greens[:4]] == ["N", "E", "S", "W"]
    assert sim.observers == []  # logger unsubscribed itself
    assert list_runs(tmp_path)[0]["run_id"] == log.run_id


def test_logs_are_deterministic(cfg, tmp_path):
    dirs = []
    for i in range(2):
        sim = Simulation(cfg, seed=4)
        with RunLogger(sim, runs_dir=tmp_path, run_id=f"r{i}") as log:
            sim.run(240)
        dirs.append(log.dir)
    for name in ("metrics.csv", "events.jsonl", "trips.csv"):
        assert (dirs[0] / name).read_text() == (dirs[1] / name).read_text()


def test_logging_does_not_change_the_simulation(cfg, tmp_path):
    a, b = Simulation(cfg, seed=6), Simulation(cfg, seed=6)
    with RunLogger(b, runs_dir=tmp_path):
        b.run(200)
    a.run(200)
    assert a.fingerprint() == b.fingerprint()


def test_heavy_demand_produces_paired_spike_and_spillback_episodes(cfg, tmp_path):
    sim = Simulation(heavy(cfg), seed=2)
    with RunLogger(sim, runs_dir=tmp_path) as log:
        sim.run(900)
    events = read_events(log.dir)
    for kind in ("queue_spike", "spillback"):
        starts = {e["episode"] for e in events if e["type"] == kind and e["status"] == "start"}
        ends = {e["episode"] for e in events if e["type"] == kind and e["status"] == "end"}
        assert starts, f"expected at least one {kind}"
        assert starts == ends  # every episode is closed (truncated ones at run end)


def test_gridlock_detected_and_cleared(cfg):
    """Force every light to red: nothing exits, everyone stops -> gridlock. Release -> it clears."""
    sim = Simulation(heavy(cfg, 600), seed=1)
    col = MetricsCollector(sim, cfg.metrics)
    events = []

    class Obs:
        def after_step(self, s):
            events.extend(col.after_step(s)[1])

    sim.observers.append(Obs())
    real_state = sim.signals.movement_state
    sim.signals.movement_state = lambda arm, m: "R"
    sim.run(150)
    starts = [e for e in events if e["type"] == "gridlock" and e["status"] == "start"]
    assert len(starts) == 1
    sim.signals.movement_state = real_state
    sim.run(60)
    ends = [e for e in events if e["type"] == "gridlock" and e["status"] == "end"]
    assert len(ends) == 1 and ends[0]["episode"] == starts[0]["episode"]


def test_no_gridlock_in_normal_operation(cfg, tmp_path):
    sim = Simulation(heavy(cfg, 700), seed=3)
    with RunLogger(sim, runs_dir=tmp_path) as log:
        sim.run(600)
    assert not [e for e in read_events(log.dir) if e["type"] == "gridlock"]
