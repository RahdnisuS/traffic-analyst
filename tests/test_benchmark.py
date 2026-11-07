"""Phase 6: stress test / capacity benchmark."""

import math

import pytest

from bench.live import LiveStressTest
from bench.stress import (Snapshot, approach_stock, judge, latest_summary, ramp, render_report, run_benchmark,
                          run_level, save_results, score_ramp, stock_growth)
from sim.config import BenchmarkSettings, load_config
from sim.engine import Simulation
from sim.metrics import MetricsCollector


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def fast(cfg, **kw):
    """Short windows so tests run in seconds (fine for pass/fail at clearly low/high demand)."""
    base = dict(warmup_s=60, measure_s=300, unserved_min=10)
    return cfg.benchmark.model_copy(update={**base, **kw})


# ---------------------------------------------------------------------------- conservation
def test_vehicles_are_conserved_per_approach(cfg):
    """The judging rests on: arrivals - departures = stock growth, exactly, per approach."""
    sim = Simulation(cfg, seed=3)
    sim.apply_settings(volume_vph=900)  # busy enough to have a backlog
    col = MetricsCollector(sim, sim.cfg.metrics)
    start = Snapshot.take(sim)
    departed = {a: 0 for a in start.arrivals}
    for _ in range(3000):
        sim.step()
        for tr in col.after_step(sim)[2]:
            departed[tr.origin] += 1
    end = Snapshot.take(sim)
    assert sim.spawner.backlog_size() > 0
    for a in start.arrivals:
        arrived = end.arrivals[a] - start.arrivals[a]
        assert arrived - departed[a] == end.stock[a] - start.stock[a]
    assert sum(approach_stock(sim).values()) == len(sim.vehicles) + sim.spawner.backlog_size()


# ---------------------------------------------------------------------------- judging (pure)
def test_stock_growth_cancels_the_signal_sawtooth():
    cycle = 120.0
    # A queue that fills during red and empties on green, with no trend: growth ~ 0
    # wherever the window happens to start in the cycle.
    # The window (1150 s) is deliberately NOT a whole number of cycles, like real windows.
    for phase in (0.0, 37.0, 90.0):
        samples = [(t, {"N": 20 * ((t + phase) % cycle) / cycle}) for t in range(0, 1151)]
        assert abs(stock_growth(samples, cycle)["N"]) < 1.0
        # The naive endpoint difference measures where in the cycle the ends fell.
        naive = samples[-1][1]["N"] - samples[0][1]["N"]
        assert abs(naive) > 5
    # Same sawtooth on top of a steady build-up of 30 vehicles over the window: recovered.
    trend = [(t, {"N": 20 * (t % cycle) / cycle + 30 * t / 1200}) for t in range(0, 1201)]
    assert stock_growth(trend, cycle)["N"] == pytest.approx(30, abs=1.0)


def _snap(t, arrivals, stock, completed, gridlocks=0):
    return Snapshot(t=t, arrivals=arrivals, stock=stock, completed=completed, gridlocks=gridlocks)


def test_judge_thresholds_and_gridlock():
    s = BenchmarkSettings(unserved_frac=0.05, unserved_min=10)
    a0 = _snap(0, {"N": 0, "S": 0}, {"N": 5, "S": 5}, 0)
    a1 = _snap(600, {"N": 400, "S": 100}, {"N": 5, "S": 5}, 495)
    # N may fall 20 behind (5% of 400), S only the 10-vehicle floor.
    ok = judge(a0, a1, s, growth={"N": 20, "S": 10})
    assert ok["passed"] and ok["approaches"]["N"]["allowed"] == 20
    bad = judge(a0, a1, s, growth={"N": 20.5, "S": 0})
    assert not bad["passed"] and bad["failing_approaches"] == ["N"]
    assert not judge(a0, a1, s, gridlock=True, growth={"N": 0, "S": 0})["passed"]
    assert ok["offered_vph"] == 3000 and ok["served_vph"] == 2970


def test_score_ramp():
    s = BenchmarkSettings()
    lv = lambda v, ok, served: {"volume_vph": v, "passed": ok, "served_vph": served}  # noqa: E731
    r = score_ramp([lv(100, True, 400), lv(150, False, 590), lv(200, True, 780), lv(250, False, 800),
                    lv(300, False, 810)], s)
    # The highest PASSING level counts, even after an earlier noisy failure.
    assert r["capacity_vph"] == 780 and r["capacity_level_vph"] == 200
    assert r["breakdown_level_vph"] == 250 and not r["censored"] and r["peak_served_vph"] == 810
    assert score_ramp([lv(100, True, 400), lv(150, True, 600)], s)["censored"]
    assert score_ramp([lv(100, False, 300)], s)["capacity_vph"] is None


# ---------------------------------------------------------------------------- real runs
def test_low_demand_passes_and_overload_fails(cfg):
    s = fast(cfg)
    low = run_level(cfg, 1, 150, s)
    assert low["passed"] and low["emergency_stops"] == 0
    high = run_level(cfg, 1, 1400, s)
    assert not high["passed"] and high["served_ratio"] < 0.8
    assert high["emergency_stops"] == 0  # overload must not break the safety invariants


def test_ramp_stops_after_consecutive_failures_and_is_deterministic(cfg):
    s = fast(cfg, start_vph=150, step_vph=600, max_vph=1500, fails_to_stop=2)
    r1 = ramp(cfg, 2, s)
    assert [lv["volume_vph"] for lv in r1["levels"]] == [150, 750, 1350]
    assert [lv["passed"] for lv in r1["levels"]] == [True, False, False]
    assert r1["capacity_level_vph"] == 150 and r1["breakdown_level_vph"] == 750
    r2 = ramp(cfg, 2, s)
    assert r1 == r2  # same seed + config -> identical result


def test_benchmark_saves_and_reloads(cfg, tmp_path):
    c = cfg.model_copy(deep=True)
    c.benchmark = fast(cfg, start_vph=150, step_vph=600, max_vph=800, fails_to_stop=1)
    result = run_benchmark(c, seeds=[1], workers=1, verbose=False)
    assert result["score"]["n"] == 1 and math.isclose(result["score"]["std_vph"], 0.0)
    out = save_results(result, tmp_path)
    for name in ("summary.json", "levels.csv", "report.md", "plot.png"):
        assert (out / name).is_file(), name
    assert "Capacity score" in render_report(result)
    latest = latest_summary(tmp_path)
    assert latest["id"] == out.name and latest["score"] == result["score"]
    assert latest_summary(tmp_path / "missing") is None


# ---------------------------------------------------------------------------- live ramp
def run_live(cfg, s, max_steps=20000):
    sim = Simulation(cfg, seed=4)
    col = MetricsCollector(sim, sim.cfg.metrics)
    original = sim.mean_volume_vph()
    st = LiveStressTest(sim, col, s, max_vph=1500)
    events = []
    for _ in range(max_steps):
        sim.step()
        col.after_step(sim)
        events += st.after_step(sim)
        if not st.active:
            break
    return sim, st, events, original


def test_live_ramp_finds_breakdown_and_restores_volume(cfg):
    s = cfg.benchmark.model_copy(update={"live_start_vph": 150, "live_step_vph": 1200, "live_step_s": 300,
                                         "live_settle_s": 60, "fails_to_stop": 1})
    sim, st, events, original = run_live(cfg, s)
    assert not st.active and st.result["reason"] == "breakdown"
    assert [x["passed"] for x in st.steps] == [True, False]
    assert st.result["capacity_level_vph"] == 150 and st.result["overloaded_level_vph"] == 1350
    assert [e["type"] for e in events] == ["stress_start", "stress_step", "stress_step", "stress_result"]
    assert sim.mean_volume_vph() == pytest.approx(original)


def test_live_ramp_can_be_stopped(cfg):
    sim = Simulation(cfg, seed=4)
    col = MetricsCollector(sim, sim.cfg.metrics)
    original = sim.mean_volume_vph()
    st = LiveStressTest(sim, col, cfg.benchmark, max_vph=1500)
    for _ in range(100):
        sim.step()
        col.after_step(sim)
        st.after_step(sim)
    events = st.stop(sim)
    assert events[-1]["type"] == "stress_result" and events[-1]["reason"] == "stopped"
    assert sim.mean_volume_vph() == pytest.approx(original) and st.stop(sim) == []


def test_benchmark_settings_validation():
    with pytest.raises(ValueError):
        BenchmarkSettings(start_vph=500, max_vph=100)
    with pytest.raises(ValueError):
        BenchmarkSettings(live_step_s=60, live_settle_s=60)
