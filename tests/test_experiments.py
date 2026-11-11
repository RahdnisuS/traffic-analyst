"""Phase 10: classic actuated control, Webster timing and the controller comparison."""

import numpy as np
import pytest

from experiments.compare import analyse, make_scenarios, paired, run_one, scenario_config, webster_plan
from sim.config import load_config
from sim.engine import Simulation
from sim.signals import ActuatedPolicy, Stage, webster_timing


def _greens(sim):
    return [(e.time, e.phase) for e in sim.signals.events if e.stage == "green"]


def test_actuated_gaps_out_on_an_empty_approach_and_maxes_out_on_a_busy_one():
    cfg = load_config()
    cfg.pedestrians.rate_ph = 0.0
    # Nobody arrives: every green gaps out at min green (+ passage time is already exceeded).
    cfg.demand.per_approach_vph = {a: 0.0 for a in "NESW"}
    sim = Simulation(cfg, seed=1, policy=ActuatedPolicy(30.0, 2.0, 60.0))
    sim.run(120)
    g = _greens(sim)
    gaps = np.diff([t for t, _ in g])
    assert len(g) > 5 and np.allclose(gaps, cfg.signals.min_green_s + cfg.signals.amber_s + cfg.signals.all_red_s,
                                       atol=0.11)
    # Heavy demand on N only: its detector stays occupied, so N maxes out at 60 s.
    cfg.demand.per_approach_vph = {"N": 1500.0, "E": 0.0, "S": 0.0, "W": 0.0}
    sim = Simulation(cfg, seed=1, policy=ActuatedPolicy(30.0, 2.0, 60.0))
    sim.run(200)
    # The very first green gaps out (the road is still empty at t = 0); the next N green, with a
    # standing queue and a stream behind it, runs to max green.
    ev = sim.signals.events
    starts = [e.time for e in ev if e.stage == "green" and e.phase == "N" and e.time > 0]
    end = next(e.time for e in ev if e.stage == "amber" and e.time > starts[0])
    assert end - starts[0] == pytest.approx(60.0, abs=0.11)


def test_actuated_is_safe_and_switches_more_than_the_fixed_timer():
    cfg = load_config()
    cfg.demand.per_approach_vph = {"N": 450.0, "E": 250.0, "S": 450.0, "W": 250.0}
    sim = Simulation(cfg, seed=3, policy=ActuatedPolicy(30.0, 2.0, 60.0))
    sim.run(900)
    assert sim.counters.emergency_stops == 0
    fixed = Simulation(cfg, seed=3)
    fixed.run(900)
    assert len(_greens(sim)) > len(_greens(fixed))
    assert sim.signals.stage in Stage


def test_webster_formula():
    # Textbook check: y = 0.2 per phase, 4 phases, 5 s lost each -> L = 20, Y = 0.8,
    # C0 = (1.5 * 20 + 5) / 0.2 = 175 -> capped at 150; greens equal.
    w = webster_timing([400.0] * 4, 2000.0, 5.0, 7.0, 150.0)
    assert w["Y"] == pytest.approx(0.8) and w["cycle_s"] == 150.0 and w["capped"]
    assert w["greens_s"] == [32.5] * 4
    # Lighter: Y = 0.4 -> C0 = 35 / 0.6 = 58.3; greens split by load (2:1:1:0.?)
    w = webster_timing([400.0, 200.0, 200.0, 0.0], 2000.0, 5.0, 7.0, 150.0)
    assert w["cycle_s"] == pytest.approx(58.3, abs=0.05)
    assert w["greens_s"][0] == pytest.approx(2 * w["greens_s"][1], abs=0.11)
    assert w["greens_s"][3] == 7.0  # min green for a phase without demand
    # A pedestrian phase that runs every cycle is lost time for vehicles: longer cycle.
    assert webster_timing([300.0] * 4, 2000.0, 5.0, 7.0, 150.0, ped_rate_ph=3600, ped_phase_s=22)["lost_s"] > 40


def test_scenarios_are_reproducible_and_cover_every_band():
    cfg = load_config()
    a, b = make_scenarios(cfg), make_scenarios(cfg)
    assert a == b and len(a) == 3 * cfg.experiments.scenarios_per_band
    for sc in a:
        lo, hi = cfg.experiments.bands[sc["band"]]
        mean = np.mean(list(sc["per_approach_vph"].values()))
        assert lo * 0.7 - 1 <= mean <= hi * 1.3 + 1
    assert len({sc["seed"] for sc in a}) == len(a)


def test_paired_statistics():
    rng = np.random.default_rng(0)
    base = {f"s{i}": {"m": 10.0 + i} for i in range(20)}
    better = {k: {"m": v["m"] * 0.9} for k, v in base.items()}
    p = paired(better, base, "m", 1000, rng)
    assert p["mean_pct"] == pytest.approx(-10.0) and p["wins"] == 20 and p["p_wilcoxon"] < 0.001
    assert p["ci_lo"] == pytest.approx(-10.0) and p["ci_hi"] == pytest.approx(-10.0)


def test_every_controller_runs_the_same_scenario(tmp_path):
    """A tiny end-to-end run: all controllers on one scenario, then the analysis."""
    cfg = load_config()
    cfg.experiments.warmup_s, cfg.experiments.measure_s = 60.0, 120.0
    controllers = ["fixed", "webster", "actuated", "gap_out"]
    cfg.experiments.controllers = controllers
    sc = make_scenarios(cfg)[-1]  # a high-demand one
    plan = webster_plan(cfg, sc, 2900.0)
    rows = [run_one(cfg.model_dump_json(), sc, c, plan if c == "webster" else None) for c in controllers]
    assert all(r["emergency_stops"] == 0 for r in rows)
    assert all(r["scenario"] == sc["id"] and r["waiting_veh_h"] > 0 for r in rows)
    assert len({r["greens_ph"] for r in rows}) > 1  # the controllers really differ
    res = analyse(rows, cfg)
    assert {s["controller"] for s in res["summary"]} == set(controllers)
    # The scenario's settings reach the config.
    c = scenario_config(cfg, sc)
    assert c.demand.per_approach_vph == sc["per_approach_vph"] and c.pedestrians.rate_ph == sc["ped_rate_ph"]


def test_rl_max_green_guard_switches_an_agent_that_never_wants_to():
    """The Phase 10 failure mode, reduced: a network that always says "keep" would hold the first
    green forever. With the guard it switches at the first decision point after 60 s, and the
    decision log says the guard did it (the network's own answer is kept)."""
    pytest.importorskip("torch")
    from rl.policy import RLPolicy

    class AlwaysKeep(RLPolicy):
        def act(self, obs):
            return 0, 0.01, -1.0

    cfg = load_config()
    cfg.pedestrians.rate_ph = 0.0
    for guard, expect_switch in ((None, False), (60.0, True)):
        sim = Simulation(cfg, seed=2, policy=AlwaysKeep(None, cfg.rl.decision_interval_s, max_green_s=guard))
        sim.run(120)
        ambers = [e.time for e in sim.signals.events if e.stage == "amber"]
        if not expect_switch:
            assert ambers == []  # unguarded: stuck on the first green
            assert not any(d.get("forced") for d in sim.decisions)
        else:
            assert ambers[0] == pytest.approx(60.0, abs=0.11)  # 60 s is on the 5 s decision grid
            forced = [d for d in sim.decisions if d.get("forced")]
            assert forced and forced[0]["action"] == "keep" and forced[0]["switched"]
