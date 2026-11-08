"""Phase 7: RL environment, decision timing, reward accounting, and train/deploy parity."""

import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from rl.env import (KEEP, SWITCH, ExternalPolicy, GapOutPolicy, SignalEnv, episode_traffic,  # noqa: E402
                    fixed_time_return, gap_out_return, make_sim)
from rl.features import DecisionClock, feature_names, observe, waiting_now  # noqa: E402
from rl.policy import RLPolicy  # noqa: E402
from sim.config import load_config  # noqa: E402
from sim.logger import RunLogger  # noqa: E402

TRAFFIC = {"per_approach_vph": {"N": 500.0, "E": 300.0, "S": 450.0, "W": 350.0}, "ped_rate_ph": 200.0, "seed": 9}


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c.rl.episode_s = 300.0  # short episodes keep the tests fast
    return c


def test_observation_matches_names_and_bounds(cfg):
    sim = make_sim(cfg, TRAFFIC, policy=ExternalPolicy())
    sim.run(120)
    obs, names = observe(sim), feature_names(sim)
    assert obs.shape == (len(names),) == (15,) and obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 3).all()


def test_observation_is_relative_to_the_green_phase(cfg):
    """Slot "green" always describes the approach that is green now, "next" the one a switch
    would serve: the same situation gives the same numbers whichever phase is green."""
    from rl.features import approach_stats, serving_order
    sim = make_sim(cfg, TRAFFIC, policy=ExternalPolicy())
    sim.run(200)
    obs, names, stats = observe(sim), feature_names(sim), approach_stats(sim)
    order = serving_order(sim)
    assert order[0] == sorted(sim.signals.phase.green)
    green, nxt = order[0][0], order[1][0]
    assert obs[names.index("queue_green")] == pytest.approx(min(stats[green][0] / 20, 3))
    assert obs[names.index("queue_next")] == pytest.approx(min(stats[nxt][0] / 20, 3))


def test_decisions_are_on_a_fixed_5s_grid_and_effective_after_min_green(cfg):
    sim = make_sim(cfg, TRAFFIC, policy=ExternalPolicy())
    clock, times, effective = DecisionClock(5.0), [], []
    while sim.time < 30:
        if clock.due(sim):
            times.append(round(sim.time, 1))
            if sim.signals.can_switch():
                effective.append(round(sim.time, 1))
        sim.step()
    assert times == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert effective[0] == 10.0  # min green is 7 s: the first grid point after it


def test_every_step_covers_the_same_time(cfg):
    env = SignalEnv(cfg, traffic=TRAFFIC)
    env.reset(seed=0)
    for action in (SWITCH, KEEP, SWITCH, SWITCH, KEEP, SWITCH):
        t0 = env.sim.time
        env.step(action)
        assert env.sim.time - t0 == pytest.approx(cfg.rl.decision_interval_s)


def test_reward_is_waiting_plus_switch_penalty(cfg):
    env = SignalEnv(cfg, traffic=TRAFFIC)
    env.reset(seed=0)
    total_reward, total_wait, switches = 0.0, 0.0, 0
    done = False
    k = 0
    while not done:
        action = SWITCH if k % 4 == 3 else KEEP
        _, r, term, trunc, info = env.step(action)
        total_reward += r
        total_wait += info["waiting_veh_s"]
        switches += info["switched"]
        done = term or trunc
        k += 1
    expected = -total_wait / cfg.rl.reward_scale - cfg.rl.switch_penalty * switches
    assert total_reward == pytest.approx(expected)
    assert info["episode_switches"] == switches > 0
    assert info["episode_waiting_veh_s"] == pytest.approx(total_wait)
    assert k == round(cfg.rl.episode_s / cfg.rl.decision_interval_s)


def test_always_keep_never_switches_and_env_passes_gym_checks(cfg):
    from gymnasium.utils.env_checker import check_env

    env = SignalEnv(cfg, traffic=TRAFFIC)
    check_env(env, skip_render_check=True)
    env.reset(seed=1)
    for _ in range(30):
        env.step(KEEP)
    assert env.sim.signals.phase_idx == 0 and len(env.sim.signals.events) == 1


def test_random_traffic_is_reproducible_and_in_range(cfg):
    a = [episode_traffic(cfg, np.random.default_rng(3)) for _ in range(2)]
    assert a[0] == a[1]
    lo, hi = cfg.rl.demand_vph
    imb = cfg.rl.demand_imbalance
    for v in a[0]["per_approach_vph"].values():
        assert lo * (1 - imb) <= v <= hi * (1 + imb)


def test_switching_cannot_break_signal_safety(cfg):
    """Even an agent that asks to switch at every decision can't shorten min green or skip
    amber/all-red: the controller enforces them. And nobody crashes."""
    env = SignalEnv(cfg, traffic=TRAFFIC)
    env.reset(seed=2)
    done = False
    while not done:
        _, _, _, done, _ = env.step(SWITCH)
    ev = env.sim.signals.events
    greens = [e.time for e in ev if e.stage == "green"]
    ambers = [e.time for e in ev if e.stage == "amber"]
    for g, a in zip(greens, ambers):
        assert a - g >= cfg.signals.min_green_s - 1e-6
    assert env.sim.counters.emergency_stops == 0


def _untrained_model(cfg):
    from sb3_contrib import MaskablePPO
    return MaskablePPO("MlpPolicy", SignalEnv(cfg, traffic=TRAFFIC), seed=4, device="cpu",
                       policy_kwargs={"net_arch": [8]})


def test_action_mask_offers_switch_only_when_it_would_work(cfg):
    env = SignalEnv(cfg, traffic=TRAFFIC)
    env.reset(seed=0)
    for _ in range(40):
        mask = env.action_masks()
        assert mask[0] and mask[1] == env.sim.signals.can_switch()
        env.step(SWITCH if mask[1] else KEEP)


def test_train_and_deploy_make_identical_decisions(cfg):
    """The SAME network driving the gym env and driving the live sim as RLPolicy must make the
    same decisions at the same times: proves observation and timing are shared, not duplicated."""
    model = _untrained_model(cfg)
    env = SignalEnv(cfg, traffic=TRAFFIC)
    obs, _ = env.reset(seed=0)
    env_decisions, done = [], False
    i_can = env.names.index("can_switch")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if obs[i_can] == 1.0:  # the live policy only consults the network when it matters
            env_decisions.append((round(env.sim.time, 1), int(action)))
        obs, _, _, done, _ = env.step(action)

    policy = RLPolicy(model, cfg.rl.decision_interval_s)
    sim = make_sim(cfg, TRAFFIC, policy=policy)
    sim.run(cfg.rl.episode_s)
    live = [(d["t"], 1 if d["action"] == "switch" else 0) for d in sim.decisions]
    assert live[:len(env_decisions)] == env_decisions
    assert {a for _, a in live} == {0, 1} or len(live) > 5


def test_rl_decisions_are_logged(cfg, tmp_path):
    model = _untrained_model(cfg)
    sim = make_sim(cfg, TRAFFIC, policy=RLPolicy(model, cfg.rl.decision_interval_s))
    with RunLogger(sim, controller="rl", runs_dir=tmp_path) as log:
        sim.run(120)
    lines = (log.dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(sim.decisions) > 5
    d = json.loads(lines[0])
    assert d["action"] in ("keep", "switch") and 0 <= d["p_switch"] <= 1
    assert set(d["queue"]) == {"N", "E", "S", "W"} and "time_in_green" in d["obs"]
    assert d["t"] == 10.0  # first grid point after the 7 s min green


def test_fixed_time_baseline_uses_the_same_accounting(cfg):
    res = fixed_time_return(cfg, TRAFFIC)
    sim = make_sim(cfg, TRAFFIC)
    w = 0.0
    for _ in range(round(cfg.rl.episode_s / sim.dt)):
        sim.step()
        w += waiting_now(sim) * sim.dt
    assert res["waiting_veh_s"] == pytest.approx(w)
    assert res["return"] == pytest.approx(-w / cfg.rl.reward_scale - cfg.rl.switch_penalty * res["switches"])


def test_gap_out_policy_matches_the_training_baseline(cfg):
    """rl.evaluate runs gap-out as a SignalPolicy; the training curve scores it through the env.
    Both must be the same controller, or the two reports would compare against different bars."""
    res = gap_out_return(cfg, TRAFFIC)
    sim = make_sim(cfg, TRAFFIC, policy=GapOutPolicy(cfg.rl.decision_interval_s))
    w = 0.0
    for _ in range(round(cfg.rl.episode_s / sim.dt)):
        sim.step()
        w += waiting_now(sim) * sim.dt
    greens = sum(1 for e in sim.signals.events if e.stage == "green") - 1
    assert w == pytest.approx(res["waiting_veh_s"])
    assert greens > 3
