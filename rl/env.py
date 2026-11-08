"""Gymnasium environment: one simulated junction, the agent controls when the green switches.

One ``step`` = one decision = exactly ``decision_interval_s`` (5 s) of simulated time. The agent
says keep (0) or switch (1); the environment runs the simulation (at its normal dt = 0.1 s) to the
next decision point. A switch is only REQUESTED: during min green, amber, all-red or a pedestrian
phase the controller ignores it (observation ``can_switch`` = 0 says so). See DecisionClock for
why every decision covers the same time.

The simulation itself is unchanged: the env installs a policy that does nothing and makes the
switch requests itself, so a trained agent runs on exactly the same engine as the fixed timer.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.config import Config, load_config
from sim.engine import Simulation

from .features import DecisionClock, feature_names, observe, waiting_now

KEEP, SWITCH = 0, 1


class ExternalPolicy:
    """A SignalPolicy that never acts on its own: the environment (or RLPolicy) decides."""

    def update(self, controller, sim=None) -> None:
        pass


def episode_traffic(cfg: Config, rng: np.random.Generator) -> dict:
    """Random traffic for one training episode: overall demand, per-approach imbalance and
    pedestrian rate. Varied so the agent can't memorise one demand level."""
    r = cfg.rl
    mean = rng.uniform(*r.demand_vph)
    arms = sorted(cfg.demand.per_approach_vph)
    vph = {a: round(mean * rng.uniform(1 - r.demand_imbalance, 1 + r.demand_imbalance), 1) for a in arms}
    return {"per_approach_vph": vph, "ped_rate_ph": round(rng.uniform(*r.ped_rate_ph), 1),
            "seed": int(rng.integers(0, 2**31 - 1))}


def make_sim(cfg: Config, traffic: dict, policy=None) -> Simulation:
    """A simulation with the given traffic. Used by training, evaluation and the baselines,
    so every controller faces identical conditions for the same ``traffic``."""
    c = cfg.model_copy(deep=True)
    c.demand.per_approach_vph = dict(traffic["per_approach_vph"])
    c.pedestrians.rate_ph = traffic["ped_rate_ph"]
    return Simulation(c, seed=traffic["seed"], policy=policy)


class SignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None, traffic: dict | None = None):
        """``traffic``: fixed traffic for every episode (evaluation); None = random per episode."""
        super().__init__()
        self.cfg = cfg or load_config()
        self.fixed_traffic = traffic
        r = self.cfg.rl
        self.steps_per_s = round(1 / self.cfg.sim.dt)
        self.end_step = round(r.episode_s / self.cfg.sim.dt)
        probe = Simulation(self.cfg, policy=ExternalPolicy())
        self.names = feature_names(probe)
        self.observation_space = spaces.Box(0.0, 3.0, shape=(len(self.names),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.sim: Simulation | None = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.traffic = self.fixed_traffic or episode_traffic(self.cfg, self.np_random)
        self.sim = make_sim(self.cfg, self.traffic, policy=ExternalPolicy())
        self.clock = DecisionClock(self.cfg.rl.decision_interval_s)
        self.waiting_veh_s = 0.0
        self.switches = 0
        return observe(self.sim), {}  # t = 0 is the first decision point

    def step(self, action):
        sim, r = self.sim, self.cfg.rl
        switched = int(action) == SWITCH and sim.signals.request_switch()
        self.switches += int(switched)
        waited = self._advance()
        reward = -waited / r.reward_scale - (r.switch_penalty if switched else 0.0)
        truncated = sim.tick >= self.end_step  # a time limit, not a terminal state
        info = {"waiting_veh_s": waited, "switched": bool(switched)}
        if truncated:
            info["episode_waiting_veh_s"] = self.waiting_veh_s
            info["episode_switches"] = self.switches
            info["episode_completed"] = sim.counters.completed
        return observe(sim), float(reward), False, truncated, info

    def action_masks(self) -> np.ndarray:
        """Which actions are real choices now (read by MaskablePPO): "switch" only when the
        controller would accept it. WHY: on the fixed 5 s grid over half the decision points fall
        in min green, amber, all-red or a pedestrian phase. There "switch" does nothing, yet plain
        PPO still learns from those coin flips: pure noise that kept the policy near 50/50 after
        40k decisions (entropy ~ ln 2) even though the critic had learned (explained variance 0.95).
        Masked, a forced "keep" has probability 1 and contributes no policy gradient."""
        return np.array([True, self.sim.signals.can_switch()])

    # ------------------------------------------------------------------ internals
    def _advance(self) -> float:
        """Step the sim until the next decision point or the episode end. Returns the
        vehicle-seconds of waiting accumulated on the way."""
        sim, dt = self.sim, self.sim.dt
        waited = 0.0
        while sim.tick < self.end_step:
            sim.step()
            waited += waiting_now(sim) * dt
            if self.clock.due(sim):
                break  # exactly one decision interval later
        self.waiting_veh_s += waited
        return waited


def fixed_time_return(cfg: Config, traffic: dict) -> dict:
    """The fixed timer on the same traffic, scored with the agent's reward. Gives the training
    curve a baseline: the agent is only useful once its return beats this line."""
    sim = make_sim(cfg, traffic)  # default policy: the fixed-time plan from config
    r, dt = cfg.rl, sim.dt
    waited = 0.0
    phase_changes = 0
    last_idx = sim.signals.phase_idx
    for _ in range(round(r.episode_s / dt)):
        sim.step()
        waited += waiting_now(sim) * dt
        if sim.signals.phase_idx != last_idx:
            phase_changes += 1
            last_idx = sim.signals.phase_idx
    return {"return": -waited / r.reward_scale - r.switch_penalty * phase_changes,
            "waiting_veh_s": waited, "switches": phase_changes, "completed": sim.counters.completed}


def agent_return(cfg: Config, traffic: dict, model) -> dict:
    """A trained (Maskable)PPO model on the same traffic, taking its MOST LIKELY action every time
    (as RLPolicy does when deployed), scored with the same reward as the baselines."""
    env = SignalEnv(cfg, traffic=traffic)
    obs, _ = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, r, _, done, info = env.step(int(action))
        total += r
    return {"return": total, "waiting_veh_s": info["episode_waiting_veh_s"], "switches": info["episode_switches"]}


GAP_OUT_MAX_GREEN = 60.0


def gap_out_return(cfg: Config, traffic: dict) -> dict:
    """A simple actuated rule on the agent's own terms: at each decision point, switch once the
    green approach has no queued vehicle left (it "gaps out"), or after 60 s of green.

    WHY it is here: with "switch = go to the NEXT phase", ending a green because ANOTHER
    approach is busy often hands the green to an empty approach. Gap-out is the classic
    actuated-signal logic that works within that constraint, and it beat the fixed timer by
    ~18% on training-style traffic. So it is the bar the agent should clear: a learned policy
    that can't beat a rule a traffic engineer writes in three lines isn't worth its complexity.
    """
    env = SignalEnv(cfg, traffic=traffic)
    obs, _ = env.reset(seed=0)
    i_queue = env.names.index("queue_green")
    total, done = 0.0, False
    while not done:
        switch = obs[i_queue] <= 0.0 or env.sim.signals.time_in_stage >= GAP_OUT_MAX_GREEN
        obs, r, _, done, info = env.step(SWITCH if switch else KEEP)
        total += r
    return {"return": total, "waiting_veh_s": info["episode_waiting_veh_s"], "switches": info["episode_switches"]}


class GapOutPolicy:
    """The same gap-out rule as ``gap_out_return``, as a SignalPolicy, so ``rl.evaluate`` can run
    it on a plain Simulation with the full metrics (delay, pedestrians). Same decision grid, same
    "queue_green" input, same 60 s cap; parity with the env version is tested."""

    name = "gap_out"

    def __init__(self, decision_interval_s: float, max_green_s: float = GAP_OUT_MAX_GREEN):
        self.clock = DecisionClock(decision_interval_s)
        self.max_green_s = max_green_s

    def update(self, controller, sim) -> None:
        if not self.clock.due(sim) or not controller.can_switch():
            return
        obs = observe(sim)
        if obs[0] <= 0.0 or controller.time_in_stage >= self.max_green_s:  # obs[0] = queue_green
            controller.request_switch()
