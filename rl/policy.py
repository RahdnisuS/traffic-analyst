"""The trained agent as a signal policy: plugs into Simulation like FixedTimePolicy does.

At every decision point (same DecisionClock as training) it builds the observation, asks the
network for keep/switch, and REQUESTS the switch; the signal controller still enforces min
green, amber, all-red and the pedestrian phase. Every decision is appended to
``sim.decisions``, which the run logger writes to decisions.jsonl: what the agent saw (in real
units, not just the scaled vector), what it chose, how sure it was (probability of switching)
and how good it judged the situation (the critic's value estimate). That is what lets the
analyst (Phase 12) answer "why did the AI switch early at 3:20?" from data, not guesswork.

Inference is deterministic (the most likely action), so a run is reproducible from its seed.

MAX-GREEN GUARD (added in Phase 10): once a green has run ``max_green_s`` the policy switches at
the next decision point whatever the network says, like the max-out of every actuated controller
(gap-out and the classic actuated rule use the same 60 s). WHY: the experiment found that in deep
overload (~135% of capacity) every input the network sees hits its cap (queues, backlogs and
waits all read "maximum"), every situation looks the same, the network keeps saying "keep", and
one green ran for 16 minutes while the other approaches overflowed. The network still decides
everything below the cap; the guard is a safety net, not the controller. The network is still
asked at a forced switch and its own answer is logged with ``forced: "max_green"``, so the
analyst can tell "the AI switched" from "the guard switched".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.metrics import clock

from .features import DecisionClock, approach_stats, feature_names, observe, serving_order

KEEP, SWITCH = 0, 1


class RLPolicy:
    name = "rl"

    def __init__(self, model, decision_interval_s: float, model_id: str = "ppo",
                 max_green_s: float | None = None):
        import torch  # imported here so plain fixed-time runs never need torch

        self._torch = torch
        self.model = model
        self.model_id = model_id
        self.clock = DecisionClock(decision_interval_s)
        self.max_green_s = max_green_s  # None = no guard (the network alone, as in training)
        self._names: list[str] | None = None

    @classmethod
    def load(cls, cfg, path: str | Path | None = None) -> "RLPolicy":
        from sb3_contrib import MaskablePPO

        p = Path(path) if path else cfg.rl_model_path()
        if not p.is_file():
            raise FileNotFoundError(f"no trained RL model at {p}: run `python -m rl.train` first")
        return cls(MaskablePPO.load(p, device="cpu"), cfg.rl.decision_interval_s, model_id=p.stem,
                   max_green_s=cfg.rl.max_green_s)

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        """(action, P(switch), value estimate) for one observation. Only called when a switch is
        possible, i.e. when the training-time action mask was [True, True]: no mask needed."""
        torch = self._torch
        with torch.no_grad():
            t, _ = self.model.policy.obs_to_tensor(obs)
            dist = self.model.policy.get_distribution(t)
            probs = dist.distribution.probs[0].cpu().numpy()
            value = float(self.model.policy.predict_values(t)[0, 0])
        return int(np.argmax(probs)), float(probs[SWITCH]), value

    def update(self, controller, sim) -> None:
        # Same decision points as training. Where a switch can't happen the action has no
        # effect, so the network isn't asked and nothing is logged.
        if not self.clock.due(sim) or not controller.can_switch():
            return
        if self._names is None:
            self._names = feature_names(sim)
        stats = approach_stats(sim)
        order = serving_order(sim)  # read BEFORE switching: the obs describes this moment
        obs = observe(sim, stats)
        action, p_switch, value = self.act(obs)
        forced = (self.max_green_s is not None and action != SWITCH
                  and controller.time_in_stage >= self.max_green_s - 1e-6)
        switched = (action == SWITCH or forced) and controller.request_switch()
        rec = self.record(sim, obs, stats, order, action, switched, p_switch, value)
        if forced:
            rec["forced"] = "max_green"
        sim.decisions.append(rec)

    def record(self, sim, obs, stats, order, action, switched, p_switch, value) -> dict:
        sig = sim.signals
        return {
            "t": round(sim.time, 1), "clock": clock(sim.time), "model": self.model_id,
            "phase": sig.phase.name, "next_phase": "+".join(order[1]) if len(order) > 1 else None,
            "time_in_green_s": round(sig.time_in_stage, 1),
            "action": "switch" if action == SWITCH else "keep", "switched": bool(switched),
            "p_switch": round(p_switch, 4), "value": round(value, 3),
            # What it saw, per approach in real units (the network gets these in serving order).
            "queue": {a: s[0] for a, s in stats.items()},
            "backlog": {a: s[1] for a, s in stats.items()},
            "longest_wait_s": {a: round(s[2], 1) for a, s in stats.items()},
            "ped_call": bool(sig.ped_call),
            "obs": dict(zip(self._names, (round(float(x), 4) for x in obs))),
        }
