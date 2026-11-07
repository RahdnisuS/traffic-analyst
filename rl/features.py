"""What the agent sees, when it decides, and how waiting is counted.

Shared by the training environment (env.py) and the live controller (policy.py). WHY one
module: if training and deployment computed the observation even slightly differently (another
scale, another lane order, deciding at other moments) the trained network would be fed inputs
it never saw, and fail silently. Here there is only one definition of each.

Observation: PHASE-RELATIVE (15 numbers, each scaled to roughly 0..1, clipped at 3).
The agent's only choice is "keep this green" or "switch to the NEXT phase", so what matters is
the approach that is green now and the ones that come after it, in that order. So the inputs are
listed in serving order, not by compass direction:

    for slot in (green, next, next2, next3):      # next = served if the agent switches now
        queue_<slot>    vehicles on that phase's lanes moving below 2 m/s, / 20
        backlog_<slot>  its vehicles waiting off the map to enter (lane full), / 20
        wait_<slot>     longest current wait of any of its vehicles, incl. the backlog, / 120 s
    time_in_green       seconds the current green has run, / 60
    ped_call            1 if a pedestrian is waiting: the next switch will insert the WALK phase
    can_switch          1 if a switch request would be accepted now (green, past min green);
                        0 during min green, amber, all-red or a pedestrian phase

WHY relative and not "queue_N, queue_E, ... + which phase is green": a first version used
that, and PPO got WORSE as it trained (fewer and fewer switches). To judge a switch it had to
learn, separately for each of the 4 phases, which inputs meant "green" and which "next".
Relative inputs make "the green is empty and the next approach is waiting" the same numbers
whatever phase is green, so one lesson applies to all four phases (4x the useful experience).
"""

from __future__ import annotations

import numpy as np

from sim.engine import WAIT_SPEED

QUEUE_SCALE = 20.0
WAIT_SCALE = 120.0
GREEN_SCALE = 60.0
CLIP = 3.0
SLOTS = ("green", "next", "next2", "next3")


def approach_stats(sim) -> dict[str, tuple[int, int, float]]:
    """Per arm, in real units: (queued vehicles, backlog, longest wait in seconds)."""
    J, now, qs = sim.junction, sim.time, sim.cfg.metrics.queue_speed
    out = {}
    for arm in J.inbound:
        queued, longest = 0, 0.0
        for lane_id in J.approach_lanes(arm):
            for veh in sim.lanes[lane_id]:
                if veh.v < qs:
                    queued += 1
                longest = max(longest, veh.wait_time)
        backlog = sim.spawner.backlog[arm]
        if backlog:  # FIFO: the head has waited longest
            longest = max(longest, now - backlog[0].arrival_time)
        out[arm] = (queued, len(backlog), longest)
    return out


def serving_order(sim) -> list[list[str]]:
    """The arms of each phase, starting from the one that is green now (then the next, ...)."""
    phases = sim.junction.phases
    i = sim.signals.phase_idx
    return [sorted(phases[(i + k) % len(phases)].green) for k in range(len(phases))]


def feature_names(sim) -> list[str]:
    names = []
    for slot in SLOTS[:len(sim.junction.phases)]:
        names += [f"queue_{slot}", f"backlog_{slot}", f"wait_{slot}"]
    return names + ["time_in_green", "ped_call", "can_switch"]


def observe(sim, stats: dict | None = None) -> np.ndarray:
    """The observation vector (float32), in ``feature_names`` order."""
    stats = stats or approach_stats(sim)
    out = []
    for arms in serving_order(sim):
        out += [sum(stats[a][0] for a in arms) / QUEUE_SCALE,
                sum(stats[a][1] for a in arms) / QUEUE_SCALE,
                max(stats[a][2] for a in arms) / WAIT_SCALE]
    sig = sim.signals
    out += [sig.time_in_stage / GREEN_SCALE, 1.0 if sig.ped_call else 0.0, 1.0 if sig.can_switch() else 0.0]
    return np.minimum(np.asarray(out, dtype=np.float32), CLIP)


def waiting_now(sim) -> int:
    """Vehicles waiting at this instant: stopped on the road (< 0.1 m/s, as in wait_time) plus
    everyone in the backlog. Times dt and summed, this is the episode's total waiting time."""
    stopped = sum(1 for v in sim.vehicles.values() if v.v < WAIT_SPEED)
    return stopped + sim.spawner.backlog_size()


class DecisionClock:
    """Decision points: every ``interval`` seconds of simulated time, on a fixed grid from t = 0.
    At a point where the light can't change (min green not reached, amber, all-red, pedestrian
    phase) the action simply has no effect, and ``can_switch`` = 0 tells the agent so.

    WHY a fixed grid: an earlier version only asked when a switch was possible, so a "keep"
    covered 5 s but a "switch" covered amber + all-red + the next min green (12-35 s). Each
    decision's reward then scaled with how much TIME it covered, and PPO discounts per decision,
    not per second, so switching looked systematically expensive: two training runs learned to
    switch less and less and got worse. With every decision covering exactly ``interval``
    seconds, rewards are comparable and the discount means the same thing everywhere (this is
    also how SUMO-RL sets up signal control).
    Side effect: greens last whole multiples of the interval past min green (7 s min green ->
    the first possible switch is at 10 s)."""

    def __init__(self, interval: float):
        self.interval = interval

    def due(self, sim) -> bool:
        k = max(1, round(self.interval / sim.dt))
        return sim.tick % k == 0
