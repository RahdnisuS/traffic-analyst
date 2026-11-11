"""Traffic-signal mechanics, kept separate from signal *policies*.

``SignalController`` knows HOW lights change: green -> amber -> all-red -> next
green, with a minimum green time it will never break. A *policy* decides WHEN
to ask for a switch. The fixed timer here is one policy. The RL agent (Phase 7)
and an actuated controller (Phase 10) are others. The safety rules live in one
place, so an RL bug can never produce an unsafe signal sequence.

Exclusive, on-demand pedestrian phase (Phase 4)
-----------------------------------------------
A pedestrian at the kerb presses the button (``call_pedestrian``). At the next phase
change (but at most once every ``min_greens_between_peds`` vehicle greens, by default once
per cycle), after the usual amber and all-red, the controller inserts:

    WALK (walk_s) -> PED_CLEAR, i.e. flashing DON'T WALK (clear_s) -> ALL_RED -> next green

Every vehicle movement is red throughout, so walkers cross every arm at once with no
turning traffic to watch for (an "exclusive" phase). WHY on demand: an empty crossing
costs vehicles nothing, so the cycle only grows when someone is actually waiting.
WHY at the next phase change and not immediately: the current green is never cut short,
so a policy's timing (and the RL agent's actions later) stay meaningful.
WHY at most once per cycle: with steady foot traffic someone is almost always waiting, and
serving them after every phase hands about a third of the clock to the pedestrian phase.
That is the known weakness of exclusive phases, and why real ones run once per cycle.
clear_s is crossing length / design walking speed, the standard MUTCD rule: someone who
steps off the kerb on the last second of WALK can still reach the far side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .junction import Phase

_EPS = 1e-6  # float tolerance when comparing accumulated times
PED_PHASE = "PED"


class Stage(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    ALL_RED = "all_red"
    WALK = "walk"
    PED_CLEAR = "ped_clear"


@dataclass(frozen=True)
class PhaseEvent:
    time: float
    phase: str
    stage: str


class SignalController:
    def __init__(self, phases: list[Phase], amber_s: float, all_red_s: float, min_green_s: float,
                 walk_s: float = 7.0, ped_clear_s: float = 13.0, min_greens_between_peds: int = 1):
        if not phases:
            raise ValueError("at least one phase required")
        self.phases = phases
        self.amber_s = amber_s
        self.all_red_s = all_red_s
        self.min_green_s = min_green_s
        self.walk_s = walk_s
        self.ped_clear_s = ped_clear_s
        self.phase_idx = 0
        self.stage = Stage.GREEN
        self.time_in_stage = 0.0
        self.t = 0.0
        self._target: int | None = None
        self.ped_call = False         # someone has pressed the button and is not yet served
        self.ped_call_time: float | None = None
        self._in_ped = False          # inside WALK / PED_CLEAR / the all-red that follows
        self.ped_phases_served = 0
        self.min_greens_between_peds = min_greens_between_peds
        self._greens_since_ped = min_greens_between_peds  # the very first call can be served
        self.events: list[PhaseEvent] = [PhaseEvent(0.0, phases[0].name, Stage.GREEN.value)]

    # ---------------------------------------------------------------- commands
    def can_switch(self) -> bool:
        return self.stage is Stage.GREEN and self.time_in_stage >= self.min_green_s - _EPS

    def request_switch(self, target: int | None = None) -> bool:
        """Ask to end the current green. Ignored (returns False) during amber, all-red or a
        pedestrian phase, or before min green has elapsed. ``target`` defaults to the next phase."""
        if not self.can_switch():
            return False
        target = (self.phase_idx + 1) % len(self.phases) if target is None else target
        if not 0 <= target < len(self.phases) or target == self.phase_idx:
            return False
        self._target = target
        self._enter(Stage.AMBER if self.amber_s > 0 else Stage.ALL_RED)
        return True

    def call_pedestrian(self) -> None:
        """Push button. Idempotent. A press during WALK needs no call: that walker just goes."""
        if self.stage is not Stage.WALK and not self.ped_call:
            self.ped_call = True
            self.ped_call_time = self.t

    def step(self, dt: float) -> None:
        self.t += dt
        self.time_in_stage += dt
        if self.stage is Stage.AMBER and self.time_in_stage >= self.amber_s - _EPS:
            self._enter(Stage.ALL_RED)
        if self.stage is Stage.WALK and self.time_in_stage >= self.walk_s - _EPS:
            self._enter(Stage.PED_CLEAR)
        if self.stage is Stage.PED_CLEAR and self.time_in_stage >= self.ped_clear_s - _EPS:
            self._enter(Stage.ALL_RED)
        if self.stage is Stage.ALL_RED and self.time_in_stage >= self.all_red_s - _EPS:
            if (self.ped_call and not self._in_ped
                    and self._greens_since_ped >= self.min_greens_between_peds):
                self._in_ped = True
                self.ped_call = False
                self.ped_call_time = None
                self.ped_phases_served += 1
                self._greens_since_ped = 0
                self._enter(Stage.WALK)
            else:
                self._in_ped = False
                self.phase_idx = self._target
                self._target = None
                self._greens_since_ped += 1
                self._enter(Stage.GREEN)

    # ---------------------------------------------------------------- queries
    @property
    def phase(self) -> Phase:
        """The vehicle phase that is green, or was green last (during a pedestrian phase)."""
        return self.phases[self.phase_idx]

    @property
    def phase_name(self) -> str:
        """Name for displays and logs: the vehicle phase, or "PED" during the pedestrian phase."""
        return PED_PHASE if self._in_ped else self.phase.name

    def movement_state(self, arm: str, movement: str) -> str:
        """'G', 'Y' or 'R' for one approach movement."""
        if self.stage not in (Stage.GREEN, Stage.AMBER) or not self.phase.allows(arm, movement):
            return "R"
        return "G" if self.stage is Stage.GREEN else "Y"

    def arm_states(self, arms) -> dict[str, str]:
        """Per-arm summary light for renderers: an arm shows the best state of its movements."""
        rank = {"G": 2, "Y": 1, "R": 0}
        return {arm: max((self.movement_state(arm, m) for m in ("left", "straight", "right")),
                         key=rank.__getitem__) for arm in arms}

    def ped_signal(self) -> str:
        """Pedestrian head: 'walk', 'flash' (flashing DON'T WALK: finish, don't start) or 'dont_walk'."""
        return {Stage.WALK: "walk", Stage.PED_CLEAR: "flash"}.get(self.stage, "dont_walk")

    def _enter(self, stage: Stage) -> None:
        self.stage = stage
        self.time_in_stage = 0.0
        self.events.append(PhaseEvent(round(self.t, 6), self.phase_name, stage.value))


class SignalPolicy(Protocol):
    def update(self, controller: SignalController, sim) -> None: ...


class ActuatedPolicy:
    """Classic fully-actuated control (Phase 10): the rule real vehicle-actuated signals run.

    Each approach has a DETECTOR covering the last ``detector_m`` metres before its stop line
    (an inductive loop / camera zone; on this layout it also covers the bike filter lane). While
    the green approach's detector is occupied, the green is extended; once it has been empty for
    ``passage_s`` (the "passage time" or "unit extension") the phase GAPS OUT and the controller
    moves to the next phase. The green never exceeds ``max_green_s`` (MAX-OUT), so a steady stream
    on one approach can't starve the others. Min green, amber, all-red and the pedestrian phase
    stay with SignalController, as for every policy.

    WHY this and gap-out both: gap-out (rl.env.GapOutPolicy) switches when the green approach has
    NO QUEUED VEHICLE, which needs to see the whole queue: easy in a simulator, but real
    controllers only know what their detectors report. This policy uses only what a loop near the
    stop line would see, so it is the fair "what a traffic engineer would install" baseline.

    Simplification: every phase is served in turn ("min recall"), even with no demand on it,
    exactly like the other controllers here (the RL agent can also only switch to the NEXT phase).
    Real controllers can skip an empty phase. Runs every simulation step (real actuated
    controllers evaluate continuously, not on the RL agent's 5 s grid).
    """

    name = "actuated"

    def __init__(self, detector_m: float = 30.0, passage_s: float = 2.0, max_green_s: float = 60.0):
        self.detector_m = detector_m
        self.passage_s = passage_s
        self.max_green_s = max_green_s
        self._last_call: float | None = None

    def _occupied(self, controller: SignalController, sim) -> bool:
        J = sim.junction
        for arm in sorted(controller.phase.green):
            for lane_id in J.approach_lanes(arm):
                lane = sim.lanes[lane_id]
                # Lanes are ordered front-first, so the first vehicle is the one nearest the line.
                if lane and lane[0].s >= J.edges[lane_id].length - self.detector_m:
                    return True
        return False

    def update(self, controller: SignalController, sim) -> None:
        if controller.stage is not Stage.GREEN:
            self._last_call = None
            return
        now = sim.time
        if self._last_call is None or self._occupied(controller, sim):
            self._last_call = now  # green just started, or a vehicle is on the detector
        if not controller.can_switch():
            return
        if now - self._last_call >= self.passage_s - _EPS or controller.time_in_stage >= self.max_green_s - _EPS:
            controller.request_switch()


def webster_timing(flows_vph: list[float], saturation_vph: float, lost_per_phase_s: float, min_green_s: float,
                   max_cycle_s: float, ped_rate_ph: float = 0.0, ped_phase_s: float = 0.0) -> dict:
    """Fixed-time plan for known demand with Webster's method (1958), the classic way to time a
    fixed signal: one phase per entry of ``flows_vph`` (split phasing: each phase serves one approach).

        y_i = q_i / s                  flow ratio of phase i (its demand / its saturation flow)
        Y   = sum(y_i)                 how loaded the junction is (>= 1: no cycle can serve it)
        L   = total lost time per cycle (amber + all-red per phase, + the pedestrian phase)
        C0  = (1.5 L + 5) / (1 - Y)    the cycle that minimises average delay
        g_i = (y_i / Y) (C0 - L)       green split in proportion to each phase's load

    The pedestrian phase (on demand, at most once per cycle) is lost time for vehicles when it
    runs; it runs in a cycle if anyone arrived during it, probability 1 - exp(-rate x C). That
    depends on C, so C is found by a few fixed-point iterations. The cycle is capped at
    ``max_cycle_s`` (when Y is near or above 1 the formula explodes; past capacity a longer cycle
    only adds delay), and every green is at least ``min_green_s``.
    """
    y = [q / saturation_vph for q in flows_vph]
    Y = sum(y)
    n = len(flows_vph)
    cycle = max_cycle_s
    lost = n * lost_per_phase_s
    for _ in range(20):
        p_ped = 1.0 - math.exp(-ped_rate_ph / 3600.0 * cycle) if ped_rate_ph > 0 else 0.0
        lost = n * lost_per_phase_s + p_ped * ped_phase_s
        c0 = (1.5 * lost + 5.0) / (1.0 - Y) if Y < 0.95 else max_cycle_s
        new = min(max(c0, lost + n * min_green_s), max_cycle_s)
        if abs(new - cycle) < 0.01:
            cycle = new
            break
        cycle = new
    green_total = cycle - lost
    greens = [max(min_green_s, green_total * yi / Y) if Y > 0 else green_total / n for yi in y]
    return {"cycle_s": round(cycle, 1), "greens_s": [round(g, 1) for g in greens], "Y": round(Y, 3),
            "lost_s": round(lost, 1), "capped": cycle >= max_cycle_s - 0.01}


class FixedTimePolicy:
    """Classic pre-timed plan: each phase gets a fixed green, in a fixed order.
    (A pedestrian phase, when called, is added between greens by the controller.)"""

    def __init__(self, green_times: list[float]):
        self.green_times = green_times

    def update(self, controller: SignalController, sim=None) -> None:
        if (controller.stage is Stage.GREEN
                and controller.time_in_stage >= self.green_times[controller.phase_idx] - _EPS):
            controller.request_switch()
