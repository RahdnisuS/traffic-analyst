"""Traffic-signal mechanics, kept separate from signal *policies*.

``SignalController`` knows HOW lights change: green -> amber -> all-red -> next
green, with a minimum green time it will never break. A *policy* decides WHEN
to ask for a switch. The fixed timer here is one policy. The RL agent (Phase 7)
and an actuated controller (Phase 10) are others. The safety rules live in one
place, so an RL bug can never produce an unsafe signal sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .junction import Phase

_EPS = 1e-6  # float tolerance when comparing accumulated times


class Stage(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    ALL_RED = "all_red"


@dataclass(frozen=True)
class PhaseEvent:
    time: float
    phase: str
    stage: str


class SignalController:
    def __init__(self, phases: list[Phase], amber_s: float, all_red_s: float, min_green_s: float):
        if not phases:
            raise ValueError("at least one phase required")
        self.phases = phases
        self.amber_s = amber_s
        self.all_red_s = all_red_s
        self.min_green_s = min_green_s
        self.phase_idx = 0
        self.stage = Stage.GREEN
        self.time_in_stage = 0.0
        self.t = 0.0
        self._target: int | None = None
        self.events: list[PhaseEvent] = [PhaseEvent(0.0, phases[0].name, Stage.GREEN.value)]

    # ---------------------------------------------------------------- commands
    def can_switch(self) -> bool:
        return self.stage is Stage.GREEN and self.time_in_stage >= self.min_green_s - _EPS

    def request_switch(self, target: int | None = None) -> bool:
        """Ask to end the current green. Ignored (returns False) during amber or
        all-red, or before min green has elapsed. ``target`` defaults to the next phase."""
        if not self.can_switch():
            return False
        target = (self.phase_idx + 1) % len(self.phases) if target is None else target
        if not 0 <= target < len(self.phases) or target == self.phase_idx:
            return False
        self._target = target
        self._enter(Stage.AMBER if self.amber_s > 0 else Stage.ALL_RED)
        return True

    def step(self, dt: float) -> None:
        self.t += dt
        self.time_in_stage += dt
        if self.stage is Stage.AMBER and self.time_in_stage >= self.amber_s - _EPS:
            self._enter(Stage.ALL_RED)
        if self.stage is Stage.ALL_RED and self.time_in_stage >= self.all_red_s - _EPS:
            self.phase_idx = self._target
            self._target = None
            self._enter(Stage.GREEN)

    # ---------------------------------------------------------------- queries
    @property
    def phase(self) -> Phase:
        return self.phases[self.phase_idx]

    def movement_state(self, arm: str, movement: str) -> str:
        """'G', 'Y' or 'R' for one approach movement."""
        if self.stage is Stage.ALL_RED or not self.phase.allows(arm, movement):
            return "R"
        return "G" if self.stage is Stage.GREEN else "Y"

    def arm_states(self, arms) -> dict[str, str]:
        """Per-arm summary light for renderers: an arm shows the best state of its movements."""
        rank = {"G": 2, "Y": 1, "R": 0}
        return {arm: max((self.movement_state(arm, m) for m in ("left", "straight", "right")),
                         key=rank.__getitem__) for arm in arms}

    def _enter(self, stage: Stage) -> None:
        self.stage = stage
        self.time_in_stage = 0.0
        self.events.append(PhaseEvent(round(self.t, 6), self.phase.name, stage.value))


class SignalPolicy(Protocol):
    def update(self, controller: SignalController, sim) -> None: ...


class FixedTimePolicy:
    """Classic pre-timed plan: each phase gets a fixed green, in a fixed order."""

    def __init__(self, green_times: list[float]):
        self.green_times = green_times

    def update(self, controller: SignalController, sim=None) -> None:
        if (controller.stage is Stage.GREEN
                and controller.time_in_stage >= self.green_times[controller.phase_idx] - _EPS):
            controller.request_switch()
