"""Demand generation: Poisson arrivals per approach, inserted when there is room.

WHY Poisson: arrivals from many independent drivers are well modelled as a
Poisson process. The number arriving in a step of length dt is Poisson(rate*dt).
WHY a backlog: if the first metres of a lane are full (the queue has spilled all
the way back), the arrival waits off-network instead of vanishing. Demand is
never silently dropped, and the backlog size shows how much demand went unserved
(this matters for the gridlock benchmark in Phase 6).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import DemandSettings, VehicleTypeParams
from .junction import Junction


@dataclass(slots=True)
class PendingVehicle:
    arrival_time: float
    vtype: str
    movement: str
    speed_factor: float


class Spawner:
    def __init__(self, junction: Junction, demand: DemandSettings,
                 vehicle_types: dict[str, VehicleTypeParams], rng: np.random.Generator):
        self.junction = junction
        self.demand = demand
        self.vehicle_types = vehicle_types
        self.rng = rng
        self.arms = sorted(junction.inbound)  # sorted: fixed RNG draw order -> determinism
        self.backlog: dict[str, deque[PendingVehicle]] = {a: deque() for a in self.arms}
        self._refresh_tables()

    def _refresh_tables(self) -> None:
        """Normalise probabilities once. Call again if demand settings change (sliders, Phase 3)."""
        tp = self.demand.turn_probabilities
        self._moves: dict[str, tuple[list[str], np.ndarray]] = {}
        for arm in self.arms:
            moves = [m for m in ("left", "straight", "right")
                     if tp.get(m, 0) > 0 and (arm, m) in self.junction.lanes_for_movement]
            if not moves:
                raise ValueError(f"arm {arm} has no possible movement for the given turn probabilities")
            p = np.array([tp[m] for m in moves], dtype=float)
            self._moves[arm] = (moves, p / p.sum())
        mix = {k: v for k, v in self.demand.vehicle_mix.items() if v > 0}
        self._types = sorted(mix)
        p = np.array([mix[t] for t in self._types], dtype=float)
        self._type_p = p / p.sum()

    def set_demand(self, demand: DemandSettings) -> None:
        self.demand = demand
        self._refresh_tables()

    def generate(self, now: float, dt: float) -> None:
        """Sample this step's arrivals into the per-approach backlogs."""
        for arm in self.arms:
            rate = self.demand.per_approach_vph.get(arm, 0.0) / 3600.0
            for _ in range(int(self.rng.poisson(rate * dt))):
                moves, p = self._moves[arm]
                movement = moves[int(self.rng.choice(len(moves), p=p))]
                vtype = self._types[int(self.rng.choice(len(self._types), p=self._type_p))]
                factor = float(np.clip(self.rng.normal(1.0, self.demand.speed_factor_std), 0.7, 1.3))
                self.backlog[arm].append(PendingVehicle(now, vtype, movement, factor))

    def backlog_size(self) -> int:
        return sum(len(q) for q in self.backlog.values())
