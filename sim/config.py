"""Typed, validated configuration loaded from config.yaml.

WHY Pydantic: a typo'd key or a negative headway should fail loudly at startup,
not quietly produce a strange simulation ten minutes later. ``extra="forbid"``
turns unknown keys (usually typos) into errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
Movement = Literal["left", "straight", "right"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimSettings(_Strict):
    dt: float = Field(0.1, gt=0, le=0.5)
    seed: int = 42
    duration_s: float = Field(600.0, gt=0)
    junction: str = "junctions/four_way_2x2.json"


class VehicleTypeParams(_Strict):
    """Physical size plus IDM parameters for one vehicle class."""

    length: float = Field(gt=0)
    width: float = Field(gt=0)
    max_accel: float = Field(gt=0)        # IDM a
    comfort_decel: float = Field(gt=0)    # IDM b
    max_decel: float = Field(gt=0)        # hard physical limit
    desired_speed: float = Field(gt=0)    # IDM v0
    time_headway: float = Field(gt=0)     # IDM T
    min_gap: float = Field(ge=0)          # IDM s0
    accel_exponent: float = Field(4.0, gt=0)  # IDM delta

    @model_validator(mode="after")
    def _braking_order(self) -> "VehicleTypeParams":
        if self.max_decel < self.comfort_decel:
            raise ValueError("max_decel must be >= comfort_decel")
        return self


class DemandSettings(_Strict):
    per_approach_vph: dict[str, float]
    turn_probabilities: dict[Movement, float]
    vehicle_mix: dict[str, float]
    speed_factor_std: float = Field(0.08, ge=0, le=0.5)

    @model_validator(mode="after")
    def _non_negative(self) -> "DemandSettings":
        for name, table in (("per_approach_vph", self.per_approach_vph),
                            ("turn_probabilities", self.turn_probabilities),
                            ("vehicle_mix", self.vehicle_mix)):
            if any(x < 0 for x in table.values()):
                raise ValueError(f"{name} values must be >= 0")
        if sum(self.turn_probabilities.values()) <= 0:
            raise ValueError("turn_probabilities must not all be zero")
        if sum(self.vehicle_mix.values()) <= 0:
            raise ValueError("vehicle_mix must not all be zero")
        return self


class SignalSettings(_Strict):
    controller: Literal["fixed"] = "fixed"
    green_s: float | list[float] = 25.0
    amber_s: float = Field(3.0, ge=0)
    all_red_s: float = Field(2.0, ge=0)
    min_green_s: float = Field(7.0, ge=0)

    def green_times(self, n_phases: int) -> list[float]:
        """Expand ``green_s`` into one value per phase and validate it."""
        greens = [float(self.green_s)] * n_phases if isinstance(self.green_s, (int, float)) \
            else [float(g) for g in self.green_s]
        if len(greens) != n_phases:
            raise ValueError(f"green_s has {len(greens)} entries but junction has {n_phases} phases")
        if any(g < self.min_green_s for g in greens):
            raise ValueError("every green_s must be >= min_green_s")
        return greens


class IntersectionSettings(_Strict):
    turn_lateral_accel: float = Field(2.5, gt=0)
    lookahead_m: float = Field(150.0, gt=0)
    conflict_distance_m: float = Field(2.5, gt=0)


class MetricsSettings(_Strict):
    queue_speed: float = Field(2.0, gt=0)
    window_s: float = Field(120.0, gt=0)
    queue_spike_on: int = Field(12, ge=1)
    queue_spike_off: int = Field(8, ge=0)
    spillback_on_frac: float = Field(0.9, gt=0, le=1)
    spillback_off_frac: float = Field(0.7, ge=0, le=1)
    gridlock_window_s: float = Field(60.0, gt=0)
    gridlock_stopped_frac: float = Field(0.9, gt=0, le=1)
    gridlock_min_vehicles: int = Field(4, ge=1)

    @model_validator(mode="after")
    def _hysteresis_order(self) -> "MetricsSettings":
        # "off" must be below "on", or an episode could start and end in the same second.
        if self.queue_spike_off >= self.queue_spike_on:
            raise ValueError("queue_spike_off must be < queue_spike_on")
        if self.spillback_off_frac >= self.spillback_on_frac:
            raise ValueError("spillback_off_frac must be < spillback_on_frac")
        return self


class LoggingSettings(_Strict):
    enabled: bool = True
    runs_dir: str = "data/runs"


class Config(_Strict):
    sim: SimSettings = SimSettings()
    vehicle_types: dict[str, VehicleTypeParams]
    demand: DemandSettings
    signals: SignalSettings = SignalSettings()
    intersection: IntersectionSettings = IntersectionSettings()
    metrics: MetricsSettings = MetricsSettings()
    logging: LoggingSettings = LoggingSettings()

    @model_validator(mode="after")
    def _mix_types_exist(self) -> "Config":
        missing = set(self.demand.vehicle_mix) - set(self.vehicle_types)
        if missing:
            raise ValueError(f"vehicle_mix references undefined vehicle types: {sorted(missing)}")
        return self

    def junction_path(self) -> Path:
        p = Path(self.sim.junction)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def runs_path(self) -> Path:
        p = Path(self.logging.runs_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> Config:
    """Read and validate a YAML config file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
