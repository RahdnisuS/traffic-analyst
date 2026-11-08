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
    # Behaviour flags (Phase 4). Size and IDM numbers alone already make a bus sluggish and a
    # bike nippy; these two capture the lane behaviour that differs by type on Indian roads.
    filters: bool = False         # may ride the gap between lanes to the front of a queue (bikes)
    kerb_lane: bool = False       # prefers the kerb-side lane when it has room (buses)

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


class DriverSettings(_Strict):
    # 0 = timid, 0.5 = the vehicle_types baseline, 1 = aggressive. Applied to newly spawned
    # drivers only; people already on the road keep their personality.
    aggressiveness: float = Field(0.5, ge=0, le=1)


class SignalSettings(_Strict):
    controller: Literal["fixed", "rl"] = "fixed"  # "rl" needs a trained model (python -m rl.train)
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
    filter_speed: float = Field(5.5, gt=0)        # m/s cap while squeezing past queued vehicles
    filter_conflict_distance_m: float = Field(1.5, gt=0)  # bike paths: narrower vehicle, smaller margin
    body_clearance_m: float = Field(0.3, ge=0)    # min gap between two vehicle BODIES in the box
    filter_trigger_gap: float = Field(12.0, gt=0)  # a bike starts filtering when a slow leader is this close
    lateral_speed: float = Field(1.2, gt=0)       # m/s sideways when changing into the filter lane (visual)


class PedestrianSettings(_Strict):
    """Social force model parameters (Helbing & Molnar 1995) plus demand and signal timing."""
    rate_ph: float = Field(300.0, ge=0)            # arrivals per hour, all crosswalks together
    desired_speed_mean: float = Field(1.34, gt=0)  # m/s, typical free walking speed
    desired_speed_std: float = Field(0.26, ge=0)
    radius: float = Field(0.25, gt=0)              # body radius (m)
    relaxation_time: float = Field(0.5, gt=0)      # tau: how quickly a walker reaches its desired velocity
    repulsion_strength: float = Field(2.5, gt=0)   # A (m/s^2): push between two people at contact
    repulsion_range: float = Field(0.3, gt=0)      # B (m): how fast that push fades with distance
    anisotropy: float = Field(0.35, ge=0, le=1)    # lambda: people react less to someone behind them
    anticipation_s: float = Field(0.5, ge=0)       # elliptical spec look-ahead (0 = circular model)
    wall_strength: float = Field(5.0, gt=0)        # kerbs and crosswalk edges
    wall_range: float = Field(0.1, gt=0)
    max_speed_factor: float = Field(1.3, ge=1)     # cap on |v| relative to desired speed
    walk_s: float = Field(7.0, gt=0)               # WALK interval (MUTCD minimum is 7 s)
    min_greens_between: int = Field(4, ge=1)       # vehicle greens between two pedestrian phases
                                                   # (= number of phases: at most once per cycle)
    clearance_speed: float = Field(1.07, gt=0)     # m/s design walking speed for the flashing
                                                   # DON'T WALK interval (MUTCD 3.5 ft/s)
    sidewalk_approach_m: float = Field(18.0, gt=0) # how far up the sidewalk people appear / vanish


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


class BenchmarkSettings(_Strict):
    """Stress test / capacity benchmark (Phase 6). See the benchmark section of config.yaml."""
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3], min_length=1)
    start_vph: float = Field(100.0, gt=0)
    step_vph: float = Field(50.0, gt=0)
    max_vph: float = Field(1500.0, gt=0)
    warmup_s: float = Field(300.0, ge=0)
    measure_s: float = Field(1200.0, gt=0)
    unserved_frac: float = Field(0.05, ge=0, lt=1)
    unserved_min: int = Field(10, ge=0)
    fails_to_stop: int = Field(2, ge=1)
    results_dir: str = "experiments/results/benchmark"
    live_start_vph: float = Field(100.0, gt=0)
    live_step_vph: float = Field(100.0, gt=0)
    live_step_s: float = Field(360.0, gt=0)
    live_settle_s: float = Field(60.0, ge=0)

    @model_validator(mode="after")
    def _ranges(self) -> "BenchmarkSettings":
        if self.max_vph < self.start_vph:
            raise ValueError("max_vph must be >= start_vph")
        if self.live_settle_s >= self.live_step_s:
            raise ValueError("live_settle_s must be < live_step_s (something must be measured)")
        return self


class PPOSettings(_Strict):
    """Stable-Baselines3 PPO hyperparameters (their documented defaults unless noted)."""
    learning_rate: float = Field(3e-4, gt=0)
    n_steps: int = Field(256, ge=8)        # decisions per env per rollout (x n_envs = batch collected)
    batch_size: int = Field(256, ge=8)
    n_epochs: int = Field(10, ge=1)
    gamma: float = Field(0.99, gt=0, le=1)
    gae_lambda: float = Field(0.95, gt=0, le=1)
    clip_range: float = Field(0.2, gt=0)
    ent_coef: float = Field(0.01, ge=0)
    net_arch: list[int] = Field(default_factory=lambda: [64, 64])


class RLSettings(_Strict):
    """Phase 7: PPO signal controller. See the rl section of config.yaml for the reasoning."""
    decision_interval_s: float = Field(5.0, gt=0)
    episode_s: float = Field(900.0, gt=0)
    reward_scale: float = Field(100.0, gt=0)
    switch_penalty: float = Field(0.2, ge=0)
    demand_vph: tuple[float, float] = (150.0, 600.0)
    demand_imbalance: float = Field(0.3, ge=0, lt=1)
    ped_rate_ph: tuple[float, float] = (0.0, 500.0)
    n_envs: int = Field(8, ge=1)
    total_timesteps: int = Field(300_000, ge=1)
    curve_eval_every: int = Field(25_000, ge=1)
    seed: int = 0
    eval_seeds: list[int] = Field(default_factory=lambda: [101, 102, 103, 104, 105])
    eval_demand_vph: dict[str, float] = Field(default_factory=lambda: {"low": 250.0, "medium": 400.0, "high": 500.0})
    model_path: str = "rl/models/ppo_signal.zip"
    results_dir: str = "experiments/results/rl"
    ppo: PPOSettings = PPOSettings()

    @model_validator(mode="after")
    def _ranges(self) -> "RLSettings":
        for name in ("demand_vph", "ped_rate_ph"):
            lo, hi = getattr(self, name)
            if not 0 <= lo <= hi:
                raise ValueError(f"{name} must be [low, high] with 0 <= low <= high")
        return self


class Config(_Strict):
    sim: SimSettings = SimSettings()
    vehicle_types: dict[str, VehicleTypeParams]
    demand: DemandSettings
    driver: DriverSettings = DriverSettings()
    signals: SignalSettings = SignalSettings()
    intersection: IntersectionSettings = IntersectionSettings()
    pedestrians: PedestrianSettings = PedestrianSettings()
    metrics: MetricsSettings = MetricsSettings()
    logging: LoggingSettings = LoggingSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
    rl: RLSettings = RLSettings()

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

    def rl_model_path(self) -> Path:
        p = Path(self.rl.model_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def rl_results_path(self) -> Path:
        p = Path(self.rl.results_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def benchmark_path(self) -> Path:
        p = Path(self.benchmark.results_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> Config:
    """Read and validate a YAML config file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
