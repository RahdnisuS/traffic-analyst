"""What the queue predictor sees: features built from the per-second metrics rows.

Shared by training (rows read back from a run's metrics.csv) and the live dashboard (rows from
the MetricsCollector's history). WHY one module, as in rl/features.py: if the two computed a
feature even slightly differently, the model would be fed inputs it never saw in training and
be quietly wrong. Here each feature has exactly one definition, and a test checks that a live
run and its own log give identical features.

WHY features from the metrics ROWS (not from the simulation's internals): the predictor is
"trained on logged runs", so it may only use what a log records. Anything it needs live must be
in the rows too. The one exception is the CONTEXT (the settings at that moment: demand, green
time, pedestrian rate, controller), which comes from the run's config: the manifest offline,
``sim.cfg`` live. Those are the slider values, so a real deployment would know them as well.

One sample = one approach at one second. The model is shared by all four approaches and the
features are APPROACH-RELATIVE ("this approach's queue", "phases until this approach's green"),
never "is this the North arm". WHY: the arms differ only in their demand, which is a feature, so
one model learns from 4x the data and works for any approach, including a busier one than it
has seen as North.

Every look-back is limited to the last ``LOOKBACK_S`` seconds, and seconds before the run
started count as an empty junction (queues of 0), which is what the junction was. So a feature
never depends on how much history happens to be kept, only on the last 5 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOOKBACK_S = 300
LAGS = (10, 30, 60, 90, 120)
CONTROLLERS = {"fixed": 0, "gap_out": 1, "rl": 2}
STAGES = {"green": 0, "amber": 1, "all_red": 2, "walk": 3, "ped_clear": 4}
PED_STAGES = (STAGES["walk"], STAGES["ped_clear"])
PED_PHASE = "PED"

FEATURES = (
    # this approach, now and recently
    ["queue_now"] + [f"queue_lag{s}" for s in LAGS] + ["queue_mean120", "queue_max120",
     "backlog_now", "backlog_lag60", "wait_now",
     "arrivals_vph60", "arrivals_vph300",
     # where this approach is in the signal cycle
     "phases_until_green", "green_now", "secs_since_green", "green_share120",
     # the signal as a whole
     "stage", "time_in_stage", "queue_green_arm", "secs_since_ped", "greens_since_ped",
     # the rest of the junction
     "queue_others", "backlog_others",
     "peds_waiting",
     # context: the settings at this moment (sliders), and how long the run has been going
     "demand_vph", "demand_others_vph", "ped_rate_ph", "green_s", "controller", "run_time"]
)
CATEGORICAL = ["controller", "stage"]  # told to LightGBM as categories, not ordered numbers


@dataclass(frozen=True)
class Context:
    """The settings in force at a moment (from the run config, i.e. the sliders)."""
    controller: str
    demand_vph: dict[str, float]
    ped_rate_ph: float
    green_s: float

    @staticmethod
    def from_config(cfg, controller: str) -> "Context":
        greens = cfg.signals.green_s
        green = float(greens if isinstance(greens, (int, float)) else np.mean(greens))
        return Context(controller=controller, demand_vph=dict(cfg.demand.per_approach_vph),
                       ped_rate_ph=float(cfg.pedestrians.rate_ph), green_s=green)


class History:
    """The columns of the metrics rows the features need, as arrays (row k = second k+1).

    ``phases``: vehicle phase names in serving order (from the junction); ``arm_phase``: the
    index of the phase that gives each arm its green. During the pedestrian phase the logged
    phase is "PED"; for "where are we in the cycle" what matters is the vehicle phase served
    last, so "PED" rows carry the previous vehicle phase forward.
    """

    def __init__(self, rows: Sequence[dict], phases: Sequence[str], arm_phase: dict[str, int]):
        self.arms = sorted(arm_phase)
        self.n_phases = len(phases)
        self.arm_phase = arm_phase
        n = len(rows)
        col = lambda key: np.fromiter((float(r[key]) for r in rows), float, n)  # noqa: E731
        self.t = col("t")
        self.queue = {a: col(f"queue_{a}") for a in self.arms}
        self.backlog = {a: col(f"backlog_{a}") for a in self.arms}
        self.arrivals = {a: col(f"arrivals_{a}") for a in self.arms}
        self.wait = {a: col(f"wait_{a}") for a in self.arms}
        self.peds_waiting = col("peds_waiting")
        self.time_in_stage = col("time_in_stage")
        self.stage = np.fromiter((STAGES[r["stage"]] for r in rows), np.int64, n)
        index = {p: i for i, p in enumerate(phases)}
        vphase = np.empty(n, np.int64)
        last = 0  # every run starts on the first phase
        for k, r in enumerate(rows):
            if r["phase"] != PED_PHASE:
                last = index[r["phase"]]
            vphase[k] = last
        self.vphase = vphase
        green = self.stage == STAGES["green"]
        self.green_arm = {a: green & (vphase == arm_phase[a]) for a in self.arms}
        self.green_onset = green & ~np.concatenate([[False], green[:-1]])

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def from_phases(cls, rows: Sequence[dict], phases: Sequence[tuple[str, Sequence[str]]]) -> "History":
        """``phases``: (name, arms given green) in serving order; see ``junction_phases`` and
        ``manifest_phases``."""
        arm_phase: dict[str, int] = {}
        for i, (_, arms) in enumerate(phases):
            for arm in sorted(arms):
                arm_phase.setdefault(arm, i)
        return cls(rows, [name for name, _ in phases], arm_phase)


def junction_phases(junction) -> list[tuple[str, list[str]]]:
    """Phases of a live Junction, for History.from_phases."""
    return [(p.name, sorted(p.green)) for p in junction.phases]


def manifest_phases(manifest: dict) -> list[tuple[str, list[str]]]:
    """The same from a run log's manifest (no need to rebuild the junction geometry)."""
    return [(p["name"], sorted(p["green"])) for p in manifest["junction"]["spec"]["phases"]]


def _lag(x: np.ndarray, i: int, lag: int) -> float:
    return float(x[i - lag]) if i >= lag else 0.0  # before the run: empty junction


def _window(x: np.ndarray, i: int, length: int) -> np.ndarray:
    """The last ``length`` seconds up to and including row i, zero-padded before the run."""
    seg = x[max(0, i - length + 1): i + 1]
    if len(seg) < length:
        seg = np.concatenate([np.zeros(length - len(seg), seg.dtype), seg])
    return seg


def _secs_since(mask: np.ndarray, i: int) -> float:
    """Seconds since ``mask`` was last true at or before row i (0 = true now), capped at the
    look-back (never true in the window = LOOKBACK_S)."""
    lo = max(0, i - LOOKBACK_S + 1)
    hits = np.flatnonzero(mask[lo: i + 1])
    return float(i - (lo + hits[-1])) if len(hits) else float(LOOKBACK_S)


def features_at(h: History, i: int, ctx: Context) -> np.ndarray:
    """Feature matrix (one row per arm, in ``h.arms`` order, columns = FEATURES) at row i."""
    lo = max(0, i - LOOKBACK_S + 1)
    stage = int(h.stage[i])
    vphase = int(h.vphase[i])
    ped_hits = np.flatnonzero(np.isin(h.stage[lo: i + 1], PED_STAGES))
    since_ped = float(i - (lo + ped_hits[-1])) if len(ped_hits) else float(LOOKBACK_S)
    first = lo + ped_hits[-1] + 1 if len(ped_hits) else lo
    greens_since_ped = float(h.green_onset[first: i + 1].sum())
    green_arm = next((a for a in h.arms if h.green_arm[a][i]), None)
    q_green = float(h.queue[green_arm][i]) if green_arm else 0.0
    q_all = sum(float(h.queue[a][i]) for a in h.arms)
    b_all = sum(float(h.backlog[a][i]) for a in h.arms)
    d_all = sum(ctx.demand_vph.get(a, 0.0) for a in h.arms)
    others = max(len(h.arms) - 1, 1)
    out = np.empty((len(h.arms), len(FEATURES)))
    for k, a in enumerate(h.arms):
        q = h.queue[a]
        w120 = _window(q, i, 120)
        green = bool(h.green_arm[a][i])
        pos = (h.arm_phase[a] - vphase) % h.n_phases
        if pos == 0 and not green:
            pos = h.n_phases  # its green has just ended: it is last in line now
        demand = ctx.demand_vph.get(a, 0.0)
        out[k] = [
            q[i], *(_lag(q, i, s) for s in LAGS), w120.mean(), w120.max(),
            h.backlog[a][i], _lag(h.backlog[a], i, 60), h.wait[a][i],
            _window(h.arrivals[a], i, 60).sum() * 60.0, _window(h.arrivals[a], i, 300).sum() * 12.0,
            pos, float(green), _secs_since(h.green_arm[a], i), _window(h.green_arm[a], i, 120).mean(),
            stage, h.time_in_stage[i], q_green, since_ped, greens_since_ped,
            q_all - q[i], b_all - h.backlog[a][i],
            h.peds_waiting[i],
            demand, (d_all - demand) / others, ctx.ped_rate_ph, ctx.green_s,
            CONTROLLERS[ctx.controller], min(h.t[i], float(LOOKBACK_S)),
        ]
    return out
