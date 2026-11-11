"""Controller comparison across random scenarios: ``python -m experiments``.

Five controllers, the same 90 random scenarios (30 per demand band), and a paired comparison:

  fixed     the config's fixed-time plan (25 s greens): what the dashboard's fixed timer does
  webster   a fixed-time plan re-timed for EACH scenario with Webster's formula, from that
            scenario's true demand and the MEASURED saturation flow: the fairest fixed timer
  actuated  classic vehicle-actuated control: a detector near the stop line, gap-out after a
            2 s passage time, 60 s max green (sim.signals.ActuatedPolicy)
  gap_out   the Phase 7 rule: switch once the green approach has no queue (sees the whole queue)
  rl        the trained PPO agent (rl/models/ppo_signal.zip)

WHY a PAIRED design: scenario-to-scenario variation (demand, imbalance, pedestrians) is far
bigger than the difference between controllers. Comparing each controller's average over
DIFFERENT random scenarios would bury the effect in that noise. Here every controller runs the
identical scenarios (same seed, same arrivals: every random draw is independent of the signals,
tested in Phase 8), so for each scenario the difference between two controllers is due to the
controllers alone. The report gives, per band, the mean +/- standard deviation of each metric
(how the junction performs), and for the AI vs each baseline the per-scenario % change with a
95% bootstrap confidence interval, the win count and a Wilcoxon signed-rank test (did it win
consistently, or on average because of a few scenarios?).

Each run: ``warmup_s`` from an empty junction (not scored), then ``measure_s`` scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim.config import PROJECT_ROOT, Config, load_config

LABEL = {"fixed": "Fixed 25 s", "webster": "Webster fixed", "actuated": "Actuated", "gap_out": "Gap-out",
         "rl": "RL agent"}
METRICS = {  # key: (label, unit, lower_is_better)
    "waiting_veh_h": ("Total waiting", "veh·h", True),
    "mean_delay_s": ("Mean control delay", "s/veh", True),
    "throughput_vph": ("Throughput", "veh/h", False),
    "served_pct": ("Demand served", "%", False),
    "max_queue": ("Worst queue on an approach", "veh", True),
    "mean_ped_wait_s": ("Pedestrian wait", "s", True),
    "greens_ph": ("Greens per hour", "", None),
}


# ---------------------------------------------------------------------- scenarios
def make_scenarios(cfg: Config) -> list[dict]:
    """``scenarios_per_band`` random scenarios per band, from one seeded generator."""
    e = cfg.experiments
    rng = np.random.default_rng(e.seed)
    arms = sorted(cfg.demand.per_approach_vph)
    types = sorted(cfg.demand.vehicle_mix)
    base_mix = np.array([cfg.demand.vehicle_mix[t] for t in types], float)
    base_mix /= base_mix.sum()
    out = []
    for band, (lo, hi) in e.bands.items():
        for k in range(e.scenarios_per_band):
            mean = rng.uniform(lo, hi)
            mix = rng.dirichlet(base_mix * e.mix_concentration)
            out.append({
                "id": f"{band}-{k:02d}", "band": band,
                "per_approach_vph": {a: round(mean * rng.uniform(1 - e.demand_imbalance, 1 + e.demand_imbalance), 1)
                                     for a in arms},
                "ped_rate_ph": round(rng.uniform(*e.ped_rate_ph), 1),
                "vehicle_mix": {t: round(float(w), 4) for t, w in zip(types, mix)},
                "seed": int(rng.integers(0, 2**31 - 1)),
            })
    return out


def scenario_config(cfg: Config, sc: dict) -> Config:
    c = cfg.model_copy(deep=True)
    c.demand.per_approach_vph = dict(sc["per_approach_vph"])
    c.demand.vehicle_mix = dict(sc["vehicle_mix"])
    c.pedestrians.rate_ph = sc["ped_rate_ph"]
    return c


# ---------------------------------------------------------------------- saturation flow
class _StopLineCounter:
    """Observer counting vehicles that cross each arm's stop line (leave its approach lanes)."""

    def __init__(self, sim, start_t: float):
        self.start_t = start_t
        self.lanes = {a: sim.junction.approach_lanes(a) for a in sorted(sim.junction.inbound)}
        self.prev = self._ids(sim)
        self.count = {a: 0 for a in self.lanes}

    def _ids(self, sim) -> dict[str, set[int]]:
        return {a: {v.id for lid in lanes for v in sim.lanes[lid]} for a, lanes in self.lanes.items()}

    def after_step(self, sim) -> None:
        now = self._ids(sim)
        if sim.time > self.start_t:
            for a in self.lanes:
                # Left the approach lanes and is now in the box (not just moved into a filter lane:
                # the filter lanes are part of the same approach, so they are in the set too).
                self.count[a] += len(self.prev[a] - now[a])
        self.prev = now


def measure_saturation_flow(cfg: Config) -> dict:
    """Vehicles per hour of green one approach discharges from a standing queue (the "s" in
    Webster's formula). Measured on this simulator, with its mixed traffic, because the textbook
    ~1,800 veh/h per lane is for cars only and our bikes discharge far faster.

    Probe: demand far past capacity on every approach (a queue is always waiting), no
    pedestrians, long fixed greens. s = vehicles crossing the stop line / green time, per arm,
    averaged. Lost time is counted separately as amber + all-red per phase (the usual
    approximation), so s is per second of DISPLAYED green and already includes the start-up delay
    and the vehicles that cross on amber."""
    from sim.engine import Simulation

    e = cfg.experiments
    c = cfg.model_copy(deep=True)
    c.demand.per_approach_vph = {a: e.saturation_probe_vph for a in c.demand.per_approach_vph}
    c.pedestrians.rate_ph = 0.0
    c.signals.green_s = e.saturation_probe_green_s
    sim = Simulation(c, seed=e.seed)
    warm, measure = 300.0, 1800.0
    counter = _StopLineCounter(sim, warm)
    sim.observers.append(counter)
    sim.run(warm + measure)
    n = len(sim.junction.phases)
    cycle = n * (e.saturation_probe_green_s + c.signals.amber_s + c.signals.all_red_s)
    green_h = measure / cycle * e.saturation_probe_green_s / 3600.0  # hours of green per arm
    per_arm = {a: round(k / green_h, 1) for a, k in counter.count.items()}
    return {"saturation_vph": round(statistics.fmean(per_arm.values()), 1), "per_arm": per_arm,
            "probe_vph": e.saturation_probe_vph, "probe_green_s": e.saturation_probe_green_s,
            "measured_s": measure}


def webster_plan(cfg: Config, sc: dict, saturation_vph: float) -> dict:
    from sim.signals import webster_timing

    sig, ped = cfg.signals, cfg.pedestrians
    phases = []
    for p in _phase_arms(cfg):
        phases.append(max(sc["per_approach_vph"][a] for a in p))  # the phase's critical approach
    ped_clear = _ped_clear_s(cfg)
    return webster_timing(phases, saturation_vph, sig.amber_s + sig.all_red_s, sig.min_green_s,
                          cfg.experiments.webster_max_cycle_s, sc["ped_rate_ph"],
                          ped.walk_s + ped_clear + sig.all_red_s)


def _phase_arms(cfg: Config) -> list[list[str]]:
    spec = json.loads(cfg.junction_path().read_text(encoding="utf-8"))
    return [sorted(p["green"]) for p in spec["phases"]]


def _ped_clear_s(cfg: Config) -> float:
    from sim.engine import Simulation
    return Simulation(cfg, seed=0).ped_clear_s


# ---------------------------------------------------------------------- one run
_RL = None


def _policy(cfg: Config, controller: str, plan: dict | None):
    global _RL
    if controller == "webster":
        from sim.signals import FixedTimePolicy
        return FixedTimePolicy(plan["greens_s"])
    if controller == "actuated":
        from sim.signals import ActuatedPolicy
        a = cfg.actuated
        return ActuatedPolicy(a.detector_m, a.passage_s, a.max_green_s)
    if controller == "gap_out":
        from rl.env import GapOutPolicy
        return GapOutPolicy(cfg.rl.decision_interval_s)
    if controller == "rl":
        from rl.policy import RLPolicy
        if _RL is None:
            _RL = RLPolicy.load(cfg)  # once per worker process
        return RLPolicy(_RL.model, cfg.rl.decision_interval_s, model_id=_RL.model_id)
    return None  # fixed: the config's plan


def run_one(cfg_json: str, sc: dict, controller: str, plan: dict | None) -> dict:
    """One scenario under one controller: warm up, then score ``measure_s``."""
    from rl.features import waiting_now
    from sim.engine import Simulation
    from sim.metrics import MetricsCollector

    cfg = scenario_config(Config.model_validate_json(cfg_json), sc)
    if controller == "webster":
        cfg.signals.green_s = list(plan["greens_s"])  # so the manifest-style config says what ran
    sim = Simulation(cfg, seed=sc["seed"], policy=_policy(cfg, controller, plan))
    col = MetricsCollector(sim, cfg.metrics)
    e = cfg.experiments
    dt = sim.dt
    warm_steps, measure_steps = round(e.warmup_s / dt), round(e.measure_s / dt)
    for _ in range(warm_steps):
        sim.step()
        col.after_step(sim)
    arrivals0 = sum(sim.spawner.arrivals_total.values())
    completed0 = sim.counters.completed
    greens0 = sum(1 for ev in sim.signals.events if ev.stage == "green")
    waited, delays, ped_waits, max_queue = 0.0, [], [], 0
    for _ in range(measure_steps):
        sim.step()
        waited += waiting_now(sim) * dt
        row, _, trips, peds = col.after_step(sim)
        delays += [t.delay for t in trips]
        ped_waits += [p.wait_time for p in peds]
        if row is not None:
            max_queue = max(max_queue, *(row[f"queue_{a}"] for a in sorted(sim.junction.inbound)))
    arrivals = sum(sim.spawner.arrivals_total.values()) - arrivals0
    departed = sim.counters.completed - completed0
    hours = e.measure_s / 3600.0
    return {
        "scenario": sc["id"], "band": sc["band"], "controller": controller, "seed": sc["seed"],
        "mean_vph": round(statistics.fmean(sc["per_approach_vph"].values()), 1),
        "ped_rate_ph": sc["ped_rate_ph"],
        "waiting_veh_h": round(waited / 3600.0, 4),
        "mean_delay_s": round(statistics.fmean(delays), 2) if delays else None,
        "throughput_vph": round(departed / hours, 1),
        "served_pct": round(100.0 * departed / arrivals, 2) if arrivals else None,
        "max_queue": max_queue,
        "mean_ped_wait_s": round(statistics.fmean(ped_waits), 1) if ped_waits else None,
        "greens_ph": round((sum(1 for ev in sim.signals.events if ev.stage == "green") - greens0) / hours, 1),
        "backlog_end": sim.spawner.backlog_size(),
        "emergency_stops": sim.counters.emergency_stops,
        "cycle_s": plan["cycle_s"] if plan else None,
    }


# ---------------------------------------------------------------------- statistics
def describe(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def paired(a_rows: dict[str, dict], b_rows: dict[str, dict], key: str, n_boot: int, rng) -> dict:
    """Per-scenario % change of A vs B in ``key`` (negative = A lower). Mean, bootstrap 95% CI of
    the mean, how many scenarios A did better, and the Wilcoxon signed-rank p-value on the raw
    differences (non-parametric: no assumption that the differences are normally distributed)."""
    from scipy.stats import wilcoxon

    common = [s for s in sorted(a_rows) if s in b_rows and a_rows[s][key] is not None and b_rows[s][key]]
    pct = np.array([100.0 * (a_rows[s][key] - b_rows[s][key]) / b_rows[s][key] for s in common])
    diff = np.array([a_rows[s][key] - b_rows[s][key] for s in common])
    if len(pct) < 2:
        return {"n": len(pct)}
    boots = rng.choice(pct, size=(n_boot, len(pct)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"n": len(pct), "mean_pct": float(pct.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "wins": int((diff < 0).sum()), "p_wilcoxon": p, "pct": pct.tolist()}


def analyse(rows: list[dict], cfg: Config) -> dict:
    e = cfg.experiments
    controllers = [c for c in e.controllers if any(r["controller"] == c for r in rows)]
    bands = list(e.bands) + ["all"]
    by = {}
    for band in bands:
        for c in controllers:
            rs = [r for r in rows if r["controller"] == c and (band == "all" or r["band"] == band)]
            by[(band, c)] = {r["scenario"]: r for r in rs}
    summary = []
    for band in bands:
        for c in controllers:
            rs = list(by[(band, c)].values())
            line = {"band": band, "controller": c, "scenarios": len(rs)}
            for k in METRICS:
                d = describe(rs, k)
                line[f"{k}_mean"] = None if d["mean"] is None else round(d["mean"], 3)
                line[f"{k}_std"] = None if d["std"] is None else round(d["std"], 3)
            summary.append(line)
    rng = np.random.default_rng(e.seed)
    comparisons = []
    focus = "rl" if "rl" in controllers else controllers[-1]
    for band in bands:
        for base in controllers:
            if base == focus:
                continue
            for key in ("waiting_veh_h", "mean_delay_s", "mean_ped_wait_s"):
                r = paired(by[(band, focus)], by[(band, base)], key, e.bootstrap, rng)
                comparisons.append({"band": band, "controller": focus, "baseline": base, "metric": key, **r})
    return {"controllers": controllers, "bands": bands, "summary": summary, "paired": comparisons, "focus": focus}


# ---------------------------------------------------------------------- report
def _fmt(v, digits=1):
    return "–" if v is None else f"{v:,.{digits}f}"


def render(res: dict, meta: dict, cfg: Config) -> str:
    e = cfg.experiments
    ctrls, focus = res["controllers"], res["focus"]
    S = {(s["band"], s["controller"]): s for s in res["summary"]}
    band_desc = ", ".join(f"{b} {lo:.0f}–{hi:.0f}" for b, (lo, hi) in e.bands.items())
    L = [
        "# Signal controllers compared across random scenarios", "",
        f"{meta['scenarios']} scenarios ({e.scenarios_per_band} per demand band: {band_desc} veh/h per approach, each "
        f"approach x U({1 - e.demand_imbalance:.1f}, {1 + e.demand_imbalance:.1f}), pedestrians "
        f"U({e.ped_rate_ph[0]:.0f}, {e.ped_rate_ph[1]:.0f})/h, random vehicle mix and seed). Every controller ran "
        f"the same scenarios: {e.warmup_s:.0f} s warm-up, then {e.measure_s:.0f} s scored. Commit "
        f"`{meta['git_commit']}`, {meta['runs']} runs in {meta['wall_s'] / 60:.1f} min.", "",
        f"Webster plans use the measured saturation flow of **{meta['saturation']['saturation_vph']:,.0f} veh/h of "
        f"green per approach** (mixed traffic; probe at {meta['saturation']['probe_vph']:.0f} veh/h, "
        f"{meta['saturation']['probe_green_s']:.0f} s greens).", "",
        "## Results by demand band (mean ± standard deviation over scenarios)", "",
    ]
    show = ["waiting_veh_h", "mean_delay_s", "served_pct", "max_queue", "mean_ped_wait_s", "greens_ph"]
    for band in res["bands"]:
        title = "All scenarios" if band == "all" else f"{band.capitalize()} demand ({e.bands[band][0]:.0f}–{e.bands[band][1]:.0f} veh/h per approach)"
        L += [f"### {title}", "",
              "| Controller | " + " | ".join(f"{METRICS[k][0]} ({METRICS[k][1]})" if METRICS[k][1] else METRICS[k][0]
                                            for k in show) + " |",
              "|---|" + "---:|" * len(show)]
        best = {}
        for k in show:
            lower = METRICS[k][2]
            if lower is None:
                continue
            vals = [(S[(band, c)][f"{k}_mean"], c) for c in ctrls if S[(band, c)][f"{k}_mean"] is not None]
            if vals:
                best[k] = (min if lower else max)(vals)[1]
        for c in ctrls:
            s = S[(band, c)]
            cells = []
            for k in show:
                d = 2 if k == "waiting_veh_h" else 1
                txt = f"{_fmt(s[f'{k}_mean'], d)} ± {_fmt(s[f'{k}_std'], d)}"
                cells.append(f"**{txt}**" if best.get(k) == c else txt)
            L.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Bold = best mean in that column.", "",
          f"## {LABEL[focus]} vs each baseline, scenario by scenario", "",
          "Per-scenario change in the metric (negative = the agent is better), mean with a 95% bootstrap confidence "
          "interval, scenarios where the agent was better, and the Wilcoxon signed-rank p-value (a small p means the "
          "difference is consistent across scenarios, not chance).", ""]
    for key in ("waiting_veh_h", "mean_delay_s", "mean_ped_wait_s"):
        L += [f"### {METRICS[key][0]}", "", "| Band | " + " | ".join(f"vs {LABEL[b]}" for b in ctrls if b != focus) + " |",
              "|---|" + "---|" * (len(ctrls) - 1)]
        for band in res["bands"]:
            cells = []
            for b in ctrls:
                if b == focus:
                    continue
                p = next(x for x in res["paired"] if x["band"] == band and x["baseline"] == b and x["metric"] == key)
                if p["n"] < 2:
                    cells.append("–")
                    continue
                pv = "<0.001" if p["p_wilcoxon"] < 0.001 else f"{p['p_wilcoxon']:.3f}"
                cells.append(f"{p['mean_pct']:+.1f}% [{p['ci_lo']:+.1f}, {p['ci_hi']:+.1f}], {p['wins']}/{p['n']}, p={pv}")
            L.append(f"| {band} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["![Results by band](plot_bands.png)", "", "![Paired differences](plot_paired.png)", ""]
    if meta["emergency_stops"]:
        L += [f"**Warning:** {meta['emergency_stops']} emergency stops across all runs.", ""]
    return "\n".join(L)


COLORS = {"fixed": "#c9c7c0", "webster": "#8a8983", "actuated": "#eb6834", "gap_out": "#1baf7a", "rl": "#2a78d6"}


def _style(ax, muted="#6b6a64", grid="#e6e4de"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    ax.yaxis.grid(True, color=grid)
    ax.set_axisbelow(True)


def plot(out: Path, res: dict, rows: list[dict], cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#2b2a27", "#6b6a64"
    ctrls, bands = res["controllers"], list(cfg.experiments.bands)
    S = {(s["band"], s["controller"]): s for s in res["summary"]}

    # 1) mean +/- std per band for the three headline metrics
    keys = ["mean_delay_s", "waiting_veh_h", "mean_ped_wait_s"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    w = 0.8 / len(ctrls)
    x = np.arange(len(bands))
    for ax, k in zip(axes, keys):
        _style(ax)
        for i, c in enumerate(ctrls):
            m = [S[(b, c)][f"{k}_mean"] or 0 for b in bands]
            sd = [S[(b, c)][f"{k}_std"] or 0 for b in bands]
            ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, m, w - 0.02, yerr=sd, color=COLORS[c], label=LABEL[c],
                   error_kw={"elinewidth": 1, "capsize": 2, "ecolor": muted})
        ax.set_xticks(x, [b.capitalize() for b in bands])
        label, unit, _ = METRICS[k]
        ax.set_title(f"{label} ({unit})", color=ink, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.suptitle(f"Mean ± standard deviation over {cfg.experiments.scenarios_per_band} scenarios per demand band "
                 "(lower is better)", color=muted,
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "plot_bands.png", dpi=130)
    plt.close(fig)

    # 2) paired % change in total waiting, agent vs each baseline, one dot per scenario
    focus = res["focus"]
    bases = [c for c in ctrls if c != focus]
    fig, axes = plt.subplots(1, len(bases), figsize=(3.4 * len(bases), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)  # jitter only
    for ax, b in zip(axes, bases):
        _style(ax)
        ax.axhline(0, color=muted, lw=1)
        for j, band in enumerate(bands):
            p = next(x for x in res["paired"] if x["band"] == band and x["baseline"] == b and x["metric"] == "waiting_veh_h")
            if p["n"] < 2:
                continue
            pct = np.array(p["pct"])
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(pct)), pct, s=10, color=COLORS[focus], alpha=0.45, lw=0)
            ax.errorbar(j + 0.3, p["mean_pct"], yerr=[[p["mean_pct"] - p["ci_lo"]], [p["ci_hi"] - p["mean_pct"]]],
                        fmt="o", color=ink, ms=4, capsize=3, lw=1.2)
        ax.set_xticks(range(len(bands)), [bd.capitalize() for bd in bands])
        ax.set_title(f"vs {LABEL[b]}", color=ink, fontsize=11, loc="left")
    axes[0].set_ylabel(f"{LABEL[focus]}: change in total waiting (%)", color=muted)
    fig.suptitle("Each dot is one scenario (below 0 = the agent waited less); black = mean with 95% CI",
                 color=muted, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "plot_paired.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Compare signal controllers across random scenarios.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--scenarios", type=int, default=None, help="override experiments.scenarios_per_band")
    ap.add_argument("--measure-s", type=float, default=None, help="override experiments.measure_s")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 scenarios per band, 300 s scored")
    ap.add_argument("--out", default=None, help="results folder (default: experiments.results_dir/<timestamp>)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    e = cfg.experiments
    if args.quick:
        e.scenarios_per_band, e.measure_s, e.warmup_s = 3, 300.0, 120.0
    if args.scenarios:
        e.scenarios_per_band = args.scenarios
    if args.measure_s:
        e.measure_s = args.measure_s
    controllers = list(e.controllers)
    if "rl" in controllers and not cfg.rl_model_path().is_file():
        print(f"warning: no trained RL model at {cfg.rl_model_path()}; running without the rl controller")
        controllers.remove("rl")
    e.controllers = controllers

    t0 = time.perf_counter()
    sat = measure_saturation_flow(cfg)
    print(f"saturation flow: {sat['saturation_vph']:.0f} veh/h of green per approach {sat['per_arm']} "
          f"[{time.perf_counter() - t0:.0f} s]")
    scenarios = make_scenarios(cfg)
    plans = {sc["id"]: webster_plan(cfg, sc, sat["saturation_vph"]) for sc in scenarios}
    jobs = [(sc, c) for sc in scenarios for c in controllers]
    workers = min(args.workers or e.workers, len(jobs))
    print(f"{len(scenarios)} scenarios x {len(controllers)} controllers = {len(jobs)} runs "
          f"({e.warmup_s:.0f} s warm-up + {e.measure_s:.0f} s scored) on {workers} workers")
    cfg_json = cfg.model_dump_json()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, cfg_json, sc, c, plans[sc["id"]] if c == "webster" else None) for sc, c in jobs]
        for n, f in enumerate(futures, 1):
            rows.append(f.result())
            if n % 50 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} runs ({time.perf_counter() - t0:.0f} s)")
    wall = time.perf_counter() - t0

    res = analyse(rows, cfg)
    from sim.logger import _git_commit
    now = datetime.now(timezone.utc)
    meta = {"created_at": now.isoformat(timespec="seconds"), "git_commit": _git_commit(),
            "config_hash": hashlib.sha256(cfg_json.encode()).hexdigest()[:12],
            "scenarios": len(scenarios), "runs": len(rows), "wall_s": round(wall, 1), "saturation": sat,
            "emergency_stops": sum(r["emergency_stops"] for r in rows), "settings": e.model_dump(mode="json"),
            "actuated": cfg.actuated.model_dump(mode="json")}
    out = Path(args.out) if args.out else cfg.experiments_path() / (f"{now:%Y%m%d-%H%M%S}" + ("_quick" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["scenario"], controllers.index(r["controller"]))))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["summary"][0]))
        w.writeheader()
        w.writerows(res["summary"])
    with open(out / "paired.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["band", "controller", "baseline", "metric", "n", "mean_pct", "ci_lo", "ci_hi", "wins", "p_wilcoxon"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in res["paired"]:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k in cols})
    (out / "scenarios.json").write_text(json.dumps({"scenarios": scenarios, "webster_plans": plans}, indent=2),
                                        encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = render(res, meta, cfg)
    (out / "report.md").write_text(report, encoding="utf-8")
    plot(out, res, rows, cfg)
    print(report)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
