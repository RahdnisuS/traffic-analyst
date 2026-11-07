"""Stress test and capacity benchmark.

WHAT "capacity" means here
--------------------------
Capacity is the most traffic the junction can serve *sustainably*. We find it by stepping
demand up and checking, at each level, whether the junction keeps up.

WHY "oversaturation" and not "gridlock"
---------------------------------------
Our drivers obey a box-junction rule: nobody enters the box without room to leave it. So
the box itself never locks up. Past capacity, the extra demand simply queues outside the
network (the spawner's backlog), growing without limit. That growing queue *is* the
breakdown, and it is what traffic engineering (HCM) calls oversaturation: demand > capacity.
Literal gridlock (nobody leaving for 60 s, see metrics.py) is still checked and fails a level.

HOW a level is judged (conservation of vehicles)
------------------------------------------------
Vehicles are never created or destroyed between arriving and leaving, so for each approach
over a measured window:

    arrivals - departures = growth of the approach's STOCK
    (stock = its vehicles on the road + its vehicles waiting to enter)

If the stock grows by more than ``max(unserved_frac x arrivals, unserved_min)`` on ANY
approach, that approach is serving less than ~95% of its demand and the level fails.
Judging per approach matters: with unequal demand (N/S busier than E/W) one approach
breaks down first, and a junction-wide total would hide it for a while.

Measuring the growth needs care. Each approach's stock is a SAWTOOTH: it fills for ~110 s of
red and empties on green. Comparing the stock at two single instants mostly measures where in
the cycle each instant fell (an early version did this: the same junction scored anywhere from
1,000 to 2,300 veh/h depending on the seed). So we compare the stock AVERAGED OVER ONE SIGNAL
CYCLE at the start of the window with the same average at the end, which cancels the sawtooth,
and scale that trend to the whole window. ``unserved_min`` only absorbs what is left (the
occasional pedestrian phase makes cycles uneven).

HOW the ramp works
------------------
Each level is an INDEPENDENT run (fresh sim, warm-up, then a measured window), so a level's
result does not depend on queues left over from the previous one. Levels go up in fixed
steps until ``fails_to_stop`` consecutive failures (one failure could be noise). Each seed is
a separate ramp; the score is reported as mean +/- std over seeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sim.config import PROJECT_ROOT, BenchmarkSettings, Config
from sim.engine import Simulation
from sim.metrics import MetricsCollector, level_of_service

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------- judging
def approach_stock(sim) -> dict[str, int]:
    """Vehicles per approach that have arrived but not yet left: waiting to enter + on the road."""
    stock = {a: len(q) for a, q in sim.spawner.backlog.items()}
    for v in sim.vehicles.values():
        stock[v.origin] += 1
    return stock


def cycle_length(sim) -> float:
    """Vehicle signal cycle (s) from the current plan: sum of green + amber + all-red per phase."""
    sg = sim.cfg.signals
    greens = getattr(sim.policy, "green_times", None) or [60.0] * len(sim.junction.phases)
    return float(sum(g + sg.amber_s + sg.all_red_s for g in greens))


@dataclass(frozen=True)
class Snapshot:
    """Counters at one instant. Two snapshots bracket a measured window."""
    t: float
    arrivals: dict[str, int]
    stock: dict[str, int]
    completed: int
    gridlocks: int

    @classmethod
    def take(cls, sim, collector: MetricsCollector | None = None) -> "Snapshot":
        return cls(t=sim.time, arrivals=dict(sim.spawner.arrivals_total), stock=approach_stock(sim),
                   completed=sim.counters.completed,
                   gridlocks=collector.event_counts["gridlock"] if collector else 0)


def stock_growth(samples: list[tuple[float, dict[str, int]]], cycle_s: float) -> dict[str, float]:
    """Growth of each approach's stock over the sampled window, with the signal sawtooth removed.

    ``samples`` are (t, stock) pairs, roughly once a second. We average the stock over the first
    and the last ``cycle_s`` seconds (capped at half the window), take the difference between
    those averages as the trend, and scale it from "centre to centre" up to the full window.
    """
    if len(samples) < 2:
        return {a: 0.0 for a in (samples[0][1] if samples else {})}
    t0, t1 = samples[0][0], samples[-1][0]
    span = min(cycle_s, (t1 - t0) / 2)
    head = [st for t, st in samples if t <= t0 + span]
    tail = [st for t, st in samples if t >= t1 - span]
    gap = (t1 - t0) - span  # distance between the centres of the two averaging blocks
    out = {}
    for a in samples[0][1]:
        h = sum(st[a] for st in head) / len(head)
        tl = sum(st[a] for st in tail) / len(tail)
        out[a] = (tl - h) * (t1 - t0) / gap if gap > 0 else tl - h
    return out


def judge(start: Snapshot, end: Snapshot, s: BenchmarkSettings, gridlock: bool = False,
          growth: dict[str, float] | None = None) -> dict:
    """Did the junction keep up with demand between two snapshots? Pure function.

    ``growth`` is the cycle-averaged stock growth per approach (see ``stock_growth``); without
    it the raw endpoint difference is used (only sensible for tests or very long windows).
    """
    dur = max(end.t - start.t, 1e-9)
    per = {}
    for a in sorted(start.arrivals):
        arrived = end.arrivals[a] - start.arrivals[a]
        # conservation: arrivals - departures = stock growth
        unserved = round(growth[a], 1) if growth is not None else end.stock[a] - start.stock[a]
        allowed = max(s.unserved_frac * arrived, s.unserved_min)
        per[a] = {"arrivals": arrived, "unserved": unserved, "allowed": round(allowed, 1),
                  "offered_vph": round(arrived / dur * 3600, 1),
                  "served_vph": round((arrived - unserved) / dur * 3600, 1),
                  "ok": unserved <= allowed}
    arrived = sum(p["arrivals"] for p in per.values())
    served = end.completed - start.completed
    failing = [a for a, p in per.items() if not p["ok"]]
    return {
        "offered_vph": round(arrived / dur * 3600, 1),
        "served_vph": round(served / dur * 3600, 1),
        # served / offered; can exceed 1 when queues left over from earlier drain during the window
        "served_ratio": round(served / arrived, 3) if arrived else 1.0,
        "gridlock": gridlock,
        "failing_approaches": failing,
        "passed": not failing and not gridlock,
        "approaches": per,
    }


# ---------------------------------------------------------------------------- one level
def run_level(cfg: Config, seed: int, volume_vph: float, s: BenchmarkSettings | None = None) -> dict:
    """One independent run at a fixed demand: warm up, then measure and judge."""
    s = s or cfg.benchmark
    sim = Simulation(cfg, seed=seed)
    sim.apply_settings(volume_vph=volume_vph, source="benchmark")
    col = MetricsCollector(sim, sim.cfg.metrics)
    steps = lambda dur: int(round(dur / sim.dt))  # noqa: E731

    for _ in range(steps(s.warmup_s)):
        sim.step()
        col.after_step(sim)
    start = Snapshot.take(sim, col)
    samples = [(start.t, start.stock)]
    gridlock_open = col.episode_active("gridlock")
    delays, max_queue = [], {a: 0 for a in sim.junction.inbound}
    for _ in range(steps(s.measure_s)):
        sim.step()
        row, _, trips, _ = col.after_step(sim)
        delays.extend(tr.delay for tr in trips)
        if row:  # once per simulated second
            samples.append((sim.time, approach_stock(sim)))
            for a in max_queue:
                max_queue[a] = max(max_queue[a], row[f"queue_{a}"])
    end = Snapshot.take(sim, col)
    gridlock = gridlock_open or end.gridlocks > start.gridlocks or col.episode_active("gridlock")
    out = judge(start, end, s, gridlock, growth=stock_growth(samples, cycle_length(sim)))
    mean_delay = statistics.fmean(delays) if delays else None
    out.update(seed=seed, volume_vph=round(volume_vph, 1), mean_delay_s=round(mean_delay, 1) if delays else None,
               los=level_of_service(mean_delay), max_queue=max_queue,
               emergency_stops=sim.counters.emergency_stops)
    return out


# ---------------------------------------------------------------------------- one ramp
def ramp(cfg: Config, seed: int, s: BenchmarkSettings | None = None, verbose: bool = False) -> dict:
    """Step demand up for one seed until ``fails_to_stop`` consecutive failures (or max_vph)."""
    s = s or cfg.benchmark
    levels, fails, vph = [], 0, s.start_vph
    while vph <= s.max_vph + 1e-9:
        r = run_level(cfg, seed, vph, s)
        levels.append(r)
        if verbose:
            mark = "pass" if r["passed"] else f"FAIL ({', '.join(r['failing_approaches']) or 'gridlock'})"
            print(f"  seed {seed}: {vph:6.0f} veh/h per approach -> offered {r['offered_vph']:6.0f}, "
                  f"served {r['served_vph']:6.0f} veh/h  {mark}", flush=True)
        fails = 0 if r["passed"] else fails + 1
        if fails >= s.fails_to_stop:
            break
        vph += s.step_vph
    return {"seed": seed, "levels": levels, **score_ramp(levels, s)}


def score_ramp(levels: list[dict], s: BenchmarkSettings) -> dict:
    """Capacity = served throughput at the highest level that passed.

    "censored" means the ramp hit max_vph without breaking down: the true capacity is at least
    the reported value, so it must not be read as an exact figure.

    "peak_served_vph" is the most the junction pushed through at ANY level, including failed
    ones. Under overload every green is fully used, so it is a bit higher than the sustainable
    capacity, but it is not a rate the junction can hold: its queues are growing.
    """
    peak = max((r["served_vph"] for r in levels), default=None)
    passed = [r for r in levels if r["passed"]]
    if not passed:
        return {"capacity_vph": None, "capacity_level_vph": None, "breakdown_level_vph": levels[0]["volume_vph"]
                if levels else None, "censored": False, "peak_served_vph": peak}
    best = max(passed, key=lambda r: r["volume_vph"])
    later_fails = [r for r in levels if not r["passed"] and r["volume_vph"] > best["volume_vph"]]
    return {"capacity_vph": best["served_vph"], "capacity_level_vph": best["volume_vph"],
            "breakdown_level_vph": later_fails[0]["volume_vph"] if later_fails else None,
            "censored": not later_fails, "peak_served_vph": peak}


# ---------------------------------------------------------------------------- full benchmark
def _ramp_job(args) -> dict:
    cfg_json, seed, verbose = args
    return ramp(Config.model_validate_json(cfg_json), seed, verbose=verbose)


def run_benchmark(cfg: Config, seeds: list[int] | None = None, workers: int = 1, verbose: bool = True) -> dict:
    """Ramp every seed (in parallel processes if workers > 1) and summarise.

    Parallelism is safe for determinism: each run owns its own seeded RNG, and results are
    collected in seed order, not completion order.
    """
    seeds = list(seeds or cfg.benchmark.seeds)
    t0 = time.perf_counter()
    jobs = [(cfg.model_dump_json(), seed, verbose) for seed in seeds]
    if workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as ex:
            ramps = list(ex.map(_ramp_job, jobs))
    else:
        ramps = [_ramp_job(j) for j in jobs]
    caps = [r["capacity_vph"] for r in ramps if r["capacity_vph"] is not None]
    score = None
    if caps:
        score = {"mean_vph": round(statistics.fmean(caps), 1),
                 "std_vph": round(statistics.stdev(caps), 1) if len(caps) > 1 else 0.0,
                 "min_vph": min(caps), "max_vph": max(caps), "n": len(caps),
                 "censored": any(r["censored"] for r in ramps),
                 "peak_served_mean_vph": round(statistics.fmean(r["peak_served_vph"] for r in ramps), 1)}
    return {"schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "config_hash": config_hash(cfg),
            "junction": Path(cfg.sim.junction).stem, "seeds": seeds,
            "settings": _settings_summary(cfg), "score": score,
            "definition": ("Capacity = total junction throughput (veh/h) at the highest demand level where "
                           "every approach served at least "
                           f"{(1 - cfg.benchmark.unserved_frac) * 100:.0f}% of its arrivals "
                           f"(comparing cycle-averaged queues, with a {cfg.benchmark.unserved_min}-vehicle noise floor) over a "
                           f"{cfg.benchmark.measure_s:.0f} s window after {cfg.benchmark.warmup_s:.0f} s "
                           "of warm-up, with no gridlock."),
            "ramps": ramps, "wall_time_s": round(time.perf_counter() - t0, 1)}


def config_hash(cfg: Config) -> str:
    """Short fingerprint of the full config: two results are comparable only if these match."""
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:12]


def _settings_summary(cfg: Config) -> dict:
    return {"benchmark": cfg.benchmark.model_dump(),
            "demand_ratios": cfg.demand.per_approach_vph, "vehicle_mix": cfg.demand.vehicle_mix,
            "turn_probabilities": cfg.demand.turn_probabilities, "controller": cfg.signals.controller,
            "green_s": cfg.signals.green_s, "aggressiveness": cfg.driver.aggressiveness,
            "ped_rate_ph": cfg.pedestrians.rate_ph}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip()
        if not commit:
            return None
        # Uncommitted changes to the code that produced the result? Say so ("-dirty", like
        # `git describe`): the commit alone would claim code that is not what actually ran.
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "sim", "bench", "config.yaml", "junctions"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------- saving
LEVEL_COLUMNS = ["seed", "volume_vph", "offered_vph", "served_vph", "served_ratio", "passed",
                 "failing_approaches", "gridlock", "mean_delay_s", "los", "emergency_stops"]


def save_results(result: dict, results_dir: Path) -> Path:
    """Write summary.json, levels.csv, report.md and (if matplotlib is installed) plot.png."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(results_dir) / f"{stamp}_s{'-'.join(map(str, result['seeds']))}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    approaches = sorted(result["ramps"][0]["levels"][0]["approaches"]) if result["ramps"] else []
    with open(out / "levels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(LEVEL_COLUMNS + [f"{k}_{a}" for a in approaches for k in ("offered", "served", "unserved", "maxq")])
        for rp in result["ramps"]:
            for r in rp["levels"]:
                base = [r[c] if c != "failing_approaches" else " ".join(r[c]) for c in LEVEL_COLUMNS]
                extra = []
                for a in approaches:
                    p = r["approaches"][a]
                    extra += [p["offered_vph"], p["served_vph"], p["unserved"], r["max_queue"][a]]
                w.writerow(base + extra)
    (out / "report.md").write_text(render_report(result), encoding="utf-8")
    try:
        plot_results(result, out / "plot.png")
    except ImportError:
        pass  # plotting is optional; the CSV holds everything needed to plot later
    return out


def render_report(result: dict) -> str:
    sc = result["score"]
    lines = ["# Capacity benchmark", "",
             f"- Junction: `{result['junction']}`, controller: `{result['settings']['controller']}`, "
             f"green {result['settings']['green_s']} s",
             f"- Seeds: {', '.join(map(str, result['seeds']))} · config `{result['config_hash']}` · "
             f"commit `{result['git_commit']}` · {result['created_at']}", ""]
    if sc:
        bound = "at least " if sc["censored"] else ""
        lines += [f"**Capacity score: {bound}{sc['mean_vph']:,.0f} ± {sc['std_vph']:,.0f} veh/h** "
                  f"(mean ± std over {sc['n']} seeds; range {sc['min_vph']:,.0f}–{sc['max_vph']:,.0f})", "",
                  f"Peak throughput while overloaded: {sc['peak_served_mean_vph']:,.0f} veh/h (mean over seeds). "
                  "Not sustainable: queues grow at that demand.", ""]
    else:
        lines += ["**No demand level passed**: the junction is oversaturated even at the first level.", ""]
    lines += [result["definition"], "", "## Per seed", "",
              "| Seed | Capacity (veh/h) | Highest passing demand (veh/h per approach) | Breakdown at | Peak served (veh/h) |",
              "|---|---|---|---|---|"]
    for r in result["ramps"]:
        cap = f"{r['capacity_vph']:,.0f}" if r["capacity_vph"] is not None else "–"
        lvl = f"{r['capacity_level_vph']:.0f}" if r["capacity_level_vph"] is not None else "–"
        brk = f"{r['breakdown_level_vph']:.0f}" if r["breakdown_level_vph"] is not None else "not reached"
        lines.append(f"| {r['seed']} | {cap} | {lvl} | {brk} | {r['peak_served_vph']:,.0f} |")
    lines += ["", "## Levels", "",
              "| Seed | Demand/approach | Offered (veh/h) | Served (veh/h) | Served % | Mean delay (s) | LOS | Result |",
              "|---|---|---|---|---|---|---|---|"]
    for rp in result["ramps"]:
        for r in rp["levels"]:
            res = "pass" if r["passed"] else "**fail** " + (", ".join(r["failing_approaches"]) or "gridlock")
            lines.append(f"| {r['seed']} | {r['volume_vph']:.0f} | {r['offered_vph']:,.0f} | {r['served_vph']:,.0f} | "
                         f"{r['served_ratio'] * 100:.0f}% | {r['mean_delay_s'] if r['mean_delay_s'] is not None else '–'} | "
                         f"{r['los'] or '–'} | {res} |")
    return "\n".join(lines) + "\n"


def plot_results(result: dict, path: Path) -> None:
    """Served vs offered throughput: the classic saturation curve, flattening at capacity."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    top = 0.0
    for rp in result["ramps"]:
        xs = [r["offered_vph"] for r in rp["levels"]]
        ys = [r["served_vph"] for r in rp["levels"]]
        top = max(top, *xs, *ys)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, label=f"seed {rp['seed']}")
        for r in rp["levels"]:
            if not r["passed"]:
                ax.plot(r["offered_vph"], r["served_vph"], "x", color="#d03b3b", ms=7, mew=1.6)
    ax.plot([0, top], [0, top], ls="--", lw=1, color="#898781", label="served = offered")
    sc = result["score"]
    if sc:
        ax.axhline(sc["mean_vph"], color="#0b0b0b", lw=1)
        ax.text(top * 0.02, sc["mean_vph"], f"  capacity {sc['mean_vph']:,.0f} ± {sc['std_vph']:,.0f} veh/h",
                va="bottom", fontsize=8.5)
    ax.set_xlabel("Offered demand, whole junction (veh/h)")
    ax.set_ylabel("Served throughput (veh/h)")
    ax.set_title("Throughput vs demand (× = level failed)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latest_summary(results_dir: Path) -> dict | None:
    """The most recent saved benchmark (directory names start with a sortable timestamp)."""
    d = Path(results_dir)
    if not d.is_dir():
        return None
    runs = sorted((p for p in d.iterdir() if (p / "summary.json").is_file()), key=lambda p: p.name)
    if not runs:
        return None
    data = json.loads((runs[-1] / "summary.json").read_text(encoding="utf-8"))
    data["id"] = runs[-1].name
    return data
