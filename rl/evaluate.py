"""Compare the trained agent with the fixed timer: ``python -m rl.evaluate``.

Same seeds, same traffic, same engine; only the controller differs. Uses the held-out
``rl.eval_seeds`` (never used in training) at the ``rl.eval_demand_vph`` levels (with the
config's per-approach ratios, so N/S stay busier than E/W). Phase 10 runs the full experiment
(30+ scenarios, plus an actuated controller); this is the Phase 7 sanity check.

Reports per controller: total waiting (what the agent optimises), mean control delay per trip,
throughput, vehicles still queued off the map at the end, and the PEDESTRIANS' mean wait: the
reward ignores pedestrians, so it is checked here that the agent doesn't win at their expense.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from sim.config import Config, load_config
from sim.metrics import MetricsCollector

from .env import make_sim
from .features import waiting_now


def eval_traffic(cfg: Config, mean_vph: float, seed: int) -> dict:
    base = cfg.demand.per_approach_vph
    m = sum(base.values()) / len(base)
    return {"per_approach_vph": {a: round(v * mean_vph / m, 1) for a, v in base.items()},
            "ped_rate_ph": cfg.pedestrians.rate_ph, "seed": seed}


def run_one(args) -> dict:
    cfg_json, controller, level, mean_vph, seed, model_path = args
    cfg = Config.model_validate_json(cfg_json)
    policy = None
    if controller == "rl":
        from .policy import RLPolicy
        policy = RLPolicy.load(cfg, model_path)
    sim = make_sim(cfg, eval_traffic(cfg, mean_vph, seed), policy=policy)
    col = MetricsCollector(sim, sim.cfg.metrics)
    waited, delays, ped_waits = 0.0, [], []
    for _ in range(round(cfg.rl.episode_s / sim.dt)):
        sim.step()
        waited += waiting_now(sim) * sim.dt
        _, _, trips, peds = col.after_step(sim)
        delays += [t.delay for t in trips]
        ped_waits += [p.wait_time for p in peds]
    phase_changes = sum(1 for e in sim.signals.events if e.stage == "green") - 1
    return {"controller": controller, "level": level, "mean_vph": mean_vph, "seed": seed,
            "waiting_veh_s": round(waited, 1), "mean_delay_s": round(statistics.fmean(delays), 2) if delays else None,
            "completed": sim.counters.completed, "backlog_end": sim.spawner.backlog_size(),
            "mean_ped_wait_s": round(statistics.fmean(ped_waits), 1) if ped_waits else None,
            "greens": phase_changes, "emergency_stops": sim.counters.emergency_stops}


def summarise(rows: list[dict], levels: list[str]) -> list[dict]:
    out = []
    for level in levels:
        line = {"level": level}
        for c in ("fixed", "rl"):
            rs = [r for r in rows if r["level"] == level and r["controller"] == c]
            for k in ("waiting_veh_s", "mean_delay_s", "completed", "backlog_end", "mean_ped_wait_s", "greens"):
                vals = [r[k] for r in rs if r[k] is not None]
                line[f"{c}_{k}"] = round(statistics.fmean(vals), 1) if vals else None
            line[f"{c}_mean_vph"] = rs[0]["mean_vph"] if rs else None
        f, a = line["fixed_waiting_veh_s"], line["rl_waiting_veh_s"]
        line["waiting_change_pct"] = round(100 * (a - f) / f, 1) if f else None
        fd, ad = line["fixed_mean_delay_s"], line["rl_mean_delay_s"]
        line["delay_change_pct"] = round(100 * (ad - fd) / fd, 1) if fd and ad is not None else None
        out.append(line)
    return out


def render(summary: list[dict], seeds: list[int], episode_s: float) -> str:
    lines = ["# RL agent vs fixed timer", "",
             f"Held-out seeds {', '.join(map(str, seeds))}; {episode_s:.0f} s per run; mean over seeds. "
             "Negative change = the agent is better.", "",
             "| Demand (veh/h per approach) | Total waiting, fixed → RL (veh·h) | Change | Mean delay, fixed → RL (s) | Change "
             "| Trips completed, fixed → RL | Ped wait, fixed → RL (s) | Greens, fixed → RL |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['level']} ({s['fixed_mean_vph']:.0f}) | {s['fixed_waiting_veh_s'] / 3600:.1f} → {s['rl_waiting_veh_s'] / 3600:.1f} "
            f"| {s['waiting_change_pct']:+.1f}% | {s['fixed_mean_delay_s']} → {s['rl_mean_delay_s']} | {s['delay_change_pct']:+.1f}% "
            f"| {s['fixed_completed']:.0f} → {s['rl_completed']:.0f} | {s['fixed_mean_ped_wait_s']} → {s['rl_mean_ped_wait_s']} "
            f"| {s['fixed_greens']:.0f} → {s['rl_greens']:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the trained RL controller against the fixed timer.")
    ap.add_argument("--model", default=None, help="model path (default: rl.model_path)")
    ap.add_argument("--out", default=None, help="results folder (default: next to the latest training run)")
    args = ap.parse_args()
    cfg = load_config()
    model = args.model or str(cfg.rl_model_path())
    levels = list(cfg.rl.eval_demand_vph)
    jobs = [(cfg.model_dump_json(), c, lvl, cfg.rl.eval_demand_vph[lvl], s, model)
            for lvl in levels for s in cfg.rl.eval_seeds for c in ("fixed", "rl")]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
        rows = list(ex.map(run_one, jobs))
    summary = summarise(rows, levels)
    report = render(summary, cfg.rl.eval_seeds, cfg.rl.episode_s)
    print(report)

    if args.out:
        out = Path(args.out)
    else:
        meta = Path(model).with_suffix(".json")
        train_dir = json.loads(meta.read_text(encoding="utf-8")).get("training_dir") if meta.is_file() else None
        out = cfg.rl_results_path() / (train_dir or datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.md").write_text(report, encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
