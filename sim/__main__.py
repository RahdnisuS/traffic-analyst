"""Headless CLI: ``python -m sim --seed 7 --duration 900``.

Every run is logged to data/runs/<run_id>/ unless ``--no-log`` is given.
Prints a progress table, the run summary and the speed relative to real time
(that speed matters for RL training).
"""

from __future__ import annotations

import argparse
import json
import time

from .config import PROJECT_ROOT, load_config
from .engine import Simulation
from .logger import RunLogger


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the traffic simulation headless.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="simulated seconds")
    ap.add_argument("--report-every", type=float, default=60.0)
    ap.add_argument("--no-log", action="store_true", help="don't write a run log")
    ap.add_argument("--runs-dir", default=None, help="override logging.runs_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = Simulation(cfg, seed=args.seed)
    log = None if (args.no_log or not cfg.logging.enabled) else \
        RunLogger(sim, controller=cfg.signals.controller, runs_dir=args.runs_dir)
    duration = args.duration or cfg.sim.duration_s
    steps_per_report = max(1, int(round(args.report_every / sim.dt)))
    total_steps = int(round(duration / sim.dt))

    print(f"junction={sim.junction.name} seed={sim.seed} dt={sim.dt}s duration={duration}s")
    print(f"{'clock':>6} {'phase':>6} {'in_net':>7} {'queue':>6} {'backlog':>8} "
          f"{'done':>6} {'thru_vph':>9} {'delay':>7} {'LOS':>4}")
    t0 = time.perf_counter()
    try:
        for i in range(1, total_steps + 1):
            sim.step()
            if log and i % steps_per_report == 0 and log.collector.latest:
                r = log.collector.latest
                print(f"{r['clock']:>6} {r['phase']:>6} {r['in_network']:>7} {r['queue_total']:>6} "
                      f"{r['backlog']:>8} {r['completed']:>6} {r['throughput_vph']:>9} "
                      f"{str(r['mean_delay_window']):>7} {r['los']:>4}")
    except BaseException as exc:
        if log:
            log.close(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    wall = time.perf_counter() - t0

    if log:
        manifest = log.close()
        print("\nsummary:", json.dumps(manifest["summary"], indent=2))
        print(f"run logged to {log.dir}")
    else:
        print("\nsummary:", sim.summary())
    print(f"wall time {wall:.2f}s -> {duration / wall:.0f}x real time")


if __name__ == "__main__":
    main()
