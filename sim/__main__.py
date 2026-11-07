"""Headless CLI: ``python -m sim --seed 7 --duration 900``.

Prints a progress table and a final summary, plus how many times faster than
real time the sim ran (that speed matters for RL training).
"""

from __future__ import annotations

import argparse
import time

from .config import PROJECT_ROOT, load_config
from .engine import Simulation


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the traffic simulation headless.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="simulated seconds")
    ap.add_argument("--report-every", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = Simulation(cfg, seed=args.seed)
    duration = args.duration or cfg.sim.duration_s
    steps_per_report = max(1, int(round(args.report_every / sim.dt)))
    total_steps = int(round(duration / sim.dt))

    print(f"junction={sim.junction.name} seed={sim.seed} dt={sim.dt}s duration={duration}s")
    print(f"{'t(s)':>7} {'phase':>6} {'in_net':>7} {'backlog':>8} {'done':>6} {'mean_wait':>10}")
    t0 = time.perf_counter()
    for i in range(1, total_steps + 1):
        sim.step()
        if i % steps_per_report == 0:
            s = sim.summary()
            print(f"{s['time_s']:>7} {sim.signals.phase.name:>6} {s['in_network']:>7} "
                  f"{s['backlog']:>8} {s['completed']:>6} {str(s['mean_wait_time_s']):>10}")
    wall = time.perf_counter() - t0
    print("\nsummary:", sim.summary())
    print(f"wall time {wall:.2f}s -> {duration / wall:.0f}x real time")


if __name__ == "__main__":
    main()
