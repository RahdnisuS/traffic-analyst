"""Capacity benchmark CLI: ``python -m bench``.

Ramps demand on every configured seed until the junction oversaturates, prints the capacity
score, and saves summary.json / levels.csv / report.md / plot.png under
``benchmark.results_dir`` (experiments/results/benchmark/ by default).
"""

from __future__ import annotations

import argparse
import os
import sys

from sim.config import PROJECT_ROOT, load_config

from .stress import render_report, run_benchmark, save_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress-test the junction and report its capacity.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override benchmark.seeds")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: one per seed, up to the CPU count)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: short windows and coarse steps (not a valid score)")
    ap.add_argument("--no-save", action="store_true", help="print only, don't write results")
    ap.add_argument("--out", default=None, help="override benchmark.results_dir")
    args = ap.parse_args()
    # Windows consoles default to a legacy code page; the report uses ± and ·.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    if args.quick:
        b = cfg.benchmark
        cfg.benchmark = b.model_copy(update={"warmup_s": 120, "measure_s": 240, "step_vph": 150})
    seeds = args.seeds or cfg.benchmark.seeds
    workers = args.workers or min(len(seeds), os.cpu_count() or 1)
    b = cfg.benchmark
    print(f"Capacity benchmark: seeds {seeds}, demand {b.start_vph:.0f}..{b.max_vph:.0f} veh/h per approach "
          f"in steps of {b.step_vph:.0f}, warm-up {b.warmup_s:.0f} s + measure {b.measure_s:.0f} s per level, "
          f"{workers} worker(s)", flush=True)

    result = run_benchmark(cfg, seeds, workers=workers)
    print()
    print(render_report(result).split("## Levels")[0].strip())
    print(f"\n(wall time {result['wall_time_s']} s)")
    if args.quick:
        result["quick"] = True
    # A --quick result is not a valid score, so it is only saved to an explicit --out folder:
    # otherwise it would become the "latest" benchmark the dashboard shows.
    if not args.no_save and not (args.quick and not args.out):
        out = save_results(result, args.out or cfg.benchmark_path())
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
