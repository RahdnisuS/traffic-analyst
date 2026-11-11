"""Generate the predictor's training data: ``python -m predict.generate``.

Logs ``predict.dataset_runs`` simulation runs with the normal RunLogger (same files as any
dashboard or CLI run) into ``predict.dataset_dir``. Each run draws its own traffic, signal
timing, driver behaviour, vehicle mix and seed, and is driven by one of the controllers in
``predict.controllers`` (round-robin, so each gets the same share and the same spread of
traffic). WHY all three controllers: the dashboard can be driven by the fixed timer or the AI,
and the queue in 2 minutes depends on when the lights will change, which depends on who is
changing them. A model trained only on fixed-timer runs would have learned the 120 s cycle and
be wrong the moment the AI drives. Gap-out adds a third, rule-based style of switching.

The split is decided here, by run index, and recorded in each manifest's tags: every 5th run is
"test", the one before it "val" (early stopping), the rest "train". So test runs are entire
runs the model never saw, with seeds it never saw.

Runs are independent, so they go to a process pool (one sim per core). The whole dataset is
reproducible from ``predict.seed``.
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sim.config import PROJECT_ROOT, Config, load_config


def split_of(index: int) -> str:
    return {4: "test", 3: "val"}.get(index % 5, "train")


def run_specs(cfg: Config, controllers: list[str]) -> list[dict]:
    """The traffic and controller of every run, drawn from one seeded generator."""
    p = cfg.predict
    rng = np.random.default_rng(p.seed)
    arms = sorted(cfg.demand.per_approach_vph)
    types = sorted(cfg.demand.vehicle_mix)
    base_mix = np.array([cfg.demand.vehicle_mix[t] for t in types], float)
    base_mix /= base_mix.sum()
    specs = []
    for k in range(p.dataset_runs):
        mean = rng.uniform(*p.demand_vph)
        mix = rng.dirichlet(base_mix * p.mix_concentration)
        specs.append({
            "index": k, "split": split_of(k), "controller": controllers[k % len(controllers)],
            "seed": int(rng.integers(0, 2**31 - 1)),
            "per_approach_vph": {a: round(mean * rng.uniform(1 - p.demand_imbalance, 1 + p.demand_imbalance), 1)
                                 for a in arms},
            "ped_rate_ph": round(rng.uniform(*p.ped_rate_ph), 1),
            "green_s": round(rng.uniform(*p.green_s), 1),
            "aggressiveness": round(rng.uniform(*p.aggressiveness), 2),
            "vehicle_mix": {t: round(float(w), 4) for t, w in zip(types, mix)},
        })
    return specs


_RL_POLICY = None  # loaded once per worker process (importing torch takes a few seconds)


def _policy(cfg: Config, controller: str):
    global _RL_POLICY
    if controller == "gap_out":
        from rl.env import GapOutPolicy
        return GapOutPolicy(cfg.rl.decision_interval_s)
    if controller == "rl":
        from rl.policy import RLPolicy
        if _RL_POLICY is None:
            _RL_POLICY = RLPolicy.load(cfg)
        # A fresh policy object per run (its decision clock), sharing the loaded network.
        return RLPolicy(_RL_POLICY.model, cfg.rl.decision_interval_s, model_id=_RL_POLICY.model_id)
    return None  # Simulation's default: the fixed-time plan from cfg.signals.green_s


def run_one(cfg_json: str, spec: dict, out_dir: str) -> dict:
    """One logged run. The settings go into the config BEFORE the sim and logger exist, so the
    manifest's config is exactly what the run used (that is what training reads back)."""
    from sim.engine import Simulation
    from sim.logger import RunLogger

    cfg = Config.model_validate_json(cfg_json)
    cfg.demand.per_approach_vph = dict(spec["per_approach_vph"])
    cfg.demand.vehicle_mix = dict(spec["vehicle_mix"])
    cfg.pedestrians.rate_ph = spec["ped_rate_ph"]
    cfg.signals.green_s = spec["green_s"]
    cfg.driver.aggressiveness = spec["aggressiveness"]
    controller = spec["controller"]
    # The config field only knows fixed|rl; gap-out is a fixed-timer config with another policy.
    cfg.signals.controller = "rl" if controller == "rl" else "fixed"
    sim = Simulation(cfg, seed=spec["seed"], policy=_policy(cfg, controller))
    run_id = f"{spec['index']:03d}_{controller}_s{spec['seed']}"
    tags = {"source": "predict_dataset", "dataset_index": spec["index"], "split": spec["split"]}
    t0 = time.perf_counter()
    with RunLogger(sim, controller=controller, runs_dir=out_dir, run_id=run_id, tags=tags):
        sim.run(cfg.predict.run_s)
    return {"run_id": run_id, "split": spec["split"], "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Log the runs the queue predictor is trained on.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--runs", type=int, default=None, help="override predict.dataset_runs")
    ap.add_argument("--run-s", type=float, default=None, help="override predict.run_s (simulated seconds)")
    ap.add_argument("--workers", type=int, default=None, help="override predict.workers")
    ap.add_argument("--out", default=None, help="override predict.dataset_dir")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing dataset first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.runs:
        cfg.predict.dataset_runs = args.runs
    if args.run_s:
        cfg.predict.run_s = args.run_s
    out = Path(args.out) if args.out else cfg.predict_dataset_path()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} already holds a dataset: pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    controllers = list(cfg.predict.controllers)
    if "rl" in controllers and not cfg.rl_model_path().is_file():
        print(f"warning: no trained RL model at {cfg.rl_model_path()}; generating without the rl controller")
        controllers.remove("rl")
    specs = run_specs(cfg, controllers)
    workers = min(args.workers or cfg.predict.workers, len(specs))
    print(f"{len(specs)} runs x {cfg.predict.run_s:.0f} s, controllers {controllers}, {workers} workers -> {out}")
    t0 = time.perf_counter()
    cfg_json = cfg.model_dump_json()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, cfg_json, s, str(out)) for s in specs]
        for n, f in enumerate(futures, 1):
            f.result()
            if n % 10 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} done ({time.perf_counter() - t0:.0f} s)")
    print(f"dataset ready in {time.perf_counter() - t0:.0f} s: {out}")


if __name__ == "__main__":
    main()
