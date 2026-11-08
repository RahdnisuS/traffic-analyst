"""Train the PPO signal controller: ``python -m rl.train``.

Writes:
  rl/models/ppo_signal.zip (+ .json metadata)    the trained network, loaded by RLPolicy
  experiments/results/rl/<timestamp>/            progress.csv (SB3 log), greedy_eval.csv,
                                                 training_curve.png, train_meta.json

WHY PPO: the action is discrete (keep / switch), the reward is noisy (random arrivals), and
we have plenty of cheap simulated experience. PPO is the standard robust choice there: its
clipped update never moves the policy too far on one noisy batch, and it runs many
environments in parallel. The network is small (2 x 64): 15 inputs and 2 actions don't need more.
The variant is MaskablePPO (sb3-contrib, SB3's official companion): PPO plus invalid-action
masking, so decisions where switching is impossible don't feed noise into the policy update.

The training curve is only meaningful next to a BASELINE: the fixed timer's return on the same
kind of random traffic, scored with the same reward. The agent is useful once it beats that line.

WHY two agent lines: SB3's "mean episode return" is the agent WHILE TRAINING, sampling its
actions at random in proportion to their probability (that is how PPO explores). Deployed, it
takes the most likely action every time. The first full run's training line sat on the fixed
timer and below gap-out, yet the deployed agent beat both: exploration was costing ~30 points.
So every ``rl.curve_eval_every`` decisions the current network is also scored greedily on the
SAME episodes as the baselines; that line is the one to compare with them.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim.config import load_config

from .env import SignalEnv, agent_return, episode_traffic, fixed_time_return, gap_out_return

BASELINE_EPISODES = 24


def _make_env(cfg):
    def thunk():
        return SignalEnv(cfg)
    return thunk


def curve_traffics(cfg, n: int) -> list[dict]:
    """The ``n`` random training-style episodes every curve line is scored on."""
    rng = np.random.default_rng(cfg.rl.seed + 10_000)  # its own stream: not the training episodes
    return [episode_traffic(cfg, rng) for _ in range(n)]


def _score_job(args):
    cfg_json, kind, traffic = args
    from sim.config import Config
    cfg = Config.model_validate_json(cfg_json)
    if kind == "fixed":
        return kind, fixed_time_return(cfg, traffic)
    if kind == "gap_out":
        return kind, gap_out_return(cfg, traffic)
    from sb3_contrib import MaskablePPO  # kind = path of a saved model
    return "agent", agent_return(cfg, traffic, MaskablePPO.load(kind, device="cpu"))


def _summary(results: list[dict]) -> dict:
    rets = [r["return"] for r in results]
    return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "mean_switches": float(np.mean([r["switches"] for r in results])), "episodes": len(results)}


def baselines(cfg, traffics: list[dict], pool: ProcessPoolExecutor) -> dict:
    """Mean return of the fixed timer and of the gap-out rule on ``traffics``
    (scored with the agent's reward)."""
    jobs = [(cfg.model_dump_json(), kind, t) for kind in ("fixed", "gap_out") for t in traffics]
    res = list(pool.map(_score_job, jobs))
    return {kind: _summary([r for k, r in res if k == kind]) for kind in ("fixed", "gap_out")}


def greedy_eval(model, cfg, traffics: list[dict], pool: ProcessPoolExecutor, scratch: Path) -> dict:
    """The current network, most-likely action every time, on ``traffics``. The model is saved to
    ``scratch`` so the worker processes can load it (the training envs are idle meanwhile)."""
    model.save(scratch)
    res = list(pool.map(_score_job, [(cfg.model_dump_json(), str(scratch), t) for t in traffics]))
    return _summary([r for _, r in res])


def _behaviour_callback(evaluate, every: int, rows: list[dict]):
    """Logs WHAT the agent does, not just its score: switches and waiting per episode. A falling
    return alone can't tell "switches constantly" from "never switches". Also runs ``evaluate``
    (the greedy evaluation) every ``every`` decisions and appends the result to ``rows``."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Behaviour(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps >= (len(rows) + 1) * every:
                res = evaluate(self.model)
                rows.append({"timesteps": self.num_timesteps, **res})
                self.logger.record("eval/greedy_return", res["mean_return"])
                print(f"  greedy eval @ {self.num_timesteps:,}: return {res['mean_return']:.1f} "
                      f"({res['mean_switches']:.0f} switches)", flush=True)
            eps = [e for e in self.model.ep_info_buffer if "episode_switches" in e]
            if eps:
                self.logger.record("rollout/ep_switches_mean", float(np.mean([e["episode_switches"] for e in eps])))
                self.logger.record("rollout/ep_waiting_veh_h", float(np.mean([e["episode_waiting_veh_s"] for e in eps])) / 3600)

    return Behaviour()


def plot_curve(progress_csv: Path, baseline: dict, greedy: list[dict], out: Path) -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = [], []
    with open(progress_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("rollout/ep_rew_mean"):
                xs.append(float(row["time/total_timesteps"]))
                ys.append(float(row["rollout/ep_rew_mean"]))
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.plot(xs, ys, lw=1.2, color="#7aa6d6", alpha=0.8, label="PPO agent while training (sampled actions)")
    if greedy:
        ax.plot([g["timesteps"] for g in greedy], [g["mean_return"] for g in greedy], "o-", lw=1.8, ms=4,
                color="#1f4e8c", label=f"PPO agent, best action, same traffic ({greedy[-1]['mean_return']:.0f})")
    ax.axhline(baseline["fixed"]["mean_return"], color="#d03b3b", ls="--", lw=1.2,
               label=f"fixed timer, same traffic ({baseline['fixed']['mean_return']:.0f})")
    ax.axhline(baseline["gap_out"]["mean_return"], color="#1baf7a", ls=":", lw=1.4,
               label=f"gap-out rule, same traffic ({baseline['gap_out']['mean_return']:.0f})")
    ax.set_xlabel("decisions (environment steps)")
    ax.set_ylabel("episode return (higher = less waiting)")
    ax.set_title("PPO training on randomised demand", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the PPO traffic-signal agent.")
    ap.add_argument("--timesteps", type=int, default=None, help="override rl.total_timesteps")
    ap.add_argument("--n-envs", type=int, default=None, help="override rl.n_envs")
    ap.add_argument("--quick", action="store_true", help="smoke test: 4k steps, short episodes, no model saved")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    cfg = load_config()
    r = cfg.rl
    if args.quick:
        cfg.rl = r.model_copy(update={"total_timesteps": 4096, "episode_s": 300.0})
        r = cfg.rl
    timesteps = args.timesteps or r.total_timesteps
    n_envs = args.n_envs or r.n_envs

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = cfg.rl_results_path() / (f"{stamp}_quick" if args.quick else stamp)
    out.mkdir(parents=True, exist_ok=True)

    n_curve = 4 if args.quick else BASELINE_EPISODES
    every = 2048 if args.quick else r.curve_eval_every
    traffics = curve_traffics(cfg, n_curve)
    # One pool for the baselines and every greedy evaluation: worker start-up (torch import) is
    # paid once, not at each evaluation.
    pool = ProcessPoolExecutor(max_workers=n_envs)
    print(f"Baselines on {n_curve} random episodes...", flush=True)
    baseline = baselines(cfg, traffics, pool)
    for k, b in baseline.items():
        print(f"  {k}: mean return {b['mean_return']:.1f} ± {b['std_return']:.1f}", flush=True)
    greedy: list[dict] = []
    scratch = out / "_eval_model.zip"

    def evaluate(m):
        return greedy_eval(m, cfg, traffics, pool, scratch)

    venv = VecMonitor(SubprocVecEnv([_make_env(cfg) for _ in range(n_envs)]),
                      info_keywords=("episode_waiting_veh_s", "episode_switches"))
    venv.seed(r.seed)
    # Reward normalisation, for TRAINING only: raw episode returns are in the hundreds and the
    # critic couldn't fit them (value loss ~2,000, a first run got worse as it trained). Rewards
    # are divided by a running std of the discounted return; observations are already scaled.
    # VecMonitor sits inside, so logged returns stay in raw, comparable units. Inference doesn't
    # use rewards at all, so the saved model needs no normalisation statistics.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=r.ppo.gamma)
    p = r.ppo
    model = MaskablePPO("MlpPolicy", venv, learning_rate=p.learning_rate, n_steps=p.n_steps, batch_size=p.batch_size,
                n_epochs=p.n_epochs, gamma=p.gamma, gae_lambda=p.gae_lambda, clip_range=p.clip_range,
                ent_coef=p.ent_coef, policy_kwargs={"net_arch": list(p.net_arch)}, seed=r.seed,
                device="cpu", verbose=0)
    model.set_logger(configure(str(out), ["csv", "stdout"]))
    print(f"Training PPO: {timesteps:,} decisions on {n_envs} parallel simulations -> {out}", flush=True)
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=False, callback=_behaviour_callback(evaluate, every, greedy))
    wall = time.perf_counter() - t0
    venv.close()
    if not greedy or greedy[-1]["timesteps"] < model.num_timesteps:  # always score the final model
        greedy.append({"timesteps": model.num_timesteps, **evaluate(model)})
        print(f"  greedy eval @ {model.num_timesteps:,}: return {greedy[-1]['mean_return']:.1f}", flush=True)
    pool.shutdown()
    scratch.unlink(missing_ok=True)
    with open(out / "greedy_eval.csv", "w", encoding="utf-8", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(greedy[0]))
        w.writeheader()
        w.writerows(greedy)

    plot_curve(out / "progress.csv", baseline, greedy, out / "training_curve.png")
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timesteps": timesteps, "n_envs": n_envs, "wall_time_s": round(wall, 1),
            "baseline_fixed_time": baseline, "greedy_eval": greedy, "rl_settings": r.model_dump(),
            "features": SignalEnv(cfg).names, "quick": args.quick}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not args.quick:
        path = cfg.rl_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        path.with_suffix(".json").write_text(json.dumps({**meta, "training_dir": out.name}, indent=2),
                                             encoding="utf-8")
        print(f"Saved model to {path}")
    print(f"Done in {wall / 60:.1f} min. Curve: {out / 'training_curve.png'}")


if __name__ == "__main__":
    main()
