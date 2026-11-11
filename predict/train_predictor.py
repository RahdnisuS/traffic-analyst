"""Train and evaluate the queue predictor: ``python -m predict.train_predictor``.

Reads the logged runs in ``predict.dataset_dir`` (from ``python -m predict.generate``), turns
them into samples (one approach at one second: features from predict/features.py, target = that
approach's queued vehicles ``horizon_s`` later), trains LightGBM and scores it on the held-out
test runs against two naive forecasts:

  persistence   the queue in 2 min = the queue now. The standard naive baseline for any
                forecast. Here it is also "same as one cycle ago", because the default cycle is
                ~120 s: a strong baseline, not a straw man.
  recent mean   the queue in 2 min = its mean over the last 2 min (one cycle): smooths the
                sawtooth instead of copying its current phase.

WHY LightGBM (gradient-boosted trees) and not a neural network: the data is a table of mixed
numbers and categories (stage, controller) of moderate size, which is where boosted trees are
the strong default. They need no feature scaling, handle interactions (e.g. "long queue AND its
green is next") by themselves, train in seconds on a CPU and report which features they use.

Writes:
  predict/models/queue_lgbm.txt (+ .json)   the model, loaded by the dashboard
  experiments/results/predict/<timestamp>/   report.md, metrics.json, feature_importance.csv,
                                             plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sim.config import PROJECT_ROOT, Config, load_config
from sim.logger import _git_commit

from .features import CATEGORICAL, FEATURES, Context, History, features_at, manifest_phases

I_NOW = FEATURES.index("queue_now")
I_MEAN = FEATURES.index("queue_mean120")
DEMAND_BANDS = ((300.0, "low (<300 veh/h)"), (450.0, "medium (300-450)"), (float("inf"), "high (>450, over capacity)"))


def demand_band(mean_vph: float) -> str:
    return next(name for limit, name in DEMAND_BANDS if mean_vph < limit)


# ---------------------------------------------------------------------- data
def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return manifest, rows


def run_samples(manifest: dict, rows: list[dict], horizon: int, every: int) -> dict:
    """All samples of one run. The settings never change inside a generated run, so the context
    comes from the manifest's config."""
    cfg = Config.model_validate(manifest["config"])
    ctx = Context.from_config(cfg, manifest["controller"])
    h = History.from_phases(rows, manifest_phases(manifest))
    if len(h) and not np.allclose(np.diff(h.t), 1.0):
        raise ValueError(f"{manifest['run_id']}: metrics rows are not one per second")
    X, y, arm, t = [], [], [], []
    for i in range(0, len(h) - horizon, every):
        X.append(features_at(h, i, ctx))
        for a in h.arms:
            y.append(h.queue[a][i + horizon])
            arm.append(a)
            t.append(h.t[i])
    n = len(y)
    mean_vph = float(np.mean(list(ctx.demand_vph.values())))
    return {"X": np.vstack(X) if X else np.empty((0, len(FEATURES))), "y": np.array(y), "arm": np.array(arm),
            "t": np.array(t), "run": np.array([manifest["run_id"]] * n),
            "controller": np.array([ctx.controller] * n), "band": np.array([demand_band(mean_vph)] * n),
            "split": manifest["tags"].get("split", "train")}


def load_dataset(dataset_dir: Path, horizon: int, every: int) -> dict[str, dict]:
    """Samples per split ("train", "val", "test"), each a dict of aligned arrays."""
    parts: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    runs = sorted(p for p in dataset_dir.iterdir() if (p / "manifest.json").is_file())
    if not runs:
        raise SystemExit(f"no runs in {dataset_dir}: run `python -m predict.generate` first")
    for p in runs:
        manifest, rows = load_run(p)
        if manifest.get("status") != "completed":
            continue  # a crashed run is evidence, not training data
        s = run_samples(manifest, rows, horizon, every)
        parts[s.pop("split")].append(s)
    out = {}
    for split, items in parts.items():
        if not items:
            raise SystemExit(f"the dataset has no '{split}' runs: generate at least 5 runs")
        out[split] = {k: np.concatenate([it[k] for it in items]) for k in items[0]}
        out[split]["n_runs"] = len(items)
    return out


# ---------------------------------------------------------------------- model
def lgb_params(cfg: Config) -> dict:
    p = cfg.predict.lgbm
    return {"objective": p.objective, "learning_rate": p.learning_rate, "num_leaves": p.num_leaves,
            "min_data_in_leaf": p.min_data_in_leaf, "feature_fraction": p.feature_fraction,
            "bagging_fraction": p.bagging_fraction, "bagging_freq": p.bagging_freq,
            "metric": "l1", "seed": cfg.predict.seed, "deterministic": True, "force_row_wise": True,
            "verbosity": -1}


def train(cfg: Config, data: dict):
    import lightgbm as lgb

    cats = [FEATURES.index(c) for c in CATEGORICAL]
    dtrain = lgb.Dataset(data["train"]["X"], data["train"]["y"], feature_name=list(FEATURES),
                         categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(data["val"]["X"], data["val"]["y"], reference=dtrain)
    p = cfg.predict.lgbm
    curve: dict = {}
    booster = lgb.train(lgb_params(cfg), dtrain, num_boost_round=p.num_boost_round, valid_sets=[dtrain, dval],
                        valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(p.early_stopping_rounds, verbose=False),
                                   lgb.record_evaluation(curve)])
    return booster, curve


def predict_clipped(booster, X: np.ndarray) -> np.ndarray:
    """A queue can't be negative (the L1 model rarely goes below 0, but it can)."""
    return np.maximum(booster.predict(X, num_iteration=booster.best_iteration), 0.0)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def score(test: dict, pred: np.ndarray) -> dict:
    """MAE of the model and both baselines, overall and broken down."""
    y = test["y"]
    fc = {"lightgbm": pred, "persistence": test["X"][:, I_NOW], "recent_mean": test["X"][:, I_MEAN]}

    def block(mask) -> dict:
        r = {k: round(mae(v[mask], y[mask]), 3) for k, v in fc.items()}
        r["n"] = int(mask.sum())
        r["skill_vs_persistence"] = round(1 - r["lightgbm"] / r["persistence"], 3) if r["persistence"] else None
        return r

    all_ = np.ones(len(y), bool)
    return {
        "overall": block(all_),
        "by_controller": {c: block(test["controller"] == c) for c in sorted(set(test["controller"]))},
        "by_demand": {b: block(test["band"] == b) for _, b in DEMAND_BANDS if (test["band"] == b).any()},
        "mean_queue": round(float(y.mean()), 2),
    }


# ---------------------------------------------------------------------- report
def plot(out: Path, scores: dict, test: dict, pred: np.ndarray, horizon: int) -> None:
    """Left: MAE per controller (lower is better), model vs the two baselines. Right: one test run
    and approach, actual queue vs what was forecast for that moment ``horizon`` s earlier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2a27", "#6b6a64", "#e6e4de"
    accent, gray, light = "#1f63b8", "#8a8983", "#c9c7c0"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)
        ax.yaxis.grid(True, color=grid)
        ax.set_axisbelow(True)

    groups = ["overall"] + list(scores["by_controller"])
    blocks = [scores["overall"]] + list(scores["by_controller"].values())
    series = [("persistence", "Persistence (queue now)", light), ("recent_mean", "Recent 2-min mean", gray),
              ("lightgbm", "LightGBM", accent)]
    x = np.arange(len(groups))
    w = 0.26
    for k, (key, label, color) in enumerate(series):
        vals = [b[key] for b in blocks]
        bars = a1.bar(x + (k - 1) * w, vals, w - 0.02, color=color, label=label)
        if key == "lightgbm":
            for b, v in zip(bars, vals):
                a1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=ink)
    a1.set_xticks(x, [g.replace("_", "-") for g in groups])
    a1.set_ylabel("MAE (queued vehicles)", color=muted)
    a1.set_title(f"Forecast error {horizon} s ahead, held-out runs", color=ink, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=8, labelcolor=ink)

    # Example trace: the test run+approach with the largest queue variance (an interesting one).
    best, best_var = None, -1.0
    for run in sorted(set(test["run"])):
        for arm in sorted(set(test["arm"])):
            m = (test["run"] == run) & (test["arm"] == arm)
            v = float(np.var(test["y"][m]))
            if v > best_var:
                best, best_var = m, v
    t_target = test["t"][best] + horizon
    a2.plot(t_target / 60, test["y"][best], color=ink, lw=2, label="Actual queue")
    a2.plot(t_target / 60, pred[best], color=accent, lw=2, ls="--", label="LightGBM forecast (made 2 min earlier)")
    a2.plot(t_target / 60, test["X"][best, I_NOW], color=gray, lw=1.2, ls=":", label="Persistence forecast")
    run_id, arm = test["run"][best][0], test["arm"][best][0]
    a2.set_title(f"Example test run {run_id}, approach {arm}", color=ink, fontsize=11, loc="left")
    a2.set_xlabel("simulated minute", color=muted)
    a2.set_ylabel("queued vehicles", color=muted)
    a2.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)
    plt.close(fig)


def write_report(out: Path, meta: dict, scores: dict, importance: list[tuple[str, float]]) -> None:
    def row(name, b):
        return (f"| {name} | {b['n']:,} | {b['persistence']:.2f} | {b['recent_mean']:.2f} | "
                f"**{b['lightgbm']:.2f}** | {100 * b['skill_vs_persistence']:.0f}% |")
    head = ["| | samples | persistence | recent 2-min mean | LightGBM | LightGBM vs persistence |",
            "|---|---:|---:|---:|---:|---:|"]
    lines = [
        f"# Queue predictor: {meta['horizon_s']} s ahead",
        "",
        f"Trained {meta['created_at']} on {meta['runs']['train']} runs (validation {meta['runs']['val']}, "
        f"test {meta['runs']['test']}), commit `{meta['git_commit']}`. {meta['best_iteration']} trees "
        f"(early stopping on the validation runs), objective `{meta['params']['objective']}`.",
        "",
        "Target: queued vehicles (below 2 m/s) on one approach, 2 minutes after the forecast is made. "
        f"Mean actual queue in the test set: {scores['mean_queue']:.2f} vehicles. "
        "All numbers are mean absolute errors (MAE, vehicles) on the held-out test runs; lower is better.",
        "",
        "## Overall", "", *head, row("all test samples", scores["overall"]),
        "", "## By controller", "", *head,
        *(row(c, b) for c, b in scores["by_controller"].items()),
        "", "## By demand (mean per approach)", "", *head,
        *(row(d, b) for d, b in scores["by_demand"].items()),
        "", "## Most used features (share of total gain)", "",
        "| feature | gain |", "|---|---:|",
        *(f"| {f} | {100 * g:.1f}% |" for f, g in importance[:12]),
        "", "![plot](plot.png)", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LightGBM queue predictor on logged runs.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--data", default=None, help="override predict.dataset_dir")
    ap.add_argument("--out", default=None, help="results folder (default: predict.results_dir/<timestamp>)")
    ap.add_argument("--model", default=None, help="where to save the model (default: predict.model_path)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    horizon = cfg.predict.horizon_s
    data_dir = Path(args.data) if args.data else cfg.predict_dataset_path()
    t0 = time.perf_counter()
    data = load_dataset(data_dir, horizon, cfg.predict.sample_every_s)
    print("samples: " + ", ".join(f"{k} {len(v['y']):,} ({v['n_runs']} runs)" for k, v in data.items())
          + f" [{time.perf_counter() - t0:.0f} s]")

    t1 = time.perf_counter()
    booster, curve = train(cfg, data)
    print(f"trained {booster.best_iteration} trees in {time.perf_counter() - t1:.0f} s")
    pred = predict_clipped(booster, data["test"]["X"])
    scores = score(data["test"], pred)
    o = scores["overall"]
    print(f"test MAE: LightGBM {o['lightgbm']:.3f} | persistence {o['persistence']:.3f} | "
          f"recent mean {o['recent_mean']:.3f}  ({100 * o['skill_vs_persistence']:.0f}% better than persistence)")

    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    importance = sorted(zip(FEATURES, (gain / gain.sum()).tolist()), key=lambda x: -x[1])
    now = datetime.now(timezone.utc)
    out = Path(args.out) if args.out else cfg.predict_results_path() / f"{now:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": now.isoformat(timespec="seconds"), "git_commit": _git_commit(),
        "horizon_s": horizon, "features": list(FEATURES), "categorical": CATEGORICAL,
        "dataset_dir": str(data_dir), "runs": {k: v["n_runs"] for k, v in data.items()},
        "samples": {k: int(len(v["y"])) for k, v in data.items()},
        "params": lgb_params(cfg), "best_iteration": booster.best_iteration,
        "val_mae_curve_every_50": [round(v, 4) for v in curve["val"]["l1"][::50]],
        "test": scores, "results_dir": str(out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out),
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(out / "feature_importance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "gain_share"])
        w.writerows((f, round(g, 5)) for f, g in importance)
    plot(out, scores, data["test"], pred, horizon)
    write_report(out, meta, scores, importance)

    model_path = Path(args.model) if args.model else cfg.predict_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"model -> {model_path}\nreport -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
