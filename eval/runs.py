"""The two simulation runs the evaluation questions are about: ``python -m eval.runs``.

WHY fixed runs: a question like "why did the AI switch at 4:35?" has a ground-truth answer only for
one specific run. These runs are regenerated deterministically (same seed + config -> bit-identical
logs, a tested invariant of the simulator), with FIXED run ids, into the normal runs folder, so the
knowledge base indexes them like any other run (``python -m rag.build``).

A fingerprint of each run's metrics + events + decisions is stored next to the questions
(eval/rag_test_questions.json, "runs"); the evaluation refuses to score if the logs no longer
match, because the ground truths were written from those exact logs (a change to the simulator
would silently make them wrong).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from sim.config import load_config

EVAL_SEED = 11
EVAL_DURATION_S = 900.0
EVAL_RUNS = {"eval_fixed_s11": "fixed", "eval_rl_s11": "rl"}
FINGERPRINT_FILES = ("metrics.csv", "events.jsonl", "decisions.jsonl")


def fingerprint(run_dir: Path) -> str:
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        p = Path(run_dir) / name
        if p.is_file():
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def make_runs(cfg=None, force: bool = False) -> dict[str, str]:
    """Create the evaluation runs (if missing) and return {run_id: fingerprint}."""
    from sim.engine import Simulation
    from sim.logger import RunLogger

    cfg = cfg or load_config()
    out = {}
    for run_id, controller in EVAL_RUNS.items():
        d = cfg.runs_path() / run_id
        if d.exists() and force:
            shutil.rmtree(d)
        if not (d / "manifest.json").is_file():
            policy = None
            if controller == "rl":
                from rl.policy import RLPolicy
                policy = RLPolicy.load(cfg)
            c = cfg.model_copy(deep=True)
            c.signals.controller = controller
            sim = Simulation(c, seed=EVAL_SEED, policy=policy)
            with RunLogger(sim, controller=controller, run_id=run_id, tags={"source": "rag_eval"}):
                sim.run(EVAL_DURATION_S)
        out[run_id] = fingerprint(d)
    return out


if __name__ == "__main__":
    for rid, fp in make_runs().items():
        print(rid, fp)
