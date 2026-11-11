"""The trained queue predictor, live: forecasts each approach's queue ``horizon_s`` ahead from a
running simulation's recent metrics rows.

Same features as training (predict/features.py), computed from the MetricsCollector's history:
the rows the dashboard charts and the run log records. The context (demand, green time,
pedestrian rate, controller) is read from ``sim.cfg`` at that moment, i.e. the current slider
values; a mid-run slider change is therefore reflected in the very next forecast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FEATURES, LOOKBACK_S, Context, History, features_at, junction_phases


class QueueForecaster:
    def __init__(self, booster, meta: dict):
        if list(meta.get("features", [])) != list(FEATURES):
            # A model trained on another feature list would silently read the wrong columns.
            raise ValueError("predictor model was trained with different features: retrain it "
                             "(python -m predict.train_predictor)")
        self.booster = booster
        self.meta = meta
        self.horizon_s = int(meta["horizon_s"])

    @classmethod
    def load(cls, cfg, path: str | Path | None = None) -> "QueueForecaster":
        import lightgbm as lgb  # only needed when a model exists

        p = Path(path) if path else cfg.predict_model_path()
        if not p.is_file():
            raise FileNotFoundError(f"no trained predictor at {p}: run `python -m predict.train_predictor`")
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(lgb.Booster(model_file=str(p)), meta)

    def brief(self) -> dict:
        """What the dashboard shows about the model: horizon and its held-out accuracy."""
        o = self.meta["test"]["overall"]
        return {"horizon_s": self.horizon_s, "mae": o["lightgbm"], "persistence_mae": o["persistence"],
                "test_runs": self.meta["runs"]["test"], "created_at": self.meta["created_at"]}

    def forecast(self, rows: Sequence[dict], sim, controller: str) -> dict:
        """Forecast made at the last row: {"t": that time + horizon, "from_t": now, arm: queue}."""
        recent = list(rows)[-(LOOKBACK_S + 1):]  # every look-back fits in this window
        h = History.from_phases(recent, junction_phases(sim.junction))
        X = features_at(h, len(h) - 1, Context.from_config(sim.cfg, controller))
        pred = np.maximum(self.booster.predict(X), 0.0)
        now = float(h.t[-1])
        return {"t": round(now + self.horizon_s, 1), "from_t": round(now, 1),
                **{a: round(float(v), 1) for a, v in zip(h.arms, pred)}}
