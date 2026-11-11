"""Phase 9: the LightGBM queue predictor (features, dataset, training, live forecasts)."""

import csv
import json

import numpy as np
import pytest

from predict.features import FEATURES, LOOKBACK_S, Context, History, features_at, junction_phases, manifest_phases
from predict.generate import run_one, run_specs, split_of
from sim.config import load_config
from sim.engine import Simulation
from sim.logger import RunLogger

PHASES = [("N", ["N"]), ("E", ["E"]), ("S", ["S"]), ("W", ["W"])]


def _row(t, phase="N", stage="green", tis=0.0, q=None):
    q = q or {}
    r = {"t": t, "phase": phase, "stage": stage, "time_in_stage": tis, "peds_waiting": 0}
    for a in "NESW":
        r.update({f"queue_{a}": q.get(a, 0), f"backlog_{a}": 0, f"arrivals_{a}": 0, f"wait_{a}": 0.0})
    return r


def _col(X, name):
    return X[:, FEATURES.index(name)]


def test_cycle_position_carries_the_vehicle_phase_through_the_ped_phase():
    rows = ([_row(t, "N", "green") for t in range(1, 21)] + [_row(t, "N", "amber") for t in range(21, 24)]
            + [_row(t, "PED", "walk") for t in range(24, 31)])
    h = History.from_phases(rows, PHASES)
    ctx = Context("fixed", {a: 300.0 for a in "NESW"}, 100.0, 25.0)
    X = features_at(h, 10, ctx)  # N green: N is served now, E next, ... (arms sorted E, N, S, W)
    assert list(_col(X, "phases_until_green")) == [1, 0, 2, 3]
    assert list(_col(X, "green_now")) == [0, 1, 0, 0]
    X = features_at(h, len(rows) - 1, ctx)  # pedestrian phase after N: N's green is over, E is next
    assert list(_col(X, "phases_until_green")) == [1, 4, 2, 3]
    assert _col(X, "secs_since_green")[1] == 10  # N last green at row 19, now row 29
    assert _col(X, "secs_since_ped")[0] == 0
    assert _col(X, "secs_since_green")[0] == LOOKBACK_S  # E never green in the window


def test_lags_before_the_run_count_as_an_empty_junction():
    rows = [_row(t, q={"N": t}) for t in range(1, 41)]
    h = History.from_phases(rows, PHASES)
    X = features_at(h, 39, Context("rl", {a: 0.0 for a in "NESW"}, 0.0, 25.0))
    n = 1  # arms sorted E, N, S, W
    assert _col(X, "queue_now")[n] == 40 and _col(X, "queue_lag10")[n] == 30
    assert _col(X, "queue_lag60")[n] == 0  # 60 s before t = 40 the run hadn't started
    assert _col(X, "queue_mean120")[n] == pytest.approx(sum(range(1, 41)) / 120)
    assert _col(X, "controller")[0] == 2


def test_live_features_match_features_from_the_run_log(tmp_path):
    """The parity guarantee: the dashboard computes features from the collector's rolling history
    (last 301 rows); training computes them from metrics.csv. They must be identical."""
    cfg = load_config()
    cfg.demand.per_approach_vph = {"N": 550.0, "E": 300.0, "S": 500.0, "W": 350.0}
    cfg.pedestrians.rate_ph = 400.0
    sim = Simulation(cfg, seed=5)
    live = {}

    class Probe:
        def after_step(self, s):
            hist = logger.collector.history
            if s.tick % 10 == 0 and hist and hist[-1]["t"] == round(s.time, 1) and int(s.time) % 37 == 0:
                recent = list(hist)[-(LOOKBACK_S + 1):]
                h = History.from_phases(recent, junction_phases(s.junction))
                live[int(s.time)] = features_at(h, len(h) - 1, Context.from_config(s.cfg, "fixed"))

    with RunLogger(sim, controller="fixed", runs_dir=tmp_path) as logger:
        sim.observers.append(Probe())  # after the logger, so the row of this second exists
        sim.run(700)
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    with open(logger.dir / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    h = History.from_phases(rows, manifest_phases(manifest))
    ctx = Context.from_config(load_config().model_validate(manifest["config"]), manifest["controller"])
    assert len(live) > 10
    assert any(t > LOOKBACK_S for t in live)  # includes times where the live window is truncated
    for t, X_live in live.items():
        np.testing.assert_array_equal(features_at(h, t - 1, ctx), X_live, err_msg=f"t={t}")


def test_dataset_split_is_by_run_and_reproducible():
    cfg = load_config()
    cfg.predict.dataset_runs = 10
    a, b = run_specs(cfg, ["fixed", "gap_out"]), run_specs(cfg, ["fixed", "gap_out"])
    assert a == b
    assert [s["split"] for s in a] == [split_of(k) for k in range(10)]
    assert [s["split"] for s in a].count("test") == 2 and [s["split"] for s in a].count("val") == 2
    assert {s["controller"] for s in a} == {"fixed", "gap_out"}


def test_generate_train_and_forecast_end_to_end(tmp_path):
    pytest.importorskip("lightgbm")
    from predict.forecaster import QueueForecaster
    from predict.train_predictor import load_dataset, predict_clipped, score, train

    cfg = load_config()
    cfg.predict.dataset_runs = 5
    cfg.predict.run_s = 200
    cfg.predict.lgbm.num_boost_round = 30
    for spec in run_specs(cfg, ["fixed", "gap_out"]):
        run_one(cfg.model_dump_json(), spec, str(tmp_path / "ds"))
    manifest = json.loads(next((tmp_path / "ds").glob("000_*/manifest.json")).read_text())
    assert manifest["tags"]["source"] == "predict_dataset"
    assert manifest["config"]["signals"]["green_s"] == run_specs(cfg, ["fixed"])[0]["green_s"]

    data = load_dataset(tmp_path / "ds", horizon=120, every=5)
    assert {k: v["n_runs"] for k, v in data.items()} == {"train": 3, "val": 1, "test": 1}
    assert len(data["train"]["y"]) == 3 * 4 * len(range(0, 200 - 120, 5))
    booster, _ = train(cfg, data)
    s = score(data["test"], predict_clipped(booster, data["test"]["X"]))
    assert set(s["overall"]) >= {"lightgbm", "persistence", "recent_mean"}

    path = tmp_path / "m.txt"
    booster.save_model(str(path))
    path.with_suffix(".json").write_text(json.dumps({
        "features": list(FEATURES), "horizon_s": 120, "created_at": "x", "runs": {"test": 1},
        "test": s}))
    fc = QueueForecaster.load(cfg, path)
    sim = Simulation(cfg, seed=3)
    with RunLogger(sim, runs_dir=tmp_path / "live") as logger:
        sim.run(30)
    f = fc.forecast(logger.collector.history, sim, "fixed")
    assert f["t"] == 150.0 and f["from_t"] == 30.0
    assert all(f[a] >= 0 for a in "NESW")

    meta = json.loads(path.with_suffix(".json").read_text())
    meta["features"] = meta["features"][:-1]
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different features"):
        QueueForecaster.load(cfg, path)  # a stale model must not silently read the wrong columns


class FakeForecaster:
    horizon_s = 120

    def brief(self):
        return {"horizon_s": 120, "mae": 3.3, "persistence_mae": 5.2, "test_runs": 30, "created_at": "x"}

    def forecast(self, rows, sim, controller):
        return {"t": rows[-1]["t"] + 120, "from_t": rows[-1]["t"], "N": 1.0, "E": 2.0, "S": 3.0, "W": 4.0}


def test_session_streams_forecasts_and_sends_them_on_connect(tmp_path):
    from server.session import SimSession

    cfg = load_config()
    cfg.predict.model_path = str(tmp_path / "none.txt")  # no real model: use a stand-in
    s = SimSession(cfg, log_runs=False)
    assert s.forecaster is None and s.init_message()["predictor"] is None
    s.forecaster = FakeForecaster()
    s._attach_forecaster()
    sent = []
    s._broadcast = sent.append
    s.playing, s.speed = True, 20.0
    for _ in range(5):
        s.tick(0.05)  # 1 simulated second per tick
    metrics = [m for m in sent if m["type"] == "metrics"]
    assert metrics and all("forecast" in m for m in metrics)
    f = metrics[-1]["forecast"][-1]
    assert f["t"] == f["from_t"] + 120
    init = s.init_message()
    assert init["predictor"]["mae"] == 3.3 and len(init["forecast_history"]) == 5


def test_session_survives_a_failing_forecaster(tmp_path):
    from server.session import SimSession

    class Broken(FakeForecaster):
        def forecast(self, rows, sim, controller):
            raise RuntimeError("boom")

    cfg = load_config()
    cfg.predict.model_path = str(tmp_path / "none.txt")
    s = SimSession(cfg, log_runs=False)
    s.forecaster = Broken()
    s._attach_forecaster()
    s.playing, s.speed = True, 20.0
    for _ in range(3):
        s.tick(0.05)
    assert s.sim.time == pytest.approx(3.0) and s.run.forecaster is None  # forecasting off, sim fine
