"""Phase 8: AI on/off toggle and the side-by-side comparison in the dashboard session."""

import asyncio

import pytest

pytest.importorskip("sb3_contrib")

from rl.env import SignalEnv  # noqa: E402
from rl.policy import RLPolicy  # noqa: E402
from server.schemas import CompareCmd, ControllerCmd, SettingsCmd, StressCmd  # noqa: E402
from server.session import SimSession  # noqa: E402
from sim.config import load_config  # noqa: E402
from sim.signals import FixedTimePolicy  # noqa: E402

TRAFFIC = {"per_approach_vph": {"N": 500.0, "E": 300.0, "S": 450.0, "W": 350.0}, "ped_rate_ph": 200.0, "seed": 9}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """A tiny untrained network: the session only needs SOME model, not a good one, and the real
    one (rl/models/*.zip) is gitignored so it doesn't exist on a fresh clone."""
    from sb3_contrib import MaskablePPO

    cfg = load_config()
    path = tmp_path_factory.mktemp("model") / "tiny.zip"
    MaskablePPO("MlpPolicy", SignalEnv(cfg, traffic=TRAFFIC), seed=4, device="cpu",
                policy_kwargs={"net_arch": [8]}).save(path)
    return path


@pytest.fixture()
def session(model_path, tmp_path):
    cfg = load_config()
    cfg.rl.model_path = str(model_path)
    return SimSession(cfg, log_runs=True, runs_dir=tmp_path)


def run(coro):
    return asyncio.run(coro)


def advance(s: SimSession, seconds: float) -> None:
    s.playing, s.speed = True, 20.0
    for _ in range(int(seconds / (20.0 / 20))):  # each tick = speed * 0.05 s = 1 s simulated
        s.tick(0.05)


def test_ai_toggle_hands_over_live_and_is_logged(session):
    advance(session, 30)
    t = session.sim.time
    run(session.handle(ControllerCmd(cmd="controller", value="rl")))
    assert isinstance(session.sim.policy, RLPolicy) and session.sim.time == t  # same run, no reset
    assert session.status()["controller"] == "rl"
    ev = [e for e in session.run.pending_events if e["type"] == "controller_change"]
    assert ev and ev[0]["from"] == "fixed" and ev[0]["to"] == "rl"
    advance(session, 60)  # long enough to outlast a pedestrian phase, where the AI can't act
    assert session.sim.decisions, "the AI should be making (logged) decisions"
    # Green time can be set while the AI drives: it is kept for the fixed-time plan.
    run(session.handle(SettingsCmd(cmd="settings", green_s=33)))
    run(session.handle(ControllerCmd(cmd="controller", value="fixed")))
    assert isinstance(session.sim.policy, FixedTimePolicy)
    assert session.sim.policy.green_times[0] == 33.0 and session.settings()["green_s"] == 33.0


def test_ai_toggle_without_a_model_is_refused(tmp_path):
    cfg = load_config()
    cfg.rl.model_path = str(tmp_path / "missing.zip")
    s = SimSession(cfg, log_runs=False)
    assert s.status()["ai_available"] is False
    with pytest.raises(ValueError, match="no trained AI model"):
        run(s.handle(ControllerCmd(cmd="controller", value="rl")))
    assert s.controller == "fixed"


def test_comparison_runs_both_controllers_on_identical_traffic(session):
    advance(session, 20)
    run(session.handle(SettingsCmd(cmd="settings", volume_vph=450)))
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    main, twin = session.run, session.twin
    # Both restart from an empty junction at t = 0, same seed, same settings.
    assert main.sim.time == twin.sim.time == 0 and main.sim.seed == twin.sim.seed
    assert isinstance(main.sim.policy, FixedTimePolicy) and isinstance(twin.sim.policy, RLPolicy)
    assert twin.sim.mean_volume_vph() == pytest.approx(450, abs=1)
    advance(session, 60)
    run(session.handle(SettingsCmd(cmd="settings", volume_vph=600, ped_rate_ph=400)))  # mid-run change
    advance(session, 60)
    assert main.sim.time == twin.sim.time == pytest.approx(120)  # lockstep
    # Every random draw is independent of the signals, so identical traffic means the two
    # generators have made exactly the same draws: their states must be bit-identical.
    assert main.sim.rng.bit_generator.state == twin.sim.rng.bit_generator.state
    assert main.sim.cfg.demand.per_approach_vph == twin.sim.cfg.demand.per_approach_vph
    c = session.comparison()
    assert c["fixed"]["controller"] == "fixed" and c["rl"]["controller"] == "rl"
    assert c["fixed"]["waiting_veh_s"] > 0 and c["change_pct"]["waiting"] is not None
    # Logged as two runs tagged with the same comparison id.
    tags = [r.logger.manifest["tags"] for r in (main, twin)]
    assert tags[0]["compare_id"] == tags[1]["compare_id"] and {t["compare_side"] for t in tags} == {"fixed", "rl"}


def test_frames_and_metrics_carry_both_runs(session):
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    frame = session._frame()
    assert frame["controller"] == "fixed" and frame["twin"]["controller"] == "rl"
    init = session.init_message()
    assert init["compare"]["active"] and init["twin_history"] == []


def test_stop_comparison_keeps_the_side_matching_the_toggle(session):
    run(session.handle(ControllerCmd(cmd="controller", value="rl")))
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    twin = session.twin
    advance(session, 15)
    run(session.handle(CompareCmd(cmd="compare", action="stop")))
    assert session.twin is None and session.run is twin and session.sim.time == pytest.approx(15)
    assert session.status()["compare"] is None


def test_stop_comparison_can_choose_which_side_to_keep(session):
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    fixed, twin = session.run, session.twin
    advance(session, 15)
    run(session.handle(CompareCmd(cmd="compare", action="stop", keep="rl")))
    assert session.run is twin and session.controller == "rl" and isinstance(session.sim.policy, RLPolicy)
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    advance(session, 5)
    fixed = session.run
    run(session.handle(CompareCmd(cmd="compare", action="stop", keep="fixed")))
    assert session.run is fixed and session.controller == "fixed"


def test_comparison_and_stress_test_exclude_each_other(session):
    run(session.handle(CompareCmd(cmd="compare", action="start")))
    with pytest.raises(ValueError, match="comparison"):
        run(session.handle(StressCmd(cmd="stress", action="start")))
    with pytest.raises(ValueError, match="stop it"):
        run(session.handle(ControllerCmd(cmd="controller", value="rl")))
    with pytest.raises(ValueError, match="already running"):
        run(session.handle(CompareCmd(cmd="compare", action="start")))
    run(session.handle(CompareCmd(cmd="compare", action="stop")))
    run(session.handle(StressCmd(cmd="stress", action="start")))
    with pytest.raises(ValueError, match="stress test"):
        run(session.handle(CompareCmd(cmd="compare", action="start")))


def test_new_commands_are_validated():
    from pydantic import ValidationError

    from server.schemas import command_adapter

    assert isinstance(command_adapter.validate_python({"cmd": "controller", "value": "rl"}), ControllerCmd)
    assert isinstance(command_adapter.validate_python({"cmd": "compare", "action": "stop"}), CompareCmd)
    for bad in ({"cmd": "controller", "value": "turbo"}, {"cmd": "compare", "action": "pause"},
                {"cmd": "controller", "value": "rl", "extra": 1}):
        with pytest.raises(ValidationError):
            command_adapter.validate_python(bad)
