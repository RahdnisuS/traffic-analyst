"""Live stress test for the dashboard: ramps demand on the shared, running simulation.

It is an observer like the metrics collector: after every sim step the session calls
``after_step``, which only looks at the clock until a step boundary is reached. All judging is
done by the same ``judge`` / ``stock_growth`` functions as the headless benchmark.

Differences from ``python -m bench`` (why this is a demo, not the score):
  * ONE continuous run: queues left over from one step carry into the next.
  * Short judged windows (5 min). After cycle-averaging, ~10 vehicles of noise remain, so a
    window needs ~200 arrivals per approach before a 5% shortfall stands out: ~20 simulated
    minutes per level. A 5-minute window can only catch CLEAR overload. Measured: the same live
    ramp put the last "kept up" level anywhere from 300 to 700 veh/h per approach depending on
    the seed (the benchmark: 500-550). So the dashboard reports the levels it saw ("kept up to
    X, clearly overloaded at Y") next to the official score, not a competing capacity figure.
When the ramp ends (breakdown, max demand, or the user stops it) the original traffic volume is
restored, so the queues can be seen draining away.
"""

from __future__ import annotations

from sim.config import BenchmarkSettings
from sim.metrics import MetricsCollector

from .stress import Snapshot, approach_stock, cycle_length, judge, stock_growth


class LiveStressTest:
    def __init__(self, sim, collector: MetricsCollector, settings: BenchmarkSettings, max_vph: float):
        self.s = settings
        self.collector = collector
        self.max_vph = min(settings.max_vph, max_vph)
        self.original_vph = round(sim.mean_volume_vph(), 1)
        self.active = True
        self.steps: list[dict] = []
        self.fails = 0
        self.result: dict | None = None
        self.events: list[dict] = [collector.custom_event("stress_start", sim.time, {
            "start_vph": settings.live_start_vph, "step_vph": settings.live_step_vph,
            "step_s": settings.live_step_s, "original_vph": self.original_vph})]
        self._begin_step(sim, settings.live_start_vph)

    # ------------------------------------------------------------------ steps
    def _begin_step(self, sim, vph: float) -> None:
        self.level = round(vph, 1)
        self.step_start = sim.time
        self._start: Snapshot | None = None   # taken once the settle period is over
        self._samples: list[tuple[float, dict]] = []
        self._gridlock = False
        sim.apply_settings(volume_vph=self.level, source="stress_test")

    def after_step(self, sim) -> list[dict]:
        """Called after every sim step. Returns any events it raised (usually none)."""
        if not self.active:
            return []
        out, self.events = self.events, []
        elapsed = sim.time - self.step_start
        if elapsed + 1e-9 >= self.s.live_settle_s:
            if self._start is None:
                self._start = Snapshot.take(sim, self.collector)
                self._gridlock = self.collector.episode_active("gridlock")
            # One stock sample per simulated second, like the headless benchmark.
            if not self._samples or sim.time - self._samples[-1][0] >= 1.0 - 1e-9:
                self._samples.append((sim.time, approach_stock(sim)))
        if elapsed + 1e-9 >= self.s.live_step_s:
            out += self._finish_step(sim)
        return out

    def _finish_step(self, sim) -> list[dict]:
        end = Snapshot.take(sim, self.collector)
        gridlock = (self._gridlock or end.gridlocks > self._start.gridlocks
                    or self.collector.episode_active("gridlock"))
        r = judge(self._start, end, self.s, gridlock, growth=stock_growth(self._samples, cycle_length(sim)))
        step = {"step": len(self.steps) + 1, "volume_vph": self.level, "offered_vph": r["offered_vph"],
                "served_vph": r["served_vph"], "served_ratio": r["served_ratio"], "passed": r["passed"],
                "failing_approaches": r["failing_approaches"], "gridlock": r["gridlock"]}
        self.steps.append(step)
        events = [self.collector.custom_event("stress_step", sim.time, step)]
        self.fails = 0 if r["passed"] else self.fails + 1
        nxt = self.level + self.s.live_step_vph
        if self.fails >= self.s.fails_to_stop:
            events += self._end(sim, "breakdown")
        elif nxt > self.max_vph + 1e-9:
            events += self._end(sim, "max_demand")
        else:
            self._begin_step(sim, nxt)
        return events

    # ------------------------------------------------------------------ ending
    def stop(self, sim) -> list[dict]:
        """User cancelled (or the session is resetting)."""
        if not self.active:
            return []
        return self.events + self._end(sim, "stopped")

    def _end(self, sim, reason: str) -> list[dict]:
        self.active = False
        self.events = []
        passed = [st for st in self.steps if st["passed"]]
        best = max(passed, key=lambda st: st["volume_vph"]) if passed else None
        self.result = {"reason": reason, "steps": len(self.steps),
                       "capacity_vph": best["served_vph"] if best else None,
                       "capacity_level_vph": best["volume_vph"] if best else None,
                       "overloaded_level_vph": next((st["volume_vph"] for st in self.steps if not st["passed"]
                                                     and (best is None or st["volume_vph"] > best["volume_vph"])),
                                                    None),
                       # reaching max demand without breaking down means "at least this much"
                       "censored": reason == "max_demand",
                       "restored_vph": self.original_vph}
        sim.apply_settings(volume_vph=self.original_vph, source="stress_test")
        return [self.collector.custom_event("stress_result", sim.time, dict(self.result))]

    # ------------------------------------------------------------------ status for the UI
    def status(self) -> dict:
        return {"active": self.active, "level_vph": self.level if self.active else None,
                "step": len(self.steps) + (1 if self.active else 0), "step_start_t": round(self.step_start, 1),
                "step_s": self.s.live_step_s, "settle_s": self.s.live_settle_s,
                "steps": self.steps, "result": self.result}
