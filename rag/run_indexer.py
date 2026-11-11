"""Turn a simulation run log (data/runs/<run_id>/) into readable text chunks for the knowledge base.

A run log is numbers (metrics.csv: one row per second) and JSON events. Neither embeds well:
the embedding model knows English, not CSV columns, and an analyst's question ("why did the
wait spike at minute 4?") is English too. So each run becomes short PROSE, three kinds of chunk:

  run_summary    one per run: controller, settings, whole-run results, event counts. Answers
                 "how did run X do?" and "which runs had spillback?"
  run_window     one per ``window_s`` (60 s): what the signals did, the queues per approach
                 (start -> end, peak with its time), arrivals, departures, delay and LOS, and
                 every notable event in words, each with its m:ss timestamp. Answers "what
                 happened at 3:20?" and "why did the wait spike at minute 4?"
  rl_decisions   one per window in which the RL agent decided: each switch (or forced switch) with
                 what it saw (queues, backlogs, longest waits per approach), its confidence and the
                 critic's value; the "keep" decisions summarised. Answers "why did the AI switch at
                 3:20?" from the logged observation, not a guess.

WHY 60 s windows: people ask about minutes ("at minute 4", "around 3:20"), and a minute covers
about half a signal cycle, short enough that a window's numbers describe ONE situation. Each chunk
starts with the run id and its time span, so the most important facts are inside the 256 word
pieces the embedding model reads, and every number in it can be cited with a timestamp.

Every number in the text comes straight from the log (rounded), never from a model: the
analyst's rule "never invent numbers" starts here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sim.metrics import clock

from .chunking import Chunk

RUN_INDEX_VERSION = 2  # bump when the summaries change: every run is re-indexed (part of its fingerprint)
CONTROLLER_NAMES = {"fixed": "fixed timer", "rl": "RL agent (PPO)", "gap_out": "gap-out rule",
                    "actuated": "actuated control"}
STAGE_WORDS = {"green": "green", "amber": "amber", "all_red": "all-red", "walk": "WALK", "ped_clear": "flashing DON'T WALK"}


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _num(x: float, digits: int = 1) -> str:
    """12.0 -> "12", 12.34 -> "12.3"."""
    r = round(float(x), digits)
    return str(int(r)) if r == int(r) else f"{r:.{digits}f}"


def _span(t0: float, t1: float) -> str:
    return f"{clock(t0)}-{clock(t1)}"


# ---------------------------------------------------------------------- loading
def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    events, decisions = [], []
    for name, out in (("events.jsonl", events), ("decisions.jsonl", decisions)):
        p = run_dir / name
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a run that crashed mid-write can end with a partial line
    return {"manifest": manifest, "rows": rows, "events": events, "decisions": decisions, "dir": run_dir}


# ---------------------------------------------------------------------- sentences
def event_sentence(ev: dict) -> str | None:
    """One event in plain English, with its timestamp. None for events not worth a sentence."""
    t, c = ev.get("type"), ev.get("clock", clock(ev.get("t", 0)))
    if t == "queue_spike":
        a = ev.get("approach")
        if ev.get("status") == "start":
            light = {"R": "red", "G": "green", "Y": "amber"}.get(ev.get("approach_light"), ev.get("approach_light"))
            since = ev.get("seconds_since_approach_green")
            since_txt = f", {_num(since)} s after its last green started" if since is not None else ""
            return (f"{c} queue spike on approach {a}: {ev.get('value')} vehicles queued while its light was {light}"
                    f"{since_txt} (phase {ev.get('phase')} {STAGE_WORDS.get(ev.get('stage'), ev.get('stage'))}); "
                    f"{ev.get('arrivals_last_60s')} arrivals in the previous minute, backlog {ev.get('backlog')}, "
                    f"mean wait of its vehicles {_num(_f(ev.get('wait_mean')))} s.")
        tail = " (still going when the run ended)" if ev.get("truncated") else ""
        return (f"{c} queue spike on approach {a} ended{tail}: it lasted {_num(_f(ev.get('duration_s')))} s from "
                f"{ev.get('started_clock')}, peak {ev.get('peak')} queued at {ev.get('peak_clock')}.")
    if t == "spillback":
        lane = ev.get("lane")
        if ev.get("status") == "start":
            return (f"{c} spillback on lane {lane} (approach {ev.get('approach')}): the queue reached "
                    f"{_num(_f(ev.get('queue_m')))} m, {_num(100 * _f(ev.get('value')), 0)}% of the lane, during phase {ev.get('phase')}.")
        return (f"{c} spillback on lane {lane} ended after {_num(_f(ev.get('duration_s')))} s (from "
                f"{ev.get('started_clock')}, peak {_num(100 * _f(ev.get('peak')), 0)}% of the lane at {ev.get('peak_clock')}).")
    if t == "gridlock":
        if ev.get("status") == "start":
            return (f"{c} GRIDLOCK: no vehicle had left for a minute with {ev.get('in_network')} vehicles in the "
                    f"network, {_num(100 * _f(ev.get('stopped_frac')), 0)}% stopped, backlog {ev.get('backlog')}.")
        return f"{c} gridlock ended after {_num(_f(ev.get('duration_s')))} s."
    if t == "ped_phase":
        return (f"{c} pedestrian phase started (all vehicle movements red): {ev.get('waiting')} pedestrians "
                f"waiting, the longest for {_num(_f(ev.get('max_wait')))} s.")
    if t == "los_change":
        frm = ev.get("from") or "none"
        return (f"{c} level of service changed from {frm} to {ev.get('to')} (mean control delay over the last "
                f"2 min {_num(_f(ev.get('mean_delay_window')))} s).")
    if t == "config_change":
        parts = []
        for k, v in (ev.get("changes") or {}).items():
            if k == "per_approach_vph":
                parts.append("demand per approach " + ", ".join(f"{a} {_num(x, 0)}" for a, x in v.items()) + " veh/h")
            elif k == "vehicle_mix":
                parts.append("vehicle mix " + ", ".join(f"{a} {_num(100 * x, 0)}%" for a, x in v.items()))
            elif k == "ped_rate_ph":
                parts.append(f"pedestrians {_num(v, 0)} per hour")
            elif k == "green_s":
                parts.append(f"fixed-timer green {_num(v)} s")
            elif k == "aggressiveness":
                parts.append(f"driver aggressiveness {v}")
            else:
                parts.append(f"{k} {v}")
        return f"{c} settings changed ({ev.get('source', 'user')}): " + "; ".join(parts) + "."
    if t == "controller_change":
        return (f"{c} signal control handed over from the {CONTROLLER_NAMES.get(ev.get('from'), ev.get('from'))} "
                f"to the {CONTROLLER_NAMES.get(ev.get('to'), ev.get('to'))}.")
    if t == "stress_start":
        return (f"{c} live stress test started: demand from {_num(_f(ev.get('start_vph')), 0)} veh/h per approach, "
                f"+{_num(_f(ev.get('step_vph')), 0)} every {_num(_f(ev.get('step_s')), 0)} s.")
    if t == "stress_step":
        ok = "passed" if ev.get("passed") else "FAILED"
        fail = ev.get("failing_approaches") or []
        return (f"{c} stress test step {ev.get('step')} at {_num(_f(ev.get('volume_vph')), 0)} veh/h per approach {ok}: "
                f"offered {_num(_f(ev.get('offered_vph')), 0)} veh/h, served {_num(_f(ev.get('served_vph')), 0)} veh/h "
                f"({_num(100 * _f(ev.get('served_ratio')), 0)}%)" + (f", falling behind on {', '.join(fail)}" if fail else "") + ".")
    if t == "stress_result":
        return (f"{c} stress test finished ({ev.get('reason')}): kept up to {_num(_f(ev.get('capacity_level_vph')), 0)} "
                f"veh/h per approach, {_num(_f(ev.get('capacity_vph')), 0)} veh/h served in total.")
    return None


def _signal_story(events: list[dict], t0: float, t1: float) -> str:
    """What the lights did in [t0, t1): which phase was green when, and pedestrian phases."""
    changes = [e for e in events if e.get("type") == "phase_change"]
    before = [e for e in changes if e["t"] <= t0]
    inside = [e for e in changes if t0 < e["t"] < t1]
    state = before[-1] if before else None
    parts = []
    if state:
        name = "pedestrian phase" if state["phase"] == "PED" else f"{state['phase']} {STAGE_WORDS.get(state['stage'], state['stage'])}"
        parts.append(f"at {clock(t0)} {name}")
    for e in inside:
        if e["stage"] == "green":
            parts.append(f"{e['phase']} green from {e['clock']}")
        elif e["stage"] == "walk":
            parts.append(f"pedestrian phase from {e['clock']}")
        elif e["stage"] == "amber":
            parts.append(f"{e['phase']} green ended {e['clock']}")
    return "; ".join(parts) if parts else "no phase change"


# ---------------------------------------------------------------------- chunks
def run_header(m: dict) -> str:
    ctl = CONTROLLER_NAMES.get(m.get("controller"), m.get("controller"))
    tags = m.get("tags") or {}
    side = f", comparison {tags['compare_id']} ({tags.get('compare_side')} side)" if tags.get("compare_id") else ""
    return f"Run {m['run_id']} ({ctl}, seed {m.get('seed')}{side})"


def summary_chunk(run: dict) -> Chunk:
    m, rows, events = run["manifest"], run["rows"], run["events"]
    cfg = m.get("config") or {}
    dem = (cfg.get("demand") or {}).get("per_approach_vph", {})
    mix = (cfg.get("demand") or {}).get("vehicle_mix", {})
    sig = cfg.get("signals") or {}
    s = m.get("summary") or {}
    dur = _f(m.get("sim_duration_s")) or (_f(rows[-1]["t"]) if rows else 0.0)
    lines = [f"{run_header(m)}: summary of the whole run, 0:00-{clock(dur)}, status {m.get('status')}, "
             f"created {m.get('created_at')}."]
    if dem:
        green = sig.get("green_s")
        green_txt = (f"{_num(green)} s" if isinstance(green, (int, float)) else ", ".join(_num(g) for g in green) + " s") if green is not None else "?"
        lines.append(f"Starting settings: demand per approach {', '.join(f'{a} {_num(v, 0)}' for a, v in dem.items())} veh/h; "
                     f"pedestrians {_num(_f((cfg.get('pedestrians') or {}).get('rate_ph')), 0)} per hour; "
                     f"fixed-timer green {green_txt}; vehicle mix {', '.join(f'{a} {_num(100 * w / max(sum(mix.values()), 1e-9), 0)}%' for a, w in mix.items())}; "
                     f"driver aggressiveness {(cfg.get('driver') or {}).get('aggressiveness')}.")
    if s:
        lines.append(f"Results: {s.get('completed')} vehicles completed, throughput {_num(_f(s.get('throughput_vph')), 0)} veh/h, "
                     f"mean control delay {_num(_f(s.get('mean_delay_s')))} s (95th percentile {_num(_f(s.get('p95_delay_s')))} s), "
                     f"level of service {s.get('los')}, mean wait {_num(_f(s.get('mean_wait_s')))} s, "
                     f"{s.get('backlog_end')} vehicles still waiting to enter at the end.")
        per = s.get("per_approach") or {}
        if per:
            lines.append("Per approach: " + "; ".join(
                f"{a} mean delay {_num(_f(v.get('mean_delay_s')))} s, worst queue {v.get('max_queue')} at {v.get('max_queue_clock')}"
                for a, v in per.items()) + ".")
        p = s.get("pedestrians") or {}
        if p:
            lines.append(f"Pedestrians: {p.get('crossings_started')} crossings, mean wait {_num(_f(p.get('mean_wait_s')))} s "
                         f"(longest {_num(_f(p.get('max_wait_s')))} s), {p.get('ped_phases')} pedestrian phases.")
    counts = {}
    for e in events:
        if e.get("type") != "phase_change" and e.get("status") != "end":
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    if counts:
        lines.append("Events: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(counts.items())) + ".")
    special = [event_sentence(e) for e in events if e.get("type") in ("controller_change", "config_change", "stress_result", "gridlock")
               and e.get("status") != "end"]
    special = [x for x in special if x]
    if special:
        lines.append("Notable: " + " ".join(special[:8]))
    return Chunk(f"run:{m['run_id']}:summary", "\n".join(lines),
                 {"kind": "run_summary", "t_start": 0.0, "t_end": round(dur, 1)})


def window_chunk(run: dict, t0: float, t1: float) -> Chunk | None:
    m, events = run["manifest"], run["events"]
    rows = [r for r in run["rows"] if t0 < _f(r["t"]) <= t1]
    if not rows:
        return None
    arms = sorted(k[len("queue_"):] for k in rows[0] if k.startswith("queue_") and k != "queue_total")
    first, last = rows[0], rows[-1]
    # A run still being written (the live run) ends mid-window: the window then ends at its last
    # logged second, not at the minute mark (measured: "5:00-6:00, stopped at 6:00" at 5:11).
    t1 = min(t1, _f(last["t"]))
    lines = [f"{run_header(m)}, {_span(t0, t1)}" + (" (the run so far)." if t1 < t0 + 59.5 else ".")]
    lines.append(f"Signals: {_signal_story(events, t0, t1)}.")
    q = []
    for a in arms:
        vals = [int(_f(r[f"queue_{a}"])) for r in rows]
        peak = max(vals)
        peak_t = clock(_f(rows[vals.index(peak)]["t"]))
        q.append(f"{a} {vals[0]}->{vals[-1]} (peak {peak} at {peak_t})")
    lines.append("Queued vehicles per approach, start->end: " + "; ".join(q) + ".")
    backlog = [(a, int(_f(last.get(f"backlog_{a}")))) for a in arms]
    if any(b for _, b in backlog):
        lines.append("Waiting off the map to enter (lane full): " + ", ".join(f"{a} {b}" for a, b in backlog if b) + ".")
    arr = {a: sum(int(_f(r.get(f"arrivals_{a}"))) for r in rows) for a in arms}
    # "completed" is cumulative; the first row of the window already includes its own second.
    done = int(_f(last["completed"])) - int(_f(first["completed"]))
    lines.append(f"Arrivals: {', '.join(f'{a} {n}' for a, n in arr.items())} ({sum(arr.values())} in total); "
                 f"{done} vehicles left the junction.")
    d0, d1 = first.get("mean_delay_window"), last.get("mean_delay_window")
    if d1 not in (None, ""):
        d0txt = f"{_num(_f(d0))} s -> " if d0 not in (None, "") else ""
        lines.append(f"Mean control delay (rolling 2 min): {d0txt}{_num(_f(d1))} s, level of service {last.get('los') or '-'}; "
                     f"throughput {_num(_f(last.get('throughput_vph')), 0)} veh/h; {_num(100 * _f(last.get('stopped_frac')), 0)}% of vehicles stopped at {clock(t1)}.")
    pw = max(int(_f(r.get("peds_waiting"))) for r in rows)
    if pw:
        lines.append(f"Pedestrians: up to {pw} waiting at the kerb, longest wait {_num(max(_f(r.get('ped_longest_wait')) for r in rows))} s.")
    evs = [event_sentence(e) for e in events if t0 < e.get("t", -1) <= t1 and e.get("type") != "phase_change"]
    evs = [x for x in evs if x]
    lines.append("Events: " + (" ".join(evs) if evs else "none."))
    return Chunk(f"run:{m['run_id']}:w{int(t0)}", "\n".join(lines),
                 {"kind": "run_window", "t_start": round(t0, 1), "t_end": round(t1, 1)})


def _green_seconds(d: dict) -> float:
    """How long the green had run at the decision. Taken from the observation the network saw
    (time_in_green, scaled by 60 s in rl/features.py) because logs written before 2026-11-11 stored
    the time AFTER a switch (0) in time_in_green_s."""
    obs = d.get("obs") or {}
    if "time_in_green" in obs and _f(obs["time_in_green"]) < 3.0:  # 3.0 = the clip (180 s)
        return 60.0 * _f(obs["time_in_green"])
    return _f(d.get("time_in_green_s"))


def decision_sentence(d: dict) -> str:
    arms = sorted(d.get("queue", {}))
    seen = "; ".join(f"{a} queue {d['queue'][a]}, backlog {d['backlog'].get(a, 0)}, longest wait "
                     f"{_num(_f(d['longest_wait_s'].get(a)))} s" for a in arms)
    if d.get("forced"):
        verdict = (f"FORCED SWITCH by the {d['forced'].replace('_', '-')} guard (the network itself said keep, "
                   f"P(switch) {_num(100 * _f(d.get('p_switch')), 0)}%)")
    elif d.get("action") == "switch":
        verdict = f"SWITCH, P(switch) {_num(100 * _f(d.get('p_switch')), 0)}%" + ("" if d.get("switched") else " (not accepted)")
    else:
        verdict = f"keep, P(switch) {_num(100 * _f(d.get('p_switch')), 0)}%"
    ped = ", pedestrian call pending" if d.get("ped_call") else ""
    return (f"{d.get('clock')}: phase {d.get('phase')} green for {_num(_green_seconds(d))} s, next would be "
            f"{d.get('next_phase')}{ped}. Decision: {verdict}; critic value {d.get('value')}. It saw: {seen}.")


def decisions_chunk(run: dict, t0: float, t1: float) -> Chunk | None:
    ds = [d for d in run["decisions"] if t0 < _f(d.get("t")) <= t1]
    if not ds:
        return None
    m = run["manifest"]
    rows = run["rows"]
    if rows:
        t1 = min(t1, max(_f(rows[-1]["t"]), _f(ds[-1].get("t"))))  # the live run ends mid-window
    switches = [d for d in ds if d.get("action") == "switch" or d.get("forced")]
    keeps = [d for d in ds if d not in switches]
    lines = [f"{run_header(m)}, RL agent decisions {_span(t0, t1)}: {len(ds)} decisions where a switch was possible, "
             f"{len(switches)} switch{'es' if len(switches) != 1 else ''}."]
    lines += [decision_sentence(d) for d in switches]
    if keeps:
        ps = [100 * _f(d.get("p_switch")) for d in keeps]
        last_keep = keeps[-1]
        lines.append(f"It kept the green at {', '.join(d.get('clock') for d in keeps)} "
                     f"(P(switch) {_num(min(ps), 0)}-{_num(max(ps), 0)}%). Last keep: {decision_sentence(last_keep)}")
    return Chunk(f"run:{m['run_id']}:d{int(t0)}", "\n".join(lines),
                 {"kind": "rl_decisions", "t_start": round(t0, 1), "t_end": round(t1, 1)})


def run_chunks(run_dir: Path, window_s: float = 60.0) -> list[Chunk]:
    """Every chunk of one run, with run-level metadata on each."""
    run = load_run(run_dir)
    m = run["manifest"]
    rows = run["rows"]
    end = _f(rows[-1]["t"]) if rows else 0.0
    chunks = [summary_chunk(run)]
    t0 = 0.0
    while t0 < end:
        t1 = t0 + window_s
        for c in (window_chunk(run, t0, t1), decisions_chunk(run, t0, t1)):
            if c is not None:
                chunks.append(c)
        t0 = t1
    base = {"source_type": "run", "run_id": m["run_id"], "controller": m.get("controller") or "",
            "seed": int(m.get("seed") or 0), "created_at": m.get("created_at") or "", "status": m.get("status") or ""}
    for c in chunks:
        c.metadata = {**base, **c.metadata, "clock_start": clock(c.metadata["t_start"]),
                      "clock_end": clock(c.metadata["t_end"]),
                      "citation": f"run {m['run_id']} @ {clock(c.metadata['t_start'])}-{clock(c.metadata['t_end'])}"}
    return chunks
