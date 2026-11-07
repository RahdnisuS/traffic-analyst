# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Traffic Analyst": a road-junction simulator with reinforcement-learning signal control, a LightGBM predictor and an LLM (RAG) analyst, built as a portfolio project in 14 phases. Phases 1–3 are done (headless sim, metrics + run logs, FastAPI/WebSocket dashboard). Next is Phase 4 (mixed traffic and pedestrians).

**`docs/SPEC.md` is the project spec: always read the relevant phase in it before starting a phase.** It holds the original brief (all 14 phases, tech stack, working rules), a decision log that overrides the brief where they differ, and a status checklist. When a phase lands, update both its Status list and the README's. Record new design decisions in its decision log.

Workflow: after each phase, stop, summarise, explain the concepts simply, commit, and wait for the user's go-ahead. Ask before major design decisions the spec doesn't cover. Keep the "why" comments in the code — the user has to explain every design decision in interviews.

## Commands

Python 3.13 venv at `.venv` (Windows: `.venv/Scripts/activate`).

```bash
pip install -r requirements.txt     # deps are grouped by phase; later phases append theirs
python -m sim --duration 900        # headless run; flags: --seed, --config, --report-every, --no-log, --runs-dir
python -m server                    # dashboard at http://127.0.0.1:8000 (--host, --port; PORT env var wins)
pytest -q                           # all tests
pytest tests/test_idm.py::test_name # single test
```

`tests/conftest.py` puts the repo root on `sys.path`, so run pytest from the root.

## Architecture

- **`config.yaml` is the single source of truth** for every tunable parameter, validated by Pydantic in `sim/config.py` (unknown keys are errors). Add new parameters there rather than hard-coding them. Units: metres, seconds, m/s, vph.
- **Junction geometry is data**: `junctions/*.json` (arms, lane counts, allowed movements per lane, signal phases) is loaded by `sim/junction.py`, which builds lane edges, turning connectors, the conflict sets between connectors, and phases.
- **`sim/engine.py` `Simulation.step()`** advances by `dt` with no I/O or wall-clock time: signal policy → spawner → leader finding → vectorised IDM (`sim/idm.py`) → integration/edge transitions. Junction entry uses a reservation ("box-junction") rule: commit only on green (or amber when unable to stop), with no conflicting path occupied or reserved, and room in the exit lane.
- **Determinism is a hard invariant**: all randomness comes from one seeded `np.random.Generator`, and loops iterate in a fixed order. Same seed + config gives a bit-identical run (`Simulation.fingerprint()`; tested). Don't introduce unseeded randomness or set/dict-order-dependent iteration.
- **Signals**: `SignalController` (`sim/signals.py`) owns the green → amber → all-red stages and enforces `min_green_s`. Policies implement the `SignalPolicy` protocol (`update(controller, sim)`) and can only *request* switches; `FixedTimePolicy` is the current one. The RL agent is meant to plug in as another policy and respect the same min-green floor.
- **Observers**: `MetricsCollector` (`sim/metrics.py`) samples queues, delay, throughput and LOS, and detects episodes (queue spike, spillback, gridlock) with on/off hysteresis. `RunLogger` (`sim/logger.py`) writes `data/runs/<timestamp>_<controller>_s<seed>/` containing `manifest.json`, `metrics.csv`, `trips.csv` and `events.jsonl`. These logs are what later phases (predictor, RAG) consume.
- **Server** (`server/`): one shared `SimSession` is advanced by a single asyncio task at 20 Hz using a "time debt" so speed is exact on average; it drops debt rather than spiralling when it falls behind. Each client has an outbox where position frames are latest-wins and metrics/events are reliable and ordered. The same collector output feeds both the logger and the browsers, so the dashboard and logs always agree. Client commands are Pydantic models in `server/schemas.py`. REST: `/api/health`, `/api/layout`, `/api/runs`, `/api/runs/{id}`; WebSocket: `/ws/sim`; `web/` is served statically at `/`.
- **Frontend** (`web/`): plain JS, no build step. `interp.js` interpolates between the two latest frames (entity interpolation), `render2d.js` draws on a canvas, and `charts.js` draws the live charts.

## Conventions / decisions

- Left-hand (Indian) traffic; 4-phase split signalling (one approach green at a time); lane counts come from the junction JSON.
- Only cars exist so far; Phase 4 adds bike/auto/bus vehicle types (bikes filter through gaps) and social-force pedestrians. Crossings use an on-demand pedestrian phase.
- `.env` holds `GROQ_API_KEY` (see `.env.example`) for the LLM phase; never commit it. `.claude/`, `data/runs/`, `chroma/` and `rl/models/*.zip` are gitignored.
