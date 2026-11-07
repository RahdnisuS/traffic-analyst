# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Traffic Analyst": a road-junction simulator with reinforcement-learning signal control, a LightGBM predictor and an LLM (RAG) analyst, built as a portfolio project in 14 phases. Phases 1–4 are done (headless sim, metrics + run logs, FastAPI/WebSocket dashboard, mixed traffic + pedestrians). Next is Phase 5 (2.5D Three.js renderer).

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

`tests/conftest.py` puts the repo root on `sys.path`, so run pytest from the root. In the Bash tool the venv activation doesn't persist between calls; call `.venv/Scripts/python -m pytest -q` (or `-m sim`, `-m server`) directly.

## Architecture

- **`config.yaml` is the single source of truth** for every tunable parameter, validated by Pydantic in `sim/config.py` (unknown keys are errors). Add new parameters there rather than hard-coding them. Units: metres, seconds, m/s, vph.
- **Junction geometry is data**: `junctions/*.json` (arms, lane counts, allowed movements per lane, signal phases) is loaded by `sim/junction.py`, which builds lane edges, turning connectors, the conflict sets between connectors, and phases.
- **`sim/engine.py` `Simulation.step()`** advances by `dt` with no I/O or wall-clock time: signal policy → spawner → pedestrians → insertion → bike filtering → leader finding → vectorised IDM (`sim/idm.py`) → integration/edge transitions. Junction entry uses a reservation ("box-junction") rule: commit only on green (or amber when unable to stop), with no conflicting path occupied or reserved, room in the exit lane, and nobody walking on a crosswalk the path crosses.
- **Edge kinds**: `inbound`, `filter` (`flt_*`: bike-only virtual lane on a lane divider; exit-side twins `oflt_*` are kind `outbound`), `connector`, `outbound`. Code that means "ends at a stop line" uses `Edge.is_approach` / `junction.approach_edges()` / `approach_lanes(arm)`, not `kind == "inbound"`. Conflicts are computed from geometry (filter-lane paths use a smaller threshold). `junction.filter_movements` derives which movements may filter.
- **Vehicle types** (car/bike/auto/bus) differ by IDM parameters plus two flags in `VehicleTypeParams`: `filters` (bikes) and `kerb_lane` (buses). `Vehicle.lat_shift` is drawing-only (smooth swerve into a filter gap); render positions come from `Simulation.lane_poses()`.
- **Pedestrians** (`sim/pedestrian.py`): `social_force()` is the pure, vectorised model (elliptical specification); `PedestrianSystem` runs the APPROACH → WAIT → CROSS → LEAVE lifecycle. Walkers and vehicles are separated by rules, not forces.
- **Determinism is a hard invariant**: all randomness comes from one seeded `np.random.Generator`, and loops iterate in a fixed order. Same seed + config gives a bit-identical run (`Simulation.fingerprint()`; tested). Don't introduce unseeded randomness or set/dict-order-dependent iteration.
- **Signals**: `SignalController` (`sim/signals.py`) owns the green → amber → all-red stages and enforces `min_green_s`. It also inserts the exclusive pedestrian phase (WALK → PED_CLEAR → all-red, phase name `"PED"`) when called, at most once every `min_greens_between` greens. Use `signals.phase_name` for display/logs (`signals.phase` is the last vehicle phase). Policies implement the `SignalPolicy` protocol (`update(controller, sim)`) and can only *request* switches; `FixedTimePolicy` is the current one. The RL agent is meant to plug in as another policy and respect the same min-green floor.
- **Observers**: `MetricsCollector` (`sim/metrics.py`) samples queues, delay, throughput and LOS, and detects episodes (queue spike, spillback, gridlock) with on/off hysteresis. `MetricsCollector.after_step()` returns `(row, events, trips, ped_trips)`. `RunLogger` (`sim/logger.py`) writes `data/runs/<timestamp>_<controller>_s<seed>/` containing `manifest.json` (schema v2), `metrics.csv`, `trips.csv`, `peds.csv` and `events.jsonl`. These logs are what later phases (predictor, RAG) consume.
- **Server** (`server/`): one shared `SimSession` is advanced by a single asyncio task at 20 Hz using a "time debt" so speed is exact on average; it drops debt rather than spiralling when it falls behind. Each client has an outbox where position frames are latest-wins and metrics/events are reliable and ordered. The same collector output feeds both the logger and the browsers, so the dashboard and logs always agree. Client commands are Pydantic models in `server/schemas.py`. REST: `/api/health`, `/api/layout`, `/api/runs`, `/api/runs/{id}`; WebSocket: `/ws/sim`; `web/` is served statically at `/`.
- **Frontend** (`web/`): plain JS, no build step. `interp.js` interpolates between the two latest frames (entity interpolation) for vehicles and pedestrians, `render2d.js` draws on a canvas, and `charts.js` draws the live charts. Renderers only consume server state, so a new renderer (Phase 5) needs just these: the `init` message (`layout` from `Junction.describe()` incl. `crosswalks` and `sidewalk_width`, `vehicle_types`, `vehicle_dims` [length, width], `ped_radius`) and each `frame`: `vehicles` as `[id, x, y, heading, speed, typeIdx]`, `peds` as `[id, x, y, state]` (0 approach, 1 wait, 2 cross, 3 leave), `signals` per arm (G/Y/R), `ped_signal` (walk/flash/dont_walk), `phase`, `stage`. Vehicle x/y is the FRONT bumper.

## Conventions / decisions

- Left-hand (Indian) traffic; 4-phase split signalling (one approach green at a time); lane counts come from the junction JSON.
- Safety invariants are tested every step (no overlaps, no red-light entry, no conflicting box paths, no vehicle onto an occupied crosswalk, `emergency_stops == 0`). If a change makes `emergency_stops` non-zero, find the cause; don't relax the test. (Example: the red-light stop obstacle is placed from each vehicle's own `min_gap`, because bikes' small s0 parked them on the line.)
- `.env` holds `GROQ_API_KEY` (see `.env.example`) for the LLM phase; never commit it. `.claude/`, `data/runs/`, `chroma/` and `rl/models/*.zip` are gitignored.
