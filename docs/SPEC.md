# Traffic Analyst — Project Spec

> Original brief, saved verbatim (2026-11-07). Decisions made during the build are logged at the bottom; where they differ from the brief, the decision log wins.

## Brief

I want to build a portfolio-grade AI product: "Traffic Analyst", a web app that simulates a road junction, controls signals with reinforcement learning, and includes an LLM-powered analyst (RAG) that explains what happened, answers questions with citations, and sets up scenarios from plain English. I'm a CSE (Gaming Technology) undergrad applying to non-gaming tech companies, so I must understand and explain every design decision in interviews. Keep it lightweight: Python backend + browser frontend. No Unity, no game engines, no external 3D assets.

## Product overview
1. Simulation: a junction with mixed Indian traffic (cars, bikes, autos, buses) and pedestrians. Sliders control traffic volume, vehicle mix, pedestrian rate, driver aggressiveness and signal timing. Live metrics and charts.
2. AI signal control: a PPO agent controls the signals. Toggle AI on/off to compare against fixed-timer signals, with % improvement shown live.
3. Benchmark: a stress test ramps traffic until gridlock and reports a junction capacity score.
4. Analyst (RAG): a chat panel that answers questions about simulation runs and traffic engineering documents, with citations, and converts plain-English scenario requests into simulation configs.

## Tech stack
- Backend: Python 3.11+, FastAPI, WebSockets, NumPy, Pydantic
- Simulation (custom, from scratch, no SUMO): Intelligent Driver Model (IDM) for car-following, lane-specific behaviour per vehicle type (bikes filter through gaps), simplified social force model for pedestrians
- RL: Gymnasium environment + Stable-Baselines3 (PPO)
- Congestion predictor: LightGBM forecasting queue length 2 minutes ahead
- RAG: plain Python pipeline (no LangChain/LlamaIndex), PyMuPDF, sentence-transformers (all-MiniLM-L6-v2), ChromaDB, rank_bm25 for hybrid search, cross-encoder re-ranker, Groq API (Llama 3) with the key in .env
- RAG evaluation: RAGAS
- Frontend: plain HTML/CSS/JavaScript, HTML Canvas (2D debug view), Three.js with an OrthographicCamera (2.5D view), Chart.js for live charts
- Config: config.yaml for all simulation, RL and RAG parameters

## Project structure (suggested)
traffic-analyst/
  sim/ (vehicle.py, idm.py, pedestrian.py, junction.py, signals.py, spawner.py, engine.py, metrics.py, logger.py)
  rl/ (env.py, train.py, evaluate.py, models/)
  predict/ (features.py, train_predictor.py)
  rag/ (ingest.py, chunking.py, embeddings.py, vectorstore.py, retrieval.py, rerank.py, generate.py, run_indexer.py, scenario_parser.py)
  server/ (main.py, routes for sim WebSocket, chat, runs)
  web/ (index.html, app.js, render2d.js, render3d.js, charts.js, chat.js, style.css)
  junctions/ (JSON layouts)
  docs/ (traffic engineering PDFs I provide)
  data/runs/ (logged simulation runs)
  eval/ (rag_test_questions.json, run_rag_eval.py, results/)
  experiments/ (results/, plots/)
  tests/
  config.yaml, requirements.txt, .env.example, README.md

## Build in phases. STOP after each phase, summarize what you built, explain the key concepts and decisions in simple terms, and wait for my go-ahead.

PART A — SIMULATION
Phase 1 – Core sim (headless): 4-way junction from JSON (lanes, stop lines, signals), vehicle spawning, IDM car-following, stopping at red lights, turning, fixed-timer signals. Deterministic with a random seed. Explain IDM.
Phase 2 – Metrics and logging: wait time, queue length per lane, throughput, travel time, congestion level. Log every run to data/runs/ (config, per-second metrics, signal phase changes, notable events such as queue spikes and gridlock) as structured JSON/CSV. These logs will later feed the analyst.
Phase 3 – Web dashboard: FastAPI + WebSocket streaming at ~20 updates/sec, 2D canvas renderer, sliders, play/pause/reset, live Chart.js charts. Mobile-friendly.
Phase 4 – Mixed traffic and pedestrians: vehicle types with different behaviours, and pedestrians using the social force model. Explain the model.
Phase 5 – 2.5D renderer: Three.js orthographic view (slight isometric tilt), procedural low-poly meshes (vehicle boxes with a cabin on top, capsule pedestrians, corner buildings, glowing signal lights, soft shadows), InstancedMesh for vehicles, interpolation between server updates for smooth 60 FPS, and a toggle back to the 2D debug view. The renderer only consumes server state; no sim logic in the frontend.
Phase 6 – Stress test and benchmark: ramp demand until gridlock (define gridlock precisely), then report a capacity score (max vehicles/hour sustained). Save results.

PART B — AI CONTROL AND PREDICTION
Phase 7 – RL signal controller: Gymnasium env (observation = queue lengths, waiting times, current phase, time in phase; action = keep or switch; reward = negative total waiting time, with a penalty for frequent switching and a minimum green time). Train PPO, log training curves, save the model. Explain the reward design. Log the agent's observations and actions per decision so they can be explained later.
Phase 8 – AI in the UI: AI on/off toggle, side-by-side comparison on the same seeds, % improvement shown live.
Phase 9 – Congestion predictor: LightGBM model forecasting queue length 2 minutes ahead, trained on logged runs, shown as a dashed forecast line on the chart. Report MAE against a naive baseline.
Phase 10 – Experiments: fixed-timer vs rule-based actuated vs RL across 30+ random scenarios at low/medium/high demand. Results table (markdown + CSV) and plots with mean and standard deviation.

PART C — RAG TRAFFIC ANALYST
Phase 11 – Knowledge base: ingest the traffic engineering PDFs in docs/ (paper/guideline name + page metadata) and the simulation run logs (summarised into readable text chunks per time window, with run ID and timestamps). Store both in ChromaDB with source-type metadata. Use hybrid retrieval (BM25 + vectors with reciprocal rank fusion) and cross-encoder re-ranking.
Phase 12 – Analyst chat panel: a chat UI in the dashboard. The analyst must:
  (a) Explain runs: "Why did wait time spike at minute 4?", answered from run logs with timestamps cited.
  (b) Answer guideline questions: "Is a 120s cycle too long here?", answered from docs with paper/page citations.
  (c) Explain RL decisions: "Why did the AI switch early at 3:20?", answered from the logged observations and actions at that time.
  (d) Set up scenarios: "Evening rush, 60% bikes, school crossing", converted into a validated config (Pydantic schema) that the user confirms before it is applied to the sliders.
  (e) Say "I don't know" when the context is insufficient. Never invent numbers; every number must come from retrieved logs or docs.
Phase 13 – RAG evaluation: create 40+ test questions across the four question types, with ground-truth answers. Measure faithfulness, answer relevancy, context precision and context recall with RAGAS. Compare vector-only vs hybrid vs hybrid + re-ranker. Save the results table.

PART D — POLISH
Phase 14 – Unit tests (IDM, signal logic, scenario parser, retrieval), error handling (API failures, bad PDFs, invalid scenario configs), a README with a Mermaid architecture diagram, setup steps, all experiment results, RL training curves, RAGAS scores, limitations and key learnings. Dockerfile and deployment to Render or Hugging Face Spaces.

## Working rules
- Ask me before making major decisions not specified here.
- Readable code with docstrings and comments explaining WHY, not just what.
- Simulation fully separated from rendering, so RL training runs headless and fast.
- Performance: 60 FPS in the 3D view with 200+ agents on a normal laptop.
- Never hardcode API keys.
- Commit to git at the end of each phase with a clear message.
- At the end, give me likely interview questions about this project, with short answers based on what we actually built.

---

## Decision log

| Date | Phase | Decision |
|---|---|---|
| 2026-11-07 | 1 | Left-hand (Indian) traffic. |
| 2026-11-07 | 1 | 4-phase split signalling: one approach green at a time. |
| 2026-11-07 | 1 | Lane counts come from the junction JSON (default 2 in + 2 out per arm). |
| 2026-11-07 | 1 | Repo root is the project root (no `traffic-analyst/` subfolder). Python 3.13 venv at `.venv`. |
| 2026-11-07 | 4 | Pedestrian crossings use an **exclusive, on-demand pedestrian phase**: a waiting pedestrian presses the button; at the next phase change the controller inserts WALK (7 s) + flashing DON'T WALK (crossing length / 1.07 m/s ≈ 13 s) + all-red, with every vehicle movement red. The current green is never cut short. |
| 2026-11-07 | 4 | The pedestrian phase runs **at most once per cycle** (`pedestrians.min_greens_between: 4`). Serving it after every phase gave vehicles only 51% green time. Trade-off: walkers wait ~60 s on average. |
| 2026-11-07 | 4 | Default vehicle mix: 35% car, 45% bike, 15% auto, 5% bus. Buses prefer the kerb lane when it has room. |
| 2026-11-07 | 4 | Bike filtering uses **virtual filter lanes** along the lane dividers (plus matching exit-side gaps), not a full sub-lane model: it reuses the 1-D lane machinery. Bikes filter only when queued behind slow traffic, only for movements whose path out of the gap crosses no neighbouring lane (straight-only on the default layout), and ride at 5.5 m/s only while alongside a queue. |
| 2026-11-07 | 4 | Pedestrians use the social force model, **elliptical specification** (Johansson, Helbing & Shukla 2007), with 0.5 s anticipation; the circular version let head-on walkers overlap. Vehicles are not force sources: walker/vehicle separation is by rules (start only on WALK with the crosswalk clear; drivers yield to anyone on it). |
| 2026-11-07 | 4 | Diagonal crossing during the exclusive phase is not modelled (adjacent crosswalks only), which keeps the clearance interval at ~13 s instead of ~22 s. |
| 2026-11-07 | 5 | Three.js r160 from a pinned CDN URL via an **import map**, loaded with a dynamic `import()`: still no build step, and if WebGL or the CDN fails the dashboard falls back to the 2D view. |
| 2026-11-07 | 5 | 3D is the default view; the 2D canvas stays as the debug view (toggle, remembered per browser; `?view=2d` overrides). Only the visible view renders. |
| 2026-11-07 | 5 | Vehicles: one `InstancedMesh` per part per type (chassis, body, cabin/canopy/window band, brake-light bar), with geometry pre-offset so the origin is the front bumper: one matrix per vehicle drives all its parts. Body colour is a stable hash of the vehicle id; type is shown by shape and livery (green autos, red/blue/orange buses). |
| 2026-11-07 | 5 | Speed state, which the 2D view shows as colour, is shown in 3D as **brake lights** (on below the 2 m/s queue speed). |
| 2026-11-07 | 5 | Camera: orthographic, 55° elevation, from the south-east, with 90° rotation steps (Q/E). Buildings are procedural from a fixed seed and kept low near the junction so they don't hide the stop lines. Sun ~42° high from the west-south-west; the 2048² shadow map covers the inner ±60% of the map only, for sharp shadows where people look. Dark theme = dusk lighting. |
| 2026-11-07 | 6 | **Breakdown = oversaturation, not literal gridlock.** The box-junction rule means the box never locks up: past capacity, extra demand queues outside the network without limit (probe at 3,000 veh/h per approach: 0 gridlock events, 2,916 vehicles waiting to enter). A level fails if any approach serves < 95% of its arrivals (`unserved_frac`, with a 10-vehicle noise floor), or if the existing gridlock detector fires. Matches the HCM notion of capacity (demand > capacity). |
| 2026-11-07 | 6 | Judged **per approach** via conservation of vehicles: arrivals - departures = growth of the approach's stock (on the road + waiting to enter). Per approach because the busier N/S approaches break down first and a junction total hides it. |
| 2026-11-07 | 6 | Stock growth is measured between **cycle-averaged** stocks (one signal cycle at each end of the window). Queues are a sawtooth over the cycle; comparing single instants gave scores of 1,044 to 2,304 veh/h for the same junction depending on the seed. |
| 2026-11-07 | 6 | **Independent runs per demand level** (fresh sim, 300 s warm-up, 1,200 s measured), +50 veh/h per approach per level, stop after 2 consecutive failures, seeds 1/2/3 in parallel processes. 600 s windows still flipped pass/fail near capacity; 1,200 s made all seeds agree within one step. Capacity score = served throughput at the highest passing level, mean ± std over seeds; peak throughput under overload is reported separately (not sustainable). |
| 2026-11-07 | 6 | `python -m bench` produces the official score (saved to `experiments/results/benchmark/<timestamp>_s1-2-3/`: summary.json, levels.csv, report.md, plot.png; records config hash and commit, `-dirty` if the code was uncommitted). The dashboard's **live stress test** ramps the running sim (+100 veh/h per approach every 360 s, first 60 s not judged), locks the settings while it runs, logs `stress_*` events, restores the user's volume at the end. Measured: 5-minute live windows can't resolve the 5% threshold (the last passing level varied 300–700 veh/h per approach across seeds; even 22-minute steps varied by a step), so the dashboard shows "kept up to X, clearly overloaded at Y" beside the official score instead of a second capacity number. |
| 2026-11-07 | 6 | **Collision fix (found by the user in the 3D view: turning buses/cars overlapping other vehicles).** The old safety test only checked spacing along each lane; a new test checks every pair of vehicle rectangles (separating-axis test) on the streamed poses. Measured at high demand with 20% buses: 57 overlap samples in 10 min, now 0 across 5 scenarios. Four causes, four fixes: (1) bodies were drawn along the path tangent at the nose, so a 12 m bus's tail swung metres off the road on a turn; now drawn along the **chord** rear-point -> front-point (`Simulation.lane_poses`). (2) A connector was "free" once a vehicle's nose left it; now it is in use while **any part of the body** is on it (`connector_users`). (3) A driver only followed the vehicle ahead on its own route, so a car going straight drove into the tail of a bus turning right from the same lane; drivers now keep following a **turned-off leader** until its body clears (`_turned_off_leaders`). (4) Conflicts are also computed **per vehicle-type pair from swept bodies** (`Junction.compute_body_conflicts`, `body_clearance_m: 0.3`), on top of the centreline rule; on this layout that adds bike-vs-bus conflicts only. A bus leaves ~5 cm to a bike in the filter gap, so bikes don't overtake a too-wide vehicle through the gap, and a turning wide vehicle waits for a bike beside or ahead of it (its tail swings out). Also fixed: a bike could swerve into the filter lane 1 m behind a stopped bike at 4 m/s (gap check now includes braking distance). |
| 2026-11-07 | 6 | Capacity after the collision fix: **1,790 ± 54 veh/h** (was 2,077 ± 115 with vehicles passing through each other). Breakdown at 500 veh/h per approach on all three seeds. |
| 2026-11-07 | 7 | **Reward = -(vehicle-seconds of waiting in the decision interval) / 100 - 0.2 per switch.** Waiting = on the road below 0.1 m/s (same as `wait_time`) plus every vehicle in the off-map backlog, so starving an approach until it spills off the map isn't free. Summed over an episode it equals total waiting time. The switch penalty is small: a switch already costs amber + all-red as waiting; it only breaks ties against flicker. Pedestrians are not in the reward; the evaluation reports their wait so any cost to them is visible. |
| 2026-11-07 | 7 | **Decisions on a fixed 5 s grid** (`DecisionClock`): every decision covers exactly 5 s. The agent can only request a switch to the next phase; min green, amber, all-red and the pedestrian phase stay with `SignalController`, so a bad policy is slow, never unsafe. Where a switch is impossible the action is masked (below). With a 7 s min green the first possible switch is at 10 s. |
| 2026-11-07 | 7 | **Phase-relative observation (15):** for the green phase, then the next, next2, next3: queued vehicles, backlog, longest wait (incl. backlog); plus time in green, pedestrian call pending, can_switch. One module (`rl/features.py`) defines observation, timing and waiting for training and the live controller; a test checks the same network makes identical decisions in both. |
| 2026-11-07 | 7 | **Randomised training traffic** per episode: mean demand U(150, 600) veh/h per approach, each approach x U(0.7, 1.3), pedestrians U(0, 500)/h; 15-min episodes from an empty junction. Evaluation uses held-out seeds 101-105 at 250/400/500 veh/h with the config's N/S-heavier ratios. **MaskablePPO** (sb3-contrib; SB3 defaults, 2x64 MLP, ent_coef 0.01) with **reward normalisation during training** (VecNormalize), 300k decisions on 14 parallel simulations (~280 decisions/s, ~20 min). CPU-only torch. |
| 2026-11-07 | 7 | **How the training setup was debugged** (each step measured, not guessed): (1) first run got WORSE as it trained (−324 -> −379 vs fixed timer −179), value loss ~2,250: raw returns in the hundreds -> reward normalisation. (2) Sanity check that the reward ranks simple strategies correctly: never switch −1,186, always switch −310, fixed timer −200, "longest queue" −265 (switching only goes to the NEXT phase), **gap-out rule** (switch once the green's queue clears) −164: the problem is learnable, and gap-out became a second baseline on the training curve. (3) Agent still learned to switch less and less: decisions that covered different amounts of time (a switch spans amber + all-red + next min green) biased per-decision discounting against switching -> fixed 5 s steps (also 3.5x faster: parallel envs no longer wait on long steps). (4) Critic then learned (explained variance 0.95) but the policy stayed ~50/50 (entropy ~ln 2): over half the decisions had no effect and only added noise -> action masking. With all four, training returns rose steadily: −351 -> −233 (68k) -> −172 (111k), past the fixed timer while still exploring. |
| 2026-11-07 | 7 | Every RL decision is logged to `decisions.jsonl` (run-log schema v3): observation in real units and as the scaled vector, action, whether the controller accepted it, P(switch), and the critic's value estimate. Inference is deterministic, so runs stay reproducible. |
| 2026-11-08 | 8 | **AI on/off = live hand-over** (user's choice over "restart the run"): the switch swaps the running sim's policy in place; safe because policies only REQUEST switches. Logged as a `controller_change` event, so a run log can contain both controllers; the manifest's `controller` is the one it started with. The dashboard always starts on the fixed timer (the model loads in a worker thread on first use, a few seconds for torch). The fixed-time green is kept in the config while the AI drives, so the slider still works and the plan resumes on switching back. |
| 2026-11-08 | 8 | **Comparison = twin simulation, split view** (user's choice over "one view + numbers"): both runs restart from an empty junction at t = 0 with the current seed and settings (neither inherits the other's queues), main = fixed timer (left), twin = AI (right), stepped in lockstep by the same tick; settings go to both at the same instant. Fairness rests on every random draw being independent of the signals (tested: the two generators end bit-identical). Each side is logged as its own run with `tags.compare_id` / `compare_side`. Stopping asks which side to keep ("Keep fixed timer" / "Keep AI"). Comparison and stress test exclude each other; the AI switch is locked while comparing. |
| 2026-11-08 | 8 | **How the comparison is shown** (dataviz + animate skills, then an impeccable critique): the result is a sentence, not a bare signed number ("5% less waiting" with a status dot as second channel), measured as `rl.evaluate` does (stopped + backlog vehicle-seconds), hidden for the first 60 s (empty road = noise), within 1% = "about the same". Pairs always read fixed timer -> AI, like the panes. In compare mode the charts switch to the emphasis form (2 lines: AI accent, fixed timer gray, legend + end labels) instead of 8 per-approach lines. Motion: only the switch thumb (160 ms) and the twin pane/panel entrance (220 ms fade + 8 px lift) move; the live numbers do not animate. Contrast fixes from the critique: light-mode `--muted` #6b6a64 (5.3:1) and a darker `--accent-fill` #1f63b8 behind white text (5.9:1). |
| 2026-11-08 | 7 | **Gap-out is the bar, and the curve shows the deployed agent.** The first full run's training curve sat on the fixed timer and below gap-out, but SB3's episode return is the agent *exploring* (sampling actions). Scored with its most likely action (as deployed) on the same 24 episodes it was ahead of both (−147 vs gap-out −156, fixed −179; wins 20/24). So: (1) `rl.train` now evaluates the network greedily every `rl.curve_eval_every` (25k) decisions on the baseline episodes and plots that line too (`greedy_eval.csv`); (2) `rl.evaluate` compares against gap-out as well as the fixed timer, via `GapOutPolicy` (same rule as the env baseline; parity tested). Known trade-off: the reward ignores pedestrians, so they wait longer than under gap-out (still less than under the fixed timer). |

## Status

- [x] Phase 1 – Core sim
- [x] Phase 2 – Metrics and logging
- [x] Phase 3 – Web dashboard
- [x] Phase 4 – Mixed traffic and pedestrians (commit f1cf44f; 69 tests). Its dashboard visuals
      (sidewalks, walkers, walk signals, mix sliders) were later checked in browser screenshots in Phase 5.
- [x] Phase 5 – 2.5D renderer (commit 62c61a0; 70 tests). Verified in headless Edge (SwiftShader) at ~130–150 vehicles: desktop,
      390 px phone, light and dark themes, 2D toggle. Open item: 60 FPS on real laptop hardware is
      not yet measured (press F in the dashboard for the frame-rate readout).
- [x] Phase 6 – Stress test and benchmark (90 tests). Capacity of the default junction (fixed time, 25 s
      green): **1,790 ± 54 veh/h** over seeds 1–3 (after the collision fix); breakdown at 500 veh/h per
      approach; peak under overload ~2,070 veh/h. Live stress test checked in the browser.
- [x] Phase 7 – RL signal controller (code bc5fb1a; trained 2026-11-08, 300k decisions, 18 min; 103 tests). Greedy
      return −148 vs gap-out −156 and fixed timer −179 (passes gap-out by 50k decisions). Held-out seeds 101–105:
      total waiting −26% / −15% / −11% vs the fixed timer and −7% / −5% / −7% vs gap-out at 250 / 400 / 500 veh/h;
      pedestrians wait longer than under gap-out. Results: `experiments/results/rl/20261108-185428/`.
- [x] Phase 8 – AI in the UI (111 tests). AI on/off switch (live hand-over), side-by-side comparison
      (twin sim, same seed, lockstep, shared camera) with the result in words after a 1-minute warm-up.
      Checked in headless Edge: 1440 px dark/light, 390 px; design critique (impeccable) 25/40 -> its 6
      priority issues fixed (keep-side stop, fixed -> AI order everywhere, contrast, one stop control,
      worded headline, focus/announcements).
