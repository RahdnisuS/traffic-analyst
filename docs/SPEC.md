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

## Status

- [x] Phase 1 – Core sim
- [x] Phase 2 – Metrics and logging
- [x] Phase 3 – Web dashboard
- [x] Phase 4 – Mixed traffic and pedestrians
- [ ] Phase 5 – 2.5D renderer (next)
