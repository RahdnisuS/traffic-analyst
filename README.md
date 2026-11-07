# Traffic Analyst

A road-junction simulator with reinforcement-learning signal control and an LLM (RAG) analyst.
Work in progress. The full README (architecture diagram, results, learnings) comes in Phase 14.

## Status
- [x] Phase 1: headless core sim (IDM car-following, fixed-time signals, left-hand traffic, deterministic)
- [x] Phase 2: metrics (queues, delay, throughput, LOS) + structured run logs in data/runs/
- [x] Phase 3: FastAPI + WebSocket dashboard (20 Hz stream, 2D canvas, sliders, live charts, mobile layout)
- [x] Phase 4: mixed Indian traffic (cars, bikes, autos, buses; bikes filter between lanes) and social-force
      pedestrians with an exclusive, on-demand pedestrian phase
- [x] Phase 5: 2.5D Three.js view (orthographic tilted camera, low-poly instanced vehicles, capsule
      pedestrians, procedural buildings, glowing signals, soft shadows) with a toggle back to the 2D view
- [x] Phase 6: stress test and capacity benchmark (`python -m bench`) plus a live stress test in the dashboard

## Quick start
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
python -m sim --duration 900  # headless run; logs to data/runs/<run_id>/ (use --no-log to skip)
python -m server              # dashboard at http://127.0.0.1:8000
python -m bench               # capacity benchmark (~70 s); results in experiments/results/benchmark/
pytest -q
```

## Dashboard
Press **Play** to start the shared simulation; sliders change demand, vehicle mix, pedestrians and
signal timing live. The junction opens in the **3D** view (drag to pan, wheel to zoom, Q/E or ⟲/⟳ to
rotate, **Fit** to reset); **2D** switches to the top-down debug view. Add `?view=2d` to the URL to
start in 2D, and press **F** for a frame-rate readout. If WebGL or the CDN is unavailable, the
dashboard stays in 2D.

## Capacity benchmark
How much traffic can the junction handle? `python -m bench` steps demand up (+50 veh/h per approach
per level, each level a fresh 25-minute run) until an approach can no longer keep up, on three seeds.

| Junction | Signals | Capacity (sustained) | Breaks down at | Peak while overloaded |
|---|---|---|---|---|
| 4-way, 2+2 lanes per arm | fixed time, 25 s green | **2,077 ± 115 veh/h** | 550–600 veh/h per approach | ~2,320 veh/h |

"Breaks down" means oversaturation: an approach serves under 95% of the vehicles arriving, so its
queue grows without limit. The junction never literally gridlocks, because drivers don't enter the
box without room to leave it; the overflow queues outside instead. The **Run stress test** button
in the dashboard runs a quicker version on the live simulation to *show* the breakdown; its short
steps only catch clear overload, so the benchmark is the number to quote.
