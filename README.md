# Traffic Analyst

A road-junction simulator with reinforcement-learning signal control and an LLM (RAG) analyst.
Work in progress. The full README (architecture diagram, results, learnings) comes in Phase 14.

## Status
- [x] Phase 1: headless core sim (IDM car-following, fixed-time signals, left-hand traffic, deterministic)
- [x] Phase 2: metrics (queues, delay, throughput, LOS) + structured run logs in data/runs/

## Quick start
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
python -m sim --duration 900  # headless run; logs to data/runs/<run_id>/ (use --no-log to skip)
pytest -q
```
