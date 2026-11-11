---
title: Traffic Analyst
emoji: 🚦
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Junction simulator with RL signals and a RAG analyst
---

# Traffic Analyst

A road-junction simulator with mixed Indian traffic, a reinforcement-learning signal controller,
a LightGBM queue forecaster and an LLM analyst that answers questions about the run and traffic
engineering manuals, with citations.

Each browser tab gets its own simulation (up to 4 at once). The analyst runs on Groq's free tier,
so it allows 10 questions per hour per visitor.

Source code, results and how it works: https://github.com/RahdnisuS/traffic-analyst
