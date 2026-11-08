# RL agent vs fixed timer and gap-out rule

Held-out seeds 101, 102, 103, 104, 105; 900 s per run; mean over seeds. Values are fixed / gap-out / RL. Negative change = the agent is better.

| Demand (veh/h per approach) | Total waiting (veh·h) | RL vs fixed | RL vs gap-out | Mean delay (s) | RL vs fixed | RL vs gap-out | Trips completed | Ped wait (s) | Greens |
|---|---|---|---|---|---|---|---|---|---|
| low (250) | 2.94 / 2.33 / 2.17 | -26.2% | -7.0% | 54.8 / 46.1 / 43.6 | -20.4% | -5.4% | 238 / 241 / 243 | 59.8 / 34.1 / 36.2 | 24 / 42 / 40 |
| medium (400) | 5.10 / 4.58 / 4.35 | -14.6% | -5.0% | 62.6 / 58.4 / 55.6 | -11.2% | -4.8% | 358 / 358 / 357 | 59.8 / 41.5 / 42.8 | 24 / 37 / 34 |
| high (500) | 8.21 / 7.83 / 7.31 | -10.9% | -6.6% | 79.9 / 78.0 / 73.1 | -8.5% | -6.3% | 432 / 432 / 436 | 61.8 / 49.0 / 56.1 | 24 / 32 / 29 |
