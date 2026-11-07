# Capacity benchmark

- Junction: `four_way_2x2`, controller: `fixed`, green 25.0 s
- Seeds: 1, 2, 3 · config `afb8d8975330` · commit `4ab6ad0` · 2026-11-07T16:04:58+00:00

**Capacity score: 2,077 ± 115 veh/h** (mean ± std over 3 seeds; range 1,944–2,145)

Peak throughput while overloaded: 2,322 veh/h (mean over seeds). Not sustainable: queues grow at that demand.

Capacity = total junction throughput (veh/h) at the highest demand level where every approach served at least 95% of its arrivals (comparing cycle-averaged queues, with a 10-vehicle noise floor) over a 1200 s window after 300 s of warm-up, with no gridlock.

## Per seed

| Seed | Capacity (veh/h) | Highest passing demand (veh/h per approach) | Breakdown at | Peak served (veh/h) |
|---|---|---|---|---|
| 1 | 2,145 | 550 | 600 | 2,349 |
| 2 | 2,142 | 550 | 600 | 2,358 |
| 3 | 1,944 | 500 | 550 | 2,259 |

## Levels

| Seed | Demand/approach | Offered (veh/h) | Served (veh/h) | Served % | Mean delay (s) | LOS | Result |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 399 | 396 | 99% | 50.9 | D | pass |
| 1 | 150 | 642 | 621 | 97% | 52.9 | D | pass |
| 1 | 200 | 810 | 819 | 101% | 52.0 | D | pass |
| 1 | 250 | 1,029 | 1,053 | 102% | 54.8 | D | pass |
| 1 | 300 | 1,161 | 1,167 | 100% | 53.2 | D | pass |
| 1 | 350 | 1,434 | 1,437 | 100% | 53.7 | D | pass |
| 1 | 400 | 1,545 | 1,548 | 100% | 61.6 | E | pass |
| 1 | 450 | 1,728 | 1,734 | 100% | 65.1 | E | pass |
| 1 | 500 | 1,914 | 1,935 | 101% | 74.3 | E | pass |
| 1 | 550 | 2,184 | 2,145 | 98% | 93.3 | F | pass |
| 1 | 600 | 2,355 | 2,244 | 95% | 120.5 | F | **fail** N, S |
| 1 | 650 | 2,565 | 2,349 | 92% | 144.8 | F | **fail** N, S |
| 2 | 100 | 360 | 351 | 98% | 43.9 | D | pass |
| 2 | 150 | 561 | 555 | 99% | 55.0 | D | pass |
| 2 | 200 | 786 | 768 | 98% | 49.2 | D | pass |
| 2 | 250 | 993 | 987 | 99% | 52.6 | D | pass |
| 2 | 300 | 1,119 | 1,104 | 99% | 51.5 | D | pass |
| 2 | 350 | 1,377 | 1,338 | 97% | 56.2 | E | pass |
| 2 | 400 | 1,623 | 1,581 | 97% | 65.9 | E | pass |
| 2 | 450 | 1,863 | 1,848 | 99% | 69.4 | E | pass |
| 2 | 500 | 2,034 | 1,995 | 98% | 79.3 | E | pass |
| 2 | 550 | 2,184 | 2,142 | 98% | 71.1 | E | pass |
| 2 | 600 | 2,364 | 2,214 | 94% | 88.4 | F | **fail** N |
| 2 | 650 | 2,469 | 2,358 | 96% | 114.6 | F | **fail** N, W |
| 3 | 100 | 366 | 378 | 103% | 49.9 | D | pass |
| 3 | 150 | 591 | 600 | 101% | 51.9 | D | pass |
| 3 | 200 | 765 | 756 | 99% | 48.9 | D | pass |
| 3 | 250 | 930 | 930 | 100% | 51.7 | D | pass |
| 3 | 300 | 1,206 | 1,188 | 98% | 52.5 | D | pass |
| 3 | 350 | 1,377 | 1,380 | 100% | 55.8 | E | pass |
| 3 | 400 | 1,617 | 1,620 | 100% | 60.6 | E | pass |
| 3 | 450 | 1,734 | 1,770 | 102% | 65.6 | E | pass |
| 3 | 500 | 1,929 | 1,944 | 101% | 65.3 | E | pass |
| 3 | 550 | 2,184 | 2,130 | 98% | 92.1 | F | **fail** N |
| 3 | 600 | 2,370 | 2,259 | 95% | 108.5 | F | **fail** N |
