# Capacity benchmark

- Junction: `four_way_2x2`, controller: `fixed`, green 25.0 s
- Seeds: 1, 2, 3 · config `f5ffa5e7f90f` · commit `79e966d` · 2026-11-07T16:37:30+00:00

**Capacity score: 1,790 ± 54 veh/h** (mean ± std over 3 seeds; range 1,749–1,851)

Peak throughput while overloaded: 2,067 veh/h (mean over seeds). Not sustainable: queues grow at that demand.

Capacity = total junction throughput (veh/h) at the highest demand level where every approach served at least 95% of its arrivals (comparing cycle-averaged queues, with a 10-vehicle noise floor) over a 1200 s window after 300 s of warm-up, with no gridlock.

## Per seed

| Seed | Capacity (veh/h) | Highest passing demand (veh/h per approach) | Breakdown at | Peak served (veh/h) |
|---|---|---|---|---|
| 1 | 1,749 | 450 | 500 | 1,983 |
| 2 | 1,851 | 450 | 500 | 2,133 |
| 3 | 1,770 | 450 | 500 | 2,085 |

## Levels

| Seed | Demand/approach | Offered (veh/h) | Served (veh/h) | Served % | Mean delay (s) | LOS | Result |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 399 | 402 | 101% | 52.2 | D | pass |
| 1 | 150 | 642 | 627 | 98% | 53.7 | D | pass |
| 1 | 200 | 810 | 822 | 101% | 53.5 | D | pass |
| 1 | 250 | 1,029 | 1,056 | 103% | 56.3 | E | pass |
| 1 | 300 | 1,161 | 1,170 | 101% | 54.5 | D | pass |
| 1 | 350 | 1,434 | 1,434 | 100% | 56.3 | E | pass |
| 1 | 400 | 1,545 | 1,557 | 101% | 65.3 | E | pass |
| 1 | 450 | 1,728 | 1,749 | 101% | 81.5 | F | pass |
| 1 | 500 | 1,914 | 1,899 | 99% | 96.5 | F | **fail** S |
| 1 | 550 | 2,184 | 1,983 | 91% | 140.3 | F | **fail** N, S |
| 2 | 100 | 360 | 351 | 98% | 44.1 | D | pass |
| 2 | 150 | 561 | 555 | 99% | 55.3 | E | pass |
| 2 | 200 | 786 | 768 | 98% | 49.7 | D | pass |
| 2 | 250 | 993 | 990 | 100% | 53.5 | D | pass |
| 2 | 300 | 1,119 | 1,104 | 99% | 52.3 | D | pass |
| 2 | 350 | 1,377 | 1,320 | 96% | 59.3 | E | pass |
| 2 | 400 | 1,623 | 1,578 | 97% | 76.2 | E | **fail** E |
| 2 | 450 | 1,863 | 1,851 | 99% | 85.6 | F | pass |
| 2 | 500 | 2,034 | 1,929 | 95% | 96.1 | F | **fail** E |
| 2 | 550 | 2,184 | 2,133 | 98% | 88.4 | F | **fail** W |
| 3 | 100 | 366 | 378 | 103% | 50.1 | D | pass |
| 3 | 150 | 591 | 603 | 102% | 52.0 | D | pass |
| 3 | 200 | 765 | 756 | 99% | 49.2 | D | pass |
| 3 | 250 | 930 | 936 | 101% | 52.9 | D | pass |
| 3 | 300 | 1,206 | 1,179 | 98% | 53.2 | D | pass |
| 3 | 350 | 1,377 | 1,380 | 100% | 57.7 | E | pass |
| 3 | 400 | 1,617 | 1,611 | 100% | 65.7 | E | pass |
| 3 | 450 | 1,734 | 1,770 | 102% | 80.2 | F | pass |
| 3 | 500 | 1,929 | 1,926 | 100% | 83.2 | F | **fail** N |
| 3 | 550 | 2,184 | 2,085 | 96% | 118.8 | F | **fail** N |
