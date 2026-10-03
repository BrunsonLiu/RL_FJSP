# Analysis Summary


## 1. Per-stage ablation (mean across 15 instances)

| Stage | mean makespan |
|---|---|
| EF           | 394.1 |
| RL_only      | 307.9 |
| RL+ILS       | 295.8 |
| RL+ILS+SA    | 301.9 |
| final        | 294.7 |
| lit          | 287.6 |

Note: `lit` includes several `null` upper bounds (instances where lit UB is not reported); the mean is taken over the available entries.


## 2. Classical rules — best per instance

- mk01: best classical = 55 (shortest_start); our pipeline = 42; lit = 40; pipeline beats best classical by -13

- mk02: best classical = 46 (mor); our pipeline = 28; lit = 26; pipeline beats best classical by -18

- mk03: best classical = 268 (shortest_start); our pipeline = 204; lit = 204; pipeline beats best classical by -64

- mk04: best classical = 78 (shortest_start); our pipeline = 73; lit = 60; pipeline beats best classical by -5

- mk05: best classical = 220 (earliest_finish); our pipeline = 176; lit = 172; pipeline beats best classical by -44

- mk06: best classical = 79 (earliest_finish); our pipeline = 68; lit = 58; pipeline beats best classical by -11

- mk07: best classical = 204 (earliest_finish); our pipeline = 143; lit = 139; pipeline beats best classical by -61

- mk08: best classical = 601 (shortest_start); our pipeline = 523; lit = 523; pipeline beats best classical by -78

- mk09: best classical = 431 (shortest_start); our pipeline = 332; lit = 307; pipeline beats best classical by -99

- mk10: best classical = 375 (shortest_start); our pipeline = 224; lit = 197; pipeline beats best classical by -151

- mk11: best classical = 706 (earliest_finish); our pipeline = 619; lit = 615; pipeline beats best classical by -87

- mk12: best classical = 599 (shortest_start); our pipeline = 508; lit = 508; pipeline beats best classical by -91

- mk13: best classical = 558 (shortest_start); our pipeline = 416; lit = 430; pipeline beats best classical by -142

- mk14: best classical = 829 (shortest_start); our pipeline = 694; lit = 694; pipeline beats best classical by -135

- mk15: best classical = 485 (shortest_start); our pipeline = 370; lit = 341; pipeline beats best classical by -115


## 3. Wall-clock comparison

| Instance | Ours (s) | OR-Tools cap (s) | Speedup |
|---|---|---|---|
| mk01 |   66.2 | 600 | 9.1x |
| mk02 |  149.8 | 600 | 4.0x |
| mk03 |  451.6 | 600 | 1.3x |
| mk04 |  137.0 | 600 | 4.4x |
| mk05 |  173.7 | 600 | 3.5x |
| mk06 | 1187.2 | 600 | 0.5x |
| mk07 |  708.4 | 600 | 0.8x |
| mk08 |  251.0 | 600 | 2.4x |
| mk09 | 1947.0 | 300 | 0.2x |
| mk10 | 3093.0 | 300 | 0.1x |
| mk11 |  266.9 | 600 | 2.2x |
| mk12 |  254.3 | 600 | 2.4x |
| mk13 |  693.5 | 300 | 0.4x |
| mk14 |  216.0 | 300 | 1.4x |
| mk15 | 3756.1 | 300 | 0.1x |

## 4. Per-difficulty gap

| Group | n | mean gap | min gap | max gap |
|---|---|---|---|---|
| small | 6 | 8.99% | 0.0% | 21.67% |
| medium | 4 | 6.18% | 0.0% | 13.71% |
| large | 5 | 1.18% | -3.26% | 8.5% |

## 5. Failure mode (4 biggest-gap instances)

| Instance | EF | RL | ILS | SA | final | lit | gap | RL% | ILS% | SA% |
|---|---|---|---|---|---|---|---|---|---|---|
| mk04 | 91 | 79 | 73 | 75 | 73 | 60 | +13 | 13.2% | 7.6% | -2.7% |
| mk09 | 433 | 339 | 332 | 339 | 332 | 307 | +25 | 21.7% | 2.1% | -2.1% |
| mk10 | 406 | 242 | 231 | 242 | 224 | 197 | +27 | 40.4% | 4.5% | -4.8% |
| mk15 | 549 | 408 | 380 | 408 | 370 | 341 | +29 | 25.7% | 6.9% | -7.4% |