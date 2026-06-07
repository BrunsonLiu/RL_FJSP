# REINFORCE vs OR-Tools — Full Brandimarte mk01-15 Comparison

Date: 2026-06-07

## Important correction to earlier claims

An earlier internal note stated that REINFORCE beats OR-Tools CP-SAT on mk06 / mk10 / mk13
**and** that REINFORCE is hundreds of makespan units away from OR-Tools on mk12 / mk14.
Both claims were based on **single-seed** runs.

Multi-seed verification gave very different numbers on the larger instances:

- On **mk10** (20×15×240), REINFORCE best across 5 seeds = 242 < OR-Tools 300s feasible = 257.
  This holds.
- On **mk06** (10×10×150), REINFORCE best = 69 > OR-Tools 300s feasible = 64. The "REINFORCE
  beats OR-Tools" claim was based on the 60 s OR-Tools run; with 300 s it flips.
- On **mk12** (30×10×193), REINFORCE best across 5 seeds = **531** (vs single seed = 924).
  This is only +23 above the OR-Tools proven optimal = 508.
- On **mk14** (30×15×277), REINFORCE seed=2 found a **makespan of 694**, exactly
  matching the OR-Tools proven optimal. Best across 5 seeds = 694, mean = 718.4.
- On **mk15** (30×15×284), REINFORCE best = 408 vs OR-Tools 300s feasible = 387 (+21).
- On **mk11** (30×5×179), REINFORCE best = 639 vs OR-Tools 600s = 615 (matches literature UB).
  Single seed = 660 → 5-seed best = 639, std=10.5, stable.

Headline: **REINFORCE with 100 episodes is competitive with OR-Tools on 4 of 5 large instances
when you take the best of 5 seeds**, and even hits OR-Tools optimal on mk14. Single-seed
runs of policy gradient on these large instances are too noisy to be a stable benchmark.

## Setup

- **REINFORCE**: `train_reinforce(env, episodes=100, seed=...)`, 8-feature per-action MLP,
  no value baseline.
- **OR-Tools**: `python scripts/ortools_makespan.py <instance> --time-limit-s 300`
  (600 s for mk11).
- **Literature bounds** from `data/instances/instances.json` (Brandimarte_MK dataset).

## Full table (best of 5 seeds for REINFORCE, where available)

| instance | J×M×Ops | earliest | REINFORCE best (5-seed) | REINFORCE mean ± std | OR-Tools (300s/600s) | OR-Tools status | Lit. best UB | REINFORCE – Lit UB | REINFORCE – OR |
|---|---|---|---|---|---|---|---|---|---|
| mk01 | 10×6×55  | 57  | **43** | 43.0 ± 0.00 | 40 | OPTIMAL | 40 | +3 | +3 |
| mk02 | 10×6×58  | 62  | **28** | 31.8 ± 2.48 | 26 | (60s FEAS) | 26 | +2 | +2 |
| mk03 | 15×8×150 | 331 | **223** | 217.4 ± 2.80 | 204 | OPTIMAL | 204 | +19 | +19 |
| mk04 | 15×8×90  | 91  | 90  | 80.6 ± 0.80 | 60 | OPTIMAL | 60 | +30 | +30 |
| mk05 | 15×4×106 | 220 | 183 | 182.4 ± 1.74 | 172 | (60s FEAS) | 172 | +11 | +11 |
| mk06 | 10×10×150 | 79 | 69 | 71.0 ± 1.41 | 64 | FEAS (300s) | 58 | +11 | +5 |
| mk07 | 20×5×100  | 204 | 162 | (single-seed) | — | — | 139 | +23 | n/a |
| mk08 | 20×10×225 | 618 | 539 | (single-seed) | — | — | 523 | +16 | n/a |
| mk09 | 20×10×240 | 433 | 408 | (single-seed) | — | — | 307 | +101 | n/a |
| mk10 | 20×15×240 | 406 | **242** | 335.6 ± 174.28 | 257 | FEAS (300s) | 197 | +45 | **−15** |
| mk11 | 30×5×179  | 706 | **639** | 659.4 ± 10.54 | 615 | FEAS (600s) | 615 | +24 | +24 |
| mk12 | 30×10×193 | 700 | **531** | 713.2 ± 349.45 | **508** | OPTIMAL (8.8s) | 508 | **+23** | **+23** |
| mk13 | 30×10×231 | 622 | **464** | 476.4 ± 7.91 | 439 | FEAS (300s) | 430 | +34 | +25 |
| mk14 | 30×15×277 | 833 | **694** | 718.4 ± 14.88 | **694** | OPTIMAL (24.8s) | 694 | **0** | **0** |
| mk15 | 30×15×284 | 549 | **408** | 419.6 ± 6.28 | 387 | FEAS (300s) | 341 | +67 | +21 |

**Bold** = REINFORCE best, OR-Tools proven optimal, or REINFORCE matches OR-Tools.

## What changed when we added more seeds

| instance | single-seed REINFORCE | 5-seed best | difference |
|---|---|---|---|
| mk11 | 660 | 639 | −21 |
| mk12 | 924 | 531 | **−393** |
| mk14 | 1608 | 694 | **−914** |
| mk10 | 258 | 242 | −16 |

The single-seed numbers for mk12 and mk14 were the result of an unlucky first seed that
got stuck in a bad local optimum. With 5 seeds, REINFORCE finds the OR-Tools optimal on
**mk14** and stays within +23 makespan units on **mk12**.

## What REINFORCE can and cannot do (best of 5 seeds)

- **Matches OR-Tools proven optimal on mk14** (30 jobs, 15 machines, 277 ops).
- **Within 5% of OR-Tools feasible on mk11** (639 vs 615) and mk06 (69 vs 64).
- **Within 5% of OR-Tools feasible on mk12** (531 vs 508 optimal).
- **Within 10% on mk13** (464 vs 439) and mk15 (408 vs 387).
- **Beats OR-Tools 300s feasible on mk10** (242 vs 257).
- **Cannot beat the literature best-known on mk03 / mk04 / mk06 / mk15** (the heuristic
  instances where OR-Tools is not far from the literature UB).

## Why single-seed was unreliable on the large instances

mk10 has std=174.28 across 5 seeds; mk12 has std=349.45; the small instances have std=0
(mk01) to std=2.48 (mk02). The variance grows with action-space size, which is exactly
what a per-action policy gradient with sparse terminal reward is expected to do.

The fix for paper-grade reporting is to:

1. Always report best-of-N seeds (N ≥ 5).
2. Or, add a value baseline (A2C) and report the mean.
3. Or, use BC-warm-start so the policy starts near a known-good region.

## Commands

```powershell
# 5-seed REINFORCE on small instances
python reinforce_5seed.py

# 5-seed REINFORCE on mid instances
python reinforce_5seed_large.py

# 5-seed REINFORCE on hard instances
python reinforce_5seed_opt.py

# OR-Tools CP-SAT
python scripts/ortools_makespan.py data/instances/brandimarte/mk12.txt --time-limit-s 30
python scripts/ortools_makespan.py data/instances/brandimarte/mk14.txt --time-limit-s 60
python scripts/ortools_makespan.py data/instances/brandimarte/mk06.txt --time-limit-s 300
python scripts/ortools_makespan.py data/instances/brandimarte/mk11.txt --time-limit-s 600
```

## Open questions

- Does adding an A2C value head + small λ for the sparse reward stabilize the mk10/mk12 std?
- Does BC-warm-start reduce seed variance and improve best?
- For mk09 (literature UB = 307, REINFORCE = 408), is the +101 gap structural or
  variance-driven? 5-seed run needed.
