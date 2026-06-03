# Large Instance Experiments

## Scope

Large Brandimarte instances in this project are currently treated as:

```text
MK11-MK15
```

These instances have 30 jobs and between 5 and 15 machines.

The single-instance `mk01`–`mk10` matrix is in
`data/results/baseline_brandimarte_matrix.md` and is referenced from
`docs/algorithm.md` §Verified Results.

## Commands

Baseline only:

```powershell
python scripts/run_benchmark.py --brandimarte-start 11 --brandimarte-count 5 --random-rollouts 5 --episodes 0
```

Short RL run:

```powershell
python scripts/run_benchmark.py --brandimarte-start 11 --brandimarte-count 5 --random-rollouts 3 --episodes 50 --seeds 0
```

## Current Result (MK11–MK15, 1 seed, 50 ep)

Result file:

```text
data/results/benchmark_mk11_mk15_rl50_full.csv
```

Summary:

```text
instance  earliest_finish  random_mean  rl_best
mk11      706              1105.667     660
mk12      700              901.333      924
mk13      622              1043.333     483
mk14      833              1347.000     1608
mk15      549              1103.000     442
```

## Interpretation

The current REINFORCE baseline can improve over the earliest-finish heuristic on some large instances, especially MK13 and MK15.

It is not stable enough for broad large-scale claims:

- MK12 is worse than both earliest-finish and random mean.
- MK14 collapses badly in the short 50-episode run.
- Results use only one seed, so they are diagnostic rather than final.

## Next Steps

To make large-instance results credible:

1. Run multiple seeds, at least `0,1,2`.
2. Increase training episodes for unstable instances.
3. Add entropy regularization or move to PPO/actor-critic.
4. Train across multiple instances instead of training each instance from scratch.
5. Save best schedule per instance and validate it externally.

## Why we are not done yet — lessons from the 10-instance matrix

`data/results/baseline_brandimarte_matrix.md` (and its CSV/JSON siblings) shows
that with the current per-instance 50-episode budget:

- REINFORCE beats every other agent on all 10 Brandimarte instances.
- The graph-based agents (Graph AC, Graph PPO) are *systematically worse than
  the random rollout mean* on the larger instances (mk03, mk05, mk07, mk08,
  mk09, mk10).

This is the central issue blocking any "graph encoder = SOTA" claim. The
remedies we plan to try first, in order of expected cost/benefit, are:

1. **Cross-instance training for the graph encoder.** Train one Graph AC / PPO
   policy across all 10 Brandimarte instances with instance-index as part of
   the input. This gives the encoder enough data to actually use its
   parameters and matches how FJSP-RL papers report results.
2. **Imitation pretraining (BC) on the earliest-finish heuristic.** Warm-start
   the graph encoder with supervised learning on `(state, action)` pairs from
   earliest-finish rollouts, then fine-tune with REINFORCE / PPO. The current
   per-instance training starts from a random policy and the graph encoder
   cannot recover within 50–100 episodes.
3. **Dense per-step reward** so the value head in PPO has a learnable signal on
   every step instead of a single terminal scalar.
4. **Per-instance fine-tuning from the cross-instance checkpoint.** After (1)
   and/or (2), do a short per-instance fine-tune and re-run the matrix. The
   expectation is that the graph agents then flip the ranking in their favor
   on at least the hard instances (mk04, mk08, mk09).

## BC + AC result on MK01 — first graph model that beats the heuristic

`data/results/bc_ac_mk01_results.md` documents a BC warm-start followed by AC
fine-tuning on a single instance:

```text
BC only (100 rollouts, 20 epochs, lr=1e-3)          -> greedy 57
BC + AC (500 ep, lr=3e-5)                           -> greedy 49
earliest-finish heuristic                            -> 57
REINFORCE from scratch (100 ep)                      -> 43
Graph AC from scratch (500 ep default)               -> 54
```

The graph encoder is no longer the worst. The next step is to do this on
all 10 Brandimarte instances and then on MK11–MK15, then add cross-instance
BC pretraining.

