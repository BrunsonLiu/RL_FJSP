# Large Instance Experiments

## Scope

Large Brandimarte instances in this project are currently treated as:

```text
MK11-MK15
```

These instances have 30 jobs and between 5 and 15 machines.

## Commands

Baseline only:

```powershell
python scripts/run_benchmark.py --brandimarte-start 11 --brandimarte-count 5 --random-rollouts 5 --episodes 0
```

Short RL run:

```powershell
python scripts/run_benchmark.py --brandimarte-start 11 --brandimarte-count 5 --random-rollouts 3 --episodes 50 --seeds 0
```

## Current Result

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

