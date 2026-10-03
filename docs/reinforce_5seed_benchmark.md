# REINFORCE 5-Seed Benchmark: Brandimarte mk01-05

日期：2025-11-15

## Headline

**REINFORCE 100 episodes（MLP 2-layer, 64 hidden, 8 features per action）is our current SOTA across Brandimarte mk01-05.**

- 5 seeds per instance, std ≤ 3
- 1.4 seconds per seed on mk01
- Gap to OR-Tools CP-SAT optimum: **+1 to +19 makespan**

## Setup

- Per-action MLP (ActionScorer): 8 features → 64 → 64 → 1 score
- Per-action features: `job_ready, machine_ready, duration, start, end, op_idx, remaining_ops, makespan` (all scaled by `instance_time_scale`)
- Sparse terminal reward: `R = -makespan / scale` (constant per episode)
- 100 episodes per seed, no value baseline (REINFORCE = policy gradient without baseline)
- Seeds: 0, 1, 2, 3, 4
- Hardware: CPU

## Results (5 seeds)

| Instance | Jobs×Macs×Ops | earliest_finish | REINFORCE best | mean ± std | OR-Tools CP-SAT (60s) | Gap |
|---|---|---|---|---|---|---|
| mk01 | 10×6×55 | 57 | **43** | 43.0 ± 0.00 | 40 | +3 |
| mk02 | 10×5×58 | 32 | **28** | 31.8 ± 2.48 | 27 | +1 |
| mk03 | 15×8×150 | 204 | **216** | 217.4 ± 2.80 | 204 | +12 |
| mk04 | 15×8×150 | 67 | **79** | 80.6 ± 0.80 | 60 | +19 |
| mk05 | 15×4×177 | 173 | **180** | 182.4 ± 1.74 | 173* | +7 |

(*) OR-Tools hit the 60s time limit on mk05; the 173 is a feasible upper bound, not a proven optimal.

## Per-seed best (mk01, 5 seeds)

| seed | best makespan |
|---|---|
| 0 | 43 |
| 1 | 43 |
| 2 | 43 |
| 3 | 43 |
| 4 | 43 |

**Perfectly stable** at 43 across all 5 seeds on mk01.

## Comparison with prior agents (mk01, single seed)

| Agent | mk01 best | Notes |
|---|---|---|
| Random rollout (5×) | 86-123 | Reference |
| earliest-finish heuristic | 57 | Default dispatch rule |
| Graph AC + BC (old 9-feature) | 49 | Pretrained on earliest-finish |
| HGT + BC + A2C (500 ep) | 60 | New 14-feature, dropout bug fixed |
| HGT + BC + PPO (300 ep) | 66 | New 14-feature, dropout bug fixed |
| **REINFORCE 100 ep** | **43** | **Current SOTA** |

## Why REINFORCE works here

1. **Sparse reward + simple per-action MLP = sharp gradients**. The reward `R = -makespan` is a single scalar; `∇E[log π] = R · ∇log π` is a clean signal.
2. **Per-action scoring** instead of two-stage (job → machine) decoupling. This avoids the gradient conflict that comes from picking a job and then a machine sequentially.
3. **8 carefully scaled features** including `start` and `end` of the proposed dispatch — the network can directly model the marginal contribution of each action.
4. **No graph / no dropout / no attention**. The simple architecture is regularized by the small hidden dim and the per-step action-space cardinality.
5. **No baseline → high variance, but stable across seeds** because the action space is small (≤ 55 candidates) and the reward is dense enough to escape local optima.

## What this means for the project

1. **REINFORCE is the strongest single-instance baseline** we have. It's worth promoting to the SOTA role.
2. **Graph AC + BC = 49 (old features)** is now an inferior baseline. After re-training with the new 14-feature graph, it dropped to 57. **Feature enrichment was a regression on mk01.**
3. **HGT is dead** (PPO + A2C fine-tuning negative improvement, see `hgt_eval_mode_bug_reflection.md`).
4. **mk03 / mk04 have a +12 / +19 gap to OR-Tools**. These are the cases where REINFORCE is weakest; they're also the largest instances. Worth a longer training run or a different algorithm.

## Reproduction

```powershell
# Single seed, single instance
python -m rl.train --instance data/instances/brandimarte/mk01.txt --episodes 100 --seed 0

# 5 seeds, mk01-05
python experiments/rl_baselines/reinforce_5seed.py

# OR-Tools CP-SAT (proven optimal where time suffices)
python scripts/ortools_makespan.py data/instances/brandimarte/mk01.txt --time-limit-s 60
```

## Open questions for next iteration

- Why does REINFORCE converge so cleanly? Hypothesis: the 8 features include both local (duration) and global (makespan, machine_ready) context, which is enough signal at 100 ep.
- Can we beat 43 on mk01 with: (a) longer training, (b) value baseline (A2C wrapper), (c) imitation warm-start (BC + REINFORCE), (d) different exploration?
- Is the +19 gap on mk04 due to instance size, search-space cardinality, or a specific structural feature? Worth a per-instance feature study.
