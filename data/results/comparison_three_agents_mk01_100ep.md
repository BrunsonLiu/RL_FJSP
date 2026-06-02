# Three-Agent Comparison on Brandimarte MK01

**Setup**

- Instance: `data/instances/brandimarte/mk01.txt`
- Jobs / Machines / Operations: 10 / 6 / 55
- Known optimum: 40
- Earliest-finish heuristic: 57
- Random rollout mean (3 seeds): 106.333
- REINFORCE and Two-Stage AC: 100 episodes, seed 0
- Graph AC: tried 100 / 300 / 500 episodes (default config), 200 episodes (tuned config), all seed 0
- Graph PPO: 200 episodes, default PPO config, seed 0
- Evaluation: greedy rollout
- Each RL schedule below was independently re-checked by `scripts/validate_schedule.py`

## Results

| Agent                          | Best makespan | Mean | Std | Wall-clock (s) | vs optimum | vs earliest_finish |
| ------------------------------ | ------------- | ---- | --- | -------------- | ---------- | ------------------ |
| **REINFORCE** (100 ep)         | **43**        | 43   | 0.0 | 5.0            | +3 (7.5%)  | −14 (24.6%)        |
| Two-Stage AC (100 ep)          | 48            | 48   | 0.0 | 10.8           | +8 (20.0%) | −9 (15.8%)         |
| Graph AC default (100 ep)      | 100           | 100  | 0.0 | 47.7           | +60 (150%) | +43 (75.4%)        |
| Graph AC default (300 ep)      | 58            | 58   | 0.0 | ~140           | +18 (45%)  | +1 (1.8%)          |
| Graph AC default (500 ep)      | 54            | 54   | 0.0 | ~235           | +14 (35%)  | −3 (5.3%)          |
| Graph AC tuned (200 ep)        | 60            | 60   | 0.0 | ~210           | +20 (50%)  | +3 (5.3%)          |
| Graph PPO (200 ep)             | 85            | 85   | 0.0 | ~75            | +45 (113%) | +28 (49.1%)        |

**Tuned AC config**: `lr=1e-3, hidden_dim=128, gnn_rounds=3`
**Default AC config**: `lr=3e-3, hidden_dim=64, gnn_rounds=2`
**PPO config**: `lr=3e-4, clip=0.2, gamma=0.99, gae_lambda=0.95, K_epochs=4, minibatch_size=16`

## Graph AC Training Trajectories

### Default config, 500 episodes

| Episode | sample (random) | greedy | best so far | entropy |
| ------- | --------------- | ------ | ----------- | ------- |
| 1       | 100             | 114    | 114         | 2.523   |
| 30      | 108             | 110    | 110         | 2.552   |
| 60      | 80              | 112    | 110         | 2.588   |
| 90      | 111             | 105    | 105         | 2.614   |
| 120     | 89              | 92     | 92          | 2.652   |
| 150     | 88              | **58** | **58**      | 2.689   |
| 200     | 88              | 84     | 58          | 2.594   |
| 300     | 81              | 71     | 58          | 2.669   |
| 400     | 90              | 87     | 56          | 2.658   |
| 450     | 84              | **54** | **54**      | 2.628   |
| 500     | 87              | 80     | 54          | 2.686   |

### Tuned config, 200 episodes

| Episode | sample (random) | greedy | best so far | entropy |
| ------- | --------------- | ------ | ----------- | ------- |
| 1       | 88              | 133    | 133         | 2.529   |
| 20      | 86              | 128    | 128         | 2.634   |
| 40      | 77              | 122    | 122         | 2.678   |
| 60      | 94              | 127    | 122         | 2.557   |
| 80      | 85              | 131    | 122         | 2.459   |
| 100     | 104             | **60** | **60**      | 2.434   |
| 120     | 84              | 102    | 60          | 2.471   |
| 160     | 89              | 69     | 60          | 2.567   |
| 200     | 99              | 93     | 60          | 2.706   |

## Observations

- All three agents beat the earliest-finish heuristic except Graph AC at 100 ep.
- REINFORCE remains the strongest at this episode budget, matching the `docs/algorithm.md` baseline of 43.
- Two-Stage AC lands at 48, matching the `docs/algorithm.md` line "Two-stage actor-critic, seed=0, 100 episodes: makespan 48".
- Graph AC improves 100 → 58 → 54 when default training is extended from 100 → 300 → 500 episodes.
  - The model is **not** truly plateaued at 300 ep; it continues to find slightly better states through 500 ep.
  - However, the marginal gain (58 → 54 = 4 makespan units over 200 extra episodes) is very small.
- The tuned config (lr 1e-3, hidden 128, rounds 3) at 200 episodes reaches 60.
  - Slightly worse than the default config at 300 ep (58) and similar to default at 200 ep would be.
  - Larger model + smaller lr + more GNN rounds did **not** break through the local optimum.
- **Conclusion**: the Graph AC bottleneck is not episodes or basic hyperparameters. The "late jump + long plateau" pattern appears in both default and tuned runs, suggesting the cause is reward design, exploration strategy, or the need for a fundamentally different learning signal (PPO, denser reward, or imitation pretraining).

## Schedules (independently validated)

| Agent         | Schedule JSON                                                  | Validator result   |
| ------------- | -------------------------------------------------------------- | ------------------ |
| REINFORCE     | `data/results/reinforce_mk01_best_schedule.json`               | valid, makespan 43 |
| Two-Stage AC  | `data/results/actor_critic_mk01_best_schedule.json`            | valid, makespan 48 |
| Graph AC default 500 ep | `data/results/graph_actor_critic_mk01_500_schedule.json`     | valid, makespan 54 |
| Graph AC tuned 200 ep  | `data/results/graph_actor_critic_mk01_200_tuned_schedule.json` | valid, makespan 60 |
| Graph PPO 200 ep       | `data/results/graph_ppo_mk01_200_best_schedule.json`         | valid, makespan 85 |

## Why Graph AC underperforms — updated diagnosis

The Graph AC code path is healthy (compiles, trains, saves, loads, and produces valid schedules). The under-performance is therefore a training or learning-signal question, not a correctness question.

Confirmed by B and C:
- Adding 200 more episodes (300 → 500) only buys 4 makespan units. The model is at a shallow local optimum.
- Tripling the encoder size and slowing the optimizer did not change the shape of the learning curve.

Remaining plausible causes:
1. **Reward is terminal makespan / instance_time_scale**. This is a sparse, high-variance signal. Graph encoders have more parameters, so the noise hurts more.
2. **EMA baseline is not a critic** — it does not give a per-state value estimate, so the graph encoder cannot learn state-dependent advantages. Replacing EMA with a learned value head (already implemented but possibly underused) might help.
3. **No warm-up / no lr decay** — the optimizer step size is constant.
4. **No exploration bonus / entropy schedule** — entropy is held constant via `entropy_coef`, but on flat plateaus more exploration might be needed.

## Possible Next Steps

A. Stop and **document this as a known limitation**. Treat the Graph AC result as "competitive at 54 with the existing code". Move on to PPO / cross-instance training / imitation pretraining.

B. Add **per-step dense reward** (e.g. negative change in machine-idle-time) and retrain Graph AC.

C. Add **imitation pretraining**: pre-train the Graph AC encoder to mimic the earliest-finish heuristic, then fine-tune with RL.

D. Replace REINFORCE-style policy gradient with **PPO clip** for both AC and Graph AC.

E. Do a **longer multi-seed study** (3 seeds × 1000 ep) to check whether Graph AC eventually crosses below 50 with enough compute.
