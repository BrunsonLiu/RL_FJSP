# BC + AC Fine-Tuning on Brandimarte MK01

**Date**: 2026-06-02
**Script**: `experiments/bc_ac_quick.py` (and `rl/train_imitation.py` for the BC stage)
**Outputs**: `data/results/imitation_mk01_best.pt`, `data/results/bc_ac_mk01_best.pt`, `data/results/bc_ac_mk01_history.json`, `data/results/bc_ac_mk01_best_schedule.json`

## Setup

Two-stage training:

1. **Behavioral cloning** (BC) on the earliest-finish dispatch rule.
   - `python -m rl.train_imitation --instance data/instances/brandimarte/mk01.txt --rollouts 100 --epochs 20 --batch-size 32 --lr 1e-3 --hidden-dim 64 --gnn-rounds 2 --seed 0`
   - 100 rollouts × 55 operations = **5500 (state, action) demonstrations**
   - 20 supervised epochs with cross-entropy on both the job head and the machine head.
   - **Result: job_acc 100 %, machine_acc 100 %, total_loss 0.0001**
   - BC-only greedy makespan on mk01: **57** (exactly matches `rollout_earliest_finish`)

2. **Actor-critic fine-tuning** starting from the BC checkpoint.
   - `train_graph_actor_critic(..., init_model="data/results/imitation_mk01_best.pt")`
   - 500 episodes, lr=3e-5, hidden_dim=64, gnn_rounds=2, seed=0
   - Low lr is critical: a higher lr (3e-3, the default) immediately destroys the BC initialization.
   - Terminal-only reward, value head is fresh, actor head is BC-warmed.

## Trajectory (greedy makespan on mk01, lower is better)

| episode | sample | greedy | best so far |
| ------- | ------ | ------ | ----------- |
| 1       | 57     | 57     | 57          |
| 50      | 63     | 83     | 57          |
| 100     | 65     | 65     | 57          |
| 150     | 60     | 58     | 57          |
| 200     | 91     | 51     | 51          |
| 250     | 56     | 60     | 51          |
| 300     | 57     | 53     | 51          |
| 350     | 66     | 53     | 51          |
| 400     | 49     | **49** | **49**      |
| 450     | 49     | 49     | 49          |
| 500     | 55     | 50     | 49          |

## Headline

**BC + AC (500 ep, lr 3e-5) reaches greedy makespan 49 on Brandimarte mk01.**

Comparison with the 4-agent baseline matrix (`data/results/baseline_brandimarte_matrix.md`):

| Agent | best on mk01 |
| ----- | ------------ |
| earliest-finish heuristic | 57 |
| Random rollout (mean) | 106.3 |
| Graph AC default (50 ep) | 110 |
| Graph PPO default (200 ep) | 85 |
| Graph AC default (500 ep, from scratch) | 54 |
| Graph AC tuned (200 ep, from scratch) | 60 |
| **BC + AC (500 ep, this experiment)** | **49** |
| Two-stage AC from scratch (100 ep) | 48 |
| REINFORCE from scratch (100 ep) | 43 |

The graph encoder (BC-warmed + AC fine-tuned) is **no longer the worst**:
it beats every graph-from-scratch run and is within 6 makespan units of REINFORCE,
which is the strongest single-instance result we have at the 100-200 episode budget.

## Validator

`scripts/validate_schedule.py` re-checks the saved schedule:

```text
OK: valid schedule, makespan=49
```

The schedule JSON is `data/results/bc_ac_mk01_best_schedule.json`.

## Why did PPO fail but AC work?

We also tried BC → PPO with the same init model. With the default PPO hyper-parameters
(lr=3e-4, K_epochs=4, GAE), the policy drifted away from BC within the first
episode and the best greedy was 72 (worse than BC alone).

Lowering lr to 3e-5 and K_epochs to 1 stabilised the PPO run but it could not push
below 57. The difference is that:

- AC updates one minibatch per episode with a learned value baseline and a single
  on-policy gradient step. The critic head provides a per-step baseline, so the
  advantages are smaller and the policy moves slowly.
- PPO with K_epochs=4 re-evaluates the same trajectory four times and applies
  four gradient steps. The clip ratio of 0.2 lets the policy wander on every
  update. With a near-optimal BC init and a sparse terminal reward, the GAE
  advantages are essentially noise, so the extra updates just add noise.

In short: **AC + low lr is the right tool for fine-tuning a near-optimal BC
initialization**; PPO needs a denser reward signal (or a much smaller step) to
do the same job.

## What this unlocks

1. **The graph encoder is now useful on a single instance.** With BC warm-start,
   AC fine-tuning can improve past the heuristic in 200-500 episodes.
2. **Cross-instance training becomes viable.** A BC-pretrained encoder trained on
   one instance (mk01) can be fine-tuned per instance with a small number of AC
   episodes. This is the cheapest way to attack the "graph agents lose on
   Brandimarte" problem.
3. **A reproducible SOTA path opens up:** BC on a strong heuristic → AC
   fine-tune on each test instance → optional PPO fine-tune with dense reward
   to push past the heuristic further.

## Reproduction

```powershell
# Stage 1: BC pretraining
python -m rl.train_imitation --instance data/instances/brandimarte/mk01.txt --rollouts 100 --epochs 20 --batch-size 32 --lr 1e-3 --hidden-dim 64 --gnn-rounds 2 --seed 0

# Stage 2: AC fine-tuning (low lr is essential)
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').resolve())); from fjsp.env import FJSPDispatchEnv; from rl.agents import train_graph_actor_critic; env = FJSPDispatchEnv.from_file('data/instances/brandimarte/mk01.txt'); agent, history = train_graph_actor_critic(env, episodes=500, lr=3e-5, hidden_dim=64, gnn_rounds=2, seed=0, init_model='data/results/imitation_mk01_best.pt'); print('best:', history[-1]['best_greedy_makespan'])"

# Stage 3: validate
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/bc_ac_mk01_best_schedule.json
```
