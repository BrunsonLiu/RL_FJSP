# RL_FJSP Agent Guide

This repository is for reinforcement learning methods for the Flexible Job Shop Scheduling Problem (FJSP).

## Project Goal

Build, train, and evaluate RL agents that produce legal FJSP schedules with low makespan.

Primary objective:
- Minimize makespan.

Secondary concerns:
- Keep schedules legal before optimizing quality.
- Make experiments reproducible with fixed seeds and recorded configs.
- Keep parsers, validators, environments, and RL policies separated.

## Repository Layout

- `data/instances/`: FJSP benchmark and toy instances.
- `data/results/`: Generated schedules, metrics, and experiment outputs (generated files are git-ignored; only curated SOTA artifacts are tracked).
- `fjsp/parser/`: Instance parsing and serialization.
- `fjsp/scheduler/`: Schedule validation, baseline dispatching, and decoding utilities.
- `fjsp/env/`: RL environment implementations.
- `rl/agents/`: RL algorithm implementations.
- `rl/models/`: Neural network modules.
- `experiments/sota/`: SOTA pipeline scripts for Brandimarte MK01–MK15.
- `experiments/rl_baselines/`: One-off RL training/evaluation experiment drivers.
- `experiments/analysis/`: Result analysis, ablation tables, and figure generation.
- `experiments/configs/`: Experiment configuration files.
- `experiments/logs/`: Training and evaluation logs.
- `scripts/`: Command-line tools for validation, smoke tests, plotting, and benchmarking.
- `scripts/dev/`: One-off debug and quick-test scripts (not part of the unit test suite).
- `tests/`: Unit tests (run with `python -m unittest`).
- `docs/`: Project notes, status reports, and algorithm documentation.
- `paper/`: Paper sources, outlines, and figures.

Scripts under `experiments/` and `scripts/dev/` locate the repository
root relative to `__file__`, so they can be run from any working
directory: `python experiments/sota/sota_reinforce_ils.py`.

## Development Rules

- Validate legality before comparing makespan.
- Do not trust an RL action sequence until `scripts/validate_schedule.py` passes.
- Keep instance parsing deterministic and independent from RL code.
- Keep schedule validators independent from model code so they can be used as a judge.
- Prefer small toy instances for quick debugging before running benchmark training.
- When changing environment dynamics, action masks, reward logic, or decoders, run the smoke test.

## Common Commands

Run the smallest end-to-end check:

```powershell
python scripts/run_smoke_test.py
```

Run dispatch environment checks:

```powershell
python scripts/run_env_smoke_test.py
python -m unittest
```

Train and evaluate the minimal REINFORCE baseline:

```powershell
python -m rl.train --episodes 200
python -m rl.evaluate
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/reinforce_mk01_schedule.json
```

Run compact benchmark experiments:

```powershell
python scripts/run_benchmark.py --brandimarte-count 5 --episodes 50 --seeds 0,1,2
python scripts/run_benchmark.py --instances mk01 --agent actor_critic --episodes 100 --seeds 0
python scripts/run_benchmark.py --instances mk01 --agent graph_actor_critic --episodes 50 --seeds 0
```

Train and evaluate the graph two-stage actor-critic baseline:

```powershell
python -m rl.train_graph_actor_critic --episodes 200
python -m rl.evaluate_graph_actor_critic
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/graph_actor_critic_mk01_best_schedule.json
```

Train and evaluate the graph two-stage PPO baseline:

```powershell
python -m rl.train_graph_ppo --episodes 200
python -m rl.evaluate_graph_ppo
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/graph_ppo_mk01_best_schedule.json
```

Behavioral cloning pretraining on the earliest-finish dispatch rule, then AC fine-tuning:

```powershell
python -m rl.train_imitation --instance data/instances/brandimarte/mk01.txt --rollouts 100 --epochs 20 --batch-size 32 --lr 1e-3 --hidden-dim 64 --gnn-rounds 2 --seed 0
python -m rl.train_imitation_ppo --instance data/instances/brandimarte/mk01.txt --init-model data/results/imitation_mk01_best.pt --episodes 200 --lr 3e-5 --K-epochs 1 --entropy-coef 0.0 --seed 0
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/bc_ppo_mk01_best_schedule.json
```

For AC fine-tuning (instead of PPO) pass `init_model` to `train_graph_actor_critic`:

```powershell
python -c "from rl.agents import train_graph_actor_critic; from fjsp.env import FJSPDispatchEnv; env = FJSPDispatchEnv.from_file('data/instances/brandimarte/mk01.txt'); agent, history = train_graph_actor_critic(env, episodes=500, lr=3e-5, hidden_dim=64, gnn_rounds=2, seed=0, init_model='data/results/imitation_mk01_best.pt'); print('best:', history[-1]['best_greedy_makespan'])"
```

BC + AC reaches greedy 49 on mk01 (see `data/results/bc_ac_mk01_results.md`).
BC + PPO stays at 57 (PPO cannot break the BC initialization with the
default sparse reward).

Run large Brandimarte diagnostics:

```powershell
python scripts/run_benchmark.py --brandimarte-start 11 --brandimarte-count 5 --random-rollouts 3 --episodes 50 --seeds 0
```

Validate an instance:

```powershell
python scripts/validate_instance.py data/instances/tiny_2x2.fjs
```

Validate a schedule JSON:

```powershell
python scripts/validate_schedule.py data/instances/tiny_2x2.fjs data/results/tiny_2x2_greedy_schedule.json
```

## SOTA Pipeline (Brandimarte MK01–MK15)

The full SOTA matrix in `paper/main.md` §6.5 is regenerated by
three scripts in sequence:

```powershell
python experiments/sota/sota_reinforce_ils.py     # 5 seeds × 100 episodes + ILS+SA
python experiments/sota/aggressive_ils_hard.py    # aggressive ILS variants on hard instances
python experiments/sota/merge_sota.py             # writes data/results/sota_final.json

# Tabu search post-processor on the three largest instances
python experiments/sota/tabu_hard_v2.py           # writes data/results/tabu_results.json
python experiments/sota/merge_ts_sota.py          # folds TS into sota_final.json
```

Final SOTA matrix (data/results/sota_final.json):

| Inst | R-best | FINAL | Lit | Status |
|---|---|---|---|---|
| mk01 | 43  | 42  | 40  | |
| mk02 | 28  | 28  | 26  | |
| mk03 | 216 | 204 | 204 | TIED OPT |
| mk04 | 79  | 73  | 60  | |
| mk05 | 180 | 176 | 172 | |
| mk06 | 69  | 68  | 58  | |
| mk07 | 154 | 143 | 139 | |
| mk08 | 533 | 523 | 523 | TIED OPT |
| mk09 | 339 | 332 | 307 | |
| mk10 | 242 | 224 | 197 | TS post-proc. |
| mk11 | 639 | 619 | 615 | |
| mk12 | 531 | 508 | 508 | TIED OPT |
| mk13 | 464 | 416 | 430 | **NEW SOTA** |
| mk14 | 694 | 694 | 694 | TIED OPT |
| mk15 | 408 | 370 | 341 | TS post-proc. |

Mean gap to literature: **5.64%** across 15 instances. Total gap:
106 makespan units. 4/15 instances tied; 1/15 instance beat
(MK13).

## Schedule JSON Contract

Schedules should use this shape:

```json
{
  "operations": [
    {"job": 0, "op": 0, "machine": 0, "start": 0, "end": 3}
  ]
}
```

Indices are zero-based in repository code and JSON outputs.
