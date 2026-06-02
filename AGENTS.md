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
- `data/results/`: Generated schedules, metrics, and experiment outputs.
- `fjsp/parser/`: Instance parsing and serialization.
- `fjsp/scheduler/`: Schedule validation, baseline dispatching, and decoding utilities.
- `fjsp/env/`: RL environment implementations.
- `rl/agents/`: RL algorithm implementations.
- `rl/models/`: Neural network modules.
- `experiments/configs/`: Experiment configuration files.
- `experiments/logs/`: Training and evaluation logs.
- `scripts/`: Command-line tools for validation, smoke tests, plotting, and benchmarking.

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
