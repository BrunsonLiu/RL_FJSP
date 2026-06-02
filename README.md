# RL_FJSP

Reinforcement learning for the Flexible Job Shop Scheduling Problem (FJSP).

This project starts with a small but reliable foundation:

- FJSP instance parser.
- Schedule legality validator.
- Tiny toy instance.
- Brandimarte MK benchmark instances.
- Greedy smoke test that generates and validates a schedule.
- Minimal REINFORCE dispatch-policy baseline.
- Two-stage actor-critic dispatch-policy baseline.

Algorithm notes:

- [docs/algorithm.md](docs/algorithm.md)
- [docs/large_instances.md](docs/large_instances.md)

Run:

```powershell
python scripts/run_smoke_test.py
```

Validate a Brandimarte benchmark instance:

```powershell
python scripts/validate_instance.py data/instances/brandimarte/mk01.txt
```

Run the first dispatch RL environment smoke test on Brandimarte MK01:

```powershell
python scripts/run_env_smoke_test.py
```

Train and evaluate the first minimal REINFORCE baseline:

```powershell
python -m rl.train --episodes 200
python -m rl.evaluate
```

Train and evaluate the first two-stage actor-critic policy:

```powershell
python -m rl.train_actor_critic --episodes 200
python -m rl.evaluate_actor_critic
```

Train and evaluate the graph two-stage actor-critic policy:

```powershell
python -m rl.train_graph_actor_critic --episodes 200
python -m rl.evaluate_graph_actor_critic
```

Run a compact benchmark table:

```powershell
python scripts/run_benchmark.py --brandimarte-count 5 --episodes 50 --seeds 0,1,2
python scripts/run_benchmark.py --instances mk01 --agent actor_critic --episodes 100 --seeds 0
python scripts/run_benchmark.py --instances mk01 --agent graph_actor_critic --episodes 50 --seeds 0
```

The first milestone is correctness: every generated schedule must satisfy operation precedence, machine capacity, processing-time compatibility, and exactly-once operation assignment.
