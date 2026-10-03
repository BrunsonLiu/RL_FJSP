# RL_FJSP

Reinforcement learning for the Flexible Job Shop Scheduling Problem (FJSP).

The repository combines dispatch policies with local search, tabu search,
and experimental learning-to-improve models. Schedule legality is checked
independently from policy code.

## Installation

Use Python 3.10 or newer. Run commands from a source checkout:

```shell
git clone https://github.com/BrunsonLiu/RL_FJSP.git
cd RL_FJSP
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install:

```shell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch and NumPy are required. Optional plotting and CP-SAT dependencies:
`python -m pip install -e ".[analysis,solver]"`. Install a suitable PyTorch
build first when using a GPU. Wheels contain the `fjsp` and `rl` modules;
benchmark data and scripts remain in the checkout. Module entry points use
checkout-relative defaults, so pass explicit paths when using a wheel.

## Components

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

## Quick Checks

```powershell
python scripts/run_smoke_test.py
python scripts/run_env_smoke_test.py
python -m unittest discover -s tests -v
python scripts/check_curated_schedules.py
```

CI runs unit tests, independent schedule validation, and package builds on
Linux and Windows. Exploratory scripts under `scripts/dev/` are not unit tests.

## Training and Evaluation

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
python -m rl.train --episodes 200 --seed 0
python -m rl.evaluate
python scripts/validate_schedule.py data/instances/brandimarte/mk01.txt data/results/reinforce_mk01_schedule.json
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

Train and evaluate the graph two-stage PPO policy:

```powershell
python -m rl.train_graph_ppo --episodes 200
python -m rl.evaluate_graph_ppo
```

Run a compact benchmark table:

```powershell
python scripts/run_benchmark.py --brandimarte-count 5 --episodes 50 --seeds 0,1,2
python scripts/run_benchmark.py --instances mk01 --agent actor_critic --episodes 100 --seeds 0
python scripts/run_benchmark.py --instances mk01 --agent graph_actor_critic --episodes 50 --seeds 0
```

The first milestone is correctness: every generated schedule must satisfy operation precedence, machine capacity, processing-time compatibility, and exactly-once operation assignment.

## Repository Map

| Path | Purpose |
| --- | --- |
| `fjsp/parser/` | Parsing and normalized zero-based indices |
| `fjsp/scheduler/` | Validation, dispatch rules, and local search |
| `fjsp/env/` | Dispatch and improvement environments |
| `rl/agents/`, `rl/models/` | Learning algorithms and networks |
| `experiments/` | Benchmark, ablation, and analysis drivers |
| `scripts/`, `tests/` | Command-line tools and unit tests |
| `data/instances/` | Toy and Brandimarte benchmark instances |
| `data/results/` | Curated results and ignored generated artifacts |
| `docs/`, `paper/` | Research notes and working manuscripts |

## Research Status

Record seeds, configurations, runtime, hardware, and validation status.
Compare policies and search-enhanced pipelines separately at matched budgets.
Learning-to-improve and neural-guided search are active experimental work.

Curated schedules and tables are retained for reproducibility. Manuscripts
are working documents. Historical SOTA/optimality labels need independent
literature and instance verification; legality checks establish feasibility
only. This repository does not certify those labels as current records.

Known artifact discrepancy: MK12 is reported as 508 in the final table, while
its saved schedule validates at 524. See the [publication notes](docs/github_release.md)
before using this entry as a reproduced result.

See [current status](docs/current_status.md) and
[instance provenance](data/instances/README.md).

## Contributing and Publication

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and experiment conventions,
and [GitHub publication](docs/github_release.md) for release preparation.
The [project roadmap](docs/ROADMAP.md) tracks the next engineering and research
tasks, their acceptance criteria, and the proposed eight-week validation plan.
Keep generated checkpoints, credentials, and transient logs outside commits.

No code license has been selected yet. Public visibility alone does not grant
reuse rights. Third-party benchmark data retains its own terms; confirm these
before redistributing it or assigning a license to the repository.
