# Contributing

Use a source checkout and Python 3.10 or newer. Create a virtual environment,
then install the project with `python -m pip install -e ".[dev]"`.
Optional plotting and CP-SAT dependencies are available through the `analysis`
and `solver` extras.

## Checks before a pull request

Run from the repository root:

```shell
python -m unittest discover -s tests -v
python scripts/run_smoke_test.py
python scripts/run_env_smoke_test.py
python scripts/check_curated_schedules.py
python -m build
```

Changes to parsers, environments, action masks, reward logic, validators, or
decoders need focused correctness tests. Keep the independent validator free
from policy/model dependencies. Validate schedules before comparing makespan.

Keep experiment drivers in `experiments/`, reusable scheduling logic in
`fjsp/`, and policies/networks in `rl/`. Exploratory scripts belong in
`scripts/dev/`; they are not part of the unit test suite.

## Reporting experiments

Record the Git revision, Python/PyTorch versions, hardware, instance path,
seed, complete configuration, runtime, and validation status. Report all seeds,
not just the best run. Separate policy performance from local-search gains and
compare methods at matched computational budgets.

Generated checkpoints and logs should stay outside commits. Curated results
need the configuration and validation evidence used to produce them. A valid
schedule proves feasibility; it does not prove optimality or a new state of the
art. Such claims need an independently checked reference and identical instances.

## Pull requests

Describe the behavioral change, motivation, commands run, and remaining
limitations. Keep unrelated cleanup in separate changes. Do not include
credentials, local environment files, or third-party data without provenance.
