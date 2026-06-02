---
name: fjsp-rl
description: Use when working on reinforcement learning for Flexible Job Shop Scheduling Problems, including FJSP instance parsing, schedule validation, action masks, reward design, legal decoders, benchmarks, and makespan evaluation.
---

# FJSP RL Workflow

Use this skill when modifying or reviewing FJSP reinforcement learning code.

## Priorities

1. Legality first: every schedule must satisfy job precedence, machine capacity, valid machine eligibility, exact processing time, and exactly-once operation assignment.
2. Keep the validator independent from RL models.
3. Treat action masks and decoders as high-risk code.
4. Compare makespan only after `scripts/validate_schedule.py` passes.
5. Start debugging with `data/instances/tiny_2x2.fjs` before larger benchmarks.

## Standard Checks

After changes to parsers, validators, environments, action masks, decoders, reward logic, or baseline scheduling, run:

```powershell
python scripts/run_smoke_test.py
```

For a generated schedule:

```powershell
python scripts/validate_schedule.py <instance.fjs> <schedule.json>
```

## RL Environment Guidance

- State should expose remaining operations, job readiness, machine readiness, candidate machine options, and current time or dispatch context.
- Actions should map cleanly to dispatch decisions, such as choosing an eligible operation-machine pair.
- Invalid actions should be impossible through masking when feasible; if accepted, they must produce a clear penalty and not corrupt state.
- Rewards should not hide legality bugs. Prefer validating terminal schedules during development.
- Keep makespan calculation in shared scheduling utilities, not duplicated inside model code.

## Experiment Guidance

- Record instance path, seed, objective, training config, final makespan, validation status, and wall-clock time.
- Compare against a deterministic baseline before claiming RL improvement.
- Use fixed seeds for smoke tests and small regression checks.

