from __future__ import annotations

from fjsp.env import DispatchAction, FJSPDispatchEnv


def choose_earliest_finish(env: FJSPDispatchEnv) -> DispatchAction:
    """Choose the valid dispatch action with the earliest resulting finish time."""
    best: tuple[int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        start = max(env.job_ready_time[action.job], env.machine_ready_time[action.machine])
        end = start + duration
        candidate = (end, start, action.job, action.machine, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best[-1]


def rollout_earliest_finish(env: FJSPDispatchEnv) -> int:
    env.reset()
    while not env.done:
        env.step(choose_earliest_finish(env))
    result = env.validate()
    if not result.is_valid:
        raise RuntimeError(f"Earliest-finish rollout produced invalid schedule: {result.errors}")
    return result.makespan

