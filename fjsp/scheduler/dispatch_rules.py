from __future__ import annotations

from typing import Callable

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


# --------------------------------------------------------------------------- #
# Additional dispatch rules used as multi-heuristic BC teachers.
#
# Each rule scores (job, machine) pairs and returns the action with the
# smallest score. Ties are broken by job index, then machine index, so the
# rule is deterministic for a given state.
# --------------------------------------------------------------------------- #


def _score_shortest_processing_time(env: FJSPDispatchEnv) -> tuple[int, int, int, int, DispatchAction]:
    """SPT: pick the (job, machine) with the shortest processing time."""
    best: tuple[int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        # Tie break: job, machine.
        candidate = (duration, action.job, action.machine, 0, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best


def choose_spt(env: FJSPDispatchEnv) -> DispatchAction:
    return _score_shortest_processing_time(env)[-1]


def choose_lpt(env: FJSPDispatchEnv) -> DispatchAction:
    """LPT: pick the (job, machine) with the longest processing time."""
    best: tuple[int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        candidate = (-duration, action.job, action.machine, 0, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best[-1]


def choose_mor(env: FJSPDispatchEnv) -> DispatchAction:
    """MOR: pick the job with the most operations remaining; among ties,
    the (job, machine) with the shortest processing time."""
    best: tuple[int, int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        op_total = len(env.instance.jobs[action.job].operations)
        op_left = op_total - op_idx
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        # Most remaining ops -> negate.
        candidate = (-op_left, duration, action.job, action.machine, 0, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best[-1]


def choose_lor(env: FJSPDispatchEnv) -> DispatchAction:
    """LOR: pick the job with the fewest operations remaining; among ties,
    the (job, machine) with the shortest processing time."""
    best: tuple[int, int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        op_total = len(env.instance.jobs[action.job].operations)
        op_left = op_total - op_idx
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        candidate = (op_left, duration, action.job, action.machine, 0, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best[-1]


def choose_shortest_ready_time(env: FJSPDispatchEnv) -> DispatchAction:
    """Pick the (job, machine) minimising max(job_ready, machine_ready) - the
    earliest *start* time. Ties broken by duration, then job/machine."""
    best: tuple[int, int, int, int, int, DispatchAction] | None = None
    for action in env.available_actions():
        op_idx = env.job_next_op[action.job]
        operation = env.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)
        start = max(env.job_ready_time[action.job], env.machine_ready_time[action.machine])
        candidate = (start, duration, action.job, action.machine, 0, action)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No valid action available.")
    return best[-1]


# Registry of named rules. ``None`` is shorthand for ``choose_earliest_finish``.
DISPATCH_RULES: dict[str, Callable[[FJSPDispatchEnv], DispatchAction]] = {
    "earliest_finish": choose_earliest_finish,
    "spt": choose_spt,
    "lpt": choose_lpt,
    "mor": choose_mor,
    "lor": choose_lor,
    "shortest_start": choose_shortest_ready_time,
}


def get_rule(name: str) -> Callable[[FJSPDispatchEnv], DispatchAction]:
    if name not in DISPATCH_RULES:
        raise KeyError(f"Unknown dispatch rule {name!r}. Choices: {sorted(DISPATCH_RULES)}")
    return DISPATCH_RULES[name]


def rollout_rule(env: FJSPDispatchEnv, rule: Callable[[FJSPDispatchEnv], DispatchAction]) -> int:
    env.reset()
    while not env.done:
        env.step(rule(env))
    result = env.validate()
    if not result.is_valid:
        raise RuntimeError(f"Rule rollout produced invalid schedule: {result.errors}")
    return result.makespan

