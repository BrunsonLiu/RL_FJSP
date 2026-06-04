"""Operation-machine graph for the FJSP dispatch environment.

The graph has two node types (operations, machines) and two edge types
(precedence, eligibility). Features are designed to capture both the local
state of an operation / machine and its global context within the partial
schedule.

Default feature dimensions
-------------------------
- ``OP_FEATURE_DIM = 14`` (was 9; +5 features for SOTA)
- ``MACHINE_FEATURE_DIM = 6`` (was 4; +2 features)
- ``GLOBAL_FEATURE_DIM = 6`` (unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fjsp.env import FJSPDispatchEnv
from fjsp.utils.scaling import instance_time_scale, scale_value


OP_FEATURE_DIM = 14
MACHINE_FEATURE_DIM = 6
GLOBAL_FEATURE_DIM = 6


@dataclass(frozen=True)
class OperationRef:
    job: int
    op: int


@dataclass(frozen=True)
class OperationMachineGraph:
    operation_refs: tuple[OperationRef, ...]
    op_features: torch.Tensor
    machine_features: torch.Tensor
    precedence_edges: torch.Tensor
    eligibility_edges: torch.Tensor
    eligibility_durations: torch.Tensor
    next_op_indices: tuple[int, ...]
    global_features: torch.Tensor


def _job_lower_bound_remaining(env: FJSPDispatchEnv, job_idx: int) -> float:
    """Shortest possible remaining processing time for ``job_idx`` assuming
    the best machine for each remaining operation is picked immediately."""
    job = env.instance.jobs[job_idx]
    cur_op = env.job_next_op[job_idx]
    return float(
        sum(min((o.duration for o in op.options), default=0.0) for op in job.operations[cur_op:])
    )


def _machine_load(env: FJSPDispatchEnv, machine: int) -> float:
    """Total amount of work still queued for this machine (its own ready
    time + the duration of any currently waiting operation that can only run
    on it). Approximated by ``machine_ready_time - current_makespan``."""
    return float(max(env.machine_ready_time[machine] - env.makespan, 0.0))


def build_operation_machine_graph(env: FJSPDispatchEnv, *, device: torch.device | str = "cpu") -> OperationMachineGraph:
    scale = instance_time_scale(env)
    operation_refs: list[OperationRef] = []
    op_index: dict[tuple[int, int], int] = {}
    op_rows: list[list[float]] = []
    precedence_edges: list[tuple[int, int]] = []
    eligibility_edges: list[tuple[int, int]] = []
    eligibility_durations: list[float] = []
    next_op_indices: list[int] = []

    # Pre-compute machine load stats.
    machine_loads = [_machine_load(env, m) for m in range(env.instance.machine_count)]
    if machine_loads:
        load_mean = sum(machine_loads) / max(len(machine_loads), 1)
        load_var = sum((l - load_mean) ** 2 for l in machine_loads) / max(len(machine_loads), 1)
    else:
        load_mean = 0.0
        load_var = 0.0

    for job_idx, job in enumerate(env.instance.jobs):
        for op_idx, operation in enumerate(job.operations):
            op_index[(job_idx, op_idx)] = len(operation_refs)
            operation_refs.append(OperationRef(job=job_idx, op=op_idx))
            if op_idx == env.job_next_op[job_idx]:
                next_op_indices.append(len(operation_refs) - 1)

            is_done = op_idx < env.job_next_op[job_idx]
            is_next = op_idx == env.job_next_op[job_idx]
            durations = [option.duration for option in operation.options]
            # Lower bound on the remaining work for this op (itself) and the
            # whole job from here on.
            op_min_dur = min(durations) if durations else 0.0
            job_lb_remaining = _job_lower_bound_remaining(env, job_idx)
            # Machine load for the best machine for this op.
            best_machine_load = 0.0
            if durations:
                best_option = min(operation.options, key=lambda o: o.duration)
                best_machine_load = machine_loads[best_option.machine] if best_option.machine < len(machine_loads) else 0.0
            # Job slack: how much later the job can start the next op
            # without delaying the global makespan estimate.
            job_slack = max(0.0, env.makespan - env.job_ready_time[job_idx])

            op_rows.append(
                [
                    scale_value(job_idx, max(env.instance.job_count - 1, 1)),
                    scale_value(op_idx, max(len(job.operations) - 1, 1)),
                    1.0 if is_done else 0.0,
                    1.0 if is_next else 0.0,
                    scale_value(env.job_ready_time[job_idx], scale),
                    scale_value(op_min_dur, scale),
                    scale_value(sum(durations) / len(durations), scale),
                    scale_value(len(operation.options), env.instance.machine_count),
                    scale_value(len(job.operations) - op_idx - 1, max(len(job.operations), 1)),
                    # --- new features ---
                    scale_value(job_lb_remaining, scale),
                    scale_value(job_slack, scale),
                    scale_value(best_machine_load, scale),
                    scale_value(load_mean, scale),
                    scale_value(load_var, scale),
                ]
            )

            if op_idx > 0:
                precedence_edges.append((op_index[(job_idx, op_idx - 1)], op_index[(job_idx, op_idx)]))
            for option in operation.options:
                eligibility_edges.append((op_index[(job_idx, op_idx)], option.machine))
                eligibility_durations.append(scale_value(option.duration, scale))

    machine_rows = [
        [
            scale_value(machine, max(env.instance.machine_count - 1, 1)),
            scale_value(env.machine_ready_time[machine], scale),
            scale_value(sum(1 for _, dst in eligibility_edges if dst == machine), max(env.instance.operation_count, 1)),
            scale_value(env.makespan, scale),
            # --- new features ---
            scale_value(machine_loads[machine], scale),
            1.0 if machine_loads[machine] <= load_mean else 0.0,
        ]
        for machine in range(env.instance.machine_count)
    ]
    unfinished_jobs = sum(1 for job_idx, job in enumerate(env.instance.jobs) if env.job_next_op[job_idx] < len(job.operations))
    global_row = [
        scale_value(env.makespan, scale),
        scale_value(env.remaining_operations, env.instance.operation_count),
        scale_value(unfinished_jobs, env.instance.job_count),
        scale_value(min(env.job_ready_time, default=0), scale),
        scale_value(max(env.job_ready_time, default=0), scale),
        scale_value(max(env.machine_ready_time, default=0), scale),
    ]

    return OperationMachineGraph(
        operation_refs=tuple(operation_refs),
        op_features=torch.tensor(op_rows, dtype=torch.float32, device=device),
        machine_features=torch.tensor(machine_rows, dtype=torch.float32, device=device),
        precedence_edges=torch.tensor(precedence_edges, dtype=torch.long, device=device),
        eligibility_edges=torch.tensor(eligibility_edges, dtype=torch.long, device=device),
        eligibility_durations=torch.tensor(eligibility_durations, dtype=torch.float32, device=device),
        next_op_indices=tuple(next_op_indices),
        global_features=torch.tensor(global_row, dtype=torch.float32, device=device),
    )
