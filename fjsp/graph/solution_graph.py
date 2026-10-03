"""Solution graph for the FJSP improvement (learning-to-improve) environment.

Unlike ``operation_machine_graph.py`` which represents a *partial* schedule
under construction, this module represents a *complete* FJSP solution as a
heterogeneous graph with critical-path annotations.

Node types
----------
- **Operation nodes**: one per (job, op) pair, all scheduled.
- **Machine nodes**: one per machine.

Edge types
----------
- **Precedence**: (job, op) -> (job, op+1) within each job.
- **Machine sequence**: (job_a, op_a) -> (job_b, op_b) if op_a is
  processed immediately before op_b on the same machine.
- **Eligibility**: (job, op) -> machine for every machine that *could*
  process this operation (including the currently assigned one).

Features
--------
Operation features (IMP_OP_FEATURE_DIM = 18):
  0. job_idx / (num_jobs - 1)
  1. op_idx / (num_ops_in_job - 1)
  2. assigned_machine / (num_machines - 1)
  3. start / scale
  4. end / scale
  5. duration / scale
  6. is_on_critical_path (0/1)
  7. num_eligible_machines / num_machines
  8. remaining_ops_in_job / num_ops_in_job
  9. job_lb_remaining / scale
 10. idle_before / scale  (gap between this op's start and the previous on same machine)
 11. idle_after / scale   (gap between this op's end and the next on same machine)
 12. machine_load_ratio   (duration / machine total load)
 13. is_bottleneck_machine (0/1)  (machine with highest load)
 14. flexibility_ratio    (1 / num_eligible_machines, low = less flexible)
 15. critical_path_depth  (position along critical path / total cp length)
 16. bottleneck_score     (BARI continuous score: cp * (0.4 + 0.3*load + 0.3*slack_inv))
 17. cp_distance          (graph distance to nearest critical-path op / N_ops)

Machine features (IMP_MACHINE_FEATURE_DIM = 8):
  0. machine_idx / (num_machines - 1)
  1. machine_ready_time / scale  (= last op's end time)
  2. total_load / scale  (sum of durations on this machine)
  3. num_ops_on_machine / num_total_ops
  4. idle_time / scale  (total idle gaps on this machine)
  5. is_bottleneck (0/1)
  6. avg_op_duration / scale
  7. num_critical_ops / num_ops_on_machine

Global features (IMP_GLOBAL_FEATURE_DIM = 8):
  0. makespan / scale
  1. critical_path_length / scale
  2. num_critical_ops / num_total_ops
  3. avg_machine_load / scale
  4. max_machine_load / scale
  5. total_idle_time / scale
  6. num_machines / max_machines  (for cross-instance generalization)
  7. num_jobs / max_jobs  (for cross-instance generalization)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from fjsp.parser.fjs_parser import FJSPInstance
from fjsp.scheduler.validator import ScheduledOperation
from fjsp.utils.scaling import scale_value


IMP_OP_FEATURE_DIM = 18
IMP_MACHINE_FEATURE_DIM = 8
IMP_GLOBAL_FEATURE_DIM = 8

# Reasonable upper bounds for cross-instance normalization
MAX_JOBS = 50
MAX_MACHINES = 30


@dataclass(frozen=True)
class SolutionGraph:
    """Heterogeneous graph representation of a complete FJSP solution."""

    operation_refs: tuple[tuple[int, int], ...]  # (job, op) per node
    op_features: torch.Tensor          # (N_ops, IMP_OP_FEATURE_DIM)
    machine_features: torch.Tensor     # (N_machines, IMP_MACHINE_FEATURE_DIM)
    precedence_edges: torch.Tensor     # (E_prec, 2) long
    machine_seq_edges: torch.Tensor    # (E_seq, 2) long  (prev_op -> next_op)
    eligibility_edges: torch.Tensor    # (E_elig, 2) long  (op -> machine)
    eligibility_durations: torch.Tensor  # (E_elig,) float
    critical_op_indices: tuple[int, ...]  # indices of ops on critical path
    global_features: torch.Tensor      # (IMP_GLOBAL_FEATURE_DIM,)


def find_critical_path(
    schedule: list[ScheduledOperation],
    instance: FJSPInstance,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Identify the critical path in a complete FJSP schedule.

    Returns
    -------
    critical_path : list of (job, op)
        Operations on the critical path, in order from source to sink.
    critical_op_set : list of int
        Indices into ``schedule`` of operations on the critical path.
    """
    if not schedule:
        return [], []

    makespan = max(op.end for op in schedule)

    # Build lookup: (job, op) -> ScheduledOperation
    op_map: dict[tuple[int, int], ScheduledOperation] = {}
    for sop in schedule:
        op_map[(sop.job, sop.op)] = sop

    # Build machine sequence: for each machine, ops sorted by start time
    by_machine: dict[int, list[ScheduledOperation]] = defaultdict(list)
    for sop in schedule:
        by_machine[sop.machine].append(sop)
    for m in by_machine:
        by_machine[m].sort(key=lambda o: o.start)

    # Build predecessor map: for each (job, op), its predecessors are:
    #   1. (job, op-1) if op > 0  (job precedence)
    #   2. The previous op on the same machine (machine precedence)
    predecessors: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            predecessors[(job_idx, op_idx)].append((job_idx, op_idx - 1))

    for m, ops in by_machine.items():
        for i in range(1, len(ops)):
            predecessors[(ops[i].job, ops[i].op)].append((ops[i - 1].job, ops[i - 1].op))

    # Compute longest path (critical path) using topological order
    # Since all edges go forward in time, we can sort by start time
    sorted_ops = sorted(schedule, key=lambda o: o.start)

    # longest_path_to[job,op] = length of longest path ending at (job,op)
    longest_path_to: dict[tuple[int, int], float] = {}
    best_pred: dict[tuple[int, int], tuple[int, int] | None] = {}

    for sop in sorted_ops:
        key = (sop.job, sop.op)
        preds = predecessors.get(key, [])
        if not preds:
            longest_path_to[key] = float(sop.duration)
            best_pred[key] = None
        else:
            best_len = -1.0
            best_p = None
            for p in preds:
                p_sop = op_map.get(p)
                if p_sop is None:
                    continue
                # Path length = longest path to predecessor + gap + duration
                gap = max(0, sop.start - p_sop.end)
                length = longest_path_to.get(p, 0.0) + float(gap) + float(sop.duration)
                if length > best_len:
                    best_len = length
                    best_p = p
            longest_path_to[key] = best_len
            best_pred[key] = best_p

    # Find the operation that ends at makespan with the longest path
    end_ops = [sop for sop in schedule if sop.end == makespan]
    if not end_ops:
        # Fallback: pick the op with the longest path
        end_key = max(longest_path_to, key=longest_path_to.get)  # type: ignore[arg-type]
        end_ops = [op_map[end_key]]

    # Trace back from the end op with the longest path
    best_end = max(end_ops, key=lambda o: longest_path_to.get((o.job, o.op), 0.0))
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = (best_end.job, best_end.op)
    while current is not None:
        path.append(current)
        current = best_pred.get(current)
    path.reverse()

    # Map to schedule indices
    schedule_index = {(sop.job, sop.op): i for i, sop in enumerate(schedule)}
    critical_indices = [schedule_index[key] for key in path if key in schedule_index]

    return path, critical_indices


def build_solution_graph(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    max_jobs: int = MAX_JOBS,
    max_machines: int = MAX_MACHINES,
    device: torch.device | str = "cpu",
) -> SolutionGraph:
    """Build a heterogeneous graph from a complete FJSP solution.

    Parameters
    ----------
    instance
        The FJSP instance.
    schedule
        A complete, valid schedule (all operations scheduled).
    max_jobs, max_machines
        Upper bounds for cross-instance normalization.
    device
        Torch device.
    """
    scale = instance_time_scale_from_schedule(instance, schedule)

    # Identify critical path
    critical_path, critical_indices = find_critical_path(schedule, instance)
    critical_set = set(critical_path)

    # Build lookup
    op_map: dict[tuple[int, int], ScheduledOperation] = {}
    for sop in schedule:
        op_map[(sop.job, sop.op)] = sop

    # Machine sequences
    by_machine: dict[int, list[ScheduledOperation]] = defaultdict(list)
    for sop in schedule:
        by_machine[sop.machine].append(sop)
    for m in by_machine:
        by_machine[m].sort(key=lambda o: o.start)

    # Machine stats
    machine_total_load: dict[int, float] = {}
    machine_idle: dict[int, float] = {}
    machine_critical_count: dict[int, int] = {}
    for m, ops in by_machine.items():
        machine_total_load[m] = sum(o.duration for o in ops)
        idle = 0
        for i in range(1, len(ops)):
            gap = ops[i].start - ops[i - 1].end
            if gap > 0:
                idle += gap
        machine_idle[m] = float(idle)
        machine_critical_count[m] = sum(1 for o in ops if (o.job, o.op) in critical_set)

    max_load = max(machine_total_load.values()) if machine_total_load else 1.0
    makespan = max((o.end for o in schedule), default=0)

    # BARI bottleneck contribution scores (continuous)
    bottleneck_scores = _compute_bottleneck_scores(
        schedule, instance, by_machine, machine_total_load, max_load, makespan, critical_set
    )

    # CP distance: graph BFS distance from each op to nearest critical-path op
    cp_distance = _compute_cp_distance(schedule, instance, by_machine, critical_set)

    # Build operation features
    operation_refs: list[tuple[int, int]] = []
    op_rows: list[list[float]] = []
    op_index: dict[tuple[int, int], int] = {}
    precedence_edges: list[tuple[int, int]] = []
    machine_seq_edges: list[tuple[int, int]] = []
    eligibility_edges: list[tuple[int, int]] = []
    eligibility_durations: list[float] = []

    # Critical path depth map
    cp_depth: dict[tuple[int, int], float] = {}
    for depth, key in enumerate(critical_path):
        cp_depth[key] = float(depth) / max(len(critical_path) - 1, 1)

    for sop in schedule:
        key = (sop.job, sop.op)
        op_index[key] = len(operation_refs)
        operation_refs.append(key)

        job = instance.jobs[sop.job]
        operation = job.operations[sop.op]
        is_critical = 1.0 if key in critical_set else 0.0

        # Idle gaps
        idle_before = 0.0
        idle_after = 0.0
        m_ops = by_machine.get(sop.machine, [])
        for i, mo in enumerate(m_ops):
            if mo.job == sop.job and mo.op == sop.op:
                if i > 0:
                    idle_before = float(sop.start - m_ops[i - 1].end)
                if i < len(m_ops) - 1:
                    idle_after = float(m_ops[i + 1].start - sop.end)
                break

        # Job lower bound remaining
        job_lb_remaining = sum(
            min(opt.duration for opt in job.operations[o].options)
            for o in range(sop.op + 1, len(job.operations))
        )

        # Machine load ratio
        m_load = machine_total_load.get(sop.machine, 1.0)
        load_ratio = float(sop.duration) / max(m_load, 1.0)

        # Is bottleneck machine
        is_bottleneck = 1.0 if machine_total_load.get(sop.machine, 0) >= max_load else 0.0

        # Flexibility ratio (inverse of number of eligible machines)
        n_eligible = len(operation.options)
        flexibility = 1.0 / max(n_eligible, 1)

        op_rows.append([
            scale_value(sop.job, max(instance.job_count - 1, 1)),
            scale_value(sop.op, max(len(job.operations) - 1, 1)),
            scale_value(sop.machine, max(instance.machine_count - 1, 1)),
            scale_value(sop.start, scale),
            scale_value(sop.end, scale),
            scale_value(sop.duration, scale),
            is_critical,
            scale_value(n_eligible, instance.machine_count),
            scale_value(len(job.operations) - sop.op - 1, max(len(job.operations), 1)),
            scale_value(job_lb_remaining, scale),
            scale_value(idle_before, scale),
            scale_value(idle_after, scale),
            load_ratio,
            is_bottleneck,
            flexibility,
            cp_depth.get(key, 0.0),
            bottleneck_scores.get(key, 0.0),
            cp_distance.get(key, 1.0),
        ])

    # Precedence edges
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            if (job_idx, op_idx - 1) in op_index and (job_idx, op_idx) in op_index:
                precedence_edges.append((op_index[(job_idx, op_idx - 1)], op_index[(job_idx, op_idx)]))

    # Machine sequence edges
    for m, ops in by_machine.items():
        for i in range(len(ops) - 1):
            a = ops[i]
            b = ops[i + 1]
            if (a.job, a.op) in op_index and (b.job, b.op) in op_index:
                machine_seq_edges.append((op_index[(a.job, a.op)], op_index[(b.job, b.op)]))

    # Eligibility edges
    for job_idx, job in enumerate(instance.jobs):
        for op_idx, operation in enumerate(job.operations):
            if (job_idx, op_idx) not in op_index:
                continue
            for option in operation.options:
                eligibility_edges.append((op_index[(job_idx, op_idx)], option.machine))
                eligibility_durations.append(scale_value(option.duration, scale))

    # Machine features
    machine_rows = []
    for m in range(instance.machine_count):
        ops = by_machine.get(m, [])
        avg_dur = sum(o.duration for o in ops) / max(len(ops), 1)
        machine_rows.append([
            scale_value(m, max(instance.machine_count - 1, 1)),
            scale_value(ops[-1].end if ops else 0, scale),
            scale_value(machine_total_load.get(m, 0), scale),
            scale_value(len(ops), len(schedule)),
            scale_value(machine_idle.get(m, 0), scale),
            1.0 if machine_total_load.get(m, 0) >= max_load else 0.0,
            scale_value(avg_dur, scale),
            scale_value(machine_critical_count.get(m, 0), max(len(ops), 1)),
        ])

    # Global features
    total_idle = sum(machine_idle.values())
    avg_load = sum(machine_total_load.values()) / max(instance.machine_count, 1)
    global_row = [
        scale_value(makespan, scale),
        scale_value(sum(o.duration for o in schedule if (o.job, o.op) in critical_set), scale),
        scale_value(len(critical_set), len(schedule)),
        scale_value(avg_load, scale),
        scale_value(max_load, scale),
        scale_value(total_idle, scale),
        float(instance.machine_count) / max_machines,
        float(instance.job_count) / max_jobs,
    ]

    return SolutionGraph(
        operation_refs=tuple(operation_refs),
        op_features=torch.tensor(op_rows, dtype=torch.float32, device=device),
        machine_features=torch.tensor(machine_rows, dtype=torch.float32, device=device),
        precedence_edges=torch.tensor(precedence_edges, dtype=torch.long, device=device) if precedence_edges else torch.zeros((0, 2), dtype=torch.long, device=device),
        machine_seq_edges=torch.tensor(machine_seq_edges, dtype=torch.long, device=device) if machine_seq_edges else torch.zeros((0, 2), dtype=torch.long, device=device),
        eligibility_edges=torch.tensor(eligibility_edges, dtype=torch.long, device=device) if eligibility_edges else torch.zeros((0, 2), dtype=torch.long, device=device),
        eligibility_durations=torch.tensor(eligibility_durations, dtype=torch.float32, device=device),
        critical_op_indices=tuple(critical_indices),
        global_features=torch.tensor(global_row, dtype=torch.float32, device=device),
    )


def instance_time_scale_from_schedule(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
) -> float:
    """Compute a normalization scale from the schedule's makespan."""
    makespan = max((o.end for o in schedule), default=1)
    return float(max(makespan, 1))


def _compute_bottleneck_scores(
    schedule: list[ScheduledOperation],
    instance: FJSPInstance,
    by_machine: dict[int, list[ScheduledOperation]],
    machine_total_load: dict[int, float],
    max_load: float,
    makespan: int,
    critical_set: set[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Compute BARI continuous bottleneck contribution score for each op.

    score = is_cp * (0.4 + 0.3 * load_ratio + 0.3 * slack_inv_ratio)

    - is_cp: 1.0 if op is on critical path, else 0.0
    - load_ratio: assigned machine load / max machine load
    - slack_inv_ratio: 1 - (idle_before + idle_after) / makespan

    High score => op is a bottleneck (on CP, busy machine, low slack).
    """
    scores: dict[tuple[int, int], float] = {}
    norm_makespan = float(makespan) if makespan > 0 else 1.0

    for sop in schedule:
        key = (sop.job, sop.op)
        is_cp = 1.0 if key in critical_set else 0.0
        if is_cp == 0.0:
            scores[key] = 0.0
            continue

        load_ratio = machine_total_load.get(sop.machine, 0.0) / max_load if max_load > 0 else 0.0

        m_ops = by_machine.get(sop.machine, [])
        idle_before = 0.0
        idle_after = 0.0
        for i, mo in enumerate(m_ops):
            if mo.job == sop.job and mo.op == sop.op:
                if i > 0:
                    idle_before = max(0.0, float(sop.start - m_ops[i - 1].end))
                if i < len(m_ops) - 1:
                    idle_after = max(0.0, float(m_ops[i + 1].start - sop.end))
                break

        slack = idle_before + idle_after
        slack_inv = 1.0 - min(1.0, slack / norm_makespan)

        scores[key] = is_cp * (0.4 + 0.3 * load_ratio + 0.3 * slack_inv)

    return scores


def _compute_cp_distance(
    schedule: list[ScheduledOperation],
    instance: FJSPInstance,
    by_machine: dict[int, list[ScheduledOperation]],
    critical_set: set[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Compute graph BFS distance from each op to nearest critical-path op.

    Distance is measured along precedence + machine-sequence edges.
    Returns normalized distance (0.0 for CP ops, up to 1.0 for far ops).
    """
    # Build adjacency (undirected for distance purposes)
    adj: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    # Job precedence edges
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            a = (job_idx, op_idx - 1)
            b = (job_idx, op_idx)
            adj[a].add(b)
            adj[b].add(a)

    # Machine sequence edges
    for m, ops in by_machine.items():
        for i in range(len(ops) - 1):
            a = (ops[i].job, ops[i].op)
            b = (ops[i + 1].job, ops[i + 1].op)
            adj[a].add(b)
            adj[b].add(a)

    # BFS from all CP ops simultaneously
    distances: dict[tuple[int, int], int] = {}
    queue: list[tuple[int, int]] = []
    for key in critical_set:
        distances[key] = 0
        queue.append(key)

    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nb in adj.get(cur, ()):
            if nb not in distances:
                distances[nb] = distances[cur] + 1
                queue.append(nb)

    # Normalize by number of ops (upper bound on diameter)
    n_ops = max(len(schedule), 1)
    return {key: float(d) / n_ops for key, d in distances.items()}
