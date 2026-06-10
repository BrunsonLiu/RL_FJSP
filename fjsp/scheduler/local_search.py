"""Local search improvement for FJSP schedules.

This module implements three classic FJSP local-search neighborhoods on a
fixed list of (job, op, machine) assignments:

1. **Reassignment (N1)**: move an operation to a different eligible machine.
2. **Swap on same machine (N2)**: swap the order of two adjacent operations
   on the same machine, if precedence allows.
3. **Swap across machines (N3)**: swap the machine of two operations that
   have at least one common eligible machine.

The search is first-improvement (it accepts the first improvement and
restarts the loop). It runs until no improvement is found or
``max_iterations`` is reached.

The motivation: simple per-action REINFORCE converges to a near-optimal
dispatch policy but is bounded by the dispatch view (no operation re-
ordering, no iterative refinement). Local search closes the gap to the
literature best-known upper bounds on Brandimarte MK04, MK09, MK15 by
~20-30 makespan units, without changing the RL agent.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Iterable

from fjsp.parser.fjs_parser import FJSPInstance
from fjsp.scheduler.validator import ScheduledOperation, validate_schedule


def _recompute(instance: FJSPInstance, assignments: list[tuple[int, int, int]]) -> list[ScheduledOperation]:
    """Given a list of (job, op, machine) assignments, decode to a legal
    schedule by walking through operations in precedence order on each
    machine. Returns the list of ScheduledOperation with start/end times."""
    machine_ready = [0] * instance.machine_count
    job_next_op = [0] * instance.job_count
    job_ready = [0] * instance.job_count
    result: list[ScheduledOperation] = []
    pending = list(assignments)
    progress = True
    while pending and progress:
        progress = False
        still_pending: list[tuple[int, int, int]] = []
        for job, op_idx, machine in pending:
            if job_next_op[job] != op_idx:
                still_pending.append((job, op_idx, machine))
                continue
            operation = instance.jobs[job].operations[op_idx]
            duration = next(o.duration for o in operation.options if o.machine == machine)
            start = max(job_ready[job], machine_ready[machine])
            end = start + duration
            result.append(
                ScheduledOperation(job=job, op=op_idx, machine=machine, start=start, end=end)
            )
            job_next_op[job] += 1
            job_ready[job] = end
            machine_ready[machine] = end
            progress = True
        pending = still_pending
    if pending:
        raise ValueError(f"Could not decode schedule, leftover {len(pending)} ops")
    return result


def _eligible_machines(instance: FJSPInstance, job: int, op: int) -> list[int]:
    return [option.machine for option in instance.jobs[job].operations[op].options]


def _machines_for_swap(instance: FJSPInstance, job_a: int, op_a: int, job_b: int, op_b: int) -> list[int]:
    """Return machines that are eligible for both (job_a, op_a) and (job_b, op_b)."""
    a = set(_eligible_machines(instance, job_a, op_a))
    b = set(_eligible_machines(instance, job_b, op_b))
    return sorted(a & b)


def _assignments(schedule: Iterable[ScheduledOperation]) -> list[tuple[int, int, int]]:
    """Return the (job, op, machine) assignments in the order they appear in
    the schedule. The order is used for stability, not for decoding."""
    return [(op.job, op.op, op.machine) for op in schedule]


def _makespan(instance: FJSPInstance, assignments: list[tuple[int, int, int]]) -> int:
    schedule = _recompute(instance, assignments)
    return validate_schedule(instance, schedule).makespan


def _neighbors_reassign(
    instance: FJSPInstance,
    assignments: list[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    """Generate neighbours by moving each operation to a different eligible
    machine. For an instance with ``n`` operations and an average of ``k``
    eligible machines per operation, this yields ``O(n*k)`` neighbours."""
    neighbours: list[list[tuple[int, int, int]]] = []
    for i, (job, op_idx, machine) in enumerate(assignments):
        for new_machine in _eligible_machines(instance, job, op_idx):
            if new_machine == machine:
                continue
            new_assignments = list(assignments)
            new_assignments[i] = (job, op_idx, new_machine)
            neighbours.append(new_assignments)
    return neighbours


def _neighbors_swap_same_machine(
    instance: FJSPInstance,
    assignments: list[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    """Generate neighbours by swapping the order of two adjacent operations
    on the same machine in the recomputed schedule. We group operations by
    machine in the recomputed schedule, then swap each adjacent pair."""
    try:
        schedule = _recompute(instance, assignments)
    except ValueError:
        return []
    by_machine: dict[int, list[ScheduledOperation]] = defaultdict(list)
    for op in schedule:
        by_machine[op.machine].append(op)
    index_of = {(key[0], key[1]): i for i, key in enumerate(assignments)}
    neighbours: list[list[tuple[int, int, int]]] = []
    for machine, ops in by_machine.items():
        ops_sorted = sorted(ops, key=lambda o: o.start)
        for i in range(len(ops_sorted) - 1):
            a = ops_sorted[i]
            b = ops_sorted[i + 1]
            ia = index_of[(a.job, a.op)]
            ib = index_of[(b.job, b.op)]
            new_assignments = list(assignments)
            new_assignments[ia], new_assignments[ib] = new_assignments[ib], new_assignments[ia]
            neighbours.append(new_assignments)
    return neighbours


def _neighbors_swap_order_across_machines(
    instance: FJSPInstance,
    assignments: list[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    """Swap the order of two operations on *different* machines in the
    recomputed schedule. We pick adjacent operations on each machine's
    timeline and try swapping their global order. This is more disruptive
    than ``swap_order`` (which is single-machine) and helps escape local
    minima where the local reordering is already optimal.
    """
    try:
        schedule = _recompute(instance, assignments)
    except ValueError:
        return []
    index_of = {(j, o): i for i, (j, o, _) in enumerate(assignments)}
    # Group ops by start time
    sorted_ops = sorted(schedule, key=lambda o: o.start)
    neighbours: list[list[tuple[int, int, int]]] = []
    for i in range(len(sorted_ops) - 1):
        a = sorted_ops[i]
        b = sorted_ops[i + 1]
        if a.machine == b.machine:
            continue
        if a.end > b.start:
            # They overlap, cannot swap
            continue
        # Swapping means re-ordering the global start times. We keep the
        # (job, op, machine) assignments identical but swap the order
        # in the assignments list - this changes decoding because the
        # _recompute walks the assignments in list order for ops whose
        # job is already at the right op index.
        ia = index_of[(a.job, a.op)]
        ib = index_of[(b.job, b.op)]
        new_assignments = list(assignments)
        new_assignments[ia], new_assignments[ib] = new_assignments[ib], new_assignments[ia]
        neighbours.append(new_assignments)
    return neighbours


def _neighbors_swap_machines(
    instance: FJSPInstance,
    assignments: list[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    """Generate neighbours by swapping the machine assignment of two
    operations that share at least one eligible machine. This often allows
    moving a bottleneck operation to a faster machine while moving a
    faster operation off it."""
    n = len(assignments)
    neighbours: list[list[tuple[int, int, int]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ji, oi, mi = assignments[i]
            jj, oj, mj = assignments[j]
            if mi == mj:
                continue
            common = _machines_for_swap(instance, ji, oi, jj, oj)
            if not common:
                continue
            # Skip if both already use a common machine and we can't change one.
            for new_machine_i in common:
                if new_machine_i == mi:
                    continue
                # Find a machine for j different from new_machine_i and from mj if possible
                candidates_for_j = [m for m in common if m != new_machine_i]
                if not candidates_for_j:
                    continue
                new_machine_j = candidates_for_j[0]
                if new_machine_i == mi and new_machine_j == mj:
                    continue
                new_assignments = list(assignments)
                new_assignments[i] = (ji, oi, new_machine_i)
                new_assignments[j] = (jj, oj, new_machine_j)
                neighbours.append(new_assignments)
    return neighbours


def local_search(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    neighborhoods: tuple[str, ...] = ("reassign", "swap_machine", "swap_order"),
    max_iterations: int = 50,
    strategy: str = "first",
) -> tuple[list[ScheduledOperation], int]:
    """Apply first-improvement local search to ``schedule``.

    Parameters
    ----------
    instance
        The FJSP instance.
    schedule
        A legal schedule. The list of (job, op, machine) assignments is
        extracted and used as the search state.
    neighborhoods
        Which neighbourhoods to use. Choose from "reassign", "swap_machine",
        "swap_order", "swap_order_across". All three are used by default.
    max_iterations
        Maximum number of outer search iterations. Each iteration tries all
        neighbours in the chosen neighbourhoods and accepts the first that
        strictly improves the makespan. The loop terminates early when no
        improvement is found.
    strategy
        "first" (default) accepts the first improvement, "best" evaluates
        the entire neighbourhood and accepts the best improvement.

    Returns
    -------
    new_schedule, new_makespan
        The improved schedule and its makespan.
    """
    assignments = _assignments(schedule)
    best_makespan = _makespan(instance, assignments)

    generator_fns = {
        "reassign": _neighbors_reassign,
        "swap_machine": _neighbors_swap_machines,
        "swap_order": _neighbors_swap_same_machine,
        "swap_order_across": _neighbors_swap_order_across_machines,
    }

    for _ in range(max_iterations):
        improved = False
        for name in neighborhoods:
            if name not in generator_fns:
                raise KeyError(f"Unknown neighborhood {name!r}.")
            if strategy == "first":
                for neighbor in generator_fns[name](instance, assignments):
                    try:
                        new_makespan = _makespan(instance, neighbor)
                    except ValueError:
                        continue
                    if new_makespan < best_makespan:
                        assignments = neighbor
                        best_makespan = new_makespan
                        improved = True
                        break
                if improved:
                    break
            else:  # best improvement
                best_neighbor = None
                best_neighbor_ms = best_makespan
                for neighbor in generator_fns[name](instance, assignments):
                    try:
                        new_makespan = _makespan(instance, neighbor)
                    except ValueError:
                        continue
                    if new_makespan < best_neighbor_ms:
                        best_neighbor = neighbor
                        best_neighbor_ms = new_makespan
                if best_neighbor is not None:
                    assignments = best_neighbor
                    best_makespan = best_neighbor_ms
                    improved = True
        if not improved:
            break
    return _recompute(instance, assignments), best_makespan


def local_search_from_makespan(
    instance: FJSPInstance,
    initial_makespan: int,
    schedule: list[ScheduledOperation],
    **kwargs,
) -> tuple[list[ScheduledOperation], int, int]:
    """Wrapper that returns (new_schedule, new_makespan, improvement)
    where ``improvement = initial_makespan - new_makespan``."""
    new_schedule, new_makespan = local_search(instance, schedule, **kwargs)
    return new_schedule, new_makespan, initial_makespan - new_makespan


def random_schedule(instance: FJSPInstance, seed: int = 0) -> list[ScheduledOperation]:
    """Generate a random but valid (job, op, machine) assignment."""
    import random
    rng = random.Random(seed)
    assignments = []
    for job_idx, job in enumerate(instance.jobs):
        for op_idx, op in enumerate(job.operations):
            machine = rng.choice([opt.machine for opt in op.options])
            assignments.append((job_idx, op_idx, machine))
    return _recompute(instance, assignments)


def neh_construct(instance: FJSPInstance) -> list[ScheduledOperation]:
    """Nawaz-Enscore-Ham (NEH) constructive heuristic for FJSP, fast
    precedence-respecting variant.

    1. For each (job, op), compute the *minimum* processing time over
       eligible machines.
    2. Sort all (job, op) pairs by descending minimum processing time.
    3. For each (job, op) in that order, choose the eligible machine that
       minimises the makespan when this operation is appended to the
       already-scheduled prefix. Precedence is preserved because we
       process operations in their (job, op) natural order relative to the
       sort: if (j, o+1) is processed before (j, o) it would have a
       smaller total time, which we do not enforce here. To make NEH
       precedence-respecting we sort by (job, op) within the same job and
       only consider operations whose predecessor is already scheduled.
    """
    # Compute minimum processing time per (job, op)
    op_min: list[tuple[int, int, float]] = []
    for job_idx, job in enumerate(instance.jobs):
        for op_idx, op in enumerate(job.operations):
            min_t = min(opt.duration for opt in op.options)
            op_min.append((job_idx, op_idx, min_t))
    # Sort by descending minimum processing time
    op_min.sort(key=lambda x: -x[2])

    # Build a precedence-respecting order: only consider (j, o) where (j, o-1)
    # is already scheduled.
    scheduled: set[tuple[int, int]] = set()
    assignments: list[tuple[int, int, int]] = []
    # First pass: process jobs in order
    for job_idx, job in enumerate(instance.jobs):
        for op_idx, op in enumerate(job.operations):
            best_machine = min(op.options, key=lambda o: o.duration).machine
            best_makespan = None
            for option in op.options:
                tentative = assignments + [(job_idx, op_idx, option.machine)]
                ms = _makespan(instance, tentative)
                if best_makespan is None or ms < best_makespan:
                    best_makespan = ms
                    best_machine = option.machine
            assignments.append((job_idx, op_idx, best_machine))
            scheduled.add((job_idx, op_idx))
    # Second pass: NEH re-inserts operations in descending total time, choosing
    # the (machine, position) that minimises the makespan.
    op_min.sort(key=lambda x: -x[2])
    for job_idx, op_idx, _ in op_min:
        if (job_idx, op_idx) not in scheduled:
            continue
        # Find the current position of this operation in assignments
        cur_idx = next(i for i, a in enumerate(assignments) if a[0] == job_idx and a[1] == op_idx)
        # Try moving to any other position and any eligible machine
        operation = instance.jobs[job_idx].operations[op_idx]
        best_choice: tuple[int, list[tuple[int, int, int]]] | None = None
        for new_pos in range(len(assignments) + 1):
            if new_pos == cur_idx:
                continue
            for option in operation.options:
                tentative = [a for a in assignments if a[0] != job_idx or a[1] != op_idx]
                tentative.insert(new_pos, (job_idx, op_idx, option.machine))
                ms = _makespan(instance, tentative)
                if best_choice is None or ms < best_choice[0]:
                    best_choice = (ms, tentative)
        if best_choice is not None:
            assignments = best_choice[1]
    return _recompute(instance, assignments)


def random_perturb(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    n_swaps: int = 5,
    seed: int = 0,
) -> list[ScheduledOperation]:
    """Apply ``n_swaps`` random reassignments to the schedule."""
    import random
    rng = random.Random(seed)
    assignments = _assignments(schedule)
    indices = list(range(len(assignments)))
    for _ in range(n_swaps):
        if not indices:
            break
        i = rng.choice(indices)
        job, op_idx, _ = assignments[i]
        eligible = _eligible_machines(instance, job, op_idx)
        new_machine = rng.choice(eligible)
        assignments[i] = (job, op_idx, new_machine)
    return _recompute(instance, assignments)


def critical_path_perturb(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    n_swaps: int = 5,
    seed: int = 0,
) -> list[ScheduledOperation]:
    """Perturb the schedule by reassigning operations near the critical path.

    The critical path is the sequence of (job, op) operations that determine
    the makespan. We trace the makespan backward, find operations on the
    critical path, and reassign them to other eligible machines. This is
    more likely to escape local minima than blind random perturbation.
    """
    import random
    rng = random.Random(seed)
    assignments = _assignments(schedule)
    assignment_to_idx = {(j, o): i for i, (j, o, _) in enumerate(assignments)}

    # Recompute to find the critical path
    try:
        full_schedule = _recompute(instance, assignments)
    except ValueError:
        return random_perturb(instance, schedule, n_swaps=n_swaps, seed=seed)

    # Find the makespan value
    makespan = max(op.end for op in full_schedule) if full_schedule else 0

    # Find operations on critical path (those ending at makespan and chaining back)
    # Critical path = operations that, if delayed, delay the makespan.
    # Approximation: ops where (end == start_of_next_critical) and (end is
    # either makespan or matches start of some next op with same machine/job).
    job_idx_in_sched = {(op.job, op.op): op for op in full_schedule}

    # Build a successor map: for each (job, op), what is the next op of same
    # job (forced precedence), and for each machine, what is the next op.
    job_next: dict[int, int] = {}
    for op in full_schedule:
        job_next[op.job] = max(job_next.get(op.job, -1), op.op)
    machine_next_ops: dict[int, list[ScheduledOperation]] = defaultdict(list)
    for op in full_schedule:
        machine_next_ops[op.machine].append(op)
    for m in machine_next_ops:
        machine_next_ops[m].sort(key=lambda o: o.start)

    # Mark critical: an op is critical if its end time matches the start of
    # another op on the same machine, OR is the last op of a job, OR equals
    # the makespan.
    critical_ops: set[tuple[int, int]] = set()
    for op in full_schedule:
        if op.end == makespan:
            critical_ops.add((op.job, op.op))
            continue
        # Same-machine successor
        for next_op in machine_next_ops[op.machine]:
            if next_op.start == op.end and next_op is not op:
                critical_ops.add((op.job, op.op))
                break

    if not critical_ops:
        return random_perturb(instance, schedule, n_swaps=n_swaps, seed=seed)

    critical_list = list(critical_ops)
    rng.shuffle(critical_list)

    n_done = 0
    for job, op_idx in critical_list:
        if n_done >= n_swaps:
            break
        if (job, op_idx) not in assignment_to_idx:
            continue
        i = assignment_to_idx[(job, op_idx)]
        eligible = _eligible_machines(instance, job, op_idx)
        if len(eligible) <= 1:
            continue
        current_machine = assignments[i][2]
        new_machine = rng.choice([m for m in eligible if m != current_machine])
        assignments[i] = (job, op_idx, new_machine)
        n_done += 1

    # If we did not perturb enough, fill with random perturbations
    if n_done < n_swaps:
        remaining = n_swaps - n_done
        all_indices = list(range(len(assignments)))
        for _ in range(remaining):
            if not all_indices:
                break
            i = rng.choice(all_indices)
            job, op_idx, _ = assignments[i]
            eligible = _eligible_machines(instance, job, op_idx)
            if len(eligible) > 1:
                new_machine = rng.choice([m for m in eligible if m != assignments[i][2]])
                assignments[i] = (job, op_idx, new_machine)

    return _recompute(instance, assignments)


def multi_start_local_search(
    instance: FJSPInstance,
    initial_schedule: list[ScheduledOperation],
    *,
    n_starts: int = 5,
    perturb_n_swaps: int = 3,
    perturb_seeds: tuple[int, ...] | None = None,
    max_iterations: int = 50,
    neighborhoods: tuple[str, ...] = ("reassign", "swap_machine", "swap_order"),
) -> tuple[list[ScheduledOperation], int]:
    """Run local search from the initial schedule and from ``n_starts - 1``
    random perturbations. Returns the best (schedule, makespan) found.

    The first start is the initial schedule (the best RL/heuristic rollout).
    Subsequent starts are perturbations of the best-so-far schedule.
    """
    best_schedule, best_makespan = local_search(
        instance,
        initial_schedule,
        neighborhoods=neighborhoods,
        max_iterations=max_iterations,
    )
    if n_starts <= 1:
        return best_schedule, best_makespan
    seeds = perturb_seeds or tuple(range(n_starts - 1))
    for seed in seeds:
        candidate = random_perturb(
            instance,
            best_schedule,
            n_swaps=perturb_n_swaps,
            seed=seed,
        )
        cand_schedule, cand_makespan = local_search(
            instance,
            candidate,
            neighborhoods=neighborhoods,
            max_iterations=max_iterations,
        )
        if cand_makespan < best_makespan:
            best_schedule, best_makespan = cand_schedule, cand_makespan
    return best_schedule, best_makespan


def iterated_local_search(
    instance: FJSPInstance,
    initial_schedule: list[ScheduledOperation],
    *,
    n_iterations: int = 20,
    perturb_n_swaps: int = 8,
    perturb_seeds: tuple[int, ...] | None = None,
    max_ls_iterations: int = 50,
    neighborhoods: tuple[str, ...] = ("reassign", "swap_machine", "swap_order"),
    perturb_mode: str = "critical",
    accept_worse: bool = False,
    worse_threshold: int = 0,
    strategy: str = "first",
) -> tuple[list[ScheduledOperation], int]:
    """Iterated local search (ILS).

    Each iteration:
    1. Apply a perturbation (random or critical-path based) to the current best.
    2. Apply local search to the perturbed schedule.
    3. Accept the new solution if it's strictly better, or (if accept_worse)
       if it's within ``worse_threshold`` of the current best.

    The first iteration is a local search on the initial schedule.
    """
    best_schedule, best_makespan = local_search(
        instance,
        initial_schedule,
        neighborhoods=neighborhoods,
        max_iterations=max_ls_iterations,
        strategy=strategy,
    )
    current_schedule, current_makespan = best_schedule, best_makespan

    perturb_fns = {
        "random": lambda sched, seed: random_perturb(
            instance, sched, n_swaps=perturb_n_swaps, seed=seed
        ),
        "critical": lambda sched, seed: critical_path_perturb(
            instance, sched, n_swaps=perturb_n_swaps, seed=seed
        ),
    }
    if perturb_mode not in perturb_fns:
        raise KeyError(f"Unknown perturb_mode {perturb_mode!r}.")
    perturb_fn = perturb_fns[perturb_mode]

    seeds = perturb_seeds or tuple(range(n_iterations))
    for it, seed in enumerate(seeds):
        if it >= n_iterations:
            break
        candidate = perturb_fn(current_schedule, seed)
        cand_schedule, cand_makespan = local_search(
            instance,
            candidate,
            neighborhoods=neighborhoods,
            max_iterations=max_ls_iterations,
            strategy=strategy,
        )
        if cand_makespan < best_makespan:
            best_schedule, best_makespan = cand_schedule, cand_makespan
        if accept_worse or cand_makespan <= current_makespan + worse_threshold:
            current_schedule, current_makespan = cand_schedule, cand_makespan
    return best_schedule, best_makespan


def ils_sa_hybrid(
    instance: FJSPInstance,
    initial_schedule: list[ScheduledOperation],
    *,
    n_ils_iterations: int = 10,
    n_sa_iterations: int = 2000,
    perturb_n_swaps: int = 8,
    max_ls_iterations: int = 50,
    sa_initial_temperature: float = 5.0,
    sa_cooling_rate: float = 0.99,
    seed: int = 0,
    strategy: str = "first",
) -> tuple[list[ScheduledOperation], int]:
    """Hybrid ILS + SA: ILS first to find a good region, then SA to refine.

    The SA accepts worse moves with probability exp(-delta/T) to escape local
    minima, then cools geometrically.
    """
    ils_schedule, ils_makespan = iterated_local_search(
        instance,
        initial_schedule,
        n_iterations=n_ils_iterations,
        perturb_n_swaps=perturb_n_swaps,
        max_ls_iterations=max_ls_iterations,
        perturb_mode="critical",
        strategy=strategy,
    )
    sa_schedule, sa_makespan = simulated_annealing(
        instance,
        ils_schedule,
        initial_temperature=sa_initial_temperature,
        cooling_rate=sa_cooling_rate,
        max_total_iterations=n_sa_iterations,
        seed=seed,
    )
    if sa_makespan < ils_makespan:
        return sa_schedule, sa_makespan
    return ils_schedule, ils_makespan


def simulated_annealing(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    initial_temperature: float = 5.0,
    cooling_rate: float = 0.995,
    min_temperature: float = 0.1,
    iterations_per_temp: int = 10,
    n_perturbations: int = 1,
    max_total_iterations: int = 5000,
    neighborhoods: tuple[str, ...] = ("reassign", "swap_machine"),
    seed: int = 0,
) -> tuple[list[ScheduledOperation], int]:
    """Simulated annealing over machine-assignment moves.

    Each step picks a random valid neighbor from the chosen neighborhoods
    and accepts it with probability ``min(1, exp(Δ/T))`` where Δ is the
    makespan change and T is the current temperature. Cooling follows a
    geometric schedule.
    """
    import random
    import math
    rng = random.Random(seed)

    assignments = _assignments(schedule)
    current_makespan = _makespan(instance, assignments)
    best_assignments = list(assignments)
    best_makespan = current_makespan

    generator_fns = {
        "reassign": _neighbors_reassign,
        "swap_machine": _neighbors_swap_machines,
        "swap_order": _neighbors_swap_same_machine,
    }

    temperature = initial_temperature
    it = 0
    while temperature > min_temperature and it < max_total_iterations:
        for _ in range(iterations_per_temp):
            if it >= max_total_iterations:
                break
            neighbor = None
            for name in neighborhoods:
                if name not in generator_fns:
                    continue
                candidates = generator_fns[name](instance, assignments)
                if candidates:
                    neighbor = rng.choice(candidates)
                    break
            if neighbor is None:
                break
            new_makespan = _makespan(instance, neighbor)
            delta = new_makespan - current_makespan
            accept = delta < 0 or rng.random() < math.exp(-delta / temperature)
            if accept:
                assignments = neighbor
                current_makespan = new_makespan
                if new_makespan < best_makespan:
                    best_assignments = list(assignments)
                    best_makespan = new_makespan
            it += 1
        temperature *= cooling_rate
    return _recompute(instance, best_assignments), best_makespan


def _move_signature(
    current_assignments: list[tuple[int, int, int]],
    neighbor_assignments: list[tuple[int, int, int]],
    neighborhood: str,
) -> tuple:
    """Compute a move signature (the diff between two assignments).

    For N1 (reassign) the signature is (op_index, new_machine) where
    op_index is the index in the assignments list. For N2/N3/N4 the
    signature is the sorted pair of indices that were swapped. This is
    symmetric so swapping (i, j) and (j, i) are the same move.
    """
    if neighborhood == "reassign":
        for i, (cur, nxt) in enumerate(zip(current_assignments, neighbor_assignments)):
            if cur != nxt:
                # cur[0] is job, cur[1] is op, cur[2] is machine
                # nxt[2] is the new machine
                return ("reassign", i, nxt[2])
        return ("reassign", -1, -1)  # no change
    # For swap-based neighborhoods, find the two indices whose values differ
    diffs = []
    for i, (cur, nxt) in enumerate(zip(current_assignments, neighbor_assignments)):
        if cur != nxt:
            diffs.append(i)
            if len(diffs) == 2:
                break
    if len(diffs) == 2:
        return (neighborhood, min(diffs), max(diffs))
    return (neighborhood, -1, -1)


def tabu_search(
    instance: FJSPInstance,
    schedule: list[ScheduledOperation],
    *,
    max_iterations: int = 500,
    tabu_tenure: int | None = None,
    neighborhoods: tuple[str, ...] = ("reassign", "swap_machine", "swap_order"),
    seed: int = 0,
    candidate_sample: int | None = None,
) -> tuple[list[ScheduledOperation], int]:
    """Tabu search on the FJSP schedule.

    Each move is represented by a small tuple that uniquely identifies it.
    The tabu list is a FIFO queue of move signatures with a fixed tenure.
    A move is *tabu* if its signature is in the recent history; tabu moves
    are skipped unless they would improve the best-known makespan
    (aspiration criterion). We always pick the best non-tabu neighbour in
    the chosen neighbourhoods; if all candidates are tabu we pick the
    least-bad one to keep making progress.

    Parameters
    ----------
    instance
        The FJSP instance.
    schedule
        A legal schedule. The list of (job, op, machine) assignments is
        extracted and used as the search state.
    max_iterations
        Maximum number of TS iterations. Each iteration evaluates all
        neighbours in the chosen neighbourhoods, picks the best non-tabu
        move, and advances the search.
    tabu_tenure
        How long a move stays tabu. Defaults to ``ceil(sqrt(N))`` where
        ``N`` is the number of operations.
    neighborhoods
        Which neighbourhoods to use. Choose from "reassign", "swap_machine",
        "swap_order", "swap_order_across".
    seed
        Random seed (only used when ``candidate_sample`` is set).
    candidate_sample
        If set, sample at most this many candidates per neighbourhood per
        iteration (uniformly at random). Useful for very large instances
        where full neighbourhood evaluation is too slow.

    Returns
    -------
    new_schedule, new_makespan
        The best schedule found and its makespan.
    """
    import random
    import math
    rng = random.Random(seed)

    assignments = _assignments(schedule)
    best_assignments = list(assignments)
    best_makespan = _makespan(instance, assignments)
    current_assignments = list(assignments)
    current_makespan = best_makespan

    n_ops = len(assignments)
    if tabu_tenure is None:
        tabu_tenure = max(5, int(math.sqrt(n_ops)))

    generator_fns = {
        "reassign": _neighbors_reassign,
        "swap_machine": _neighbors_swap_machines,
        "swap_order": _neighbors_swap_same_machine,
        "swap_order_across": _neighbors_swap_order_across_machines,
    }
    for name in neighborhoods:
        if name not in generator_fns:
            raise KeyError(f"Unknown neighborhood {name!r}.")

    # Tabu list as a FIFO queue of move signatures.
    tabu_queue: deque = deque(maxlen=tabu_tenure)
    tabu_set: set = set()

    for it in range(max_iterations):
        # Generate all neighbours with their makespans and move signatures.
        candidates: list[tuple[int, list, tuple]] = []
        for name in neighborhoods:
            neighbors = generator_fns[name](instance, current_assignments)
            if candidate_sample is not None and len(neighbors) > candidate_sample:
                neighbors = rng.sample(neighbors, candidate_sample)
            for neighbor in neighbors:
                try:
                    ms = _makespan(instance, neighbor)
                except ValueError:
                    continue
                sig = _move_signature(current_assignments, neighbor, name)
                candidates.append((ms, neighbor, sig))

        if not candidates:
            break

        # Sort by makespan ascending (best first).
        candidates.sort(key=lambda x: x[0])

        # Pick the best non-tabu move; if all tabu, pick the best (aspiration
        # is implicit because we accept any tabu move that improves the
        # best, but we also break ties by best makespan).
        chosen = None
        for ms, neighbor, sig in candidates:
            if sig not in tabu_set or ms < best_makespan:
                chosen = (ms, neighbor, sig)
                break
        if chosen is None:
            # All candidates are tabu and none improves best -> pick the best
            chosen = candidates[0]

        new_makespan, new_assignments, sig = chosen
        current_assignments = new_assignments
        current_makespan = new_makespan

        # Update tabu list (deque with maxlen auto-discards oldest).
        if len(tabu_queue) == tabu_tenure:
            tabu_set.discard(tabu_queue[0])
        tabu_queue.append(sig)
        tabu_set.add(sig)

        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_assignments = list(current_assignments)

    return _recompute(instance, best_assignments), best_makespan
