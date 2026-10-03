"""Compute the optimal makespan of an FJSP instance with OR-Tools CP-SAT.

Usage:
    python scripts/ortools_makespan.py data/instances/brandimarte/mk01.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  -- sets sys.path
from ortools.sat.python import cp_model

from fjsp.parser.fjs_parser import parse_fjs


def solve_makespan(instance_path: str | Path, time_limit_s: float = 30.0) -> tuple[int, bool] | None:
    """Returns (makespan, is_optimal) or None if no solution found."""
    instance = parse_fjs(instance_path)
    model = cp_model.CpModel()

    horizon = 0
    for job in instance.jobs:
        for op in job.operations:
            horizon += max(option.duration for option in op.options)

    # Variables
    starts: dict[tuple[int, int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int, int], cp_model.IntVar] = {}
    machine_assign: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    op_intervals: dict[tuple[int, int, int], cp_model.IntervalVar] = {}
    machine_intervals: dict[int, list[cp_model.IntervalVar]] = {
        m: [] for m in range(instance.machine_count)
    }

    for job_idx, job in enumerate(instance.jobs):
        for op_idx, op in enumerate(job.operations):
            for opt_idx, option in enumerate(op.options):
                suffix = (job_idx, op_idx, opt_idx)
                suffix_str = f"_{job_idx}_{op_idx}_{opt_idx}"
                start = model.NewIntVar(0, horizon, f"start{suffix_str}")
                end = model.NewIntVar(0, horizon, f"end{suffix_str}")
                interval = model.NewIntervalVar(start, option.duration, end, f"interval{suffix_str}")
                starts[suffix] = start
                ends[suffix] = end
                op_intervals[suffix] = interval
                machine_assign[suffix] = model.NewBoolVar(f"assigned{suffix_str}")
                machine_intervals[option.machine].append(interval)

    # Exactly one option per operation
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(len(job.operations)):
            opts = [machine_assign[(job_idx, op_idx, o)] for o in range(len(job.operations[op_idx].options))]
            model.AddExactlyOne(opts)

    # Precedence within a job
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            # End of op-1 <= start of op
            # The end is option-dependent; we need a per-op end variable
            pass  # handled below

    # Define an "end_of_op" for each op = end of whichever option was chosen
    op_starts: dict[tuple[int, int], cp_model.IntVar] = {}
    op_ends: dict[tuple[int, int], cp_model.IntVar] = {}
    op_durations: dict[tuple[int, int], cp_model.IntVar] = {}
    for job_idx, job in enumerate(instance.jobs):
        for op_idx, op in enumerate(job.operations):
            key = (job_idx, op_idx)
            s = model.NewIntVar(0, horizon, f"op_start_{job_idx}_{op_idx}")
            e = model.NewIntVar(0, horizon, f"op_end_{job_idx}_{op_idx}")
            d = model.NewIntVar(0, horizon, f"op_dur_{job_idx}_{op_idx}")
            op_starts[key] = s
            op_ends[key] = e
            op_durations[key] = d
            for opt_idx, option in enumerate(op.options):
                suffix = (job_idx, op_idx, opt_idx)
                # When this option is selected: s = start, d = duration, e = end
                model.Add(starts[suffix] == s).OnlyEnforceIf(machine_assign[suffix])
                model.Add(ends[suffix] == e).OnlyEnforceIf(machine_assign[suffix])
                model.Add(option.duration == d).OnlyEnforceIf(machine_assign[suffix])

    # Precedence
    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            model.Add(op_ends[(job_idx, op_idx - 1)] <= op_starts[(job_idx, op_idx)])

    # No overlap on a machine
    for machine, intervals in machine_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # Minimize makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    for job_idx, job in enumerate(instance.jobs):
        last_op = len(job.operations) - 1
        model.Add(makespan >= op_ends[(job_idx, last_op)])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        is_optimal = status == cp_model.OPTIMAL
        return int(solver.Value(makespan)), is_optimal
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = solve_makespan(args.instance, time_limit_s=args.time_limit_s)
    elapsed = time.perf_counter() - t0
    if result is None:
        print(f"NO_SOLUTION: {args.instance} ({elapsed:.2f}s)", file=sys.stderr)
        sys.exit(1)
    makespan, is_optimal = result
    status_str = "OPTIMAL" if is_optimal else "FEASIBLE (time limit)"
    print(f"{status_str}: {args.instance} makespan={makespan} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
