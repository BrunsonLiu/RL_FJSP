from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fjsp.parser.fjs_parser import FJSPInstance


@dataclass(frozen=True)
class ScheduledOperation:
    job: int
    op: int
    machine: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    makespan: int
    errors: tuple[str, ...]


def schedule_from_dict(payload: dict[str, Any]) -> list[ScheduledOperation]:
    raw_ops = payload.get("operations")
    if not isinstance(raw_ops, list):
        raise ValueError("Schedule JSON must contain an 'operations' list.")

    operations: list[ScheduledOperation] = []
    for idx, item in enumerate(raw_ops):
        if not isinstance(item, dict):
            raise ValueError(f"operations[{idx}] must be an object.")
        try:
            operations.append(
                ScheduledOperation(
                    job=int(item["job"]),
                    op=int(item["op"]),
                    machine=int(item["machine"]),
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"operations[{idx}] is missing field {exc.args[0]!r}.") from exc
    return operations


def validate_schedule(instance: FJSPInstance, schedule: list[ScheduledOperation]) -> ValidationResult:
    errors: list[str] = []
    expected = {
        (job_idx, op_idx)
        for job_idx, job in enumerate(instance.jobs)
        for op_idx, _ in enumerate(job.operations)
    }
    seen: dict[tuple[int, int], ScheduledOperation] = {}

    for item in schedule:
        key = (item.job, item.op)
        if key in seen:
            errors.append(f"Duplicate operation job={item.job} op={item.op}.")
            continue
        seen[key] = item

        if item.start < 0 or item.end <= item.start:
            errors.append(f"Invalid time window for job={item.job} op={item.op}: {item.start}->{item.end}.")
            continue

        if item.job < 0 or item.job >= instance.job_count:
            errors.append(f"Unknown job index {item.job}.")
            continue
        job = instance.jobs[item.job]
        if item.op < 0 or item.op >= len(job.operations):
            errors.append(f"Unknown operation index job={item.job} op={item.op}.")
            continue
        if item.machine < 0 or item.machine >= instance.machine_count:
            errors.append(f"Unknown machine index {item.machine}.")
            continue

        options = job.operations[item.op].options
        option_duration = {option.machine: option.duration for option in options}
        expected_duration = option_duration.get(item.machine)
        if expected_duration is None:
            errors.append(f"job={item.job} op={item.op} cannot run on machine={item.machine}.")
        elif item.duration != expected_duration:
            errors.append(
                f"job={item.job} op={item.op} on machine={item.machine} has duration {item.duration}, "
                f"expected {expected_duration}."
            )

    missing = expected.difference(seen)
    for job, op in sorted(missing):
        errors.append(f"Missing operation job={job} op={op}.")

    extra = set(seen).difference(expected)
    for job, op in sorted(extra):
        errors.append(f"Unexpected operation job={job} op={op}.")

    for job_idx, job in enumerate(instance.jobs):
        for op_idx in range(1, len(job.operations)):
            prev = seen.get((job_idx, op_idx - 1))
            current = seen.get((job_idx, op_idx))
            if prev is not None and current is not None and current.start < prev.end:
                errors.append(
                    f"Precedence violation job={job_idx}: op={op_idx} starts at {current.start} "
                    f"before op={op_idx - 1} ends at {prev.end}."
                )

    by_machine: dict[int, list[ScheduledOperation]] = {}
    for item in seen.values():
        by_machine.setdefault(item.machine, []).append(item)

    for machine, items in by_machine.items():
        items.sort(key=lambda op: (op.start, op.end, op.job, op.op))
        for left, right in zip(items, items[1:]):
            if right.start < left.end:
                errors.append(
                    f"Machine overlap on machine={machine}: "
                    f"job={left.job} op={left.op} [{left.start},{left.end}) and "
                    f"job={right.job} op={right.op} [{right.start},{right.end})."
                )

    makespan = max((item.end for item in schedule), default=0)
    return ValidationResult(is_valid=not errors, makespan=makespan, errors=tuple(errors))


def greedy_schedule(instance: FJSPInstance) -> list[ScheduledOperation]:
    """Simple deterministic baseline: schedule operations job-by-job at earliest feasible time."""
    machine_ready = [0 for _ in range(instance.machine_count)]
    job_ready = [0 for _ in range(instance.job_count)]
    schedule: list[ScheduledOperation] = []

    for job_idx, job in enumerate(instance.jobs):
        for op_idx, operation in enumerate(job.operations):
            candidates: list[tuple[int, int, int, int]] = []
            for option in operation.options:
                start = max(job_ready[job_idx], machine_ready[option.machine])
                end = start + option.duration
                candidates.append((end, start, option.machine, option.duration))
            end, start, machine, _ = min(candidates)
            schedule.append(ScheduledOperation(job=job_idx, op=op_idx, machine=machine, start=start, end=end))
            job_ready[job_idx] = end
            machine_ready[machine] = end

    return schedule


def schedule_to_dict(schedule: list[ScheduledOperation]) -> dict[str, list[dict[str, int]]]:
    return {
        "operations": [
            {
                "job": item.job,
                "op": item.op,
                "machine": item.machine,
                "start": item.start,
                "end": item.end,
            }
            for item in schedule
        ]
    }

