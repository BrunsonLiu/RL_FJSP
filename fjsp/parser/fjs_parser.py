from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperationOption:
    machine: int
    duration: int


@dataclass(frozen=True)
class Operation:
    options: tuple[OperationOption, ...]


@dataclass(frozen=True)
class Job:
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class FJSPInstance:
    jobs: tuple[Job, ...]
    machine_count: int

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def operation_count(self) -> int:
        return sum(len(job.operations) for job in self.jobs)


def _meaningful_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def parse_fjs(path: str | Path) -> FJSPInstance:
    """Parse a common FJSP instance.

    Public benchmark files may use either 0-based or 1-based machine ids.
    Returned objects always use 0-based ids.
    """
    instance_path = Path(path)
    lines = _meaningful_lines(instance_path)
    if not lines:
        raise ValueError(f"Empty instance file: {instance_path}")

    header = [int(token) for token in lines[0].split()]
    if len(header) < 2:
        raise ValueError("First non-comment line must contain at least jobs and machines.")

    job_count, machine_count = header[0], header[1]
    if job_count <= 0 or machine_count <= 0:
        raise ValueError("Job count and machine count must be positive.")
    if len(lines) - 1 < job_count:
        raise ValueError(f"Expected {job_count} job lines, found {len(lines) - 1}.")

    parsed_jobs: list[list[list[tuple[int, int]]]] = []
    machine_ids: list[int] = []
    for job_idx in range(job_count):
        values = [int(token) for token in lines[job_idx + 1].split()]
        if not values:
            raise ValueError(f"Job {job_idx} line is empty.")

        cursor = 0
        op_count = values[cursor]
        cursor += 1
        if op_count <= 0:
            raise ValueError(f"Job {job_idx} must contain at least one operation.")

        operations: list[list[tuple[int, int]]] = []
        for op_idx in range(op_count):
            if cursor >= len(values):
                raise ValueError(f"Job {job_idx} operation {op_idx} is missing option count.")
            option_count = values[cursor]
            cursor += 1
            if option_count <= 0:
                raise ValueError(f"Job {job_idx} operation {op_idx} has no machine options.")

            options: list[tuple[int, int]] = []
            for _ in range(option_count):
                if cursor + 1 >= len(values):
                    raise ValueError(f"Job {job_idx} operation {op_idx} has an incomplete option pair.")
                machine_id = values[cursor]
                duration = values[cursor + 1]
                cursor += 2

                if duration <= 0:
                    raise ValueError(f"Processing time must be positive, got {duration}.")
                machine_ids.append(machine_id)
                options.append((machine_id, duration))

            operations.append(options)

        if cursor != len(values):
            extra = " ".join(str(value) for value in values[cursor:])
            raise ValueError(f"Job {job_idx} has trailing tokens: {extra}")
        parsed_jobs.append(operations)

    machine_base = 0 if 0 in machine_ids else 1
    jobs: list[Job] = []
    for job_idx, parsed_operations in enumerate(parsed_jobs):
        operations: list[Operation] = []
        for op_idx, parsed_options in enumerate(parsed_operations):
            options: list[OperationOption] = []
            for machine_id, duration in parsed_options:
                machine = machine_id - machine_base
                if machine < 0 or machine >= machine_count:
                    valid = f"0..{machine_count - 1}" if machine_base == 0 else f"1..{machine_count}"
                    raise ValueError(
                        f"Job {job_idx} operation {op_idx} references machine {machine_id}, "
                        f"but valid ids are {valid}."
                    )
                options.append(OperationOption(machine=machine, duration=duration))
            operations.append(Operation(options=tuple(options)))
        jobs.append(Job(operations=tuple(operations)))

    return FJSPInstance(jobs=tuple(jobs), machine_count=machine_count)


def summarize_instance(instance: FJSPInstance) -> str:
    return (
        f"jobs={instance.job_count}, machines={instance.machine_count}, "
        f"operations={instance.operation_count}"
    )
