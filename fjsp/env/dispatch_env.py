from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal, TypedDict

from fjsp.parser.fjs_parser import FJSPInstance, parse_fjs
from fjsp.scheduler.validator import ScheduledOperation, ValidationResult, validate_schedule


RewardMode = Literal["makespan_delta", "zero_until_done"]


@dataclass(frozen=True)
class DispatchAction:
    """Schedule the next unscheduled operation of a job on a chosen machine."""

    job: int
    machine: int


class Observation(TypedDict):
    job_next_op: list[int]
    job_ready_time: list[int]
    machine_ready_time: list[int]
    remaining_operations: int
    makespan: int
    valid_actions: list[tuple[int, int]]


class FJSPDispatchEnv:
    """Deterministic FJSP dispatch environment.

    The action chooses a job and an eligible machine for that job's next operation.
    The environment decodes the action by scheduling the operation at its earliest
    feasible start time.
    """

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        invalid_action_penalty: float = -100.0,
        reward_mode: RewardMode = "makespan_delta",
    ) -> None:
        self.instance = instance
        self.invalid_action_penalty = invalid_action_penalty
        self.reward_mode = reward_mode
        self.reset()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        invalid_action_penalty: float = -100.0,
        reward_mode: RewardMode = "makespan_delta",
    ) -> "FJSPDispatchEnv":
        return cls(
            parse_fjs(path),
            invalid_action_penalty=invalid_action_penalty,
            reward_mode=reward_mode,
        )

    @property
    def done(self) -> bool:
        return self.remaining_operations == 0

    @property
    def makespan(self) -> int:
        return max(self.machine_ready_time, default=0)

    @property
    def schedule(self) -> list[ScheduledOperation]:
        return list(self._schedule)

    def reset(self) -> Observation:
        self.job_next_op = [0 for _ in self.instance.jobs]
        self.job_ready_time = [0 for _ in self.instance.jobs]
        self.machine_ready_time = [0 for _ in range(self.instance.machine_count)]
        self.remaining_operations = self.instance.operation_count
        self._schedule: list[ScheduledOperation] = []
        return self.observe()

    def observe(self) -> Observation:
        return {
            "job_next_op": list(self.job_next_op),
            "job_ready_time": list(self.job_ready_time),
            "machine_ready_time": list(self.machine_ready_time),
            "remaining_operations": self.remaining_operations,
            "makespan": self.makespan,
            "valid_actions": [(action.job, action.machine) for action in self.available_actions()],
        }

    def available_actions(self) -> list[DispatchAction]:
        if self.done:
            return []

        actions: list[DispatchAction] = []
        for job_idx, job in enumerate(self.instance.jobs):
            op_idx = self.job_next_op[job_idx]
            if op_idx >= len(job.operations):
                continue
            for option in job.operations[op_idx].options:
                actions.append(DispatchAction(job=job_idx, machine=option.machine))
        return actions

    def valid_action_mask(self) -> list[list[bool]]:
        mask = [[False for _ in range(self.instance.machine_count)] for _ in self.instance.jobs]
        for action in self.available_actions():
            mask[action.job][action.machine] = True
        return mask

    def is_valid_action(self, action: DispatchAction) -> bool:
        return any(action == valid for valid in self.available_actions())

    def step(self, action: DispatchAction | tuple[int, int]) -> tuple[Observation, float, bool, dict[str, object]]:
        if not isinstance(action, DispatchAction):
            action = DispatchAction(job=int(action[0]), machine=int(action[1]))

        if self.done:
            return self.observe(), 0.0, True, {"error": "episode already finished"}

        if not self.is_valid_action(action):
            info = {"invalid_action": True, "action": (action.job, action.machine)}
            return self.observe(), self.invalid_action_penalty, False, info

        old_makespan = self.makespan
        op_idx = self.job_next_op[action.job]
        operation = self.instance.jobs[action.job].operations[op_idx]
        duration = next(option.duration for option in operation.options if option.machine == action.machine)

        start = max(self.job_ready_time[action.job], self.machine_ready_time[action.machine])
        end = start + duration
        scheduled = ScheduledOperation(
            job=action.job,
            op=op_idx,
            machine=action.machine,
            start=start,
            end=end,
        )
        self._schedule.append(scheduled)

        self.job_next_op[action.job] += 1
        self.job_ready_time[action.job] = end
        self.machine_ready_time[action.machine] = end
        self.remaining_operations -= 1

        reward = self._reward(old_makespan=old_makespan, new_makespan=self.makespan, done=self.done)
        info: dict[str, object] = {"scheduled": scheduled}
        if self.done:
            info["validation"] = self.validate()
        return self.observe(), reward, self.done, info

    def validate(self) -> ValidationResult:
        return validate_schedule(self.instance, self._schedule)

    def sample_valid_action(self, rng: Random | None = None) -> DispatchAction:
        actions = self.available_actions()
        if not actions:
            raise ValueError("No valid actions available.")
        random_source = rng if rng is not None else Random()
        return random_source.choice(actions)

    def _reward(self, *, old_makespan: int, new_makespan: int, done: bool) -> float:
        if self.reward_mode == "zero_until_done":
            return float(-new_makespan if done else 0.0)
        return float(-(new_makespan - old_makespan))

