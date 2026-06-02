from __future__ import annotations

from fjsp.env import FJSPDispatchEnv


def scale_value(value: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(value) / float(denominator)


def instance_time_scale(env: FJSPDispatchEnv) -> int:
    total = 0
    for job in env.instance.jobs:
        for operation in job.operations:
            total += min(option.duration for option in operation.options)
    return max(total, 1)

