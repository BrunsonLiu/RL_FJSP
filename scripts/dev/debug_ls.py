"""Debug test for local search."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from fjsp.scheduler.local_search import _recompute, _assignments, _neighbors_swap_same_machine

env = FJSPDispatchEnv.from_file(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))

env.reset()
schedule = []
while not env.done:
    action = choose_earliest_finish(env)
    _, _, _, info = env.step(action)
    schedule.append(info['scheduled'])

print(f"EF schedule length: {len(schedule)}")
print(f"EF schedule unique (job, op): {len(set((op.job, op.op) for op in schedule))}")

assignments = _assignments(schedule)
print(f"Assignments length: {len(assignments)}")
print(f"Assignments unique (job, op): {len(set((a[0], a[1]) for a in assignments))}")

recomputed = _recompute(env.instance, assignments)
print(f"Recomputed length: {len(recomputed)}")
print(f"Recomputed unique (job, op): {len(set((op.job, op.op) for op in recomputed))}")

# Find (job, op) in recomputed but not in assignments
a_keys = set((a[0], a[1]) for a in assignments)
r_keys = set((op.job, op.op) for op in recomputed)
print(f"In recomputed but not in assignments: {r_keys - a_keys}")
print(f"In assignments but not in recomputed: {a_keys - r_keys}")
