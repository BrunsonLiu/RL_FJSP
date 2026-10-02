"""Test NEH heuristic + LS on Brandimarte instances.

Compares EF, NEH, REINFORCE, and NEH+LS to see if NEH gives a better
starting point for LS than REINFORCE.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish, choose_earliest_finish
from fjsp.scheduler.local_search import local_search, neh_construct, multi_start_local_search

ROOT = Path(__file__).resolve().parents[2] / "data" / "instances" / "brandimarte"
INSTANCES = [1, 4, 9, 10, 12, 15]

print(f"{'inst':<6} {'EF':>5} {'NEH':>5} {'NEH+LS':>7} {'time':>6}")

with open(ROOT.parent / 'instances.json', 'r') as f:
    meta = json.load(f)
lit_index = {}
for inst_data in meta:
    if not inst_data.get('path', '').startswith('brandimarte/'):
        continue
    name = inst_data['name']
    bounds = inst_data.get('bounds') or {}
    lit_index[name] = {
        'optimum': inst_data.get('optimum'),
        'ub': bounds.get('upper'),
    }

for mk in INSTANCES:
    inst = ROOT / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))

    ef = rollout_earliest_finish(env)

    t0 = time.perf_counter()
    neh_schedule = neh_construct(env.instance)
    neh_makespan = env.validate().makespan  # not used

    # Use NEH schedule
    from fjsp.scheduler.validator import validate_schedule
    neh_makespan = validate_schedule(env.instance, neh_schedule).makespan

    # Apply LS to NEH schedule
    _, ls_makespan = local_search(env.instance, neh_schedule, max_iterations=30)
    elapsed = time.perf_counter() - t0

    name = f"mk{mk:02d}"
    print(f"{name:<6} {ef:>5} {neh_makespan:>5} {ls_makespan:>7} {elapsed:>6.1f}")
