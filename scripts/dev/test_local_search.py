"""Quick smoke test: EF heuristic + local search on a few Brandimarte instances.

Confirms that local_search.py is wired up correctly and reports the
improvement achieved over the EF baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from fjsp.scheduler.local_search import local_search, multi_start_local_search

ROOT = Path(__file__).resolve().parents[2] / "data" / "instances" / "brandimarte"
INSTANCES = [1, 4, 9, 15]  # mix of small/medium/large

print(f"{'instance':<10} {'EF':>6} {'LS':>6} {'MS-LS':>8} {'Δ':>6} {'OR-Tools':>10} {'lit_UB':>8} {'sec':>6}")

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
        'lb': bounds.get('lower'),
        'ub': bounds.get('upper'),
    }

for mk in INSTANCES:
    inst = ROOT / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))

    # 1. EF rollout
    env.reset()
    schedule = []
    while not env.done:
        action = choose_earliest_finish(env)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    ef_makespan = env.validate().makespan

    # 2. Simple local search
    t0 = time.perf_counter()
    _, ls_makespan = local_search(env.instance, schedule, max_iterations=50)
    ls_elapsed = time.perf_counter() - t0

    # 3. Multi-start local search
    t0 = time.perf_counter()
    _, ms_makespan = multi_start_local_search(
        env.instance, schedule, n_starts=5, perturb_n_swaps=3, max_iterations=30,
    )
    ms_elapsed = time.perf_counter() - t0

    improvement = ef_makespan - ms_makespan
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    opt_str = f"{opt}" if opt is not None else "-"
    ub_str = f"{ub}" if ub is not None else "-"
    print(f"{name:<10} {ef_makespan:>6} {ls_makespan:>6} {ms_makespan:>8} {improvement:>+6} {opt_str:>10} {ub_str:>8} {ms_elapsed:>6.1f}")
