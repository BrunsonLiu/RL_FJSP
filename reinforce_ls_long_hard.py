"""Longer REINFORCE training (500 ep) + LS on hard instances.

Saves to data/results/reinforce_ls_long_hard.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, r'd:\desktop2\RL_FJSP')

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from fjsp.scheduler.local_search import local_search

ROOT = Path(r'd:\desktop2\RL_FJSP')
SEEDS = [0, 1, 2]
INSTANCES = [4, 9, 10, 12, 15]  # hard instances

with open(ROOT / 'data' / 'instances' / 'instances.json', 'r') as f:
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


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


print(f"{'inst':<6} {'R-100':>6} {'R+LS-100':>9} {'R-500':>6} {'R+LS-500':>9} {'lit':>5} {'Δ-lit':>6} {'time':>6}")
results = []
for mk in INSTANCES:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    r100_results = []
    r100_ls_results = []
    r500_results = []
    r500_ls_results = []
    for seed in SEEDS:
        # 100 episodes
        agent100, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        r100_makespan, r100_schedule = collect_schedule(env, agent100)
        r100_results.append(r100_makespan)
        _, ls100 = local_search(env.instance, r100_schedule, max_iterations=30)
        r100_ls_results.append(ls100)

        # 500 episodes
        agent500, _ = train_reinforce(env, episodes=500, seed=seed, hidden_dim=64)
        r500_makespan, r500_schedule = collect_schedule(env, agent500)
        r500_results.append(r500_makespan)
        _, ls500 = local_search(env.instance, r500_schedule, max_iterations=30)
        r500_ls_results.append(ls500)

    r100_best = min(r100_results)
    r100_ls_best = min(r100_ls_results)
    r500_best = min(r500_results)
    r500_ls_best = min(r500_ls_results)
    elapsed = time.perf_counter() - t0

    gap = r500_ls_best - lit_target if lit_target else None
    gap_str = f"{gap:+d}" if gap is not None else "-"
    lit_str = f"{lit_target}" if lit_target is not None else "-"
    print(f"{name:<6} {r100_best:>6} {r100_ls_best:>9} {r500_best:>6} {r500_ls_best:>9} {lit_str:>5} {gap_str:>6} {elapsed:>6.1f}")
    results.append({
        'instance': name,
        'r100_best': r100_best,
        'r100_ls_best': r100_ls_best,
        'r500_best': r500_best,
        'r500_ls_best': r500_ls_best,
        'lit_target': lit_target,
        'gap_to_lit_500ls': r500_ls_best - lit_target if lit_target else None,
    })

out = ROOT / 'data' / 'results' / 'reinforce_ls_long_hard.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
