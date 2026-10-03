"""5-seed REINFORCE benchmark on Brandimarte mk06, mk10, mk13, mk15.

Saves to data/results/reinforce_mk06_mk10_mk13_mk15_5seed.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = [6, 10, 13, 15]

results = []
for mk in INSTANCES:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    env_meta = {
        'instance': inst.stem,
        'jobs': env.instance.job_count,
        'machines': env.instance.machine_count,
        'operations': env.instance.operation_count,
        'earliest_finish': rollout_earliest_finish(env),
    }
    seed_runs = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        agent, history = train_reinforce(env, episodes=100, seed=seed)
        elapsed = time.perf_counter() - t0
        best = int(history[-1]['best_greedy_makespan'])
        seed_runs.append({'seed': seed, 'best': best, 'seconds': round(elapsed, 2)})
        print(f"  {inst.stem} seed={seed}: best={best} ({elapsed:.1f}s)")
    values = [r['best'] for r in seed_runs]
    env_meta['reinforce_best'] = min(values)
    env_meta['reinforce_mean'] = round(mean(values), 2)
    env_meta['reinforce_std'] = round(pstdev(values), 2) if len(values) > 1 else 0.0
    env_meta['seed_results'] = seed_runs
    results.append(env_meta)
    print(f"  >>> {inst.stem}: best={env_meta['reinforce_best']} mean={env_meta['reinforce_mean']} std={env_meta['reinforce_std']}")

out = ROOT / 'data' / 'results' / 'reinforce_mk06_mk10_mk13_mk15_5seed.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
