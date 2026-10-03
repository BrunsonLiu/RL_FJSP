"""REINFORCE 5-seed best + multi-start local search on Brandimarte mk01-15.

For each instance, trains REINFORCE for 5 seeds × 100 episodes, takes each
greedy schedule, applies multi-start local search, and reports the best
result across seeds × local search restarts.

Saves to data/results/reinforce_ls_mk01_mk15.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from fjsp.scheduler.local_search import local_search, multi_start_local_search

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = list(range(1, 16))

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
        'lb': bounds.get('lower'),
        'ub': bounds.get('upper'),
    }


def collect_schedule(env: FJSPDispatchEnv, agent) -> tuple[int, list]:
    """Greedy rollout and return (makespan, schedule)."""
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


results = []
print(f"{'inst':<6} {'EF':>5} {'R-best':>7} {'R+LS':>7} {'lit_UB':>7} {'Δ-to-lit':>9} {'time':>7}")
for mk in INSTANCES:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    ef = rollout_earliest_finish(env)

    seed_results = []
    seed_ls_results = []
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        r_makespan, r_schedule = collect_schedule(env, agent)
        seed_results.append(r_makespan)
        # Multi-start local search on the REINFORCE schedule
        _, ls_makespan = multi_start_local_search(
            env.instance, r_schedule, n_starts=3, perturb_n_swaps=3, max_iterations=20,
        )
        seed_ls_results.append(ls_makespan)

    r_best = min(seed_results)
    r_ls_best = min(seed_ls_results)
    elapsed = time.perf_counter() - t0
    gap = r_ls_best - lit_target if lit_target else None
    gap_str = f"{gap:+d}" if gap is not None else "-"
    lit_str = f"{lit_target}" if lit_target is not None else "-"
    print(f"{name:<6} {ef:>5} {r_best:>7} {r_ls_best:>7} {lit_str:>7} {gap_str:>9} {elapsed:>7.1f}")
    results.append({
        'instance': name,
        'jobs': env.instance.job_count,
        'machines': env.instance.machine_count,
        'operations': env.instance.operation_count,
        'ef_makespan': ef,
        'reinforce_best': r_best,
        'reinforce_mean': round(mean(seed_results), 2),
        'reinforce_std': round(pstdev(seed_results), 2) if len(seed_results) > 1 else 0,
        'reinforce_ls_best': r_ls_best,
        'reinforce_ls_mean': round(mean(seed_ls_results), 2),
        'reinforce_ls_std': round(pstdev(seed_ls_results), 2) if len(seed_ls_results) > 1 else 0,
        'lit_optimum': opt,
        'lit_ub': ub,
        'lit_target': lit_target,
        'gap_to_lit': r_ls_best - lit_target if lit_target else None,
        'time_seconds': round(elapsed, 2),
    })

out = ROOT / 'data' / 'results' / 'reinforce_ls_mk01_mk15.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
