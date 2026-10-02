"""REINFORCE 5-seed + (LS + SA) on hard Brandimarte instances.

For each instance, trains REINFORCE for 5 seeds × 100 episodes, takes each
greedy schedule, applies LS to first improvement, then runs SA from the
LS result. Reports the best result.

Saves to data/results/reinforce_ls_sa_hard.json
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
from fjsp.scheduler.local_search import (
    local_search,
    multi_start_local_search,
    simulated_annealing,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = [4, 9, 10, 12, 15]  # hard instances with largest remaining gaps

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
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


results = []
print(f"{'inst':<6} {'EF':>5} {'R':>5} {'R+LS':>6} {'R+LS+SA':>9} {'lit':>5} {'Δ-lit':>6} {'time':>7}")
for mk in INSTANCES:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    ef = rollout_earliest_finish(env)

    r_results = []
    rls_results = []
    rls_sa_results = []
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        r_makespan, r_schedule = collect_schedule(env, agent)
        r_results.append(r_makespan)

        # LS on REINFORCE schedule
        _, ls_schedule_makespan = local_search(env.instance, r_schedule, max_iterations=30)
        rls_results.append(ls_schedule_makespan)

        # SA from LS result (limited iterations to control time)
        fresh_ls_schedule, fresh_ls = local_search(env.instance, r_schedule, max_iterations=30)
        sa_schedule, sa_makespan = simulated_annealing(
            env.instance, fresh_ls_schedule,
            initial_temperature=10.0,
            cooling_rate=0.99,
            min_temperature=0.1,
            iterations_per_temp=20,
            max_total_iterations=2000,
            seed=seed,
        )
        rls_sa_results.append(sa_makespan)

    r_best = min(r_results)
    rls_best = min(rls_results)
    rls_sa_best = min(rls_sa_results)
    elapsed = time.perf_counter() - t0
    gap = rls_sa_best - lit_target if lit_target else None
    gap_str = f"{gap:+d}" if gap is not None else "-"
    lit_str = f"{lit_target}" if lit_target is not None else "-"
    print(f"{name:<6} {ef:>5} {r_best:>5} {rls_best:>6} {rls_sa_best:>9} {lit_str:>5} {gap_str:>6} {elapsed:>7.1f}")
    results.append({
        'instance': name,
        'jobs': env.instance.job_count,
        'machines': env.instance.machine_count,
        'operations': env.instance.operation_count,
        'ef_makespan': ef,
        'reinforce_best': r_best,
        'reinforce_ls_best': rls_best,
        'reinforce_ls_sa_best': rls_sa_best,
        'lit_target': lit_target,
        'gap_to_lit': rls_sa_best - lit_target if lit_target else None,
        'time_seconds': round(elapsed, 2),
    })

out = ROOT / 'data' / 'results' / 'reinforce_ls_sa_hard.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
