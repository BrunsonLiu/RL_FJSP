"""A2C + LS on hard Brandimarte instances, 5 seeds.

Compares REINFORCE+LS vs A2C+LS to see if A2C value baseline helps.
Saves to data/results/a2c_ls_hard.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents import train_reinforce, train_a2c
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from fjsp.scheduler.local_search import local_search

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = [1, 4, 9, 10, 12, 13, 15]  # mix of small/medium/large/hard

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


def collect_schedule(env, agent) -> tuple[int, list]:
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def collect_schedule_a2c(env, agent) -> tuple[int, list]:
    env.reset()
    schedule = []
    while not env.done:
        actions = env.available_actions()
        from rl.agents.per_action_a2c import action_features, global_state_features, FEATURE_DIM
        from fjsp.utils.scaling import instance_time_scale
        import torch
        scale = instance_time_scale(env)
        features = torch.tensor(
            [action_features(env, a, scale=scale) for a in actions],
            dtype=torch.float32,
        )
        logits = agent.policy(features)
        idx = int(torch.argmax(logits).item())
        _, _, _, info = env.step(actions[idx])
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


print(f"{'inst':<6} {'R':>5} {'R+LS':>6} {'A2C':>5} {'A2C+LS':>7} {'lit':>5} {'Δ-lit':>6} {'time':>6}")
results = []
for mk in INSTANCES:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    r_results = []
    rls_results = []
    a2c_results = []
    a2c_ls_results = []
    for seed in SEEDS:
        # REINFORCE
        agent_r, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        r_makespan, r_schedule = collect_schedule(env, agent_r)
        r_results.append(r_makespan)
        _, ls_r = local_search(env.instance, r_schedule, max_iterations=30)
        rls_results.append(ls_r)

        # A2C
        agent_a2c, _ = train_a2c(env, episodes=100, seed=seed, hidden_dim=64)
        a2c_makespan, a2c_schedule = collect_schedule_a2c(env, agent_a2c)
        a2c_results.append(a2c_makespan)
        _, ls_a2c = local_search(env.instance, a2c_schedule, max_iterations=30)
        a2c_ls_results.append(ls_a2c)

    r_best = min(r_results)
    rls_best = min(rls_results)
    a2c_best = min(a2c_results)
    a2c_ls_best = min(a2c_ls_results)
    elapsed = time.perf_counter() - t0

    # Use the best of (R+LS, A2C+LS) as the final
    final_best = min(rls_best, a2c_ls_best)
    gap = final_best - lit_target if lit_target else None
    gap_str = f"{gap:+d}" if gap is not None else "-"
    lit_str = f"{lit_target}" if lit_target is not None else "-"
    print(f"{name:<6} {r_best:>5} {rls_best:>6} {a2c_best:>5} {a2c_ls_best:>7} {lit_str:>5} {gap_str:>6} {elapsed:>6.1f}")
    results.append({
        'instance': name,
        'reinforce_best': r_best,
        'reinforce_ls_best': rls_best,
        'a2c_best': a2c_best,
        'a2c_ls_best': a2c_ls_best,
        'final_best': final_best,
        'lit_target': lit_target,
        'gap_to_lit': final_best - lit_target if lit_target else None,
    })

out = ROOT / 'data' / 'results' / 'a2c_ls_hard.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out}")
