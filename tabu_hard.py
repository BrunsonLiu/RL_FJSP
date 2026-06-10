"""Tabu search post-processor for the worst-gap Brandimarte instances.

For each of mk10/mk15/mk09 (the three with the largest gap to literature):
1. Build a starting schedule with REINFORCE 5 seeds x 100 episodes
2. Run a first-improvement ILS to get a local optimum
3. Run tabu search from the ILS result
4. Run simulated annealing after TS for final refinement
5. Report best
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r'd:\desktop2\RL_FJSP')

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import (
    local_search,
    iterated_local_search,
    simulated_annealing,
    tabu_search,
)
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from rl.agents import train_reinforce

ROOT = Path(r'd:\desktop2\RL_FJSP')

# Hard instances with biggest gap to literature
HARD_INSTANCES = [10, 15, 9, 4, 6]


def collect_schedule(env, agent) -> tuple[int, list]:
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def collect_ef_schedule(env):
    env.reset()
    schedule = []
    while not env.done:
        action = choose_earliest_finish(env)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return schedule, env.validate().makespan


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


def run_tabu_for_instance(mk: int, *, n_ils: int = 10, n_tabu: int = 300,
                          n_sa: int = 2000, tabu_tenure: int = 15,
                          tabu_sample: int = 200) -> dict:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    ef_schedule, ef = collect_ef_schedule(env)

    # Train REINFORCE for each seed, collect the best
    best_rl_ms = float('inf')
    best_rl_sched = None
    for seed in range(5):
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        ms, sched = collect_schedule(env, agent)
        if ms < best_rl_ms:
            best_rl_ms, best_rl_sched = ms, sched

    # Also try EF + ILS
    ef_ils_sched, ef_ils_ms = iterated_local_search(
        env.instance, ef_schedule, n_iterations=n_ils,
        perturb_n_swaps=8, max_ls_iterations=30,
        perturb_mode='critical',
    )

    # ILS from best RL
    ils_sched, ils_ms = iterated_local_search(
        env.instance, best_rl_sched, n_iterations=n_ils,
        perturb_n_swaps=8, max_ls_iterations=30,
        perturb_mode='critical',
    )

    # Take the best of (RL+ILS, EF+ILS) as TS starting point
    if ef_ils_ms < ils_ms:
        ts_start, ils_best = ef_ils_sched, ef_ils_ms
    else:
        ts_start, ils_best = ils_sched, ils_ms

    # Tabu search with candidate sampling (essential for large instances)
    ts_sched, ts_ms = tabu_search(
        env.instance, ts_start,
        max_iterations=n_tabu,
        tabu_tenure=tabu_tenure,
        neighborhoods=("reassign", "swap_machine", "swap_order"),
        candidate_sample=tabu_sample,
    )

    # Final SA refinement from TS best
    sa_sched, sa_ms = simulated_annealing(
        env.instance, ts_sched,
        initial_temperature=8.0, cooling_rate=0.99,
        min_temperature=0.1, iterations_per_temp=20,
        max_total_iterations=n_sa,
    )

    final_best = min(ils_best, ts_ms, sa_ms)
    elapsed = time.perf_counter() - t0
    return {
        'instance': name,
        'lit_target': lit_target,
        'ef_makespan': ef,
        'reinforce_best': best_rl_ms,
        'ef_ils': ef_ils_ms,
        'ils_best': ils_best,
        'ts_makespan': ts_ms,
        'sa_makespan': sa_ms,
        'final_best': final_best,
        'gap_to_lit': final_best - lit_target if lit_target else None,
        'time_seconds': round(elapsed, 2),
    }


def main():
    results = []
    print(
        f"{'inst':<6} {'EF':>5} {'R-best':>7} {'ILS':>5} {'+TS':>5} {'+SA':>5} "
        f"{'FINAL':>6} {'lit':>5} {'Δ':>4} {'time':>7}"
    )
    for mk in HARD_INSTANCES:
        try:
            r = run_tabu_for_instance(mk)
        except Exception as e:
            print(f"mk{mk:02d}: FAILED - {e}")
            continue
        results.append(r)
        gap = r['gap_to_lit']
        gap_str = f"{gap:+d}" if gap is not None else "-"
        lit_str = f"{r['lit_target']}" if r['lit_target'] is not None else "-"
        print(
            f"{r['instance']:<6} {r['ef_makespan']:>5} {r['reinforce_best']:>7} "
            f"{r['ils_best']:>5} {r['ts_makespan']:>5} {r['sa_makespan']:>5} "
            f"{r['final_best']:>6} {lit_str:>5} {gap_str:>4} {r['time_seconds']:>7.1f}"
        )

    out = ROOT / 'data' / 'results' / 'tabu_hard.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
