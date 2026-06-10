"""Aggressive SOTA on hard Brandimarte instances using best-improvement
local search, cross-machine swap, and a longer ILS with multiple
perturbation strengths.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r'd:\desktop2\RL_FJSP')

from fjsp.scheduler import (
    local_search,
    iterated_local_search,
    simulated_annealing,
    neh_construct,
    random_schedule,
)
from fjsp.scheduler.validator import validate_schedule
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from rl.agents import train_reinforce

ROOT = Path(r'd:\desktop2\RL_FJSP')

# Hard instances where we still have a gap to literature.
HARD_INSTANCES = [4, 5, 6, 7, 9, 10, 11, 12, 15]

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


def aggressive_ils_for_instance(env, rl_schedule, *, n_ils: int = 30, n_sa: int = 3000) -> tuple[list, int]:
    """Run several aggressive ILS+SA variants and return the best."""
    best_schedule, best_ms = None, float('inf')
    # Variant 1: critical perturbation + best LS + cross-machine swap
    s1, m1 = iterated_local_search(
        env.instance, rl_schedule,
        n_iterations=n_ils, perturb_n_swaps=8, max_ls_iterations=30,
        perturb_mode='critical',
        neighborhoods=("reassign", "swap_machine", "swap_order", "swap_order_across"),
        strategy='best',
    )
    if m1 < best_ms:
        best_ms, best_schedule = m1, s1

    # Variant 2: random perturbation + first LS
    s2, m2 = iterated_local_search(
        env.instance, rl_schedule,
        n_iterations=n_ils, perturb_n_swaps=12, max_ls_iterations=30,
        perturb_mode='random',
        neighborhoods=("reassign", "swap_machine", "swap_order"),
        strategy='first',
    )
    if m2 < best_ms:
        best_ms, best_schedule = m2, s2

    # Variant 3: critical perturbation + first LS, more iterations
    s3, m3 = iterated_local_search(
        env.instance, rl_schedule,
        n_iterations=n_ils + 10, perturb_n_swaps=10, max_ls_iterations=30,
        perturb_mode='critical',
        neighborhoods=("reassign", "swap_machine", "swap_order"),
        strategy='first',
    )
    if m3 < best_ms:
        best_ms, best_schedule = m3, s3

    # Variant 4: critical perturbation with large kick
    s4, m4 = iterated_local_search(
        env.instance, rl_schedule,
        n_iterations=n_ils, perturb_n_swaps=20, max_ls_iterations=30,
        perturb_mode='random',
        neighborhoods=("reassign", "swap_machine", "swap_order"),
        strategy='first',
    )
    if m4 < best_ms:
        best_ms, best_schedule = m4, s4

    # Final SA refinement
    sa_sched, sa_ms = simulated_annealing(
        env.instance, best_schedule,
        initial_temperature=15.0, cooling_rate=0.99,
        min_temperature=0.1, iterations_per_temp=20,
        max_total_iterations=n_sa,
    )
    if sa_ms < best_ms:
        best_ms, best_schedule = sa_ms, sa_sched

    return best_schedule, best_ms


def main() -> None:
    results = []
    print(
        f"{'inst':<6} {'EF':>5} {'R-best':>7} {'ILS':>5} {'+SA':>5} {'FINAL':>6} "
        f"{'lit':>5} {'Δ':>4} {'time':>7}"
    )
    for mk in HARD_INSTANCES:
        inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
        env = FJSPDispatchEnv.from_file(str(inst))
        name = f"mk{mk:02d}"
        opt = lit_index.get(name, {}).get('optimum')
        ub = lit_index.get(name, {}).get('ub')
        lit_target = opt if opt is not None else ub

        t0 = time.perf_counter()
        ef = rollout_earliest_finish(env)

        # Train REINFORCE 5 seeds, take best
        best_rl_ms = float('inf')
        best_rl_sched = None
        for seed in range(5):
            agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
            ms, sched = collect_schedule(env, agent)
            if ms < best_rl_ms:
                best_rl_ms, best_rl_sched = ms, sched

        # Run aggressive ILS+SA
        ils_sched, ils_ms = aggressive_ils_for_instance(env, best_rl_sched)

        # Also run SA directly from RL schedule
        sa_sched, sa_ms = simulated_annealing(
            env.instance, best_rl_sched,
            initial_temperature=15.0, cooling_rate=0.99, max_total_iterations=3000,
        )

        final_best = min(ils_ms, sa_ms)
        elapsed = time.perf_counter() - t0
        gap = final_best - lit_target if lit_target else None
        gap_str = f"{gap:+d}" if gap is not None else "-"
        lit_str = f"{lit_target}" if lit_target is not None else "-"
        print(
            f"{name:<6} {ef:>5} {best_rl_ms:>7} {ils_ms:>5} {sa_ms:>5} {final_best:>6} "
            f"{lit_str:>5} {gap_str:>4} {elapsed:>7.1f}"
        )
        results.append({
            'instance': name,
            'lit_target': lit_target,
            'reinforce_best': best_rl_ms,
            'ils_best': ils_ms,
            'sa_makespan': sa_ms,
            'final_best': final_best,
            'gap_to_lit': final_best - lit_target if lit_target else None,
            'time_seconds': round(elapsed, 2),
        })

    out = ROOT / 'data' / 'results' / 'aggressive_ils_hard.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
