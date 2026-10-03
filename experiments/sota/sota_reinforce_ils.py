"""SOTA: REINFORCE + ILS+SA on Brandimarte MK01-MK15.

For each instance:
1. Train REINFORCE for 5 seeds × 100 episodes.
2. Pick the best greedy schedule.
3. Run iterated local search (ILS) with critical-path perturbation
   for many iterations with several perturbation strengths.
4. Run simulated annealing (SA) from the best.
5. Report the best.

Saves to data/results/sota_reinforce_ils.json
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
    iterated_local_search,
    ils_sa_hybrid,
    neh_construct,
    random_schedule,
    critical_path_perturb,
    random_perturb,
)
from fjsp.scheduler.validator import validate_schedule

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2, 3, 4]
INSTANCES = list(range(1, 16))

# ILS iteration counts scaled by instance size
ILS_ITERS = {
    1: 25, 2: 25, 3: 20, 4: 25, 5: 25, 6: 25, 7: 25, 8: 20, 9: 15, 10: 15,
    11: 15, 12: 15, 13: 15, 14: 15, 15: 15,
}
SA_ITERS = {
    1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000, 6: 2000, 7: 2000, 8: 2000,
    9: 1500, 10: 1500, 11: 1500, 12: 1500, 13: 1500, 14: 1500, 15: 1500,
}

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
print(
    f"{'inst':<6} {'EF':>5} {'R-best':>7} {'ILS-best':>9} {'+SA':>5} {'FINAL':>6} "
    f"{'lit':>5} {'Δ':>4} {'time':>7}"
)


_LOG_HANDLE = None


def _log(msg: str) -> None:
    print(msg)
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.write(msg + "\n")
        _LOG_HANDLE.flush()


def run_sota_on_instance(mk: int, *, n_ils: int, n_sa: int) -> dict:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    opt = lit_index.get(name, {}).get('optimum')
    ub = lit_index.get(name, {}).get('ub')
    lit_target = opt if opt is not None else ub

    t0 = time.perf_counter()
    ef = rollout_earliest_finish(env)

    # Train REINFORCE for each seed, collect the schedule with the best
    # greedy makespan. Track all 5 seed results for mean ± std reporting.
    best_rl_ms = float('inf')
    best_rl_schedule = None
    all_rl_ms = []
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        r_makespan, r_schedule = collect_schedule(env, agent)
        all_rl_ms.append(r_makespan)
        if r_makespan < best_rl_ms:
            best_rl_ms = r_makespan
            best_rl_schedule = r_schedule

    # Try ILS with different perturbation strengths and modes
    best_ils_ms = float('inf')
    best_ils_schedule = None
    for mode, n_swaps in [('critical', 8), ('random', 10)]:
        cand_sched, cand_ms = iterated_local_search(
            env.instance, best_rl_schedule,
            n_iterations=n_ils, perturb_n_swaps=n_swaps,
            max_ls_iterations=30, perturb_mode=mode,
        )
        if cand_ms < best_ils_ms:
            best_ils_ms = cand_ms
            best_ils_schedule = cand_sched

    # Final SA refinement from the best ILS schedule
    sa_sched, sa_ms = simulated_annealing(
        env.instance, best_ils_schedule,
        initial_temperature=10.0, cooling_rate=0.99,
        min_temperature=0.1, iterations_per_temp=20,
        max_total_iterations=n_sa,
    )
    final_best = min(best_ils_ms, sa_ms)

    elapsed = time.perf_counter() - t0
    return {
        'instance': name,
        'jobs': env.instance.job_count,
        'machines': env.instance.machine_count,
        'operations': env.instance.operation_count,
        'ef_makespan': ef,
        'reinforce_best': best_rl_ms,
        'reinforce_mean': round(mean(all_rl_ms), 2),
        'reinforce_std': round(pstdev(all_rl_ms), 2) if len(all_rl_ms) > 1 else 0.0,
        'reinforce_all_seeds': all_rl_ms,
        'ils_best': best_ils_ms,
        'sa_makespan': sa_ms,
        'final_best': final_best,
        'lit_optimum': opt,
        'lit_ub': ub,
        'lit_target': lit_target,
        'gap_to_lit': final_best - lit_target if lit_target else None,
        'time_seconds': round(elapsed, 2),
    }


def main(instances: list[int] | None = None) -> None:
    global _LOG_HANDLE
    log_path = ROOT / 'data' / 'results' / 'sota_reinforce_ils.log'
    _LOG_HANDLE = open(log_path, 'w')
    _log(
        f"{'inst':<6} {'EF':>5} {'R-best':>7} {'ILS-best':>9} {'+SA':>5} {'FINAL':>6} "
        f"{'lit':>5} {'Δ':>4} {'time':>7}"
    )
    for mk in instances or INSTANCES:
        n_ils = ILS_ITERS.get(mk, 20)
        n_sa = SA_ITERS.get(mk, 1500)
        result = run_sota_on_instance(mk, n_ils=n_ils, n_sa=n_sa)
        results.append(result)
        gap = result['gap_to_lit']
        gap_str = f"{gap:+d}" if gap is not None else "-"
        lit_str = f"{result['lit_target']}" if result['lit_target'] is not None else "-"
        _log(
            f"{result['instance']:<6} {result['ef_makespan']:>5} {result['reinforce_best']:>7} "
            f"{result['ils_best']:>9} {result['sa_makespan']:>5} {result['final_best']:>6} "
            f"{lit_str:>5} {gap_str:>4} {result['time_seconds']:>7.1f}"
        )
    out = ROOT / 'data' / 'results' / 'sota_reinforce_ils.json'
    out.write_text(json.dumps(results, indent=2))
    _log(f"\nWrote {out}")
    _LOG_HANDLE.close()
    _LOG_HANDLE = None


if __name__ == "__main__":
    main()
