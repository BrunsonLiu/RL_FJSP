"""Tabu search from the SOTA ILS-best schedule for hard instances.

For each hard instance, load the current best SOTA result from
data/results/sota_reinforce_ils.json, then run TS to see if it can
improve further. This avoids re-training REINFORCE.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import (
    iterated_local_search, tabu_search, simulated_annealing, local_search,
    critical_path_perturb, random_perturb, _assignments, _recompute,
)
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from rl.agents import train_reinforce
from fjsp.scheduler.validator import validate_schedule

ROOT = Path(__file__).resolve().parents[2]

# Run only one at a time to avoid timeout
HARD_INSTANCES = [int(sys.argv[1])] if len(sys.argv) > 1 else [15]


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def train_and_collect_best(env, *, n_seeds: int = 5, episodes: int = 100):
    best_ms = float('inf')
    best_sched = None
    for seed in range(n_seeds):
        agent, _ = train_reinforce(env, episodes=episodes, seed=seed, hidden_dim=64)
        ms, sched = collect_schedule(env, agent)
        if ms < best_ms:
            best_ms, best_sched = ms, sched
    return best_ms, best_sched


def main():
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

    results = []
    for mk in HARD_INSTANCES:
        inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
        env = FJSPDispatchEnv.from_file(str(inst))
        name = f"mk{mk:02d}"
        opt = lit_index.get(name, {}).get('optimum')
        ub = lit_index.get(name, {}).get('ub')
        lit_target = opt if opt is not None else ub
        print(f'\n=== {name} (lit={lit_target}, '
              f'{env.instance.job_count}j, {env.instance.machine_count}m, '
              f'{env.instance.operation_count}ops) ===')

        # Train REINFORCE 5 seeds
        print('  Training REINFORCE 5 seeds...')
        t0 = time.perf_counter()
        best_rl_ms, best_rl_sched = train_and_collect_best(env)
        print(f'  Best RL: {best_rl_ms} (took {time.perf_counter()-t0:.1f}s)')

        # ILS from best RL
        print('  Running ILS (n=8)...')
        t0 = time.perf_counter()
        ils_sched, ils_ms = iterated_local_search(
            env.instance, best_rl_sched, n_iterations=8,
            perturb_n_swaps=8, max_ls_iterations=30,
            perturb_mode='critical',
        )
        print(f'  ILS: {ils_ms} (took {time.perf_counter()-t0:.1f}s)')

        # TS from ILS
        print('  Running TS (n=300, sample=200)...')
        t0 = time.perf_counter()
        ts_sched, ts_ms = tabu_search(
            env.instance, ils_sched, max_iterations=300,
            tabu_tenure=20,
            neighborhoods=("reassign", "swap_machine", "swap_order"),
            candidate_sample=200,
        )
        print(f'  TS: {ts_ms} (took {time.perf_counter()-t0:.1f}s)')

        # Save the TS-improved schedule
        best = min(ils_ms, ts_ms)
        gap = best - lit_target if lit_target else None
        print(f'  FINAL: {best} (gap to lit {lit_target}: {gap:+d})')

        # Save the TS result
        out = ROOT / 'data' / 'results' / f'ts_{name}.json'
        with open(out, 'w') as f:
            json.dump({
                'instance': name,
                'lit_target': lit_target,
                'rl_best': best_rl_ms,
                'ils_ms': ils_ms,
                'ts_ms': ts_ms,
                'final_best': best,
                'gap_to_lit': gap,
            }, f, indent=2)

        results.append({
            'instance': name,
            'lit_target': lit_target,
            'rl_best': best_rl_ms,
            'ils_ms': ils_ms,
            'ts_ms': ts_ms,
            'final_best': best,
            'gap_to_lit': gap,
        })

    print('\n=== SUMMARY ===')
    print(f"{'inst':<6} {'RL':>5} {'ILS':>5} {'TS':>5} {'FINAL':>6} {'lit':>5} {'Δ':>4}")
    for r in results:
        lit = r['lit_target']
        gap = r['gap_to_lit']
        print(
            f"{r['instance']:<6} {r['rl_best']:>5} {r['ils_ms']:>5} {r['ts_ms']:>5} "
            f"{r['final_best']:>6} {lit:>5} {gap:+d}"
        )

    out = ROOT / 'data' / 'results' / 'tabu_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
