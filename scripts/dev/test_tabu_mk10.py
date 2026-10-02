"""Single-instance test: TS on mk10 from current best (RL+ILS)."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import (
    iterated_local_search, tabu_search, simulated_annealing, local_search,
)
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from rl.agents import train_reinforce
from fjsp.scheduler.validator import validate_schedule


def collect_ef_schedule(env):
    env.reset()
    schedule = []
    while not env.done:
        action = choose_earliest_finish(env)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return schedule, env.validate().makespan


def collect_schedule(env, agent):
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    return env.validate().makespan, schedule


def main():
    inst = str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk15.txt")
    env = FJSPDispatchEnv.from_file(inst)
    print(f'Instance: mk15 ({env.instance.job_count} jobs, '
          f'{env.instance.machine_count} machines, '
          f'{env.instance.operation_count} ops)')
    print('Lit UB: 341')

    # Train REINFORCE
    print('Training REINFORCE 5 seeds...')
    best_rl_ms = float('inf')
    best_rl_sched = None
    for seed in range(5):
        t0 = time.perf_counter()
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        ms, sched = collect_schedule(env, agent)
        t1 = time.perf_counter()
        print(f'  seed {seed}: RL={ms} (took {t1-t0:.1f}s)')
        if ms < best_rl_ms:
            best_rl_ms, best_rl_sched = ms, sched

    # ILS from best RL
    print(f'\nBest RL: {best_rl_ms}. Running ILS (n_ils=10)...')
    t0 = time.perf_counter()
    ils_sched, ils_ms = iterated_local_search(
        env.instance, best_rl_sched, n_iterations=10,
        perturb_n_swaps=8, max_ls_iterations=30,
        perturb_mode='critical',
    )
    t1 = time.perf_counter()
    print(f'  ILS: {ils_ms} (took {t1-t0:.1f}s)')

    # TS from ILS
    print(f'\nRunning TS (n_tabu=500, sample=300)...')
    t0 = time.perf_counter()
    ts_sched, ts_ms = tabu_search(
        env.instance, ils_sched, max_iterations=500,
        tabu_tenure=20,
        neighborhoods=("reassign", "swap_machine", "swap_order"),
        candidate_sample=300,
    )
    t1 = time.perf_counter()
    print(f'  TS: {ts_ms} (took {t1-t0:.1f}s)')

    # SA after TS
    print(f'\nRunning SA...')
    t0 = time.perf_counter()
    sa_sched, sa_ms = simulated_annealing(
        env.instance, ts_sched,
        initial_temperature=8.0, cooling_rate=0.99,
        max_total_iterations=2000,
    )
    t1 = time.perf_counter()
    print(f'  SA: {sa_ms} (took {t1-t0:.1f}s)')

    best = min(ils_ms, ts_ms, sa_ms)
    print(f'\nFinal best: {best} (gap to lit 341: {best - 341:+d})')


if __name__ == '__main__':
    main()
