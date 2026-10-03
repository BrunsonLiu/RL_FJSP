"""Quick timing test: SOTA approach on mk01 and mk04."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents import train_reinforce
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.local_search import (
    iterated_local_search,
    simulated_annealing,
    neh_construct,
    random_schedule,
)
from fjsp.scheduler.validator import validate_schedule

ROOT = Path(__file__).resolve().parents[2]

for mk in [1, 4, 9]:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    print(f"\n=== {name} ({env.instance.job_count}x{env.instance.machine_count} = {env.instance.operation_count} ops) ===")

    t0 = time.perf_counter()
    neh = neh_construct(env.instance)
    neh_ms = validate_schedule(env.instance, neh).makespan
    print(f"NEH: {neh_ms} ({time.perf_counter()-t0:.2f}s)")

    # RL for 1 seed
    t1 = time.perf_counter()
    agent, _ = train_reinforce(env, episodes=100, seed=0, hidden_dim=64)
    rl_ms, rl_sched = env.validate().makespan, env.schedule
    # Actually we need a proper greedy rollout
    env.reset()
    schedule = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        schedule.append(info['scheduled'])
    rl_ms = env.validate().makespan
    print(f"REINFORCE 1 seed: {rl_ms} ({time.perf_counter()-t1:.2f}s)")

    t2 = time.perf_counter()
    ils_sched, ils_ms = iterated_local_search(
        env.instance, rl_sched,
        n_iterations=10, perturb_n_swaps=8, max_ls_iterations=20, perturb_mode='critical',
    )
    print(f"ILS(10): {ils_ms} ({time.perf_counter()-t2:.2f}s)")

    t3 = time.perf_counter()
    sa_sched, sa_ms = simulated_annealing(
        env.instance, ils_sched,
        initial_temperature=10.0, cooling_rate=0.99, max_total_iterations=1000,
    )
    print(f"+SA(1000): {sa_ms} ({time.perf_counter()-t3:.2f}s)")

    print(f"TOTAL: {time.perf_counter()-t0:.2f}s")
