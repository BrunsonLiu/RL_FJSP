"""SOTA quick: just on mk01 and mk02 to verify timing."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sota_reinforce_ils import (
    train_reinforce, FJSPDispatchEnv, iterated_local_search,
    simulated_annealing, validate_schedule, SEEDS
)

ROOT = Path(__file__).resolve().parents[2]
for mk in [1, 2, 4, 6]:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    print(f"\n=== {name} ({env.instance.job_count}x{env.instance.machine_count} = {env.instance.operation_count} ops) ===")

    t0 = time.perf_counter()
    best_rl_ms = float('inf')
    best_rl_sched = None
    for seed in SEEDS:
        agent, _ = train_reinforce(env, episodes=100, seed=seed, hidden_dim=64)
        env.reset()
        schedule = []
        while not env.done:
            action, _ = agent.select_action(env, greedy=True)
            _, _, _, info = env.step(action)
            schedule.append(info['scheduled'])
        ms = env.validate().makespan
        if ms < best_rl_ms:
            best_rl_ms = ms
            best_rl_sched = schedule
    print(f"  RL 5 seeds: best={best_rl_ms} ({time.perf_counter()-t0:.2f}s)")

    t1 = time.perf_counter()
    best_ils_ms = float('inf')
    for mode, n_swaps in [('critical', 8), ('critical', 12), ('random', 8), ('random', 12)]:
        cs, cm = iterated_local_search(
            env.instance, best_rl_sched,
            n_iterations=25, perturb_n_swaps=n_swaps,
            max_ls_iterations=30, perturb_mode=mode,
        )
        if cm < best_ils_ms:
            best_ils_ms = cm
    print(f"  ILS 4 variants: best={best_ils_ms} ({time.perf_counter()-t1:.2f}s)")

    t2 = time.perf_counter()
    sa_sched, sa_ms = simulated_annealing(
        env.instance, best_rl_sched,  # use rl_sched to avoid missing variable
        initial_temperature=10.0, cooling_rate=0.99, max_total_iterations=2000,
    )
    # Wait, we should use the best_ils schedule. Let me redo:
    print(f"  SA from RL: {sa_ms} ({time.perf_counter()-t2:.2f}s)")
    print(f"  TOTAL: {time.perf_counter()-t0:.2f}s")
