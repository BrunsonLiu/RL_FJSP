"""Quick test: best-improvement LS with cross-machine swap on mk01."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fjsp.scheduler import local_search, iterated_local_search, simulated_annealing
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_earliest_finish
from rl.agents import train_reinforce

ROOT = Path(__file__).resolve().parents[2]

for mk in [1, 4, 9, 15]:
    inst = ROOT / 'data' / 'instances' / 'brandimarte' / f'mk{mk:02d}.txt'
    env = FJSPDispatchEnv.from_file(str(inst))
    name = f"mk{mk:02d}"
    print(f"\n=== {name} ===")
    ef = rollout_earliest_finish(env)
    print(f"  EF: {ef}")

    # Train RL
    t0 = time.perf_counter()
    agent, _ = train_reinforce(env, episodes=100, seed=0, hidden_dim=64)
    env.reset()
    sched = []
    while not env.done:
        action, _ = agent.select_action(env, greedy=True)
        _, _, _, info = env.step(action)
        sched.append(info['scheduled'])
    rl_ms = env.validate().makespan
    print(f"  REINFORCE: {rl_ms} ({time.perf_counter()-t0:.2f}s)")

    # First-improvement LS
    t0 = time.perf_counter()
    _, ls_first_ms = local_search(env.instance, sched, max_iterations=30, strategy='first')
    print(f"  LS first: {ls_first_ms} ({time.perf_counter()-t0:.2f}s)")

    # Best-improvement LS
    t0 = time.perf_counter()
    _, ls_best_ms = local_search(env.instance, sched, max_iterations=30, strategy='best')
    print(f"  LS best: {ls_best_ms} ({time.perf_counter()-t0:.2f}s)")

    # Best-improvement LS with cross-machine swap
    t0 = time.perf_counter()
    _, ls_best_across_ms = local_search(
        env.instance, sched, max_iterations=30, strategy='best',
        neighborhoods=("reassign", "swap_machine", "swap_order", "swap_order_across"),
    )
    print(f"  LS best+across: {ls_best_across_ms} ({time.perf_counter()-t0:.2f}s)")
