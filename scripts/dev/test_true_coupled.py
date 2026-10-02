"""Test true coupled_reassign: random rollout with full insertion optimization."""
import sys
import time
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv


def random_improve(inst_path, n_steps=50, seed=0):
    random.seed(seed)
    inst = parse_fjs(inst_path)
    initial = earliest_finish_schedule(inst)
    env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=n_steps, patience=20)
    obs = env.reset()

    best = env.makespan
    t0 = time.time()
    while not env.done:
        if not obs["valid_moves"]:
            break
        idx = random.randint(0, len(obs["valid_moves"]) - 1)
        move = obs["valid_moves"][idx]
        obs, reward, done, info = env.step(idx)
        if env.best_makespan < best:
            best = env.best_makespan
            print(f"  step {env.step_count} ({move.move_type}): {info['makespan']} -> best {best}")
    dt = time.time() - t0
    print(f"Final: {env.makespan}, best: {env.best_makespan}, EF: {max(o.end for o in initial)}, time: {dt:.2f}s")
    return best


if __name__ == "__main__":
    for name in ["mk01", "mk02", "mk03", "mk04"]:
        print(f"\n=== {name} ===")
        random_improve(f"data/instances/brandimarte/{name}.txt", n_steps=50, seed=0)
