"""Quick performance test for BARI environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.env.improvement_env import FJSPImprovementEnv
from fjsp.scheduler.validator import earliest_finish_schedule
import time

inst = parse_fjs("data/instances/brandimarte/mk01.txt")
initial = earliest_finish_schedule(inst)
print(f"EF makespan: {max(o.end for o in initial)}")

env = FJSPImprovementEnv(inst, initial_schedule=initial, max_steps=50, patience=10)
t0 = time.perf_counter()
obs = env.reset()
print(f"Reset time: {time.perf_counter()-t0:.3f}s")
print(f"Valid moves: {len(obs['valid_moves'])}")

t0 = time.perf_counter()
for i in range(20):
    if env.done:
        break
    action = env.sample_valid_action()
    obs, reward, done, info = env.step(action)
print(f"20 steps time: {time.perf_counter()-t0:.3f}s")
print(f"Makespan: {env.makespan}, Best: {env.best_makespan}")
