"""Test PA-A2C on mk01 to verify the agent works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents import train_a2c
from fjsp.env import FJSPDispatchEnv

env = FJSPDispatchEnv.from_file(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
agent, history = train_a2c(env, episodes=100, seed=0, hidden_dim=64)
print(f"A2C best: {history[-1]['best_greedy_makespan']}")
print(f"A2C final greedy: {history[-1]['greedy_makespan']}")
print(f"Number of history points: {len(history)}")
for h in history:
    print(h)
