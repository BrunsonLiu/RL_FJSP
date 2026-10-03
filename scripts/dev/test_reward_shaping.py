"""Test reward shaping in improvement env."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv


def main():
    inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    initial = earliest_finish_schedule(inst)
    print(f"Initial makespan: {max(o.end for o in initial)}")

    for shaping in ["sparse", "dense", "potential"]:
        env = FJSPImprovementEnv(
            inst, initial_schedule=initial, max_steps=10, reward_shaping=shaping
        )
        rewards = []
        for _ in range(5):
            obs = env.observe()
            if not obs["valid_moves"]:
                break
            _, reward, done, info = env.step(0)
            rewards.append(reward)
            if done:
                break
        print(f"{shaping:10s}: rewards={rewards}")

    print("OK")


if __name__ == "__main__":
    main()
