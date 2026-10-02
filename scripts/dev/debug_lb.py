"""Debug lower bound calculation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fjsp.parser.fjs_parser import parse_fjs
from fjsp.scheduler.validator import earliest_finish_schedule
from fjsp.env.improvement_env import FJSPImprovementEnv


def main():
    inst = parse_fjs(str(Path(__file__).resolve().parents[2] / "data/instances/brandimarte/mk01.txt"))
    initial = earliest_finish_schedule(inst)
    env = FJSPImprovementEnv(inst, initial_schedule=initial)
    lb = env._lower_bound_makespan()
    print(f"makespan={env.makespan}, lb={lb}")


if __name__ == "__main__":
    main()
