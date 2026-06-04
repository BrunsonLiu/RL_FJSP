"""Quick smoke test for the new dispatch rules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import (
    DISPATCH_RULES,
    rollout_rule,
)


def main() -> int:
    env = FJSPDispatchEnv.from_file("data/instances/brandimarte/mk01.txt")
    for name, rule in DISPATCH_RULES.items():
        ms = rollout_rule(env, rule)
        print(f"{name:>18s}: makespan={ms}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
