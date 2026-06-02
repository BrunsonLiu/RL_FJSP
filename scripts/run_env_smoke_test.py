from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import _bootstrap  # noqa: F401
from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import choose_earliest_finish
from fjsp.scheduler.validator import schedule_to_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "env_smoke_schedule.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic dispatch-environment smoke test.")
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE), help="Path to an FJSP instance.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the generated schedule JSON.")
    parser.add_argument("--policy", choices=["earliest_finish", "random"], default="earliest_finish")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    rng = Random(args.seed)
    total_reward = 0.0

    while not env.done:
        action = env.sample_valid_action(rng) if args.policy == "random" else choose_earliest_finish(env)
        _, reward, _, _ = env.step(action)
        total_reward += reward

    result = env.validate()
    if not result.is_valid:
        print("Environment smoke test failed:")
        for error in result.errors:
            print(f"- {error}")
        raise SystemExit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")

    print(
        f"OK: instance={args.instance}, policy={args.policy}, "
        f"operations={len(env.schedule)}, makespan={result.makespan}, total_reward={total_reward}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
