"""Evaluate a saved HGT-FJSP model on a given instance (greedy rollout)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.dispatch_rules import rollout_rule, choose_earliest_finish, choose_shortest_ready_time
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents.hgt_fjsp import HGTActorCriticAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--schedule-out", default=None)
    _root = Path(__file__).resolve().parents[1]
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    agent = HGTActorCriticAgent.load(args.model, device="cpu")
    # Use 0 as instance id (single instance training).
    agent.instance_id = 0
    result = agent.rollout(env, greedy=True, seed=0)
    ef = rollout_rule(env, choose_earliest_finish)
    ss = rollout_rule(env, choose_shortest_ready_time)
    print(
        json.dumps(
            {
                "model": args.model,
                "instance": args.instance,
                "hgt_makespan": result.makespan,
                "earliest_finish_makespan": ef,
                "shortest_start_makespan": ss,
                "valid": result.is_valid,
            },
            indent=2,
        )
    )
    if args.schedule_out:
        Path(args.schedule_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.schedule_out).write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
