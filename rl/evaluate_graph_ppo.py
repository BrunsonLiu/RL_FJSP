from __future__ import annotations

import argparse
import json
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents.graph_ppo import GraphPPOAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
DEFAULT_MODEL = ROOT / "data" / "results" / "graph_ppo_mk01.pt"
DEFAULT_SCHEDULE = ROOT / "data" / "results" / "graph_ppo_mk01_schedule.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a graph two-stage PPO FJSP policy.")
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE), help="Path to an FJSP instance.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to a saved graph PPO model.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--schedule-out", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    agent = GraphPPOAgent.load(args.model, hidden_dim=args.hidden_dim)
    result = agent.rollout(env, greedy=True)

    if not result.is_valid:
        raise SystemExit("Evaluation produced an invalid schedule.")

    output = Path(args.schedule_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")
    print(f"OK: graph_ppo_greedy_makespan={result.makespan}, schedule={output}")


if __name__ == "__main__":
    main()
