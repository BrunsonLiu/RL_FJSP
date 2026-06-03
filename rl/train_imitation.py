"""CLI: train a graph two-stage actor-critic policy with behavioral cloning
on earliest-finish demonstrations, then optionally save the best checkpoint.

Usage:
    python -m rl.train_imitation --instance data/instances/brandimarte/mk01.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents import train_imitation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
DEFAULT_MODEL = ROOT / "data" / "results" / "imitation_mk01_best.pt"
DEFAULT_HISTORY = ROOT / "data" / "results" / "imitation_mk01_history.json"
DEFAULT_SCHEDULE = ROOT / "data" / "results" / "imitation_mk01_best_schedule.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a graph two-stage actor-critic policy with behavioral cloning on FJSP dispatch demonstrations.")
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE), help="Path to an FJSP instance.")
    parser.add_argument("--rollouts", type=int, default=100, help="Number of expert rollouts to collect.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of supervised passes over the demonstrations.")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for supervised updates.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY))
    parser.add_argument("--schedule-out", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    agent, history = train_imitation(
        env,
        rollouts=args.rollouts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        gnn_rounds=args.gnn_rounds,
        seed=args.seed,
    )

    # Greedy rollout to verify the policy is functional and to emit a schedule.
    result = agent.rollout(env, greedy=True)
    if not result.is_valid:
        raise SystemExit("BC policy produced an invalid greedy schedule.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(history, indent=2), encoding="utf-8")

    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")

    print(f"OK: bc_greedy_makespan={result.makespan}, model={args.model_out}")
    print(f"OK: history={args.history_out}, schedule={args.schedule_out}")


if __name__ == "__main__":
    main()
