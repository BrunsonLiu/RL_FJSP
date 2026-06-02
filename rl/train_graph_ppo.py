from __future__ import annotations

import argparse
import json
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents import train_graph_ppo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
DEFAULT_MODEL = ROOT / "data" / "results" / "graph_ppo_mk01.pt"
DEFAULT_HISTORY = ROOT / "data" / "results" / "graph_ppo_mk01_history.json"
DEFAULT_SCHEDULE = ROOT / "data" / "results" / "graph_ppo_mk01_best_schedule.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a graph two-stage PPO policy for FJSP.")
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE), help="Path to an FJSP instance.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--K-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY))
    parser.add_argument("--schedule-out", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    agent, history = train_graph_ppo(
        env,
        episodes=args.episodes,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        gnn_rounds=args.gnn_rounds,
        seed=args.seed,
        clip_ratio=args.clip_ratio,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        K_epochs=args.K_epochs,
        minibatch_size=args.minibatch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
    )

    agent.save(args.model_out)
    history_path = Path(args.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    best_result = agent.rollout(env, greedy=True)
    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")

    final = history[-1]
    print(
        f"OK: trained graph_ppo episodes={args.episodes}, "
        f"best_greedy_makespan={best_result.makespan}, "
        f"best_seen={int(final['best_greedy_makespan'])}, "
        f"model={args.model_out}"
    )


if __name__ == "__main__":
    main()
