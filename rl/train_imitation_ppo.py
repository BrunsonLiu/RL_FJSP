"""CLI: load a BC-pretrained graph PPO model and fine-tune it with PPO on FJSP.

Usage:
    python -m rl.train_imitation_ppo --instance data/instances/brandimarte/mk01.txt
                                   --init-model data/results/imitation_mk01_best.pt
                                   --episodes 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from rl.agents import GraphPPOAgent, train_graph_ppo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = ROOT / "data" / "instances" / "brandimarte" / "mk01.txt"
DEFAULT_INIT_MODEL = ROOT / "data" / "results" / "imitation_mk01_best.pt"
DEFAULT_MODEL_OUT = ROOT / "data" / "results" / "bc_ppo_mk01_best.pt"
DEFAULT_HISTORY = ROOT / "data" / "results" / "bc_ppo_mk01_history.json"
DEFAULT_SCHEDULE = ROOT / "data" / "results" / "bc_ppo_mk01_best_schedule.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a BC-pretrained graph PPO policy on FJSP.")
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE))
    parser.add_argument("--init-model", default=str(DEFAULT_INIT_MODEL), help="Path to a BC-pretrained GraphTwoStageActorCriticNet checkpoint.")
    parser.add_argument("--episodes", type=int, default=100)
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
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY))
    parser.add_argument("--schedule-out", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    env = FJSPDispatchEnv.from_file(args.instance)
    agent = GraphPPOAgent.load(args.init_model, hidden_dim=args.hidden_dim)
    agent.optimizer = __import__("torch").optim.Adam(agent.model.parameters(), lr=args.lr)

    # Run PPO fine-tuning in-place by reusing train_graph_ppo.
    # We need a custom loop because train_graph_ppo always creates a fresh agent.
    # Instead, manually run the same PPO training but with our preloaded agent.
    from random import Random
    import torch
    from copy import deepcopy
    from rl.agents.graph_ppo import _collect_rollout, _ppo_update

    torch.manual_seed(args.seed)
    rng = Random(args.seed)
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state = None

    for episode in range(1, args.episodes + 1):
        steps, makespan, is_valid = _collect_rollout(env, agent, rng=Random(rng.randrange(0, 2**31 - 1)))
        if not is_valid or not steps:
            raise RuntimeError("BC-PPO rollout produced an invalid or empty episode.")
        metrics = _ppo_update(
            env, agent, steps,
            clip_ratio=args.clip_ratio, gamma=args.gamma, gae_lambda=args.gae_lambda,
            K_epochs=args.K_epochs, minibatch_size=args.minibatch_size,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        )
        if episode == 1 or episode == args.episodes or episode % max(1, args.episodes // 10) == 0:
            greedy_result = agent.rollout(env, greedy=True)
            if best_makespan is None or greedy_result.makespan < best_makespan:
                best_makespan = greedy_result.makespan
                best_episode = episode
                best_state = deepcopy(agent.model.state_dict())
            history.append({
                "episode": float(episode),
                "sample_makespan": float(makespan),
                "greedy_makespan": float(greedy_result.makespan),
                "best_greedy_makespan": float(best_makespan),
                "best_episode": float(best_episode),
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
            })

    if best_state is not None:
        agent.model.load_state_dict(best_state)

    final_result = agent.rollout(env, greedy=True)
    if not final_result.is_valid:
        raise SystemExit("BC-PPO produced an invalid final schedule.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(history, indent=2), encoding="utf-8")

    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(env.schedule), indent=2), encoding="utf-8")

    print(
        f"OK: bc_ppo_greedy_makespan={final_result.makespan} "
        f"best_seen={best_makespan}@ep{best_episode} model={args.model_out}"
    )


if __name__ == "__main__":
    main()
