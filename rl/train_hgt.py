"""Training entry point for the HGT-FJSP agent (single instance or cross-instance)."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from random import Random

import torch
from torch import nn

from fjsp.env import FJSPDispatchEnv
from fjsp.scheduler.validator import schedule_to_dict
from fjsp.utils.scaling import instance_time_scale
from rl.agents.hgt_fjsp import HGTActorCriticAgent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES = [
    ROOT / "data" / "instances" / "brandimarte" / f"mk{i:02d}.txt" for i in range(1, 11)
]
DEFAULT_MODEL = ROOT / "data" / "results" / "hgt_mk01_best.pt"
DEFAULT_HISTORY = ROOT / "data" / "results" / "hgt_mk01_history.json"
DEFAULT_SCHEDULE = ROOT / "data" / "results" / "hgt_mk01_best_schedule.json"


def _instance_id_for_path(path: Path) -> int:
    """Stable integer ID for an instance file path (used as embedding key)."""
    return abs(hash(path.stem)) % 10_000


def train_hgt_actor_critic(
    envs: list[FJSPDispatchEnv],
    instance_paths: list[Path],
    *,
    episodes: int,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    num_blocks: int = 4,
    num_heads: int = 8,
    ffn_dim: int = 512,
    dropout: float = 0.1,
    seed: int = 0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    init_model: str | Path | None = None,
    device: str = "cpu",
    use_instance_embed: bool = True,
) -> tuple[HGTActorCriticAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    if init_model is not None:
        agent = HGTActorCriticAgent.load(init_model, hidden_dim=hidden_dim, device=device)
    else:
        num_known = len(envs) if use_instance_embed else 0
        agent = HGTActorCriticAgent.create(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_known_instances=num_known if use_instance_embed else None,
            device=device,
        )
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(1, episodes + 1):
        # Sample an instance uniformly for this episode.
        idx = rng.randrange(len(envs))
        env = envs[idx]
        if use_instance_embed:
            agent.instance_id = _instance_id_for_path(instance_paths[idx]) if init_model is None else _instance_id_for_path(instance_paths[idx])
        # Stochastic rollout.
        result = agent.rollout(env, greedy=False, seed=rng.randrange(0, 2**31 - 1))
        if not result.is_valid or not result.log_probs:
            raise RuntimeError("HGT actor-critic rollout produced an invalid or empty episode.")

        # Discounted returns over per-step rewards.
        gamma = 0.99
        returns: list[float] = []
        running = 0.0
        for r in reversed(result.rewards):
            running = r + gamma * running
            returns.insert(0, running)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=agent.device)
        # Advantage normalization.
        advantages = returns_t - (returns_t.mean() + 1e-8)
        if advantages.numel() > 1:
            advantages = advantages / (advantages.std() + 1e-8)

        log_probs = torch.stack(result.log_probs)
        values = torch.stack(result.values).reshape(-1)
        entropies = torch.stack(result.entropies)

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns_t)
        entropy_loss = -entropies.mean()
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=1.0)
        optimizer.step()

        if episode == 1 or episode == episodes or episode % max(1, episodes // 10) == 0:
            # Greedy eval on the same instance for monitoring.
            greedy_result = agent.rollout(env, greedy=True)
            greedy_makespan = greedy_result.makespan
            if best_makespan is None or greedy_makespan < best_makespan:
                best_makespan = greedy_makespan
                best_episode = episode
                best_state = deepcopy(agent.net.state_dict())
            history.append(
                {
                    "episode": float(episode),
                    "instance": instance_paths[idx].stem,
                    "sample_makespan": float(result.makespan),
                    "greedy_makespan": float(greedy_makespan),
                    "best_greedy_makespan": float(best_makespan),
                    "best_episode": float(best_episode),
                    "actor_loss": float(actor_loss.detach().cpu().item()),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "entropy": float(entropies.mean().detach().cpu().item()),
                }
            )
            print(
                f"[HGT] ep {episode:3d}/{episodes} | inst={instance_paths[idx].stem} | "
                f"sample={result.makespan} greedy={greedy_makespan} best={best_makespan} "
                f"actor={float(actor_loss):.4f} critic={float(critic_loss):.4f} "
                f"entropy={float(entropies.mean()):.4f}"
            )

    if best_state is not None:
        agent.net.load_state_dict(best_state)
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an HGT-FJSP actor-critic policy.")
    parser.add_argument("--instances", nargs="+", default=[str(p) for p in DEFAULT_INSTANCES])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--no-instance-embed", action="store_true")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL))
    parser.add_argument("--history-out", default=str(DEFAULT_HISTORY))
    parser.add_argument("--schedule-out", default=str(DEFAULT_SCHEDULE))
    args = parser.parse_args()

    instance_paths = [Path(p) for p in args.instances]
    envs = [FJSPDispatchEnv.from_file(p) for p in instance_paths]
    agent, history = train_hgt_actor_critic(
        envs, instance_paths,
        episodes=args.episodes,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        seed=args.seed,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        init_model=args.init_model,
        use_instance_embed=not args.no_instance_embed,
    )

    # Save the best model on the first instance (single-instance eval).
    primary = instance_paths[0]
    eval_env = FJSPDispatchEnv.from_file(primary)
    if not args.no_instance_embed:
        agent.instance_id = _instance_id_for_path(primary)
    result = agent.rollout(eval_env, greedy=True)
    if not result.is_valid:
        raise SystemExit("HGT policy produced an invalid final schedule.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(history, indent=2), encoding="utf-8")

    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(eval_env.schedule), indent=2), encoding="utf-8")

    print(
        f"OK: hgt_greedy_makespan={result.makespan} on {primary.name} | "
        f"model={args.model_out}"
    )


if __name__ == "__main__":
    main()
