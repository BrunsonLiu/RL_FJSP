"""HGT-A2C: A2C-style synchronous actor-critic for the HGT-FJSP agent.

This mirrors ``rl.agents.graph_actor_critic.train_graph_actor_critic``: a
single stochastic rollout per update, a sparse return of ``-makespan`` at
the final step, and a REINFORCE-with-baseline policy gradient with
advantage = (return - value). It is intentionally simple (no PPO clip,
no GAE) and has been shown to work well for the simpler
``GraphTwoStageActorCriticNet`` from a BC init.

The HGT version supports:
  * an optional init from a BC-pretrained checkpoint,
  * cross-instance training over a list of envs.
"""
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


def train_hgt_a2c(
    envs: list[FJSPDispatchEnv],
    instance_paths: list[Path],
    *,
    episodes: int,
    lr: float = 3e-4,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 128,
    dropout: float = 0.0,
    seed: int = 0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    init_model: str | Path | None = None,
    use_instance_embed: bool = True,
    device: str = "cpu",
    log_every: int = 10,
) -> tuple[HGTActorCriticAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    if init_model is not None:
        agent = HGTActorCriticAgent.load(
            init_model,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            device=device,
        )
    else:
        num_known = max(1, len(envs)) if use_instance_embed else 0
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
    best_ie: dict[str, torch.Tensor] | None = None

    instance_id_map: dict[str, int] = (
        {p.stem: i for i, p in enumerate(instance_paths)} if use_instance_embed else {}
    )

    def _instance_id_for_path(path: Path) -> int:
        if not use_instance_embed:
            return 0
        return instance_id_map.get(path.stem, 0)

    for episode in range(1, episodes + 1):
        idx = rng.randrange(len(envs))
        env = envs[idx]
        if use_instance_embed:
            agent.instance_id = _instance_id_for_path(instance_paths[idx])
        result = agent.rollout(env, greedy=False, seed=rng.randrange(0, 2**31 - 1))
        if not result.is_valid or not result.log_probs:
            raise RuntimeError("HGT-A2C rollout produced an invalid or empty episode.")
        # Sparse return: same negative normalised makespan at every step.
        scale = float(instance_time_scale(env))
        returns = torch.full(
            (len(result.log_probs),),
            -float(result.makespan) / scale,
            dtype=torch.float32,
            device=agent.device,
        )
        log_probs = torch.stack(result.log_probs)
        values = torch.stack(result.values).reshape(-1)
        entropies = torch.stack(result.entropies).reshape(-1)
        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=1.0)
        optimizer.step()

        if (
            episode == 1
            or episode == episodes
            or episode % log_every == 0
        ):
            greedy_result = agent.rollout(env, greedy=True, seed=0)
            if best_makespan is None or greedy_result.makespan < best_makespan:
                best_makespan = greedy_result.makespan
                best_episode = episode
                best_state = deepcopy(agent.net.state_dict())
                best_ie = (
                    deepcopy(agent.instance_embed.state_dict())
                    if agent.instance_embed is not None
                    else None
                )
            history.append(
                {
                    "episode": float(episode),
                    "instance": instance_paths[idx].stem,
                    "sample_makespan": float(result.makespan),
                    "greedy_makespan": float(greedy_result.makespan),
                    "best_greedy_makespan": float(best_makespan),
                    "best_episode": float(best_episode),
                    "actor_loss": float(actor_loss.detach().cpu().item()),
                    "critic_loss": float(critic_loss.detach().cpu().item()),
                    "entropy": float(entropies.mean().detach().cpu().item()),
                }
            )
            print(
                f"[HGT-A2C] ep {episode:3d}/{episodes} | inst={instance_paths[idx].stem} | "
                f"sample={result.makespan} greedy={greedy_result.makespan} best={best_makespan} "
                f"actor={actor_loss.item():.4f} critic={critic_loss.item():.4f} "
                f"entropy={entropies.mean().item():.4f}"
            )

    if best_state is not None:
        agent.net.load_state_dict(best_state)
        if best_ie is not None and agent.instance_embed is not None:
            agent.instance_embed.load_state_dict(best_ie)
    return agent, history


def main() -> None:
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train an HGT-FJSP policy with A2C (REINFORCE-with-baseline).")
    parser.add_argument("--instances", nargs="+", default=[str(_root / "data" / "instances" / "brandimarte" / "mk01.txt")])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--no-instance-embed", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--model-out", default=str(_root / "data" / "results" / "hgt_a2c_mk01.pt"))
    parser.add_argument("--history-out", default=str(_root / "data" / "results" / "hgt_a2c_mk01_history.json"))
    parser.add_argument("--schedule-out", default=str(_root / "data" / "results" / "hgt_a2c_mk01_best_schedule.json"))
    args = parser.parse_args()

    instance_paths = [Path(p) for p in args.instances]
    envs = [FJSPDispatchEnv.from_file(p) for p in instance_paths]
    agent, history = train_hgt_a2c(
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
        log_every=args.log_every,
    )

    primary = instance_paths[0]
    eval_env = FJSPDispatchEnv.from_file(primary)
    if not args.no_instance_embed:
        agent.instance_id = abs(hash(primary.stem)) % max(1, len(instance_paths))
    result = agent.rollout(eval_env, greedy=True, seed=0)
    if not result.is_valid:
        raise SystemExit("HGT-A2C policy produced an invalid final schedule.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(history, indent=2), encoding="utf-8")
    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(eval_env.schedule), indent=2), encoding="utf-8")
    print(f"OK: hgt_a2c_greedy_makespan={result.makespan} on {primary.name} | model={args.model_out}")


if __name__ == "__main__":
    main()
