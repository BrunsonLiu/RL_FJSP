"""HGT-PPO: Proximal Policy Optimization with GAE on the HGT-FJSP agent.

The training loop:
  1. Collect a stochastic rollout of the current policy, storing
     (state_snapshot, action, log_prob_old, value_old, reward).
  2. Compute GAE advantages and returns from the per-step rewards and
     values.
  3. For K epochs, restore each state, re-evaluate the action under the
     *current* policy, and apply the clipped PPO objective:
        L = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
            + value_coef * MSE(V(s), R)
            - entropy_coef * H(pi)
     where ratio = exp(log_prob_new - log_prob_old).
  4. Save the best greedy-eval checkpoint.

The state-snapshot + restore approach mirrors
``rl.agents.graph_ppo.GraphPPOAgent``. We re-build the HGT graph on the
fly from the snapshot, so the policy is always evaluated on the *current*
parameters, not the ones that produced the rollout.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Optional

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.scheduler.validator import ScheduledOperation, schedule_to_dict
from fjsp.utils.scaling import instance_time_scale
from rl.agents.hgt_fjsp import HGTActorCriticAgent


ROOT = Path(__file__).resolve().parents[1]


def _snapshot_env(env: FJSPDispatchEnv) -> tuple:
    # ``makespan`` is a derived property, so we only snapshot the mutable state
    # that actually drives the schedule.
    return (
        list(env.job_next_op),
        list(env.job_ready_time),
        list(env.machine_ready_time),
        int(env.remaining_operations),
        [ScheduledOperation(**vars(op)) for op in env._schedule],
    )


def _restore_env(env: FJSPDispatchEnv, snapshot: tuple) -> None:
    # ``makespan`` and ``done`` are derived; do not assign them.
    env.job_next_op = list(snapshot[0])
    env.job_ready_time = list(snapshot[1])
    env.machine_ready_time = list(snapshot[2])
    env.remaining_operations = int(snapshot[3])
    env._schedule = [ScheduledOperation(**vars(op)) for op in snapshot[4]]


def _collect_rollout(
    env: FJSPDispatchEnv,
    agent: HGTActorCriticAgent,
    *,
    rng: Random,
    reward_shaping: str = "delta_makespan",
) -> tuple[list[tuple], int, bool]:
    """Returns (steps, makespan, is_valid). Each step is
    (snapshot, action, log_prob_old, value_old)."""
    env.reset()
    steps: list[tuple] = []
    prev_makespan = 0
    while not env.done:
        snap = _snapshot_env(env)
        try:
            action, log_prob, value, entropy = agent.select_action(env, greedy=False, rng=rng)
        except (RuntimeError, ValueError):
            return steps, env.makespan, False
        steps.append((snap, action, log_prob.detach(), value.detach(), entropy.detach()))
        env.step(action)
        new_makespan = env.makespan
        # Reward shaping.
        delta = new_makespan - prev_makespan
        prev_makespan = new_makespan
        scale = float(instance_time_scale(env))
        if reward_shaping == "delta_makespan":
            r = -float(delta) / scale
        elif reward_shaping == "delta_makespan_idle":
            # Sum of new machine-idle times (over machines that became busy).
            idle = sum(
                max(env.machine_ready_time[m] - new_makespan, 0.0) for m in range(env.instance.machine_count)
            )
            r = -float(delta) / scale - 0.01 * (idle / scale)
        elif reward_shaping == "delta_makespan_idle_wait":
            # Sum of new job-wait times.
            wait = sum(env.job_ready_time[j] for j in range(env.instance.job_count))
            idle = sum(
                max(env.machine_ready_time[m] - new_makespan, 0.0) for m in range(env.instance.machine_count)
            )
            r = -float(delta) / scale - 0.005 * (idle / scale) - 0.001 * (wait / scale)
        elif reward_shaping == "terminal":
            r = 0.0
        else:
            raise ValueError(f"Unknown reward_shaping={reward_shaping}")
        # Add a per-step terminal reward component at the very end.
        if env.done:
            r += -float(env.makespan - prev_makespan) / scale
        # Append reward to the last step (we don't store it per-step in `steps`).
        steps[-1] = (snap, action, log_prob.detach(), value.detach(), entropy.detach(), r)
    return steps, env.makespan, True


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE. ``last_value`` is V(s_{T}) (0 if terminal else bootstrap)."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0  # we treat all steps as non-terminal (reward shaping already accounts for terminal)
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[t] = last_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    env: FJSPDispatchEnv,
    agent: HGTActorCriticAgent,
    steps: list[tuple],
    *,
    clip_ratio: float,
    gamma: float,
    gae_lambda: float,
    K_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
) -> dict[str, float]:
    # Build tensors from the rollout. ``values`` come from the critic as shape
    # (1,) per step, so we flatten to (n,) to keep shapes clean.
    rewards = torch.tensor([s[5] for s in steps], dtype=torch.float32, device=agent.device)
    log_probs_old = torch.stack([s[2] for s in steps]).reshape(-1).to(agent.device)
    values = torch.stack([s[3] for s in steps]).reshape(-1).to(agent.device)
    entropies_old = torch.stack([s[4] for s in steps]).reshape(-1).to(agent.device)

    # GAE. Treat the rollout as one episode (non_episodic value bootstrap = 0).
    advantages, returns = _compute_gae(rewards, values, torch.zeros((), device=agent.device), gamma=gamma, gae_lambda=gae_lambda)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(steps)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0.0}

    for _ in range(K_epochs):
        order = list(range(n))
        Random(sum(hash(s[1]) for s in steps) + id(steps)).shuffle(order)
        for start in range(0, n, minibatch_size):
            mb = order[start:start + minibatch_size]

            new_log_probs: list[torch.Tensor] = []
            new_values: list[torch.Tensor] = []
            new_entropies: list[torch.Tensor] = []
            for idx in mb:
                snap, action, _, _, _, _ = steps[idx]
                _restore_env(env, snap)
                new_lp, new_v, new_ent = agent.evaluate_action(env, action)
                new_log_probs.append(new_lp)
                new_values.append(new_v)
                new_entropies.append(new_ent)
            new_lp_t = torch.stack(new_log_probs)
            new_v_t = torch.stack(new_values).reshape(-1)
            new_ent_t = torch.stack(new_entropies)

            ratio = torch.exp(new_lp_t - log_probs_old[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_v_t, returns[mb])
            entropy_loss = -new_ent_t.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=1.0)
            agent.optimizer.step()

            metrics["policy_loss"] += float(policy_loss.detach().cpu().item())
            metrics["value_loss"] += float(value_loss.detach().cpu().item())
            metrics["entropy"] += float(new_ent_t.mean().detach().cpu().item())
            metrics["n_updates"] += 1.0

    n_up = max(1.0, metrics["n_updates"])
    for k in ("policy_loss", "value_loss", "entropy"):
        metrics[k] /= n_up
    return metrics


def train_hgt_ppo(
    envs: list[FJSPDispatchEnv],
    instance_paths: list[Path],
    *,
    episodes: int,
    lr: float = 3e-4,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 256,
    dropout: float = 0.0,
    seed: int = 0,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    K_epochs: int = 4,
    minibatch_size: int = 32,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    init_model: str | Path | None = None,
    reward_shaping: str = "delta_makespan",
    use_instance_embed: bool = True,
    device: str = "cpu",
) -> tuple[HGTActorCriticAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    if init_model is not None:
        agent = HGTActorCriticAgent.load(init_model, hidden_dim=hidden_dim, device=device)
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
    agent.optimizer = torch.optim.Adam(agent.net.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str,torch.Tensor] | None = None

    # Build a stable per-instance id mapping so the embedding lookup is always
    # in-range. The order matches ``envs``/``instance_paths``.
    instance_id_map: dict[str, int] = {}
    if use_instance_embed:
        for path in instance_paths:
            stem = path.stem
            if stem not in instance_id_map:
                instance_id_map[stem] = len(instance_id_map)

    def _instance_id_for_path(path: Path) -> int:
        if not use_instance_embed:
            return 0
        return instance_id_map.get(path.stem, 0)

    for episode in range(1, episodes + 1):
        idx = rng.randrange(len(envs))
        env = envs[idx]
        if use_instance_embed:
            agent.instance_id = _instance_id_for_path(instance_paths[idx])
        steps, makespan, is_valid = _collect_rollout(env, agent, rng=Random(rng.randrange(0, 2**31 - 1)), reward_shaping=reward_shaping)
        if not is_valid or not steps:
            raise RuntimeError("HGT-PPO rollout produced an invalid or empty episode.")
        metrics = _ppo_update(
            env, agent, steps,
            clip_ratio=clip_ratio, gamma=gamma, gae_lambda=gae_lambda,
            K_epochs=K_epochs, minibatch_size=minibatch_size,
            value_coef=value_coef, entropy_coef=entropy_coef,
        )
        if episode == 1 or episode == episodes or episode % max(1, episodes // 20) == 0:
            greedy_result = agent.rollout(env, greedy=True)
            if best_makespan is None or greedy_result.makespan < best_makespan:
                best_makespan = greedy_result.makespan
                best_episode = episode
                best_state = deepcopy(agent.net.state_dict())
            history.append(
                {
                    "episode": float(episode),
                    "instance": instance_paths[idx].stem,
                    "sample_makespan": float(makespan),
                    "greedy_makespan": float(greedy_result.makespan),
                    "best_greedy_makespan": float(best_makespan),
                    "best_episode": float(best_episode),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "n_ppo_updates": metrics["n_updates"],
                }
            )
            print(
                f"[HGT-PPO] ep {episode:3d}/{episodes} | inst={instance_paths[idx].stem} | "
                f"sample={makespan} greedy={greedy_result.makespan} best={best_makespan} "
                f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )

    if best_state is not None:
        agent.net.load_state_dict(best_state)
    return agent, history, instance_id_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an HGT-FJSP policy with PPO+GAE.")
    _root = ROOT
    parser.add_argument("--instances", nargs="+", default=[str(_root / "data" / "instances" / "brandimarte" / "mk01.txt")])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--K-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--reward-shaping", choices=["terminal", "delta_makespan", "delta_makespan_idle", "delta_makespan_idle_wait"], default="delta_makespan")
    parser.add_argument("--no-instance-embed", action="store_true")
    parser.add_argument("--model-out", default=str(_root / "data" / "results" / "hgt_ppo_mk01_best.pt"))
    parser.add_argument("--history-out", default=str(_root / "data" / "results" / "hgt_ppo_mk01_history.json"))
    parser.add_argument("--schedule-out", default=str(_root / "data" / "results" / "hgt_ppo_mk01_best_schedule.json"))
    args = parser.parse_args()

    instance_paths = [Path(p) for p in args.instances]
    envs = [FJSPDispatchEnv.from_file(p) for p in instance_paths]
    agent, history, instance_id_map = train_hgt_ppo(
        envs, instance_paths,
        episodes=args.episodes,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        seed=args.seed,
        clip_ratio=args.clip_ratio,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        K_epochs=args.K_epochs,
        minibatch_size=args.minibatch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        init_model=args.init_model,
        reward_shaping=args.reward_shaping,
        use_instance_embed=not args.no_instance_embed,
    )

    # Save + emit schedule on the first instance.
    primary = instance_paths[0]
    eval_env = FJSPDispatchEnv.from_file(primary)
    if not args.no_instance_embed:
        agent.instance_id = instance_id_map.get(primary.stem, 0)
    result = agent.rollout(eval_env, greedy=True)
    if not result.is_valid:
        raise SystemExit("HGT-PPO policy produced an invalid final schedule.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_out)

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_out).write_text(json.dumps(history, indent=2), encoding="utf-8")

    schedule_path = Path(args.schedule_out)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule_to_dict(eval_env.schedule), indent=2), encoding="utf-8")

    print(f"OK: hgt_ppo_greedy_makespan={result.makespan} on {primary.name} | model={args.model_out}")


if __name__ == "__main__":
    main()
