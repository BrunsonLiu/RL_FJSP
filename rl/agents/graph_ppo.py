from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.scheduler.validator import ScheduledOperation
from fjsp.utils.scaling import instance_time_scale
from rl.agents.graph_actor_critic import GraphTwoStageActorCriticAgent


@dataclass(frozen=True)
class PPOStep:
    state_snapshot: tuple
    action: DispatchAction
    log_prob_old: torch.Tensor
    value_old: torch.Tensor
    reward: float
    done: bool


def _snapshot_env(env: FJSPDispatchEnv) -> tuple:
    return (
        list(env.job_next_op),
        list(env.job_ready_time),
        list(env.machine_ready_time),
        int(env.remaining_operations),
        list(env._schedule),
    )


def _restore_env(env: FJSPDispatchEnv, snapshot: tuple) -> None:
    env.job_next_op = list(snapshot[0])
    env.job_ready_time = list(snapshot[1])
    env.machine_ready_time = list(snapshot[2])
    env.remaining_operations = int(snapshot[3])
    env._schedule = [ScheduledOperation(**vars(op)) for op in snapshot[4]]


class GraphPPOAgent(GraphTwoStageActorCriticAgent):
    """PPO variant of the graph two-stage actor-critic agent.

    Uses the same network architecture, save/load protocol, and rollout
    method as GraphTwoStageActorCriticAgent. Only the training loop differs.
    """


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[t] = last_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def _collect_rollout(
    env: FJSPDispatchEnv,
    agent: GraphPPOAgent,
    *,
    rng: Random,
) -> tuple[list[PPOStep], int, bool]:
    env.reset()
    steps: list[PPOStep] = []
    while not env.done:
        snapshot = _snapshot_env(env)
        action, log_prob, value, _ = agent.select_action(env, greedy=False, rng=rng)
        env.step(action)
        reward = -float(env.makespan) / float(instance_time_scale(env)) if env.done else 0.0
        steps.append(
            PPOStep(
                state_snapshot=snapshot,
                action=action,
                log_prob_old=log_prob.detach(),
                value_old=value.detach(),
                reward=reward,
                done=env.done,
            )
        )
    validation = env.validate()
    return steps, int(validation.makespan), bool(validation.is_valid)


def _ppo_update(
    env: FJSPDispatchEnv,
    agent: GraphPPOAgent,
    steps: list[PPOStep],
    *,
    clip_ratio: float,
    gamma: float,
    gae_lambda: float,
    K_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
) -> dict[str, float]:
    old_log_probs = torch.stack([s.log_prob_old for s in steps]).to(agent.device)
    old_values = torch.stack([s.value_old for s in steps]).to(agent.device)
    rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32, device=agent.device)
    dones = torch.tensor([float(s.done) for s in steps], dtype=torch.float32, device=agent.device)
    advantages, returns = _compute_gae(rewards, old_values, dones, gamma=gamma, gae_lambda=gae_lambda)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(steps)
    indices = list(range(n))

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(K_epochs):
        rng = Random(0)
        rng.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb = indices[start:start + minibatch_size]
            mb_log_probs_old = old_log_probs[mb]
            mb_advantages = advantages[mb]
            mb_returns = returns[mb]

            new_log_probs_list: list[torch.Tensor] = []
            new_values_list: list[torch.Tensor] = []
            new_entropy_list: list[torch.Tensor] = []
            for idx in mb:
                step = steps[idx]
                _restore_env(env, step.state_snapshot)
                log_prob, value, entropy = agent.evaluate_action(env, step.action)
                new_log_probs_list.append(log_prob)
                new_values_list.append(value)
                new_entropy_list.append(entropy)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            new_entropy = torch.stack(new_entropy_list)

            ratio = torch.exp(new_log_probs - mb_log_probs_old)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values, mb_returns)
            entropy_bonus = new_entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
            agent.optimizer.step()

            total_policy_loss += float(policy_loss.detach().cpu().item())
            total_value_loss += float(value_loss.detach().cpu().item())
            total_entropy += float(entropy_bonus.detach().cpu().item())
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, n_updates),
        "value_loss": total_value_loss / max(1, n_updates),
        "entropy": total_entropy / max(1, n_updates),
    }


def train_graph_ppo(
    env: FJSPDispatchEnv,
    *,
    episodes: int,
    lr: float = 3e-4,
    hidden_dim: int = 64,
    gnn_rounds: int = 2,
    seed: int = 0,
    clip_ratio: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    K_epochs: int = 4,
    minibatch_size: int = 32,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> tuple[GraphPPOAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    agent = GraphPPOAgent.create(hidden_dim=hidden_dim, gnn_rounds=gnn_rounds, device=device)
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(1, episodes + 1):
        steps, makespan, is_valid = _collect_rollout(
            env, agent, rng=Random(rng.randrange(0, 2**31 - 1))
        )
        if not is_valid or not steps:
            raise RuntimeError("Graph PPO rollout produced an invalid or empty episode.")
        metrics = _ppo_update(
            env,
            agent,
            steps,
            clip_ratio=clip_ratio,
            gamma=gamma,
            gae_lambda=gae_lambda,
            K_epochs=K_epochs,
            minibatch_size=minibatch_size,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        if episode == 1 or episode == episodes or episode % max(1, episodes // 10) == 0:
            greedy_result = agent.rollout(env, greedy=True)
            if best_makespan is None or greedy_result.makespan < best_makespan:
                best_makespan = greedy_result.makespan
                best_episode = episode
                best_state = deepcopy(agent.model.state_dict())
            history.append(
                {
                    "episode": float(episode),
                    "sample_makespan": float(makespan),
                    "greedy_makespan": float(greedy_result.makespan),
                    "best_greedy_makespan": float(best_makespan),
                    "best_episode": float(best_episode),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                }
            )

    if best_state is not None:
        agent.model.load_state_dict(best_state)
    return agent, history
