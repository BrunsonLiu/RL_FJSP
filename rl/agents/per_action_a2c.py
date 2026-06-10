"""Per-action A2C agent with GAE advantage and value head.

The policy is identical to PA-REINFORCE (a small MLP over 8 handcrafted
per-action features). The value head is a separate MLP over the global
state (job ready times, machine ready times, makespan, remaining
operations). The advantage is computed by GAE with γ=0.99, λ=0.95.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn

from fjsp.env import DispatchAction, FJSPDispatchEnv
from fjsp.scheduler.validator import validate_schedule
from fjsp.utils.scaling import instance_time_scale, scale_value
from rl.models.action_scorer import ActionScorer


FEATURE_DIM = 8


@dataclass(frozen=True)
class StepRecord:
    features: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float


@dataclass(frozen=True)
class EpisodeResult:
    makespan: int
    is_valid: bool
    log_prob_sum: torch.Tensor
    value_sum: torch.Tensor
    return_: torch.Tensor


def action_features(
    env: FJSPDispatchEnv,
    action: DispatchAction,
    *,
    scale: int | None = None,
) -> list[float]:
    op_idx = env.job_next_op[action.job]
    job = env.instance.jobs[action.job]
    operation = job.operations[op_idx]
    duration = next(option.duration for option in operation.options if option.machine == action.machine)
    start = max(env.job_ready_time[action.job], env.machine_ready_time[action.machine])
    end = start + duration
    scale = instance_time_scale(env) if scale is None else scale
    return [
        scale_value(env.job_ready_time[action.job], scale),
        scale_value(env.machine_ready_time[action.machine], scale),
        scale_value(duration, scale),
        scale_value(start, scale),
        scale_value(end, scale),
        scale_value(op_idx, max(len(job.operations) - 1, 1)),
        scale_value(len(job.operations) - op_idx - 1, max(len(job.operations), 1)),
        scale_value(env.makespan, scale),
    ]


def global_state_features(env: FJSPDispatchEnv) -> list[float]:
    """Compute a fixed-size global state feature vector.

    Includes the mean and max of job ready times, mean and max of machine
    ready times, the makespan, the number of remaining operations, and
    the total workload (sum of remaining processing times on the critical
    machine). All scaled by the instance time scale.
    """
    scale = instance_time_scale(env)
    jr = env.job_ready_time
    mr = env.machine_ready_time
    rem = env.remaining_operations
    return [
        scale_value(sum(jr) / max(len(jr), 1), scale),
        scale_value(max(jr, default=0), scale),
        scale_value(sum(mr) / max(len(mr), 1), scale),
        scale_value(max(mr, default=0), scale),
        scale_value(env.makespan, scale),
        scale_value(rem, max(env.instance.operation_count, 1)),
        scale_value(max(jr, default=0) - min(jr, default=0), scale),
        scale_value(max(mr, default=0) - min(mr, default=0), scale),
    ]


GLOBAL_STATE_DIM = 8


class PerActionA2CAgent:
    def __init__(
        self,
        policy: ActionScorer,
        value_net: nn.Module,
        *,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.device = torch.device(device)

    @classmethod
    def create(
        cls,
        *,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> "PerActionA2CAgent":
        policy = ActionScorer(FEATURE_DIM, hidden_dim=hidden_dim)
        value_net = nn.Sequential(
            nn.Linear(GLOBAL_STATE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        return cls(policy, value_net, device=device)

    def select_action(
        self,
        env: FJSPDispatchEnv,
        *,
        greedy: bool,
        rng: Random | None = None,
    ) -> tuple[DispatchAction, torch.Tensor | None, torch.Tensor]:
        actions = env.available_actions()
        if not actions:
            raise ValueError("No valid actions available.")
        scale = instance_time_scale(env)
        features = torch.tensor(
            [action_features(env, action, scale=scale) for action in actions],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.policy(features)
        state = torch.tensor(
            global_state_features(env),
            dtype=torch.float32,
            device=self.device,
        )
        value = self.value_net(state).squeeze(-1)
        if greedy:
            index = int(torch.argmax(logits).item())
            return actions[index], None, value
        from torch.distributions import Categorical
        distribution = Categorical(logits=logits)
        if rng is not None:
            torch.manual_seed(rng.randrange(0, 2**31 - 1))
        sample = distribution.sample()
        index = int(sample.item())
        return actions[index], distribution.log_prob(sample), value

    def rollout(
        self,
        env: FJSPDispatchEnv,
        *,
        greedy: bool = False,
        seed: int | None = None,
    ) -> tuple[int, list[StepRecord], list[torch.Tensor], bool]:
        """Returns (makespan, step_records, values, is_valid)."""
        env.reset()
        rng = Random(seed) if seed is not None else None
        step_records: list[StepRecord] = []
        total_reward = 0.0
        last_value = torch.tensor(0.0, device=self.device)
        while not env.done:
            actions = env.available_actions()
            if not actions:
                break
            scale = instance_time_scale(env)
            features_all = torch.tensor(
                [action_features(env, a, scale=scale) for a in actions],
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.policy(features_all)
            state = torch.tensor(
                global_state_features(env),
                dtype=torch.float32,
                device=self.device,
            )
            value = self.value_net(state).squeeze(-1)
            if greedy:
                index = int(torch.argmax(logits).item())
                log_prob = None
            else:
                from torch.distributions import Categorical
                distribution = Categorical(logits=logits)
                if rng is not None:
                    torch.manual_seed(rng.randrange(0, 2**31 - 1))
                sample = distribution.sample()
                index = int(sample.item())
                log_prob = distribution.log_prob(sample)
            action = actions[index]
            chosen_features = features_all[index]
            _, reward, _, _ = env.step(action)
            step_reward = float(reward)
            total_reward += step_reward
            if log_prob is not None:
                step_records.append(
                    StepRecord(
                        features=chosen_features,
                        log_prob=log_prob,
                        value=value,
                        reward=step_reward,
                    )
                )
            last_value = value
        validation = env.validate()
        return validation.makespan, step_records, [r.value for r in step_records] + [last_value], validation.is_valid


def train_a2c(
    env: FJSPDispatchEnv,
    *,
    episodes: int,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
) -> tuple[PerActionA2CAgent, list[dict[str, float]]]:
    torch.manual_seed(seed)
    rng = Random(seed)
    agent = PerActionA2CAgent.create(hidden_dim=hidden_dim, device=device)
    params = list(agent.policy.parameters()) + list(agent.value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    history: list[dict[str, float]] = []
    best_makespan: int | None = None
    best_episode = 0
    best_state: dict[str, torch.Tensor] | None = None

    for episode in range(1, episodes + 1):
        makespan, step_records, values, _is_valid = agent.rollout(env, greedy=False, seed=rng.randrange(0, 2**31 - 1))
        if not _is_valid or not step_records:
            continue
        # Compute GAE
        rewards = [r.reward for r in step_records]
        advantages: list[torch.Tensor] = []
        gae = torch.tensor(0.0, device=agent.device)
        next_value = torch.tensor(0.0, device=agent.device)
        for t in reversed(range(len(step_records))):
            td = rewards[t] + gamma * next_value - step_records[t].value
            gae = td + gamma * gae_lambda * gae
            advantages.append(gae)
            next_value = step_records[t].value
        advantages.reverse()
        advantages_t = torch.stack(advantages)
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        # Returns for value loss
        returns_t = advantages_t + torch.stack([r.value for r in step_records])

        log_probs = torch.stack([r.log_prob for r in step_records])
        values_t = torch.stack([r.value for r in step_records])

        policy_loss = -(log_probs * advantages_t.detach()).sum()
        value_loss = (values_t - returns_t.detach()).pow(2).sum()
        # Entropy bonus (we use a uniform prior over actions; for now skip)
        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if episode == 1 or episode == episodes or episode % max(1, episodes // 10) == 0:
            greedy_makespan, _, _, _ = agent.rollout(env, greedy=True)
            if best_makespan is None or greedy_makespan < best_makespan:
                best_makespan = greedy_makespan
                best_episode = episode
                best_state = {
                    "policy": deepcopy(agent.policy.state_dict()),
                    "value": deepcopy(agent.value_net.state_dict()),
                }
            history.append({
                "episode": float(episode),
                "sample_makespan": float(makespan),
                "greedy_makespan": float(greedy_makespan),
                "best_greedy_makespan": float(best_makespan),
                "best_episode": float(best_episode),
                "loss": float(loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
            })

    if best_state is not None:
        agent.policy.load_state_dict(best_state["policy"])
        agent.value_net.load_state_dict(best_state["value"])
    return agent, history
